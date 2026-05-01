import sys
import argparse
import pandas as pd
import numpy as np
import kornia.augmentation as K
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import warnings
import os
import csv
from collections import Counter
warnings.filterwarnings('ignore')

# ПАРАМЕТРЫ
parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=str, required=True, help='путь к csv с колонками image_path, age, gender, bite')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--patience', type=int, default=10, help='остановка обучения если val_loss не улучшалась 10 эпох')
parser.add_argument('--output_dir', type=str, default='./outputs', help='папка для сохранения модели и метрик')
parser.add_argument('--use_wandb', action='store_true', help='логировать в wandb')
parser.add_argument('--wandb_project', type=str, default='dental_classification', help='название проекта в WandB')
parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity (username или team name)')
parser.add_argument('--weights_path', type=str, default='./efficientnet_b0_weights.pth',
                    help='путь к файлу весов EfficientNet-B0 (можно скачать с https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth)')
args = parser.parse_args()

# инициализация wandb (если используется)
if args.use_wandb:
    import wandb

    wandb_key = None

    # проверяем файл с ключом в текущей директории
    if os.path.exists('.wandb_key'):
        with open('.wandb_key', 'r') as f:
            wandb_key = f.read().strip()
            print("Используем WandB ключ из файла .wandb_key")
    # проверяем файл в домашней директории (дополнительно)
    elif os.path.exists(os.path.expanduser('~/.wandb_key')):
        with open(os.path.expanduser('~/.wandb_key'), 'r') as f:
            wandb_key = f.read().strip()
            print("Используем WandB ключ из файла ~/.wandb_key")

    if wandb_key:
        os.environ['WANDB_API_KEY'] = wandb_key
        wandb.login()
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
    else:
        print("WandB ключ не найден => Логирование в WandB отключено")
        args.use_wandb = False

os.makedirs(args.output_dir, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ДАННЫЕ
def age_to_group(age): # 0 (0-20), 1 (21-44), 2 (45+)
    if age <= 20: return 0
    elif age <= 44: return 1
    else: return 2

# загрузка данных
df = pd.read_csv(args.csv)
df = df.dropna(subset=['age', 'gender', 'bite']).reset_index(drop=True)
print(f"В таблице: {len(df)} образцов")

df['age_group'] = df['age'].apply(age_to_group).astype(int)
df['gender_bin'] = df['gender'].map({'M': 1, 'F': 0})
df['bite_class'] = df['bite'] - 1

print("Распределение возрастных групп:\n", df['age_group'].value_counts().sort_index())

# разбиение на train/val/test по id пациента со стратификацией по возрасту (если возможно)
df['patient_id'] = df['image_path'].str.split('/').str[-1].str.split('_').str[0]

df['age_group_stratify'] = df['age_group'] # временная колонка для стратификации
patients = df.groupby('patient_id').first().reset_index() # группируем по пациентам для стратификации

print("Распределение пациентов по возрастным группам (для стратификации):")
print(patients['age_group_stratify'].value_counts().sort_index())

train_patients, temp_patients = train_test_split(
    patients['patient_id'],
    test_size=0.2,
    random_state=42,
    stratify=patients['age_group_stratify']
)

temp_patients_df = patients[patients['patient_id'].isin(temp_patients)]
temp_age_counts = temp_patients_df['age_group_stratify'].value_counts()

if (temp_age_counts >= 2).all(): # если в каком-то классе меньше 2 образцов, стратификацию не используем
    print("Используем стратификацию для разбиения val/test")
    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=0.5,
        random_state=42,
        stratify=temp_patients_df['age_group_stratify']
    )
else:
    print(f"Недостаточно образцов для стратификации. Распределение: {temp_age_counts.to_dict()}")
    print(" => Используем разбиение без стратификации")
    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=0.5,
        random_state=42
    )

train_df = df[df['patient_id'].isin(train_patients)]
val_df   = df[df['patient_id'].isin(val_patients)]
test_df  = df[df['patient_id'].isin(test_patients)]

# удаляем временную колонку
train_df = train_df.drop(columns=['age_group_stratify'])
val_df = val_df.drop(columns=['age_group_stratify'])
test_df = test_df.drop(columns=['age_group_stratify'])

# проверка отсутствия пересечений
train_ids = set(train_df['patient_id'])
val_ids   = set(val_df['patient_id'])
test_ids  = set(test_df['patient_id'])
print(f"Пересечение train-val: {len(train_ids & val_ids)} (должно быть 0)")
print(f"Пересечение train-test: {len(train_ids & test_ids)} (должно быть 0)")
print(f"Пересечение val-test: {len(val_ids & test_ids)} (должно быть 0)")

train_df.to_csv('train.csv', index=False)
val_df.to_csv('valid.csv', index=False)
print(f"Сохранены train.csv ({len(train_df)} записей) и valid.csv ({len(val_df)} записей)")

print(f"Train: {len(train_df)} (пациентов: {len(train_ids)}), Val: {len(val_df)} (пациентов: {len(val_ids)}), Test: {len(test_df)} (пациентов: {len(test_ids)})")

# WeightedRandomSampler для балансировки возраста
class_counts = train_df['age_group'].value_counts().sort_index().values
class_weights = 1.0 / class_counts
sample_weights = train_df['age_group'].map(lambda x: class_weights[x]).values
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)
print("Веса классов для сэмплера:", dict(zip(train_df['age_group'].unique(), class_weights)))

# веса классов для функций потерь
age_classes = np.array(sorted(train_df['age_group'].unique()))
age_weights = compute_class_weight('balanced', classes=age_classes, y=train_df['age_group'].values)
age_weight_tensor = torch.tensor(age_weights, dtype=torch.float32).to(DEVICE)

bite_classes = np.array(sorted(train_df['bite_class'].unique()))
bite_weights = compute_class_weight('balanced', classes=bite_classes, y=train_df['bite_class'].values)
bite_weight_tensor = torch.tensor(bite_weights, dtype=torch.float32).to(DEVICE)

print("Веса возраста:", age_weight_tensor)
print("Веса прикуса:", bite_weight_tensor)

# ДАТАСЕТ
class DentalDataset(Dataset):
    def __init__(self, df, transform=None, preload=False, use_gpu_augment=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.preload = preload
        self.use_gpu_augment = use_gpu_augment
        self.device = DEVICE

        if self.preload:
            self.images = []
            for idx in tqdm(range(len(self.df)), desc="Preloading images"):
                img = Image.open(self.df.iloc[idx]['image_path']).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                self.images.append(img)

        if self.use_gpu_augment:
            self.gpu_aug = K.AugmentationSequential(
                K.RandomHorizontalFlip(p=0.5),
                K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.8),
                data_keys=["input"],
                same_on_batch=False,
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.preload:
            img = self.images[idx]
        else:
            img = Image.open(row['image_path']).convert('RGB')
            if self.transform:
                img = self.transform(img)

        if self.use_gpu_augment:
            img = img.unsqueeze(0).to(self.device)
            img = self.gpu_aug(img)
            img = img.squeeze(0).cpu()

        age = torch.tensor(row['age_group'], dtype=torch.long)
        gender = torch.tensor(row['gender_bin'], dtype=torch.float32)
        bite = torch.tensor(row['bite_class'], dtype=torch.long)
        return img, age, gender, bite

preload_transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_ds = DentalDataset(train_df, transform=preload_transform, preload=True, use_gpu_augment=True)
val_ds   = DentalDataset(val_df, transform=preload_transform, preload=True, use_gpu_augment=False)
test_ds  = DentalDataset(test_df, transform=preload_transform, preload=True, use_gpu_augment=False)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

print(f"Батчи трейна: {len(train_loader)}, Батчи валидации: {len(val_loader)}, Батчи теста: {len(test_loader)}")

# МОДЕЛЬ
class MultiTaskModel(nn.Module):
    def __init__(self, num_age_classes=3, weights_path=None):
        super().__init__()

        if weights_path is None:
            weights_path = args.weights_path
        # пытаемся загрузить веса
        if os.path.exists(weights_path):
            print(f"Загружаем веса EfficientNet-B0 из файла: {weights_path}")
            backbone = models.efficientnet_b0(weights=None)
            state_dict = torch.load(weights_path, map_location='cpu')
            backbone.load_state_dict(state_dict)
            print("Веса успешно загружены")
        else:
            # если файл не найден:
            print(f"Файл с весами не найден: {weights_path}\n")
            print("Пожалуйста, скачайте предобученные веса EfficientNet-B0:")
            print("  wget https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth")
            print(f"  mv efficientnet_b0_rwightman-3dd342df.pth {weights_path}\n")
            print("Или укажите правильный путь через аргумент --weights_path")
            raise FileNotFoundError(f"Файл с весами не найден: {weights_path}")

        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.age_head = nn.Linear(in_features, num_age_classes)
        self.gender_head = nn.Linear(in_features, 1)
        self.bite_head = nn.Linear(in_features, 3)
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        age_out = self.age_head(features)
        gender_out = self.gender_head(features).squeeze()
        bite_out = self.bite_head(features)
        return age_out, gender_out, bite_out

model = MultiTaskModel(num_age_classes=3, weights_path=args.weights_path).to(DEVICE)

criterion_age = nn.CrossEntropyLoss(weight=age_weight_tensor)
criterion_gender = nn.BCEWithLogitsLoss()
criterion_bite = nn.CrossEntropyLoss(weight=bite_weight_tensor)

optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# логирование и метрики
if args.use_wandb:
    import wandb
    wandb.init(project="dental_classification", config=vars(args))

metrics_file = os.path.join(args.output_dir, "metrics.csv")
with open(metrics_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_loss', 'val_loss',
                     'age_acc', 'age_precision', 'age_recall', 'age_logloss', 'age_auc',
                     'gender_acc', 'gender_precision', 'gender_recall', 'gender_logloss', 'gender_auc',
                     'bite_acc', 'bite_precision', 'bite_recall', 'bite_logloss', 'bite_auc'])

best_metric = -float('inf')
patience_counter = 0

for epoch in range(1, args.epochs + 1):
    model.train()
    train_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for images, ages, genders, bites in pbar:
        images = images.to(DEVICE)
        ages = ages.to(DEVICE)
        genders = genders.to(DEVICE)
        bites = bites.to(DEVICE)
        optimizer.zero_grad()
        age_out, gender_out, bite_out = model(images)
        loss = criterion_age(age_out, ages) + criterion_gender(gender_out, genders) + criterion_bite(bite_out, bites)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    age_logits, age_trues = [], []
    gender_logits, gender_trues = [], []
    bite_logits, bite_trues = [], []
    with torch.no_grad():
        for images, ages, genders, bites in val_loader:
            images = images.to(DEVICE)
            ages = ages.to(DEVICE)
            genders = genders.to(DEVICE)
            bites = bites.to(DEVICE)
            age_out, gender_out, bite_out = model(images)
            loss = criterion_age(age_out, ages) + criterion_gender(gender_out, genders) + criterion_bite(bite_out, bites)
            val_loss += loss.item()
            age_logits.append(age_out.cpu())
            age_trues.append(ages.cpu())
            gender_logits.append(gender_out.cpu())
            gender_trues.append(genders.cpu())
            bite_logits.append(bite_out.cpu())
            bite_trues.append(bites.cpu())
    avg_val_loss = val_loss / len(val_loader)

    # Метрики
    age_probs = torch.softmax(torch.cat(age_logits), dim=1).numpy()
    age_true = torch.cat(age_trues).numpy()
    age_pred = np.argmax(age_probs, axis=1)
    age_acc = accuracy_score(age_true, age_pred)
    age_prec = precision_score(age_true, age_pred, average='macro', zero_division=0)
    age_rec = recall_score(age_true, age_pred, average='macro', zero_division=0)
    age_logloss = log_loss(age_true, age_probs, labels=[0,1,2])
    age_auc = roc_auc_score(age_true, age_probs, multi_class='ovr', average='macro', labels=[0,1,2])

    gender_probs = torch.sigmoid(torch.cat(gender_logits)).numpy()
    gender_true = torch.cat(gender_trues).numpy()
    gender_pred = (gender_probs > 0.5).astype(int)
    gender_acc = accuracy_score(gender_true, gender_pred)
    gender_prec = precision_score(gender_true, gender_pred, zero_division=0)
    gender_rec = recall_score(gender_true, gender_pred, zero_division=0)
    gender_logloss = log_loss(gender_true, gender_probs)
    gender_auc = roc_auc_score(gender_true, gender_probs)

    bite_probs = torch.softmax(torch.cat(bite_logits), dim=1).numpy()
    bite_true = torch.cat(bite_trues).numpy()
    bite_pred = np.argmax(bite_probs, axis=1)
    bite_acc = accuracy_score(bite_true, bite_pred)
    bite_prec = precision_score(bite_true, bite_pred, average='macro', zero_division=0)
    bite_rec = recall_score(bite_true, bite_pred, average='macro', zero_division=0)
    bite_logloss = log_loss(bite_true, bite_probs)
    bite_auc = roc_auc_score(bite_true, bite_probs, multi_class='ovr', average='macro')

    print(f"\nЭпоха {epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")
    print(f"Возраст: Acc={age_acc:.4f}, Prec={age_prec:.4f}, Rec={age_rec:.4f}, LogLoss={age_logloss:.4f}, AUC={age_auc:.4f}")
    print(f"Пол    : Acc={gender_acc:.4f}, Prec={gender_prec:.4f}, Rec={gender_rec:.4f}, LogLoss={gender_logloss:.4f}, AUC={gender_auc:.4f}")
    print(f"Прикус : Acc={bite_acc:.4f}, Prec={bite_prec:.4f}, Rec={bite_rec:.4f}, LogLoss={bite_logloss:.4f}, AUC={bite_auc:.4f}")

    with open(metrics_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, avg_train_loss, avg_val_loss,
                         age_acc, age_prec, age_rec, age_logloss, age_auc,
                         gender_acc, gender_prec, gender_rec, gender_logloss, gender_auc,
                         bite_acc, bite_prec, bite_rec, bite_logloss, bite_auc])

    if args.use_wandb:
        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "age_acc": age_acc,
            "age_precision": age_prec,
            "age_recall": age_rec,
            "age_log_loss": age_logloss,
            "age_auc": age_auc,
            "gender_acc": gender_acc,
            "gender_precision": gender_prec,
            "gender_recall": gender_rec,
            "gender_log_loss": gender_logloss,
            "gender_auc": gender_auc,
            "bite_acc": bite_acc,
            "bite_precision": bite_prec,
            "bite_recall": bite_rec,
            "bite_log_loss": bite_logloss,
            "bite_auc": bite_auc,
            "lr": optimizer.param_groups[0]['lr']
        })

    scheduler.step(avg_val_loss)

    current_metric = (age_acc + bite_acc)/2
    if current_metric > best_metric:
        best_metric = current_metric
        torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
        print("  -> сохранена лучшая модель")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= args.patience:
            print(f"Ранняя остановка после {epoch} эпох")
            break

# ТЕСТИРОВАНИЕ
model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pth")))
model.eval()
age_logits, age_trues = [], []
gender_logits, gender_trues = [], []
bite_logits, bite_trues = [], []
with torch.no_grad():
    for images, ages, genders, bites in test_loader:
        images = images.to(DEVICE)
        ages = ages.to(DEVICE)
        genders = genders.to(DEVICE)
        bites = bites.to(DEVICE)
        age_out, gender_out, bite_out = model(images)
        age_logits.append(age_out.cpu())
        age_trues.append(ages.cpu())
        gender_logits.append(gender_out.cpu())
        gender_trues.append(genders.cpu())
        bite_logits.append(bite_out.cpu())
        bite_trues.append(bites.cpu())

age_probs = torch.softmax(torch.cat(age_logits), dim=1).numpy()
age_true = torch.cat(age_trues).numpy()
age_pred = np.argmax(age_probs, axis=1)
age_auc = roc_auc_score(age_true, age_probs, multi_class='ovr', average='macro', labels=[0,1,2])

gender_probs = torch.sigmoid(torch.cat(gender_logits)).numpy()
gender_true = torch.cat(gender_trues).numpy()
gender_pred = (gender_probs > 0.5).astype(int)
gender_auc = roc_auc_score(gender_true, gender_probs)

bite_probs = torch.softmax(torch.cat(bite_logits), dim=1).numpy()
bite_true = torch.cat(bite_trues).numpy()
bite_pred = np.argmax(bite_probs, axis=1)
bite_auc = roc_auc_score(bite_true, bite_probs, multi_class='ovr', average='macro', labels=[0,1,2])

with open(os.path.join(args.output_dir, "test_metrics.txt"), 'w') as f:
    f.write(f"Возраст: Acc={accuracy_score(age_true, age_pred):.4f},\n"
            f"         Prec={precision_score(age_true, age_pred, average='macro', zero_division=0):.4f},\n"
            f"         Rec={recall_score(age_true, age_pred, average='macro', zero_division=0):.4f},\n"
            f"         LogLoss={log_loss(age_true, age_probs, labels=[0,1,2]):.4f},\n"
            f"         AUC={age_auc:.4f}\n")
    f.write(f"Пол    : Acc={accuracy_score(gender_true, gender_pred):.4f},\n"
            f"         Prec={precision_score(gender_true, gender_pred, zero_division=0):.4f},\n"
            f"         Rec={recall_score(gender_true, gender_pred, zero_division=0):.4f},\n"
            f"         LogLoss={log_loss(gender_true, gender_probs):.4f},\n"
            f"         AUC={gender_auc:.4f}\n")
    f.write(f"Прикус : Acc={accuracy_score(bite_true, bite_pred):.4f},\n"
            f"         Prec={precision_score(bite_true, bite_pred, average='macro', zero_division=0):.4f},\n"
            f"         Rec={recall_score(bite_true, bite_pred, average='macro', zero_division=0):.4f},\n"
            f"         LogLoss={log_loss(bite_true, bite_probs):.4f},\n"
            f"         AUC={bite_auc:.4f}\n")

print("Результаты сохранены в ", args.output_dir)