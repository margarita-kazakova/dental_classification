# Dental Classification: Age, Gender, Bite
Многозадачная нейросетевая классификация возраста, пола и типа прикуса по фотографиям полости рта.

## Полные результаты экспериментов

### EfficientNet-B0 (вход: 224×224)
| Функция потерь | γ | Возраст Acc | Пол Acc | Прикус Acc | Возраст AUC | Пол AUC | Прикус AUC |
|----------------|---|-------------|---------|------------|-------------|---------|------------|
| CrossEntropy | – | 0.644 | 0.818 | 0.727 | 0.545 | 0.864 | 0.798 |
| Focal Loss | 1.0 | 0.629 | 0.811 | 0.621 | – | 0.901 | – |
| Focal Loss | 1.5 | 0.621 | **0.826** | 0.720 | 0.624 | **0.911** | 0.813 |
| Focal Loss | 2.0 | 0.614 | 0.811 | **0.735** | – | – | – |

### MaxViT-Tiny-512 (вход: 512×512)
| Функция потерь | γ | Возраст Acc | Пол Acc | Прикус Acc | Возраст AUC | Пол AUC | Прикус AUC |
|----------------|---|-------------|---------|------------|-------------|---------|------------|
| CrossEntropy | – | **0.674** | 0.811 | **0.871** | 0.712 | 0.910 | 0.856 |
| Focal Loss | 1.0 | 0.636 | 0.803 | 0.833 | 0.651 | 0.891 | 0.898 |
| Focal Loss | 1.5 | 0.644 | 0.826 | 0.750 | 0.705 | 0.938 | 0.872 |
| Focal Loss | 2.0 | 0.636 | **0.856** | 0.856 | 0.696 | **0.940** | 0.862 |

### Лучшие результаты
| Задача | Модель | Функция потерь | γ | Accuracy | AUC |
|--------|--------|----------------|---|----------|-----|
| **Возраст** | MaxViT | CrossEntropy | – | **0.674** | 0.712 |
| **Пол** | MaxViT | Focal Loss | 2.0 | **0.856** | **0.940** |
| **Прикус** | MaxViT | CrossEntropy | – | **0.871** | 0.856 |

### Ключевые выводы
1. **MaxViT превосходит EfficientNet** по всем задачам классификации:
   - Возраст: +3.0% (0.674 vs 0.644)
   - Пол: +3.0% (0.856 vs 0.826)
   - Прикус: **+13.6%** (0.871 vs 0.735)
2. **Focal Loss эффективен для бинарной классификации** пола:
   - Улучшение Accuracy на 4.5% (0.811 → 0.856) для MaxViT
3. **CrossEntropy с весами классов лучше для многоклассовых задач**:
   - Возраст: 0.674 (CrossEntropy) vs 0.636 (Focal Loss)
   - Прикус: 0.871 (CrossEntropy) vs 0.856 (Focal Loss)

## Архитектура
- **Backbone**: EfficientNet-B0 или MaxViT-Tiny-512 (предобучены на ImageNet)
- **Головы**: 3 параллельных классификатора
- **Аугментация**: GPU-ускоренная (Kornia)
- **Функция потерь**: CrossEntropyLoss + Focal Loss (для возраста)

## Данные
- 1320 изображений (- 1320 изображений ([AlphaDent](https://www.kaggle.com/competitions/alpha-dent)))
- Разбиение по пациентам (train/val/test = 80/10/10)
- Стратификация по возрасту

## Установка и запуск

### 1. Клонирование репозитория
```bash
git clone https://github.com/margarita-kazakova/dental-classification.git
cd dental-classification
```
### 2. Создание виртуального окружения
```bash
# Windows
python -m venv venv
venv\Scripts\activate
# Linux / Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 4. Скачивание весов
Для EfficientNet-B0:
```bash
wget https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth
mv efficientnet_b0_rwightman-3dd342df.pth efficientnet_b0_weights.pth
```
Для MaxViT:
```bash
wget https://huggingface.co/timm/maxvit_tiny_tf_512.in1k/resolve/main/model.safetensors
mv model.safetensors maxvit_tiny_tf_512_weights.safetensors
```

### 5. Скачивание датасета AlphaDent
Скачайте датасет - 1320 изображений ([AlphaDent](https://www.kaggle.com/competitions/alpha-dent)) с Kaggle и распакуйте в папку dental_classification.

### 6. Запуск обучения
С EfficientNet-B0:
```bash
python train_classifier.py \
    --csv dataset_relative.csv \
    --backbone efficientnet \
    --weights_path ./efficientnet_b0_weights.pth \
    --img_size 224 \
    --batch_size 64 \
    --epochs 30 \
    --lr 1e-3 \
    --output_dir ./outputs
```
С MaxViT:
```bash
python train_classifier.py \
    --csv dataset_relative.csv \
    --backbone maxvit \
    --weights_path ./maxvit_tiny_tf_512_weights.safetensors \
    --img_size 512 \
    --batch_size 16 \
    --epochs 30 \
    --lr 5e-5 \
    --output_dir ./outputs
```
### 7. Запуск с WandB логированием (опционально)
```bash
# Создайте файл с ключом
echo "ваш_wandb_ключ" > .wandb_key

# Запустите с флагом --use_wandb
python train_classifier.py \
    # ... \
    --use_wandb
```

### 8. Запуск на суперкомпьютере cHARISMa
Отредактируйте файл run_dental_classification_sample.slurm, затем выполните:
```bash
sbatch run_dental_classification_sample.slurm
```

## Выходные файлы
После завершения обучения в папке **outputs_classifier_номер** появятся:
- best_model.pth — веса лучшей модели 
- metrics.csv — метрики по эпохам 
- test_metrics.txt — итоговые метрики на тесте

Также в корневой папке будут созданы train.csv и valid.csv — разбиение датасета по пациентам с сегментацией по возрасту.