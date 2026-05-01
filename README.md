# Dental Classification: Age, Gender, Bite
Многозадачная нейросетевая классификация возраста, пола и типа прикуса по фотографиям полости рта.

## Результаты (на тестовой выборке 10% данных, 132 изображения)
| Задача                 | Accuracy | Precision | Recall | AUC |
|------------------------|----------|-----------|--------|-----|
| **Возраст** (3 класса) | 0.67 | 0.51 | 0.43 | 0.56 |
| **Пол** (2 класса)     | 0.75 | 0.64 | 0.61 | 0.82 |
| **Прикус** (3 класса)  | 0.64 | 0.55 | 0.62 | 0.80 |

### Интерпретация
- **Пол** предсказывается лучше всего (AUC 0.82)
- **Возраст** и **прикус** — более сложные задачи (AUC 0.56 и 0.80 соответственно)
- Дисбаланс классов влияет на точность

## Архитектура
- **Backbone**: EfficientNet-B0 (предобучен на ImageNet)
- **Головы**: 3 параллельных классификатора
- **Аугментация**: GPU-ускоренная (Kornia)

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

### 4. Скачивание весов EfficientNet-B0
```bash
wget https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth
mv efficientnet_b0_rwightman-3dd342df.pth efficientnet_b0_weights.pth
```

### 5. Скачивание датасета AlphaDent
Скачайте датасет - 1320 изображений ([AlphaDent](https://www.kaggle.com/competitions/alpha-dent)) с Kaggle и распакуйте в папку dental_classification.

### 6. Запуск обучения
```bash
python train_classifier.py \
    --csv dataset_relative.csv \
    --weights_path ./efficientnet_b0_weights.pth \
    --batch_size 64 \
    --epochs 30 \
    --output_dir ./outputs
```

### 7. Запуск с WandB логированием (опционально)
```bash
# Создайте файл с ключом
echo "ваш_wandb_ключ" > .wandb_key

# Запустите с флагом --use_wandb
python train_classifier.py \
    --csv dataset_relative.csv \
    --weights_path ./efficientnet_b0_weights.pth \
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