# Это шаблон. Вставьте свои пути, а затем переименуйте этот файл в run_dental_classification

#!/bin/bash
#SBATCH --job-name="dental_classifier"
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --time=0-12:0
#SBATCH --mail-user=YOUR_EMAIL@edu.hse.ru
#SBATCH --mail-type=END,FAIL
#SBATCH --output=out_classifier_%j.out
#SBATCH --error=err_classifier_%j.out

module load CUDA/12.4
module load Python/Anaconda

source activate YOUR_ENV_NAME

# работа с локальным SSD
if [ -n "$SLURM_TMPDIR" ]; then
    LOCAL_SCRATCH="$SLURM_TMPDIR"
else
    LOCAL_SCRATCH="/home/YOUR_USERNAME/tmp_slurm_${SLURM_JOB_ID}"
    mkdir -p "$LOCAL_SCRATCH"
fi
echo "Используем локальный scratch: $LOCAL_SCRATCH"

# копируем данные (укажите свои пути)
rsync -avh --progress /path/to/your/AlphaDent/ "$LOCAL_SCRATCH/AlphaDent/"
rsync -avh --progress /path/to/your/dataset_relative.csv "$LOCAL_SCRATCH/"

OUTPUT_LOCAL="$LOCAL_SCRATCH/outputs"
mkdir -p "$OUTPUT_LOCAL"

echo "=== Начинаем обучение: ==="
srun python /path/to/your/train_classifier.py \
    --csv "$LOCAL_SCRATCH/dataset_relative.csv" \
    --weights_path /path/to/your/efficientnet_b0_weights.pth \
    --batch_size 64 \
    --num_workers 0 \
    --epochs 30 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --patience 10 \
    --output_dir "$OUTPUT_LOCAL"
    # --use_wandb

FINAL_OUTPUT="/home/YOUR_USERNAME/dental_classification/outputs_classifier_${SLURM_JOB_ID}"
mkdir -p "$FINAL_OUTPUT"
echo "Копируем результаты в: $FINAL_OUTPUT"
rsync -avh "$OUTPUT_LOCAL/" "$FINAL_OUTPUT/"

if [ -z "$SLURM_TMPDIR" ]; then
    rm -rf "$LOCAL_SCRATCH"
fi

echo "=== Конец ==="