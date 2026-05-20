#!/bin/bash
#SBATCH --job-name=deepfake_modelA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err

# ─── Load modules (adapt to your HPC environment) ───────────
# module load python/3.10
# module load cuda/11.8
# source ~/venv/bin/activate

mkdir -p logs

echo "=============================="
echo " Job ID     : $SLURM_JOB_ID"
echo " Node       : $SLURMD_NODENAME"
echo " Start time : $(date)"
echo "=============================="

# ─── Edit these paths ────────────────────────────────────────
DATA_DIR="/scratch/$USER/ASVspoof2019/LA/LA"
OUTPUT_DIR="/scratch/$USER/output_modelA"

# ─── Optional: WaveFake paths (leave empty to skip) ─────────
WAVEFAKE_REAL=""
WAVEFAKE_FAKE=""

# ─── Run ─────────────────────────────────────────────────────
python train.py \
  --data_dir       $DATA_DIR \
  --output_dir     $OUTPUT_DIR \
  --epochs         50 \
  --batch_size     32 \
  --lr             1e-4 \
  --weight_decay   1e-4 \
  --embed_dim      128 \
  --transformer_layers   2 \
  --num_heads      4 \
  --dropout        0.3 \
  --patience       10 \
  --warmup         3 \
  --num_workers    8 \
  --seed           42 \
  ${WAVEFAKE_REAL:+--wavefake_real $WAVEFAKE_REAL} \
  ${WAVEFAKE_FAKE:+--wavefake_fake $WAVEFAKE_FAKE}

echo "=============================="
echo " End time : $(date)"
echo "=============================="
