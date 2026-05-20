
# Hybrid Deepfake Audio Detection — Model A

**Architecture:** MFCC + Mel-Spectrogram + F0 → CNN Branches → Weighted Fusion → Multi-Head Attention → TransformerEncoderBlock → MLPClassifierHead

---

## Requirements

```bash
pip install -r requirements.txt
```

> **Note:** `pyworld` is recommended for accurate F0 extraction.
> If it fails to install, the code automatically falls back to `librosa`.

---

## Dataset Structure

Download **ASVspoof 2019 LA**   from this link :  https://datashare.ed.ac.uk/handle/10283/3336 
and place it so the structure looks like:

```
/your/data/path/
└── LA/
    └── LA/
        ├── ASVspoof2019_LA_train/
        │   └── flac/
        │       ├── LA_T_0000001.flac
        │       └── ...
        ├── ASVspoof2019_LA_dev/
        │   └── flac/
        ├── ASVspoof2019_LA_eval/
        │   └── flac/
        └── ASVspoof2019_LA_cm_protocols/
            ├── ASVspoof2019.LA.cm.train.trn.txt
            ├── ASVspoof2019.LA.cm.dev.trl.txt
            └── ASVspoof2019.LA.cm.eval.trl.txt
```

For cross-dataset testing, also download **WaveFake** from this link : https://zenodo.org/record/5525342 
```
/wavefake/
├── REAL/
│   ├── file1.wav
│   └── ...
└── FAKE/
    ├── file1.wav
    └── ...
```

---

## Quick Start

### Minimal run (default settings):
```bash
python train.py --data_dir /path/to/LA/LA
```

### Full run with all options:
```bash
python train.py \
  --data_dir      /path/to/LA/LA \
  --output_dir    ./output \
  --epochs        100 \
  --batch_size    16 \
  --lr            1e-4 \
  --embed_dim     128 \
  --transformer_layers  2 \
  --dropout       0.3 \
  --patience      8 \
  --num_workers   4
```

### With WaveFake cross-dataset test:
```bash
python train.py \
  --data_dir       /path/to/LA/LA \
  --wavefake_real  /path/to/WaveFake/REAL \
  --wavefake_fake  /path/to/WaveFake/FAKE \
  --epochs         100
```

### Resume from checkpoint:
```bash
python train.py \
  --data_dir /path/to/LA/LA \
  --resume   ./output/best_model_A.pth \
  --epochs   100
```

---

## All Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `./data/LA/LA` | ASVspoof 2019 LA root directory |
| `--wavefake_real` | `None` | WaveFake REAL folder  |
| `--wavefake_fake` | `None` | WaveFake FAKE folder |
| `--output_dir` | `./output` | Output folder for model and results |
| `--epochs` | `100` | Number of training epochs |
| `--batch_size` | `16` | Batch size (reduce to 8 if OOM) |
| `--lr` | `1e-4` | Learning rate |
| `--weight_decay` | `1e-4` | AdamW weight decay |
| `--patience` | `8` | Early stopping patience |
| `--warmup` | `3` | LR warmup epochs |
| `--embed_dim` | `128` | Embedding dimension |
| `--num_heads` | `4` | Attention heads |
| `--transformer_layers` | `2` | TransformerEncoderBlock |
| `--dropout` | `0.3` | Dropout probability |
| `--sample_rate` | `16000` | Audio sample rate |
| `--duration` | `4` | Clip duration (seconds) |
| `--n_mfcc` | `40` | MFCC coefficients |
| `--n_mels` | `128` | Mel filter banks |
| `--aug_noise` | `0.3` | Noise augmentation probability |
| `--aug_codec` | `0.2` | Codec augmentation probability |
| `--aug_pitch` | `0.15` | Pitch shift probability |
| `--aug_time` | `0.15` | Time stretch probability |
| `--num_workers` | `4` | DataLoader workers |
| `--seed` | `42` | Random seed |
| `--no_cuda` | `False` | Force CPU (add flag to disable GPU) |
| `--resume` | `None` | Path to checkpoint to resume from |

---

## Running on HPC / Supercalculator (SLURM)

Create a file `run.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=deepfake_det
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.log

module load python/3.10
module load cuda/11.8

source ~/venv/bin/activate

python train.py \
  --data_dir      /scratch/$USER/ASVspoof2019/LA/LA \
  --output_dir    /scratch/$USER/output \
  --epochs        100 \
  --batch_size    32 \
  --lr            1e-4 \
  --num_workers   8
```

Submit with:
```bash
sbatch run.slurm
```

Monitor:
```bash
squeue -u $USER
tail -f logs/train_<JOB_ID>.log
```

---

## Output Files

After training, `--output_dir` will contain:

```
output/
├── best_model_A.pth       ← best checkpoint (by Dev EER)
├── results.json           ← all metrics (dev, eval, wavefake)
└── training_curves.png    ← EER and Loss plots
```

### results.json structure:
```json
{
  "dev":  { "eer": 1.23, "auc": 99.5, "f1": 98.7, "acc": 98.9, "fpr": 1.1 },
  "eval": { "eer": 2.01, "auc": 99.1, "f1": 97.8, "acc": 98.1, "fpr": 2.0 },
  "wavefake": { "eer": 5.3, "auc": 97.2, "f1": 94.1, "acc": 95.0, "fpr": 4.9 },
  "history": [ { "epoch": 1, "loss": 0.62, "eer": 14.2, ... }, ... ]
}
```

---

## Architecture Summary

```
Raw Audio [B, 1, 64000]
        │
        ▼
  Data Augmentation
  (Noise · Codec · Pitch · Time Stretch)
        │
        ▼
  ┌─────────────────────────────┐
  │      Feature Extraction     │
  │  MFCC      Mel-Spec    F0   │
  │ [B,40,T]  [B,128,T]  [B,1,T]│
  └─────────────────────────────┘
        │
        ▼
  CNN 2D     CNN 2D     CNN 1D
        │
        ▼
  Weighted Fusion  [B, 128, T']
        │
        ▼
  Multi-Head Attention  [B, T', 128]
        │
        ▼
  TransformerEncoderBlock ×2  [B, T', 128]
        │
        ▼
  Global Average Pooling  [B, 128]
        │
        ▼
  MLPClassifierHead  [B, 1]
        │
        ▼
  σ(logit) > 0.5  →  FAKE
  σ(logit) ≤ 0.5  →  REAL
```

---

## Baselines Comparison (ASVspoof 2019 LA)

| Method | EER% |
|--------|------|
| GMM baseline (LFCC) | 8.09 |
| LCNN | 5.06 |
| RawNet2 | 1.12 |
| AASIST | 0.83 |
| **Our Model A** | *see results.json* |

---

## Troubleshooting

**CUDA out of memory:**
```bash
python train.py --batch_size 8 ...
```

**pyworld install fails:**
```
pip install pyworld --no-build-isolation
```
The code works without it using librosa as fallback.

**File not found errors:**
Run this to check your paths:
```python
import os
base = "/your/data/path/LA/LA"
print(os.path.exists(f"{base}/ASVspoof2019_LA_train/flac"))
print(os.path.exists(f"{base}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"))
```
