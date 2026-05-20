"""
 Hybrid Deepfake Audio Detection — Model A (MFCC + Mel + F0)
 CNN Branches → Weighted Fusion → Multi-Head Attention → Transformer Encoder → MLP Classifier



"""

import os
import sys
import math
import json
import time
import random
import argparse
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  F0 backend
# ─────────────────────────────────────────────
try:
    import pyworld as pw
    HAS_PYWORLD = True
except ImportError:
    HAS_PYWORLD = False


# =============================================================
#  ARGUMENT PARSER
# =============================================================
def get_args():
    p = argparse.ArgumentParser(
        description='Hybrid Deepfake Audio Detector — Model A (MFCC+Mel+F0)'
    )

    # ── Paths ──────────────────────────────────────────────
    p.add_argument('--data_dir', type=str,
                   default='./data/LA/LA',
                   help='Root of ASVspoof2019 LA (contains ASVspoof2019_LA_train/)')
    p.add_argument('--wavefake_real', type=str, default=None,
                   help='(Optional) WaveFake REAL audio folder')
    p.add_argument('--wavefake_fake', type=str, default=None,
                   help='(Optional) WaveFake FAKE audio folder')
    p.add_argument('--output_dir', type=str, default='./output',
                   help='Where to save model checkpoints and results')

    # ── Audio ──────────────────────────────────────────────
    p.add_argument('--sample_rate', type=int, default=16000)
    p.add_argument('--duration',    type=int, default=4,
                   help='Clip duration in seconds')
    p.add_argument('--n_mfcc',      type=int, default=40)
    p.add_argument('--n_mels',      type=int, default=128)
    p.add_argument('--n_fft',       type=int, default=1024)
    p.add_argument('--hop_length',  type=int, default=256)

    # ── Model ──────────────────────────────────────────────
    p.add_argument('--embed_dim',    type=int, default=128)
    p.add_argument('--num_heads',    type=int, default=4)
    p.add_argument('--transformer_layers', type=int, default=2,
                   help='Number of transformer encoder layers')
    p.add_argument('--dropout',      type=float, default=0.3)

    # ── Training ───────────────────────────────────────────
    p.add_argument('--epochs',      type=int,   default=100)
    p.add_argument('--batch_size',  type=int,   default=16)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--weight_decay',type=float, default=1e-4)
    p.add_argument('--patience',    type=int,   default=8,
                   help='Early stopping patience (epochs)')
    p.add_argument('--warmup',      type=int,   default=3,
                   help='LR warmup epochs')
    p.add_argument('--num_workers', type=int,   default=4)
    p.add_argument('--seed',        type=int,   default=42)

    # ── Augmentation ───────────────────────────────────────
    p.add_argument('--aug_noise',   type=float, default=0.3)
    p.add_argument('--aug_codec',   type=float, default=0.2)
    p.add_argument('--aug_pitch',   type=float, default=0.15)
    p.add_argument('--aug_time',    type=float, default=0.15)

    # ── Misc ───────────────────────────────────────────────
    p.add_argument('--no_cuda',     action='store_true')
    p.add_argument('--resume',      type=str, default=None,
                   help='Path to checkpoint to resume from')

    args, unknown = p.parse_known_args()
    return args


# =============================================================
#  AUGMENTATION
# =============================================================
class Augmentor:
    def __init__(self, args):
        self.args = args
        self.N    = args.sample_rate * args.duration

    def _fix(self, wav):
        if wav.shape[-1] >= self.N:
            return wav[:, :self.N]
        return F.pad(wav, (0, self.N - wav.shape[-1]))

    def add_noise(self, wav):
        snr = random.uniform(15, 35)
        n   = torch.randn_like(wav)
        sc  = wav.norm() / (n.norm() * 10 ** (snr / 20) + 1e-8)
        return (wav + n * sc).clamp(-1, 1)

    def apply_codec(self, wav):
        try:
            out = torchaudio.functional.apply_codec(
                wav, self.args.sample_rate,
                format=random.choice(['mp3', 'vorbis']),
                compression=random.choice([48, 64, 96])
            )
            return out if not torch.isnan(out).any() else wav
        except Exception:
            return wav

    def pitch_shift(self, wav):
        try:
            s = librosa.effects.pitch_shift(
                wav.squeeze().numpy(), sr=self.args.sample_rate,
                n_steps=random.uniform(-1.5, 1.5)
            )
            out = torch.tensor(s, dtype=wav.dtype).unsqueeze(0)
            return self._fix(out) if not torch.isnan(out).any() else wav
        except Exception:
            return wav

    def time_stretch(self, wav):
        try:
            s = librosa.effects.time_stretch(
                wav.squeeze().numpy(), rate=random.uniform(0.92, 1.08)
            )
            out = torch.tensor(s, dtype=wav.dtype).unsqueeze(0)
            return self._fix(out) if not torch.isnan(out).any() else wav
        except Exception:
            return wav

    def __call__(self, wav, training=True):
        if not training:
            return self._fix(wav)
        if random.random() < self.args.aug_noise:
            wav = self.add_noise(wav)
        if random.random() < self.args.aug_codec:
            wav = self.apply_codec(wav)
        if random.random() < self.args.aug_pitch:
            wav = self.pitch_shift(wav)
        if random.random() < self.args.aug_time:
            wav = self.time_stretch(wav)
        mx = wav.abs().max()
        if mx > 1e-6:
            wav = wav / mx * 0.9
        return self._fix(wav)


# =============================================================
#  FEATURE EXTRACTOR
# =============================================================
class FeatureExtractor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args       = args
        self.target_len = args.sample_rate * args.duration
        self.target_frm = self.target_len // args.hop_length + 1

        self.mfcc_tf = T.MFCC(
            sample_rate=args.sample_rate,
            n_mfcc=args.n_mfcc,
            melkwargs=dict(
                n_fft=args.n_fft, hop_length=args.hop_length,
                n_mels=args.n_mels, f_min=0, f_max=8000,
            )
        )
        self.mel_tf = T.MelSpectrogram(
            sample_rate=args.sample_rate, n_fft=args.n_fft,
            hop_length=args.hop_length, win_length=args.n_fft,
            n_mels=args.n_mels,
        )
        self.to_db = T.AmplitudeToDB(top_db=80)

    def _pad(self, x, L):
        return x[..., :L] if x.shape[-1] >= L else F.pad(x, (0, L - x.shape[-1]))

    def _extract_f0(self, wav_np):
        sr = self.args.sample_rate
        try:
            if HAS_PYWORLD:
                wav_d  = wav_np.astype(np.float64)
                f0, t  = pw.dio(wav_d, sr, frame_period=self.args.hop_length / sr * 1000)
                f0     = pw.stonemask(wav_d, f0, t, sr)
            else:
                f0, _  = librosa.piptrack(y=wav_np, sr=sr, hop_length=self.args.hop_length)
                f0     = f0.max(axis=0)
            f0 = np.log1p(np.maximum(f0, 0))
            return torch.tensor(f0, dtype=torch.float32).unsqueeze(0)
        except Exception:
            return torch.zeros(1, self.target_frm)

    def forward(self, wav):
        device = wav.device
        wav    = self._pad(wav, self.target_len)

        mfcc = self._pad(self.mfcc_tf(wav.squeeze(1)), self.target_frm)
        mel  = self._pad(self.to_db(self.mel_tf(wav.squeeze(1))), self.target_frm)

        f0s = []
        for i in range(wav.shape[0]):
            f0i = self._extract_f0(wav[i, 0].detach().cpu().numpy())
            f0s.append(self._pad(f0i, self.target_frm))
        f0 = torch.stack(f0s).to(device)

        return mfcc.to(device), mel.to(device), f0


# =============================================================
#  CNN BRANCHES
# =============================================================
class CNNBranch(nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 1)), nn.Dropout2d(dropout),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2, 1)), nn.Dropout2d(dropout),
            nn.Conv2d(64, embed_dim, 3, padding=1), nn.BatchNorm2d(embed_dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None)),
        )

    def forward(self, x):        
        return self.net(x.unsqueeze(1)).squeeze(2)


class F0Branch(nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(64, embed_dim, 3, padding=1), nn.BatchNorm1d(embed_dim), nn.ReLU(),
        )

    def forward(self, x):          
        return self.net(x)


# =============================================================
#  MULTIMODAL ENCODER
# =============================================================
class MultiModalEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        D = args.embed_dim
        self.mfcc_br = CNNBranch(D, args.dropout)
        self.mel_br  = CNNBranch(D, args.dropout)
        self.f0_br   = F0Branch(D // 4, args.dropout)
        self.f0_proj = nn.Linear(D // 4, D)
        self.attn    = nn.MultiheadAttention(D, args.num_heads,
                                              dropout=args.dropout, batch_first=True)
        self.weights = nn.Parameter(torch.ones(3))

    def forward(self, mfcc, mel, f0):
        fm = self.mfcc_br(mfcc)
        fl = self.mel_br(mel)
        ff = self.f0_br(f0)

        T  = min(fm.shape[-1], fl.shape[-1], ff.shape[-1])
        fm, fl, ff = fm[..., :T], fl[..., :T], ff[..., :T]

        ff = self.f0_proj(ff.permute(0, 2, 1)).permute(0, 2, 1)
        w  = F.softmax(self.weights, dim=0)
        x  = w[0] * fm + w[1] * fl + w[2] * ff

        x  = x.permute(0, 2, 1)
        x, _ = self.attn(x, x, x)
        return x                  


# =============================================================
#  TRANSFORMER ENCODER BLOCK (formerly AASISTGraphBlock)
# =============================================================
class TransformerLayer(nn.Module):
    """Single Transformer encoder layer with self-attention and feed-forward"""
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.attn  = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff    = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim)
        )
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x    = self.norm1(x + self.drop(a))
        x    = self.norm2(x + self.drop(self.ff(x)))
        return x


class TransformerEncoderBlock(nn.Module):
    """Stack of Transformer self-attention layers for spectro-temporal modeling"""
    def __init__(self, dim, n_layers, heads, dropout):
        super().__init__()
        self.layers = nn.ModuleList([TransformerLayer(dim, heads, dropout) for _ in range(n_layers)])
        self.pool   = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):        
        for layer in self.layers:
            x = layer(x)
        return self.pool(x.permute(0, 2, 1)).squeeze(-1)


# =============================================================
#  MLP CLASSIFIER HEAD (formerly GANHead)
# =============================================================
class MLPClassifierHead(nn.Module):
    """Standard MLP classifier (not a GAN discriminator)"""
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256), nn.LeakyReLU(0.2), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.Dropout(dropout),
            nn.Linear(128, 64),  nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


# =============================================================
#  FULL MODEL A
# =============================================================
class ModelA(nn.Module):
    """Full model: MFCC + Mel-Spectrogram + F0 → CNN Branches → Weighted Fusion 
       → Multi-Head Attention → Transformer Encoder → MLP Classifier"""

    def __init__(self, args):
        super().__init__()
        self.features = FeatureExtractor(args)
        self.encoder  = MultiModalEncoder(args)
        self.transformer = TransformerEncoderBlock(args.embed_dim, args.transformer_layers,
                                                    args.num_heads, args.dropout)
        self.head     = MLPClassifierHead(args.embed_dim, args.dropout)

    def forward(self, wav):
        mfcc, mel, f0 = self.features(wav)
        fused  = self.encoder(mfcc, mel, f0)
        pooled = self.transformer(fused)
        return self.head(pooled)


# =============================================================
#  DATASET
# =============================================================
def parse_protocol(path):
    samples = []
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 5:
                label = 0 if p[4].lower() == 'bonafide' else 1
                samples.append((p[1], label))
    return samples


class ASVspoofDataset(Dataset):
    def __init__(self, audio_dir, protocol_path, args, training=False):
        self.audio_dir  = audio_dir
        self.samples    = parse_protocol(protocol_path)
        self.args       = args
        self.training   = training
        self.aug        = Augmentor(args)
        self.target_len = args.sample_rate * args.duration

    def _load(self, utt_id):
        for ext in ['.flac', '.wav']:
            path = os.path.join(self.audio_dir, utt_id + ext)
            if os.path.exists(path):
                wav, sr = torchaudio.load(path)
                if sr != self.args.sample_rate:
                    wav = torchaudio.functional.resample(wav, sr, self.args.sample_rate)
                if wav.shape[0] > 1:
                    wav = wav.mean(0, keepdim=True)
                L = self.target_len
                wav = wav[:, :L] if wav.shape[-1] >= L else F.pad(wav, (0, L - wav.shape[-1]))
                return wav
        raise FileNotFoundError(utt_id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        utt_id, label = self.samples[idx]
        try:
            wav = self._load(utt_id)
            wav = self.aug(wav, training=self.training)
            return wav, torch.tensor(label, dtype=torch.float32)
        except Exception:
            return torch.zeros(1, self.target_len), torch.tensor(label, dtype=torch.float32)


class WaveFakeDataset(Dataset):
    """Cross-dataset test on WaveFake"""

    def __init__(self, real_dir, fake_dir, args):
        self.samples    = []
        self.args       = args
        self.target_len = args.sample_rate * args.duration
        self.aug        = Augmentor(args)

        if not os.path.exists(real_dir):
            raise FileNotFoundError(f"WaveFake real dir not found: {real_dir}")
        if not os.path.exists(fake_dir):
            raise FileNotFoundError(f"WaveFake fake dir not found: {fake_dir}")

        for f in sorted(os.listdir(real_dir)):
            if f.endswith(('.wav', '.flac')):
                self.samples.append((os.path.join(real_dir, f), 0))
        for f in sorted(os.listdir(fake_dir)):
            if f.endswith(('.wav', '.flac')):
                self.samples.append((os.path.join(fake_dir, f), 1))

        print(f'WaveFake: {len(self.samples)} files '
              f'({sum(1 for _, l in self.samples if l == 0)} real / '
              f'{sum(1 for _, l in self.samples if l == 1)} fake)')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            wav, sr = torchaudio.load(path)
            if sr != self.args.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, self.args.sample_rate)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            wav = self.aug(wav, training=False)
            return wav, torch.tensor(label, dtype=torch.float32)
        except Exception:
            return torch.zeros(1, self.target_len), torch.tensor(label, dtype=torch.float32)


# =============================================================
#  METRICS
# =============================================================
def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return round(float(eer) * 100, 4)


@torch.no_grad()
def evaluate(model, loader, device, tag=''):
    model.eval()
    all_labels, all_scores = [], []
    for wav, labels in loader:
        probs = torch.sigmoid(model(wav.to(device)).squeeze(1))
        all_scores.extend(probs.cpu().tolist())
        all_labels.extend(labels.int().tolist())

    preds = [1 if s > 0.5 else 0 for s in all_scores]
    acc   = sum(p == l for p, l in zip(preds, all_labels)) / len(all_labels) * 100
    auc   = roc_auc_score(all_labels, all_scores) * 100
    eer   = compute_eer(all_labels, all_scores)
    f1    = f1_score(all_labels, preds) * 100
    fp    = sum(1 for l, p in zip(all_labels, preds) if l == 0 and p == 1)
    fpr   = fp / max(1, sum(1 for l in all_labels if l == 0)) * 100

    metrics = dict(acc=round(acc, 2), auc=round(auc, 2),
                   eer=eer, f1=round(f1, 2), fpr=round(fpr, 2))

    if tag:
        print(f'[{tag}] EER={eer:.2f}% | AUC={auc:.2f}% | '
              f'Acc={acc:.2f}% | F1={f1:.2f}% | FPR={fpr:.2f}%')
    return metrics


# =============================================================
#  TRAINING LOOP
# =============================================================
def train(args, model, tr_loader, dv_loader, device):
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, 'best_model_A.pth')

    n_real  = sum(1 for _, l in tr_loader.dataset.samples if l == 0)
    n_fake  = sum(1 for _, l in tr_loader.dataset.samples if l == 1)
    pos_w   = torch.tensor([n_real / max(n_fake, 1)], device=device)
    crit    = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    optim   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(ep):
        if ep < args.warmup:
            return (ep + 1) / args.warmup
        prog = (ep - args.warmup) / max(1, args.epochs - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))

    sched   = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    best_eer  = 999.0
    patience  = 0
    history   = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0   = time.time()
        loss_sum = 0.0

        for step, (wav, labels) in enumerate(tr_loader):
            wav, labels = wav.to(device), labels.to(device)
            optim.zero_grad()
            loss = crit(model(wav).squeeze(1), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            loss_sum += loss.item()

            if step % 200 == 0:
                lr = optim.param_groups[0]['lr']
                print(f'  Ep{epoch:02d}[{step:4d}/{len(tr_loader)}] '
                      f'loss={loss.item():.4f} lr={lr:.2e}')

        sched.step()
        avg_loss = loss_sum / len(tr_loader)
        elapsed  = time.time() - t0
        print(f'\nEp {epoch:02d} | loss={avg_loss:.4f} | time={elapsed:.0f}s')

        metrics         = evaluate(model, dv_loader, device, tag=f'Dev ep{epoch}')
        metrics['epoch'] = epoch
        metrics['loss']  = round(avg_loss, 4)
        history.append(metrics)

        if metrics['eer'] < best_eer:
            best_eer = metrics['eer']
            torch.save({'epoch': epoch, 'state': model.state_dict(),
                        'eer': best_eer}, ckpt_path)
            print(f'  Best model saved — EER={best_eer:.2f}%')
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f'    Early stopping at epoch {epoch}')
                break

    return history, best_eer, ckpt_path


# =============================================================
#  PLOTS
# =============================================================
def save_plots(history, output_dir):
    epochs = [h['epoch'] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, [h['eer']  for h in history], 'b-o', ms=3)
    ax1.axhline(0.83, color='gray', ls='--', alpha=0.6, label='AASIST ref')
    ax1.set_title('EER % on Dev (lower=better)')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('EER %')
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, [h['loss'] for h in history], 'r-o', ms=3)
    ax2.set_title('Training Loss')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Plot saved: {out}')


# =============================================================
#  MAIN
# =============================================================
def main():
    args   = get_args()
    device = torch.device('cpu' if args.no_cuda or not torch.cuda.is_available() else 'cuda')

    # reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    print('=' * 60)
    print('  Hybrid Deepfake Detector — Model A (MFCC + Mel + F0)')
    print('  Architecture: CNN Branches → Weighted Fusion → MHA → Transformer → MLP')
    print('=' * 60)
    print(f'  Device      : {device}')
    print(f'  Epochs      : {args.epochs}')
    print(f'  Batch size  : {args.batch_size}')
    print(f'  LR          : {args.lr}')
    print(f'  Embed dim   : {args.embed_dim}')
    print(f'  Transformer layers : {args.transformer_layers}')
    print(f'  pyworld     : {"YES" if HAS_PYWORLD else "NO (librosa fallback)"}')
    print('=' * 60)

    # ── Paths ──────────────────────────────────────────────
    base        = args.data_dir
    TRAIN_AUDIO = os.path.join(base, 'ASVspoof2019_LA_train', 'flac')
    DEV_AUDIO   = os.path.join(base, 'ASVspoof2019_LA_dev',   'flac')
    EVAL_AUDIO  = os.path.join(base, 'ASVspoof2019_LA_eval',  'flac')
    TRAIN_PROTO = os.path.join(base, 'ASVspoof2019_LA_cm_protocols',
                               'ASVspoof2019.LA.cm.train.trn.txt')
    DEV_PROTO   = os.path.join(base, 'ASVspoof2019_LA_cm_protocols',
                               'ASVspoof2019.LA.cm.dev.trl.txt')
    EVAL_PROTO  = os.path.join(base, 'ASVspoof2019_LA_cm_protocols',
                               'ASVspoof2019.LA.cm.eval.trl.txt')

    for name, p in [('Train audio', TRAIN_AUDIO), ('Train proto', TRAIN_PROTO),
                    ('Dev audio',   DEV_AUDIO),   ('Dev proto',   DEV_PROTO)]:
        status = 'OK' if os.path.exists(p) else 'MISSING '
        print(f'  [{status}] {name}: {p}')

    # ── Datasets ───────────────────────────────────────────
    tr_ds = ASVspoofDataset(TRAIN_AUDIO, TRAIN_PROTO, args, training=True)
    dv_ds = ASVspoofDataset(DEV_AUDIO,   DEV_PROTO,   args, training=False)
    ev_ds = ASVspoofDataset(EVAL_AUDIO,  EVAL_PROTO,  args, training=False)

    print(f'\n  Train: {len(tr_ds):,} | Dev: {len(dv_ds):,} | Eval: {len(ev_ds):,}')

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, pin_memory=True)
    dv_loader = DataLoader(dv_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
    ev_loader = DataLoader(ev_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)

    # ── Model ──────────────────────────────────────────────
    model = ModelA(args).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f'\n  Parameters: {n_par:,}')

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['state'])
        print(f'  Resumed from: {args.resume}')

    # ── Train ──────────────────────────────────────────────
    history, best_dev_eer, ckpt_path = train(args, model, tr_loader, dv_loader, device)

    # ── Final Eval ─────────────────────────────────────────
    print('\n' + '=' * 60)
    print('  FINAL EVALUATION')
    print('=' * 60)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['state'])

    dev_m  = evaluate(model, dv_loader, device, tag='Dev  (best ckpt)')
    eval_m = evaluate(model, ev_loader, device, tag='Eval (ASVspoof)')

    results = {'dev': dev_m, 'eval': eval_m, 'history': history}

    # ── WaveFake cross-test ────────────────────────────────
    if args.wavefake_real and args.wavefake_fake:
        wf_ds     = WaveFakeDataset(args.wavefake_real, args.wavefake_fake, args)
        wf_loader = DataLoader(wf_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True)
        wf_m      = evaluate(model, wf_loader, device, tag='WaveFake (cross-dataset)')
        results['wavefake'] = wf_m
    else:
        print('\n  [INFO] WaveFake paths not provided — skipping cross-dataset test')
        print('         Add --wavefake_real /path/REAL --wavefake_fake /path/FAKE to enable')

    # ── Save results ───────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    out_json = os.path.join(args.output_dir, 'results.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Results saved: {out_json}')

    save_plots(history, args.output_dir)

    # ── Summary table ──────────────────────────────────────
    print('\n' + '=' * 60)
    print('  COMPARISON WITH BASELINES (ASVspoof 2019 LA)')
    print('=' * 60)
    print(f'  {"Method":<30} {"EER%":>8} {"AUC%":>8} {"F1%":>8}')
    print(f'  {"-"*54}')
    print(f'  {"GMM baseline (LFCC)":<30} {"8.09":>8} {"-":>8} {"-":>8}')
    print(f'  {"LCNN":<30} {"5.06":>8} {"-":>8} {"-":>8}')
    print(f'  {"RawNet2":<30} {"1.12":>8} {"-":>8} {"-":>8}')
    print(f'  {"AASIST":<30} {"0.83":>8} {"-":>8} {"-":>8}')
    print(f'  {"-"*54}')
    print(f'  {"Our Model A (MFCC+Mel+F0)":<30} '
          f'{eval_m["eer"]:>8} {eval_m["auc"]:>8} {eval_m["f1"]:>8}')
    print('=' * 60)


if __name__ == '__main__':
    main()
