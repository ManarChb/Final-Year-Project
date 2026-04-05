#  Audio Deepfake Detection Project

##  Description
This project aims to detect fake (spoofed) vs real (genuine) audio using 
machine learning techniques, based on the ASVspoof2019 LA dataset.

This repository covers:
- Dataset subsetting
- Audio preprocessing (MFCC, Spectrogram & Prosodic feature extraction)

---

##  Repository Structure
```
Final-Year-Project/
│
├── data/
│   └── subset_dataset.csv         # file_path + label 
│
├── features/
│   ├── mfcc/
│   │   └── mfcc_features.csv      # 13 MFCC coefficients per file
│   ├── spectrograms/
│   │   ├── real/                  # .png spectrogram images
│   │   └── fake/                  # .png spectrogram images
│   └── prosodic/
│       └── prosodic_features.csv  # pitch, energy, ZCR, tempo
│
├── notebooks/
│   ├── 01_create_subset.ipynb
│   ├── 02_mfcc_extraction.ipynb
│   ├── 03_spectrogram_extraction.ipynb
│   └── 04_prosodic_extraction.ipynb
│
├── README.md
└── requirements.txt
```

---

##  Dataset
- **Source:** ASVspoof2019 LA (Logical Access)
- **Subset:** ~1000 real + ~1000 fake audio samples
- Raw audio files are **NOT** included due to size limitations.
- The `subset_dataset.csv` contains file paths, labels, split info, and speaker IDs.

---

## Features Extracted

| Feature | Description | Details | Format |
|---------|-------------|---------|--------|
| MFCC | Mel-Frequency Cepstral Coefficients | 13 coefficients per frame | .csv / .npy |
| Spectrogram | Visual frequency representation over time | Mel-scale, saved as images | .png |
| Prosodic | Low-level speech characteristics | Pitch mean/std, Energy mean, ZCR mean, Tempo | .csv |

---

##  Project Status

| Phase | Task | Status |
|-------|------|--------|
| Phase 1 | Dataset Subsetting |  Done |
| Phase 1 | MFCC Extraction |  Done |
| Phase 1 | Spectrogram Extraction |  Done |
| Phase 1 | Prosodic Extraction | Done |
| Phase 2 | Feature Fusion |  In Progress |
| Phase 2 | Model Training |  Pending |
| Phase 3 | Evaluation & Metrics |  Pending |

---

##  How to Use

1. Clone the repository:
```bash
   git clone https://github.com/ManarChb/Final-Year-Project.git
   cd Final-Year-Project
```

2. Install dependencies:
```bash
   pip install -r requirements.txt
```

3. Run notebooks in order:
   - `01_create_subset.ipynb` — Build the dataset subset
   - `02_mfcc_extraction.ipynb` — Extract MFCC features
   - `03_spectrogram_extraction.ipynb` — Generate spectrogram images
   - `04_prosodic_extraction.ipynb` — Extract prosodic features

---

##  Requirements
```txt
librosa
numpy
pandas
matplotlib
scikit-learn
soundfile
tqdm
```

> Install all at once: `pip install -r requirements.txt`

---

##  Notes
- Audio files are excluded from this repo (too large for GitHub).
- Features were extracted and saved in the `features/` directory.
- Spectrograms are organized by label (real/fake) for easier loading.
- Dataset split follows the official ASVspoof2019 train/dev/eval protocol.

---

##  Next Steps
- [ ] Feature Fusion (combine MFCC + Prosodic + Spectrogram)
- [ ] Model Training (CNN / LSTM / SVM)
- [ ] Evaluation (Accuracy, EER, ROC Curve)
