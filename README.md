# Final-Year-Project
# Audio Deepfake Detection Project

## Description
This project aims to detect fake (spoofed) vs real (genuine) audio using 
machine learning techniques, based on the ASVspoof2019 LA dataset.

This repository covers:
- Dataset subsetting
- Audio preprocessing (MFCC & Spectrogram extraction)

---
##  Dataset
- **Source:** ASVspoof2019 LA (Logical Access)
- **Subset:** ~1000 real + ~1000 fake audio samples
-  Raw audio files are NOT included due to size limitations.
- The `subset_dataset.csv` contains file paths, labels, and metadata.

---

## Features Extracted

| Feature | Description | Format |
|---------|-------------|--------|
| MFCC | Mel-Frequency Cepstral Coefficients | .csv / .npy |
| Spectrogram | Visual frequency over time | .png |
| Prosodic | Pitch, Energy, ZCR, Tempo | .csv |

---
##  Requirements
- Python 3.x
- librosa
- numpy
- pandas
- matplotlib
- scikit-learn

---
## Notes
- Audio files are excluded from this repo (too large).
- Features were extracted and saved locally / uploaded separately.
