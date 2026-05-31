# Distorted Visual Sequence Pattern Recognition

Read a 6-character code from noisy, distorted grayscale images using a **CRNN + CTC** deep-learning model. Scored by **Character Error Rate (CER)**.

- **Main deliverable:** [`notebook/solution.ipynb`](notebook/solution.ipynb) — a self-contained, documented workflow that runs on Kaggle GPU, Colab, or locally.
- **Beginner-friendly explanation + tuning guide:** [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md).
- **Reusable code:** the [`src/`](src/) package (same solution as clean modules).

## Approach (TL;DR)

`Image → CNN (visual features) → BiLSTM (sequence) → CTC (alignment-free text)`

The CNN turns the image into a sequence of vertical "slices", the BiLSTM reads them in order, and CTC lets us train on the final string alone — no per-character boxes needed. See the docs for the full, plain-language walkthrough.

## Dataset

- 20,000 train images + labels, 5,000 test images. Images are `200×100` grayscale.
- Labels are 6-char codes over a 31-symbol alphabet `23456789ABCDEFGHJKMNPQRSTUVWXYZ` (ambiguous `0/1/I/L/O` excluded).
- Expected layout:

```
data/cig_ps/
├── train_images/   # train-0.png ...
├── test_images/    # test-0.png ...
└── train-labels.csv
```

Download (one-off):

```bash
pip install gdown
python -c "import gdown; gdown.download(id='1e0Bpbmjp-Pc1Oz7luC1g8JhfzKpGZRHu', output='data/cig_ps.zip')"
python -c "import zipfile; zipfile.ZipFile('data/cig_ps.zip').extractall('data')"
```

## Run on Kaggle (recommended — free GPU)

1. Create a **Kaggle Dataset** by uploading `cig_ps.zip` (it unzips to `cig_ps/`).
2. New Notebook → **Add Data** (your dataset) → Settings → **Accelerator: GPU**.
3. Upload `notebook/solution.ipynb` and **Run All**. The data path is auto-detected under `/kaggle/input`.
4. Download `outputs/submission.csv`, rename to `submission_<name>_<enroll no.>.csv`, and submit.

## Run locally

```bash
pip install -r requirements.txt

# Option A: the notebook
jupyter notebook notebook/solution.ipynb

# Option B: the scripts
python -m src.train      # trains, saves outputs/crnn_best.pt
python -m src.predict    # writes outputs/submission.csv
```

> CPU works but is slow (~17 min/epoch). Use a GPU (Kaggle/Colab) for real training.

## Tuning

All hyper-parameters live in the notebook's `CFG` block (and `src/config.py`). The ranked, step-by-step tuning guide is in [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md#5-tuning-guide-when-you-want-a-better-score).
