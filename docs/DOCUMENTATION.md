# Understanding the Solution — A Beginner's Guide

This document explains *what* we built, *why* each piece exists, and *how to tune it*. It assumes you are new to data science, so every term is introduced in plain language. It is meant to be read top-to-bottom once; afterwards the notebook will make full sense.

---

## 1. The problem in one paragraph

We are given small grayscale images, each hiding a **6-character code** (like `BU522X`). The images are deliberately messed up: background speckles, a big black blob covering part of the text, blur, and characters that lean or overlap. Our job: look at an image and output the correct string. We are graded by **Character Error Rate (CER)** — basically "how many single-character edits are needed to fix our guess." Lower is better; `0` is perfect.

---

## 2. The key idea: treat it as a *sequence*, not a classification

A naive approach would be "cut the image into 6 boxes and classify each box." That fails here because spacing is irregular and characters overlap — we don't know where one ends and the next begins.

Instead we treat the image as a **left-to-right sequence** and use a model that reads it like a line of text. This is the **CRNN + CTC** approach, the industry-standard recipe for reading text from images (license plates, captchas, scanned words).

Three building blocks:

| Block | Plain-language job | Technical name |
|-------|--------------------|----------------|
| Eyes  | Scan the image and summarise each vertical "slice" into numbers | **CNN** (Convolutional Neural Network) |
| Reader | Go through those slices in order and understand character-to-character flow | **BiLSTM** (Bidirectional Long Short-Term Memory) |
| Translator | Turn the slice-by-slice scores into a clean final string | **CTC** (Connectionist Temporal Classification) |

---

## 3. Walking through the pipeline

### 3.1 Data preparation (`src/dataset.py`)
- **Convert to grayscale & resize** every image to a fixed `32×160`. Models need consistent input sizes; height `32` is the standard the CNN is tuned for, and width `160` gives enough horizontal "slices" for a 6-character code.
- **Normalise** pixel values to roughly `[-1, 1]`. Networks train faster and more stably when inputs are centered around zero.
- **Augmentation (training only):** we apply tiny random rotations/shifts/scales. This shows the model slightly different versions of each image every epoch, so it learns the *character shapes* rather than memorising exact pixels — which helps it generalise to the unseen test images.
- **Cleaning:** 2 of the 20,000 labels were corrupted by spreadsheet auto-formatting (a number became `5.40E+12`, a code became a date `04-Mar-54`). We drop those 2 rows.

### 3.2 Turning text into numbers (`src/charset.py`)
Computers work with numbers, not letters. We map each of the 31 possible characters to an integer `1..31`. Index `0` is reserved for a special **blank** symbol that CTC needs (explained below).

### 3.3 The model (`src/model.py`)
- **CNN:** a stack of convolution + pooling layers. Convolutions detect visual patterns (edges → strokes → character parts). Pooling gradually shrinks the image height from 32 down to **1**, while keeping the width. The output is a sequence of about **41 feature vectors**, one per horizontal position — think of it as 41 thin vertical slices, each described by 512 numbers.
- **BiLSTM:** an LSTM is a network that processes a sequence step-by-step while remembering context. "Bidirectional" means it reads the slices both left→right and right→left, so each position is informed by what comes before *and* after it. This is how it handles overlapping/leaning characters.
- **Linear layer:** at every one of the 41 slices it outputs a score for each of the 32 classes (31 characters + blank).

### 3.4 CTC — the clever part
We only know the *final string*, not which slice corresponds to which character. CTC handles this mismatch.

- The model emits a label for all 41 slices, e.g. `B B _ U _ 5 5 2 2 _ 2 X` (`_` = blank).
- **Decoding rule:** merge neighbouring duplicates, then delete blanks → `BU522X`.
- The blank is essential: to output a real double like `22`, the model puts a blank between them (`2 _ 2`) so they are not merged into one `2`.
- **CTC loss** during training automatically considers *all* the ways the slices could line up to produce the correct string, so we never have to label positions by hand.

### 3.5 Training (`src/train.py`)
- We hold out **10%** of the data as a **validation set** — images the model never trains on, used to honestly measure progress.
- **Optimizer (Adam):** the algorithm that nudges the model's millions of internal numbers to reduce the loss.
- **Learning rate:** how big those nudges are. Too big = unstable, too small = slow. We start at `1e-3` and **halve it automatically** when validation stops improving.
- **Gradient clipping:** caps the size of updates so an early CTC spike can't blow up training.
- **Best-checkpoint saving:** after each epoch we measure validation CER and only save the model when it improves. So even if later epochs over-fit, our saved model is the best one.

> **Normal behaviour:** for the first 1–3 epochs the model often predicts *all blanks* (CER ≈ 1.0). Then it "breaks through" and CER drops quickly. Don't panic during those first epochs.

### 3.6 Prediction & submission (`src/predict.py`)
We run the best saved model over the 5,000 test images, decode each output to a string, and write a CSV with two columns: `image,prediction`. Rename it to `submission_<name>_<enroll no.>.csv` before submitting.

---

## 4. How to read the results

- **CER** (primary metric): mean edit-distance per character. `0.05` means ~5% of characters need fixing.
- **Exact-match accuracy** (shown in the notebook): fraction of codes predicted *perfectly*. A useful sanity check that's easy to feel.
- **Learning curves:** training loss should fall steadily; validation CER should fall then flatten. If validation CER rises while training loss keeps dropping, the model is **over-fitting** (memorising) — use the saved best checkpoint and/or more augmentation.

---

## 5. Tuning guide (when you want a better score)

Change values in the notebook's `CFG` block (or `src/config.py`). Tune **one thing at a time** and compare validation CER. Ordered by typical impact:

1. **More training.** `epochs = 40–60`. The cheapest win. Watch that validation CER is still improving.
2. **More augmentation** (biggest robustness gain against the blob/noise). In `build_transforms`, add after the affine line:
   - `transforms.RandomErasing(p=0.3)` (applied after `ToTensor`) — simulates the occlusion blob.
   - `transforms.GaussianBlur(3)` — simulates blur.
3. **Model capacity.** `rnn_hidden = 384` or `512`; or widen the image with `img_width = 200` for more time steps. Bigger = slower but can be more accurate.
4. **Learning rate.** Try `lr` in `{3e-4, 5e-4, 1e-3, 2e-3}`. If loss is jumpy, lower it; if learning is very slow, raise it.
5. **Batch size.** On a bigger GPU try `batch_size = 256` (faster epochs). Very small batches train noisier.
6. **Better decoding.** Replace greedy decoding with **beam search**, or exploit the fact that every code is length 6 (keep the 6 most confident non-blank slots).
7. **Ensemble / TTA.** Train 3 models with different `seed`s and take a majority vote per character, or average predictions over light test-time augmentations.

**Practical tuning loop:**
1. Pick one hyper-parameter and a small set of values.
2. Train, record the **best validation CER**.
3. Keep the winner, move to the next hyper-parameter.
4. Only when satisfied, regenerate the submission CSV.

**If you are over-fitting** (val CER ≫ train performance): add augmentation, increase `dropout` (e.g. `0.1 → 0.3`), or increase `weight_decay`.
**If you are under-fitting** (both train and val are poor): train longer, increase model capacity, or raise the learning rate.

---

## 6. Project map

```
CIG_open_project_aiml/
├── notebook/solution.ipynb   # main deliverable: full workflow, runs on Kaggle GPU
├── src/                      # the same logic as reusable modules
│   ├── config.py             # every hyper-parameter
│   ├── charset.py            # text <-> integer mapping (CTC)
│   ├── dataset.py            # loading, transforms, augmentation
│   ├── model.py              # the CRNN architecture
│   ├── utils.py              # CTC decoding, CER metric, seeding
│   ├── train.py              # training loop
│   └── predict.py            # writes submission.csv
├── docs/DOCUMENTATION.md     # this file
├── requirements.txt
└── README.md                 # quick start (local + Kaggle)
```

The notebook is self-contained (it does not import `src/`), so it runs anywhere. The `src/` package is the same solution organised as clean, reusable code for local runs and future experiments.
