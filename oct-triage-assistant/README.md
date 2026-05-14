# Image-Based Retinal OCT Triage Assistant

Educational medical computer vision project: classify retinal **OCT** (optical coherence tomography) B-scans into four **triage-style** categories (**CNV**, **DME**, **DRUSEN**, **NORMAL**). Built for learning, portfolio work, and responsible AI practice.

**This is not a clinical diagnostic tool.** Outputs are **educational predictions** only.

## Disclaimer

This software is for **education and research demonstration** only. It is **not** a medical device, **not** FDA-cleared (or equivalent), and **must not** be used for diagnosis, treatment decisions, or patient triage in real care. Always rely on qualified clinicians and appropriate imaging workflows.

## Project layout

| Path | Role |
|------|------|
| `data/raw/` | Train / val / test images in class-named subfolders (see below). |
| `data/processed/` | Optional space for preprocessed exports. |
| `data/sample_images/` | A few de-identified examples for demos (add your own). |
| `src/config.py` | Paths, hyperparameters, class names, device. |
| `src/dataset.py` | PyTorch `DataLoader` wiring via `ImageFolder`. |
| `src/transforms.py` | Train vs validation image preprocessing (ImageNet norms for ResNet). |
| `src/model.py` | ResNet18 backbone + replaced classifier head (transfer learning). |
| `src/train.py` | Training + validation loops and checkpoint saving. |
| `src/evaluate.py` | Test-set metrics, confusion matrix, classification report. |
| `src/predict.py` | Single-image inference from a saved checkpoint. |
| `app/streamlit_app.py` | Local demo UI with safety copy and escalation hints. |
| `models/saved_models/` | Default location for `best_resnet18.pt` (created after training). |
| `reports/` | Figures, metric tables, and error-analysis notes. |
| `scripts/prepare_kermany2018.py` | Download **Kermany 2018** from Kaggle via `kagglehub` and populate `data/raw/`. |

## Dataset layout (expected)

Place images under `data/raw/` like this (supported formats: common image types readable by Pillow, e.g. `.png`, `.jpg`):

```text
data/raw/
├── train/
│   ├── CNV/
│   ├── DME/
│   ├── DRUSEN/
│   └── NORMAL/
├── val/
│   ├── CNV/
│   ├── DME/
│   ├── DRUSEN/
│   └── NORMAL/
└── test/
    ├── CNV/
    ├── DME/
    ├── DRUSEN/
    └── NORMAL/
```

`torchvision.datasets.ImageFolder` learns class names from folder names. Use **exact** folder names above so labels stay consistent with `src/config.py`.

## Option: Kermany 2018 (Kaggle) via `kagglehub`

The public dataset [`paultimothymooney/kermany2018`](https://www.kaggle.com/datasets/paultimothymooney/kermany2018) matches the four classes used here. The archive is **large (~11 GB)**; `kagglehub` caches it under your user directory after the first successful download.

1. Install deps (includes `kagglehub`) and stay in the project root `oct-triage-assistant/`.
2. If the dataset is not public for your account, configure Kaggle credentials as described in the [kagglehub README](https://github.com/Kaggle/kagglehub/blob/main/README.md) (for example `KAGGLE_USERNAME` / `KAGGLE_KEY`, or `kagglehub login`).
3. Run the preparation script (default: **symlinks** into `data/raw/` so you do not duplicate the whole cache on disk):

   ```bash
   python3 scripts/prepare_kermany2018.py
   ```

   Useful flags:

   - `--source /path/to/extracted/root` — skip download; point at an already-extracted folder (often contains `OCT2017/train` and `OCT2017/test`).
   - `--val-fraction 0.15` — fraction of **official train** images reserved for validation when an official `test/` split exists.
   - `--mode copy` — physically copy files into `data/raw/` instead of symlinking (uses a lot of disk).
   - `--download-only` — only download/resolve the dataset path and exit (no changes under `data/raw/`).

When the archive contains **official `train/`, `val/`, and `test/`** (typical `OCT2017/` layout), the script maps those directly into `data/raw/` (publisher split). If only `train/` + `test/` exist, it maps **test** into `data/raw/test/` and randomly holds out **`--val-fraction`** of **train** for `data/raw/val/`. The published **val** set can be small; that is normal for this dataset.

## Quickstart

1. Create a virtual environment and install dependencies:

   ```bash
   cd oct-triage-assistant
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Add your split dataset under `data/raw/` as shown, **or** run `python3 scripts/prepare_kermany2018.py` (see above).

3. Train a baseline ResNet18 classifier:

   ```bash
   python -m src.train
   ```

4. Evaluate on the held-out test set:

   ```bash
   python -m src.evaluate
   ```

5. Run inference on one image:

   ```bash
   python -m src.predict --image path/to/image.png
   ```

6. Launch the Streamlit demo:

   ```bash
   streamlit run app/streamlit_app.py
   ```

## Responsible AI

- Language in the app and docs avoids claiming **diagnosis**; we describe **educational** / **triage-style** outputs.
- Low-confidence predictions trigger a message recommending **human expert review**.
- Predictions of **CNV** or **DME** show a reminder that **clinical review** is appropriate (still not a diagnosis).
- Report metrics honestly on a fixed test split; do not overstate generalization to new devices or populations.

## License

Add a license of your choice before publishing to GitHub.
