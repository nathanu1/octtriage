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

## Quickstart

1. Create a virtual environment and install dependencies:

   ```bash
   cd oct-triage-assistant
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Add your split dataset under `data/raw/` as shown.

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
