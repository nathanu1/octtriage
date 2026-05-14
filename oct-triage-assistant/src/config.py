"""
Central configuration for paths, label vocabulary, and training hyperparameters.

Keeping constants here avoids magic numbers scattered across scripts and makes
experiments easier to reproduce: you change one file, not six.
"""

from __future__ import annotations

from pathlib import Path

import torch

# Project root = parent of `src/` (run scripts from repo root: `oct-triage-assistant/`)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Data locations (match README tree) ---
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
TRAIN_DIR = RAW_DATA_DIR / "train"
VAL_DIR = RAW_DATA_DIR / "val"
TEST_DIR = RAW_DATA_DIR / "test"

PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_IMAGES_DIR = DATA_DIR / "sample_images"

# --- Label space (must match subdirectory names under train/val/test) ---
# ImageFolder sorts folder names; this order matches sorted(["CNV","DME","DRUSEN","NORMAL"]).
CLASSES: tuple[str, ...] = ("CNV", "DME", "DRUSEN", "NORMAL")
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}

# --- Model IO ---
MODELS_DIR = PROJECT_ROOT / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved_models"
CHECKPOINT_FILENAME = "best_resnet18.pt"
CHECKPOINT_PATH = SAVED_MODELS_DIR / CHECKPOINT_FILENAME

# --- Training / image defaults ---
IMG_SIZE = 224  # Standard for torchvision ResNet weights trained on ImageNet
BATCH_SIZE = 32
NUM_WORKERS = 4  # Set to 0 on Windows if DataLoader workers cause issues
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42

# --- Device ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Inference / demo thresholds (not clinical thresholds) ---
LOW_CONFIDENCE_THRESHOLD = 0.55  # Below this max probability → "human review" hint
