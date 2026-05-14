# Error analysis (educational project)

This file is a **placeholder** for structured error analysis after you have a trained model and evaluation outputs.

## What belongs here

- **Confusion pairs**: which classes are most often confused (use `reports/figures/confusion_matrix_test.png` and per-class metrics from `src/evaluate.py`).
- **Qualitative buckets**: examples where the model is confident but wrong (high-risk for demos) vs low-confidence cases.
- **Data issues**: label noise, scanner differences, motion artifact, cropping — all affect metrics and generalization.

## Clinical / product safety framing (non-exhaustive)

- This repository demonstrates **software + ML workflow literacy**, not regulatory clearance.
- Avoid language like “the model diagnoses …”; prefer “**educational triage-style prediction**” or “**prototype output**.”
- Any performance numbers are **dataset-specific** and do not automatically transfer to new populations, scanners, or acquisition protocols.
- If you show results publicly, pair metrics with **limitations** and a clear **non-clinical** scope.

When you finish a training run, paste short notes and links to saved figures under `reports/figures/` and JSON under `reports/metrics/`.
