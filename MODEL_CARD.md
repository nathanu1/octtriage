# Model Card: OCTTriage EfficientNet Baseline

## Status

**Prototype specification; no trained checkpoint is distributed.** Any checkpoint produced by
this repository is an experimental artifact and not a cleared medical device.

## Intended use

This model is intended to support research on reordering an ophthalmologist's OCT reading queue.
It produces a provisional urgency tier and supporting model-attention overlay for mandatory human
review. It does not diagnose disease, recommend treatment, replace image interpretation, or
communicate results directly to patients.

## Training data

The initial pipeline targets the Kermany OCT2017 Kaggle distribution and its `CNV`, `DME`,
`DRUSEN`, and `NORMAL` image labels. The preparation code derives patient groups from filenames,
consolidates upstream folders, and creates deterministic train/validation/test splits with no
patient appearing in multiple splits.

The dataset does not establish performance across scanner manufacturers, acquisition protocols,
care settings, demographic groups, comorbidities, or contemporary clinical prevalence. It also
does not provide trustworthy severity labels for the proposed severity output.

## Model and outputs

- ImageNet-initialized EfficientNet-B0, replaceable with ResNet-18 through configuration
- Four-class softmax output with a temperature fitted on the validation split (constrained to
  T ≥ 1, so it can only soften confidence)
- Horizontal-flip test-time augmentation, with agreement measured within each B-scan
- Grad-CAM overlay for the finding that drives the tier, on the B-scan with the strongest evidence
- Separate heuristic quality gate
- Config-driven tier mapping with uncertainty escalation

For multi-frame DICOM and multi-page TIFF input, at most 32 usable frames are sampled.
Pathology-class evidence uses the maximum observed class probability across frames and views so
focal evidence is not averaged away; reassuring `NORMAL` evidence uses the minimum, so a volume
reads as normal only when every view supports it. This choice may increase false positives and
requires validation.

## Primary evaluation target

Urgent-class recall, where `CNV` and `DME` constitute the prototype urgent set, is the primary
checkpoint-selection metric. Required reporting also includes:

- Per-class sensitivity and specificity
- Per-class one-vs-rest AUROC
- Confusion matrix
- Expected calibration error (calibrated and uncalibrated) and reliability diagram
- Performance stratified by site, device, demographic group, and image quality when metadata exists
- Policy-level outcomes from `oct-evaluate`: the share of urgent cases reaching the `URGENT` tier
  and the share of non-urgent cases flagged `URGENT` after the quality gate, aggregation, and
  uncertainty escalation are applied
- Patient-level bootstrap confidence intervals for those rates and for urgent recall
- Promotion thresholds chosen on the validation split with `oct-tune-thresholds`, locked, and only
  then reported on the test split

Accuracy alone is not an acceptable performance claim.

## Known limitations and risks

- OCT2017 is an image-classification dataset, not a prospective urgency-triage cohort.
- Kermany filenames are used as prototype patient identifiers; the derived grouping must be
  independently verified against source documentation before claiming patient independence.
  A patient with images under several labels is kept in one split and stratified by their most
  frequent label.
- DME severity is unavailable and therefore remains `UNKNOWN`; the default rule escalates it.
- The model has no validated capability for macular hole, retinal detachment, vitreomacular
  traction, or pathology absent from the training labels.
- Temperature scaling calibrates confidence to accuracy on the Kermany validation split only. It
  is not a probability of disease, and calibration is expected to degrade under scanner,
  population, and prevalence shift. The uncertainty thresholds act on this calibrated confidence.
- Grad-CAM can be plausible while a prediction is wrong and must not be interpreted as causal.
- Quality heuristics may reject valid scans or accept artifacts.
- Scanner shift, demographic shift, prevalence shift, and adversarial/corrupted inputs are not
  comprehensively addressed.
- A high urgent recall can coexist with an unusably high false-positive rate; both must be reviewed.

## Fail-safe policy

Quality failure, low confidence, high predictive entropy, and augmentation disagreement escalate
to urgent clinician review. Multiple candidate findings resolve to the highest urgency. The model
cannot automatically downgrade a case; clinician changes create append-only override records.

## Out-of-scope use

- Clinical diagnosis or treatment selection
- Screening or patient-facing reassurance
- Autonomous queue removal or delayed care
- Use on modalities other than retinal OCT
- Use on pediatric, emergency, or population-screening cohorts without specific validation

## Requirements before clinical deployment

Deployment would require, at minimum, a formal intended-use statement, clinical risk management,
locked data and model governance, cybersecurity and privacy controls, human-factors testing,
multi-site external validation, prospective workflow evaluation, monitoring and rollback plans,
and applicable FDA Software as a Medical Device and/or CE-marking review. These activities are not
completed by this repository.
