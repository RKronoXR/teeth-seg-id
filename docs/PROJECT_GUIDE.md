# Teeth Segmentation and Identification Project Guide

Instance segmentation and FDI tooth identification in panoramic dental radiographs.

This project trains, evaluates, and analyzes Mask R-CNN models for individual tooth segmentation and tooth numbering using the UFBA-425 dataset.

## Project progress

Current estimated progress:

| Area | Estimated progress |
|---|---:|
| Full project | ~65% |
| Reproducible academic baseline | ~80% |
| User-facing software/tooling | ~45-50% |

The current pipeline is strong enough for reproducible academic experimentation. The remaining work is mainly focused on model improvement, external-image inference, packaging, documentation refinement, and usability for non-technical users.

## Main objectives

1. Segment individual teeth in panoramic radiographs.
2. Assign FDI tooth numbers to each segmented tooth.
3. Evaluate detection and segmentation performance using COCO metrics.
4. Analyze performance per tooth.
5. Visualize predictions and worst-case errors.
6. Keep all model runs reproducible and separated so checkpoints are not overwritten.
7. Build toward a usable academic and possibly commercial tool.

## Dataset

Primary dataset:

- UFBA-425
- 425 panoramic radiographs
- Instance-level tooth masks
- Bounding boxes
- FDI tooth numbering annotations
- License: CC BY 4.0

The dataset is not stored in this repository. It is downloaded and processed through the project scripts.

Expected local dataset structure after preparation:

```text
data/
  raw/
  interim/
    UFBA-425/
  processed/
    UFBA-425/
      coco/
        instances_train.json
        instances_val.json
        instances_test.json
```

## Repository structure

```text
configs/
data/
  raw/
  interim/
  processed/
  external/
docs/
notebooks/
outputs/
  checkpoints/
  figures/
  logs/
  predictions/
  reports/
requirements/
scripts/
src/
  teeth_seg_id/
tests/
```

Large generated files are ignored by git, including datasets, checkpoints, logs, predictions, figures, and reports.

## Environment setup

Create and activate a virtual environment:

```bash
cd /home/rkronoxr/Workspace/ACTA-AI-Lab/teeth-seg-id
python3 -m venv .venv
source .venv/bin/activate
```

Install base dependencies:

```bash
pip install -r requirements/base.txt
```

Install PyTorch for CUDA 13:

```bash
pip install -r requirements/torch-cu130.txt
```

Check that PyTorch can access the GPU:

```bash
PYTHONPATH=src .venv/bin/python scripts/check_torch_gpu.py
```

Expected result should show:

```text
CUDA available: True
Device: NVIDIA GB10
```

## Data preparation

Download UFBA-425:

```bash
PYTHONPATH=src .venv/bin/python scripts/download_ufba425.py
```

Extract the dataset:

```bash
PYTHONPATH=src .venv/bin/python scripts/extract_ufba425.py
```

Summarize the dataset:

```bash
PYTHONPATH=src .venv/bin/python scripts/summarize_ufba425.py
```

Create reproducible train, validation, and test splits:

```bash
PYTHONPATH=src .venv/bin/python scripts/create_ufba425_splits.py
```

Convert UFBA-425 annotations to COCO format:

```bash
PYTHONPATH=src .venv/bin/python scripts/convert_ufba425_to_coco.py
```

Check the dataset loader:

```bash
PYTHONPATH=src .venv/bin/python scripts/check_dataset_loader.py
```

## FDI label mapping

The model uses 32 tooth classes plus background.

Class IDs map to FDI numbers in this order:

```python
list(range(11, 19)) + list(range(21, 29)) + list(range(31, 39)) + list(range(41, 49))
```

That means:

```text
1  -> 11
2  -> 12
...
8  -> 18
9  -> 21
...
16 -> 28
17 -> 31
...
24 -> 38
25 -> 41
...
32 -> 48
```

The model uses:

```text
num_classes = 33
```

because the 32 FDI classes are plus one background class.

## Baseline model

Current architecture:

```text
Torchvision Mask R-CNN ResNet-50 FPN
```

Current setup:

```python
maskrcnn_resnet50_fpn(
    weights=None,
    weights_backbone=None,
    num_classes=33,
)
```

No pretrained weights are used in the current baseline.

## Training principles

Every training run should use a unique `RUN_NAME`.

This prevents overwriting previous models:

```bash
RUN_NAME=maskrcnn_experiment_name_$(date +%Y%m%d_%H%M%S)
```

Each run writes to:

```text
outputs/checkpoints/<RUN_NAME>/
outputs/logs/<RUN_NAME>_train_log.csv
outputs/logs/<RUN_NAME>_terminal.log
```

The most important file is:

```text
outputs/checkpoints/<RUN_NAME>/best_model.pth
```

The best model information is stored in:

```text
outputs/checkpoints/<RUN_NAME>/best_model_info.json
```

## Recommended training script

Use:

```text
scripts/train_maskrcnn_val_ap.py
```

This script:

- trains Mask R-CNN
- evaluates validation performance every N epochs
- selects the best model by validation mask AP
- saves `best_model.pth`
- supports early stopping
- supports periodic checkpoints
- avoids saving every epoch unless requested

Recommended command:

```bash
cd /home/rkronoxr/Workspace/ACTA-AI-Lab/teeth-seg-id

RUN_NAME=maskrcnn_b8_lr0002_valap_$(date +%Y%m%d_%H%M%S)

nohup env PYTHONPATH=src .venv/bin/python scripts/train_maskrcnn_val_ap.py \
  --epochs 300 \
  --batch-size 8 \
  --lr 0.0002 \
  --run-name "$RUN_NAME" \
  --val-every 5 \
  --early-stop-patience 25 \
  --save-every 25 \
  > outputs/logs/${RUN_NAME}_terminal.log 2>&1 &

echo "$RUN_NAME"
```

Monitor training:

```bash
tail -f outputs/logs/${RUN_NAME}_terminal.log
```

## Early stopping

`--early-stop-patience` controls how many epochs to wait after the best validation mask AP before stopping.

Example:

```bash
--early-stop-patience 25
```

If the best validation mask AP is found at epoch 85 and there is no improvement for 25 epochs, training stops at epoch 110.

Use:

```bash
--early-stop-patience 0
```

to disable early stopping.

## Periodic checkpoints

By default, the validation AP training script always saves:

```text
best_model.pth
best_model_info.json
```

It only saves epoch checkpoints when `--save-every` is greater than zero.

Example:

```bash
--save-every 25
```

This saves:

```text
epoch_25.pth
epoch_50.pth
epoch_75.pth
...
```

Use:

```bash
--save-every 0
```

to disable periodic checkpoints.

## Photometric augmentation training

Photometric augmentation changes the image appearance but does not change tooth location or FDI labels.

Safe augmentations currently used:

- brightness
- contrast
- gamma
- Gaussian noise
- slight Gaussian blur

Use:

```text
scripts/train_maskrcnn_val_ap_aug.py
```

Recommended command:

```bash
cd /home/rkronoxr/Workspace/ACTA-AI-Lab/teeth-seg-id

RUN_NAME=maskrcnn_b8_lr0002_valap_augphoto_$(date +%Y%m%d_%H%M%S)

nohup env PYTHONPATH=src .venv/bin/python scripts/train_maskrcnn_val_ap_aug.py \
  --epochs 300 \
  --batch-size 8 \
  --lr 0.0002 \
  --run-name "$RUN_NAME" \
  --val-every 5 \
  --early-stop-patience 25 \
  --save-every 25 \
  --augment-photometric \
  --aug-prob 0.8 \
  --brightness 0.15 \
  --contrast 0.15 \
  --gamma 0.15 \
  --noise-std 0.01 \
  --blur-prob 0.15 \
  --blur-sigma 1.0 \
  > outputs/logs/${RUN_NAME}_terminal.log 2>&1 &

echo "$RUN_NAME"
```

Monitor:

```bash
tail -f outputs/logs/${RUN_NAME}_terminal.log
```

Do not use horizontal flip yet unless FDI labels are correctly remapped.

## Evaluation on test set

Evaluate the best model on the test set:

```bash
cd /home/rkronoxr/Workspace/ACTA-AI-Lab/teeth-seg-id

RUN_NAME=<RUN_NAME>

PYTHONPATH=src .venv/bin/python scripts/evaluate_maskrcnn_coco.py \
  --split test \
  --max-images 65 \
  --checkpoint "outputs/checkpoints/${RUN_NAME}/best_model.pth" \
  > "outputs/reports/eval_${RUN_NAME}_best_model_test.txt" 2>&1

grep "Average Precision" "outputs/reports/eval_${RUN_NAME}_best_model_test.txt"
```

COCO output appears twice:

1. First block: bounding boxes.
2. Second block: segmentation masks.

Important metrics:

| Metric | Meaning |
|---|---|
| bbox AP | strict bounding-box detection performance |
| bbox AP50 | easier bounding-box detection at IoU 0.50 |
| mask AP | strict segmentation performance |
| mask AP50 | easier segmentation at IoU 0.50 |

Interpretation:

```text
mask AP50 = did the model find/segment the tooth reasonably well?
mask AP   = how precise are the mask borders across stricter IoU thresholds?
```

## Per-tooth AP evaluation

Evaluate model performance per FDI tooth:

```bash
cd /home/rkronoxr/Workspace/ACTA-AI-Lab/teeth-seg-id

RUN_NAME=<RUN_NAME>

PYTHONPATH=src .venv/bin/python scripts/evaluate_per_tooth_ap.py \
  --split test \
  --checkpoint "outputs/checkpoints/${RUN_NAME}/best_model.pth" \
  --output-csv "outputs/reports/per_tooth_ap_${RUN_NAME}_test.csv" \
  > "outputs/reports/per_tooth_ap_${RUN_NAME}_test.txt" 2>&1
```

The CSV contains:

```text
FDI
n_gt
bbox_AP
bbox_AP50
bbox_AP75
mask_AP
mask_AP50
mask_AP75
```

This helps identify weak teeth.

## Prediction visualization

Generate side-by-side ground truth and prediction images:

```bash
cd /home/rkronoxr/Workspace/ACTA-AI-Lab/teeth-seg-id

RUN_NAME=<RUN_NAME>

PYTHONPATH=src .venv/bin/python scripts/visualize_predictions.py \
  --split test \
  --checkpoint "outputs/checkpoints/${RUN_NAME}/best_model.pth" \
  --count 20 \
  --threshold 0.5 \
  --show-scores
```

Open output folder:

```bash
xdg-open outputs/figures/predictions/test
```

If scores are not needed, remove:

```bash
--show-scores
```

With `--show-scores`, confidence is shown below the tooth label.

## Worst-case tooth visualization

Visualize worst cases for selected teeth:

```bash
cd /home/rkronoxr/Workspace/ACTA-AI-Lab/teeth-seg-id

RUN_NAME=<RUN_NAME>

PYTHONPATH=src .venv/bin/python scripts/visualize_tooth_errors.py \
  --split test \
  --checkpoint "outputs/checkpoints/${RUN_NAME}/best_model.pth" \
  --teeth 24,14,31,41,18 \
  --threshold 0.5 \
  --top-k 10 \
  --output-dir "outputs/figures/tooth_errors_${RUN_NAME}" \
  --csv "outputs/reports/tooth_error_cases_${RUN_NAME}_test.csv"
```

Open output folder:

```bash
xdg-open outputs/figures/tooth_errors_${RUN_NAME}/test
```

The output shows:

1. Original image.
2. Ground truth mask for the selected FDI.
3. Best prediction for that FDI.

If IoU is 0 and score is 0, the model missed the tooth at the chosen threshold.

## Automatic run analysis

Generate a Markdown report for a completed run:

```bash
cd /home/rkronoxr/Workspace/ACTA-AI-Lab/teeth-seg-id

RUN_NAME=<RUN_NAME>

PYTHONPATH=src .venv/bin/python scripts/analyze_training_run.py \
  --run-name "$RUN_NAME" \
  --test-eval "outputs/reports/eval_${RUN_NAME}_best_model_test.txt" \
  --per-tooth-csv "outputs/reports/per_tooth_ap_${RUN_NAME}_test.csv" \
  --tooth-errors-csv "outputs/reports/tooth_error_cases_${RUN_NAME}_test.csv"
```

Report path:

```text
outputs/reports/run_analysis/<RUN_NAME>/<RUN_NAME>_analysis.md
```

Figures path:

```text
outputs/reports/run_analysis/<RUN_NAME>/figures/
```

## Current stable baseline

Current stable baseline:

```text
Run: maskrcnn_b8_lr0002_valap_20260625_210821
Best epoch: 85
Best validation mask AP: 0.5990
Best validation mask AP50: 0.9498
Best validation bbox AP: 0.6269
Test bbox AP: 0.599
Test bbox AP50: 0.941
Test bbox AP75: 0.701
Test mask AP: 0.584
Test mask AP50: 0.936
Test mask AP75: 0.699
```

Earlier baseline:

```text
Run: maskrcnn_b8_lr0008_200epochs_20260625_173140
Best epoch selected manually: 175
Test bbox AP: 0.616
Test bbox AP50: 0.932
Test mask AP: 0.583
Test mask AP50: 0.930
Test mask AP75: 0.715
```

The newer model is preferred as the official baseline because it uses validation AP model selection, early stopping, and stable training, even though the test-set performance is only marginally different.

## Known weak teeth

In the current stable baseline, the weakest teeth by mask AP were:

```text
FDI 24
FDI 14
FDI 12
FDI 32
FDI 41
```

Worst-case analysis also showed IoU 0 cases for:

```text
FDI 18
FDI 41
FDI 14
FDI 31
```

The main pattern is:

```text
mask AP50 is high, but mask AP is lower.
```

This means the model usually finds the teeth, but fine border precision can still improve.

## Current project status

Completed:

- repository structure
- Apache 2.0 license
- dataset documentation
- UFBA-425 download and extraction scripts
- dataset summary
- reproducible train/validation/test split
- COCO conversion
- dataset loader
- GPU verification
- Mask R-CNN baseline
- validation and test evaluation
- per-tooth AP evaluation
- worst-case visualization
- early stopping
- validation AP model selection
- reduced checkpoint storage
- automatic run analysis
- photometric augmentation training script

In progress:

- photometric augmentation experiment

Next steps:

1. Finish photometric augmentation training.
2. Evaluate the augmented model on the test set.
3. Run per-tooth AP for the augmented model.
4. Generate automatic analysis report.
5. Compare augmented model with the stable baseline.
6. If improved, promote it to the new official baseline.
7. If not improved, test:
   - learning-rate scheduling
   - stronger backbone
   - safe geometric augmentation
   - external pretraining
   - post-processing
8. Build inference script for external panoramic radiographs.
9. Add a simple command-line workflow for non-technical users.
10. Prepare a model card.

## Recommended decision workflow after each training run

For every new run:

1. Check `best_model_info.json`.
2. Evaluate `best_model.pth` on test.
3. Run per-tooth AP.
4. Generate prediction visualizations.
5. Generate worst-case visualizations.
6. Run `analyze_training_run.py`.
7. Compare against the official baseline.
8. Promote the new model only if it improves relevant metrics or gives better stability/generalization.

## Avoiding accidental overwrites

Always use:

```bash
RUN_NAME=some_descriptive_name_$(date +%Y%m%d_%H%M%S)
```

Do not reuse an old run name unless the intention is to resume or modify that specific run.

Safe examples:

```bash
RUN_NAME=maskrcnn_b8_lr0002_valap_$(date +%Y%m%d_%H%M%S)
RUN_NAME=maskrcnn_b8_lr0002_valap_augphoto_$(date +%Y%m%d_%H%M%S)
RUN_NAME=maskrcnn_b8_lr0001_resume_$(date +%Y%m%d_%H%M%S)
```

Each creates a separate folder in:

```text
outputs/checkpoints/
```

## License

Code in this repository is licensed under Apache 2.0.

Dataset licensing follows the original UFBA-425 dataset license.
