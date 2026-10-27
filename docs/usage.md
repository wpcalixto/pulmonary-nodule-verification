# Usage

All scripts resolve their folders from the project base directory given by the environment variable `PULMONARY_BASE`
(see `docs/parameters.md`). The stages are executed in numerical order; each stage reads the products of the previous ones.

```bash
export PULMONARY_BASE=/path/to/project            # empty working directory
python scripts/00_check_environment.py
python scripts/01_download_datasets.py            # LIDC-IDRI, RIDER Lung CT, LNDb
# ... continue in the order below
```

| stage | file | function |
|---|---|---|
| 00 | `scripts/00_check_environment.py` | Records the execution environment (Python, CUDA, GPU, PyTorch, SimpleITK, nnU-Net, LungMask, statistical libraries, disk space and folder access) before any processing stage. |
| 01 | `scripts/01_download_datasets.py` | Downloads LIDC-IDRI and RIDER Lung CT (TCIA/NBIA API, series by series) and LNDb (Zenodo, file by file) with automatic resumption. |
| 02 | `scripts/02_audit_raw.py` | Audits the raw data: inventory and selection of patients, studies, series, slices and XML annotations, orientation and geometry checks. |
| 03 | `scripts/03_convert_dicom_to_nifti.py` | Converts DICOM series (LIDC-IDRI, RIDER) and MetaImage volumes (LNDb) to NIfTI in Hounsfield units, with slice ordering by physical position. |
| 04 | `scripts/04_build_lidc_annotations.py` | Builds the canonical LIDC-IDRI nodule annotations from the XML route (pylidc 0.2.1) on the canonical image grid. |
| 05 | `scripts/05_build_lndb_annotations.py` | Builds the LNDb annotations from the official masks and challenge CSV files, with manifest-driven case discovery. |
| 06 | `scripts/06_generate_lung_masks_and_crop.py` | Generates lung masks (LungMask R231) and crops LIDC-IDRI, RIDER and LNDb volumes to the lung bounding box. |
| 07 | `scripts/07_draw_bd1_partition.py` | Draws the BD1 partition into development and internal-validation sets and the five cross-validation folds. |
| 08 | `scripts/08_prepare_nnunet.py` | Prepares the nnU-Net v2 dataset tree for the canonical cohort under the adjudicated-or-negative-confirmed labelling policy. |
| 08a | `scripts/08a_calibrate_collapse_monitor.py` | Calibrates the training-collapse monitor rule against archived training logs (no GPU, no image access). |
| 08b | `scripts/08b_probe_gpu_fit.py` | Probes GPU memory to verify that each planned configuration fits the available GPU with the planned batch and patch sizes. |
| 08c | `scripts/08c_freeze_training_config.py` | Freezes, by introspection, the effective training configuration of the ten architectures into the manifest consumed by stage 09. |
| 09 | `scripts/09_train_nnunet.py` | Trains the fifty definitive nnU-Net models (ten configurations, five folds) through the shared training core. |
| 10 | `scripts/10_infer_and_export_predictions.py` | Runs nnU-Net v2 inference and exports binary masks, probability maps and connected components for BD1, RIDER and LNDb. |
| 10a | `scripts/10a_audit_prediction_outputs.py` | Audits the completeness and integrity of the exported prediction products before the post-detection analyses. |
| 11 | `scripts/11_match_and_score_bd1.py` | Performs one-to-one Hungarian matching (IoU) between predicted and reference components in BD1 and computes detection scores. |
| 11b | `scripts/11b_recompute_bd1_size_fields.py` | Recomputes the size fields of the BD1 matched components used by the downstream stages. |
| 12 | `scripts/12_interobserver_analysis.py` | Computes inter-observer agreement (pairwise Dice and mean surface distance) for LIDC-IDRI nodules and LNDb objects. |
| 13 | `scripts/13_prepare_filter_objects.py` | Prepares predicted (TP/FP) and reference (TP/FN) objects of BD1 as regions of interest for the morphological-filter analysis. |
| 14 | `scripts/14_compute_filter_maps_gpu.py` | Computes multiscale 3D morphological filter responses (LoG, Hessian blobness and vesselness, among others) on the BD1 regions of interest using the GPU. |
| 15 | `scripts/15_extract_filter_features.py` | Extracts per-component morphological filter features and the distance to the pleura for BD1, LNDb and RIDER. |
| 16 | `scripts/16_analyze_filters_bd1.py` | Statistical analysis of the filter features in BD1 (TP versus FP predictions; TP versus FN references). |
| 17 | `scripts/17_train_filter_verifier.py` | Trains the morphological post-detection verifier on out-of-fold predictions and freezes its decision threshold. |
| 17a | `scripts/17a_apply_filter_verifier_internal.py` | Applies the frozen verifier to the internal-validation set without re-tuning. |
| 18 | `scripts/18_prepare_rider_registration.py` | Rigidly registers the second RIDER scan to the first scan per patient and architecture. |
| 19 | `scripts/19_match_rider_hungarian_iou.py` | Matches RIDER test-retest components by IoU with the Hungarian algorithm. |
| 20 | `scripts/20_extract_rider_morphometry_and_filters.py` | Extracts morphometry and filter features for each matched RIDER test-retest pair. |
| 21 | `scripts/21_analyze_rider_repeatability.py` | Repeatability analysis on RIDER (Bland-Altman, bootstrap confidence intervals, ICC(3,1), repeatability coefficient). |
| 22 | `scripts/22_evaluate_lndb.py` | External evaluation of the nnU-Net configurations on LNDb without retraining (matching, TP, FP, FN and derived metrics). |
| 22a | `scripts/22a_prepare_lndb_filter_objects.py` | Prepares LNDb predicted and reference objects for the external morphological-filter analysis. |
| 22b | `scripts/22b_compute_lndb_filter_maps.py` | Computes the morphological filter features on the LNDb objects with the same scales and parameters as BD1. |
| 23 | `scripts/23_validate_filters_lndb.py` | External validation of the morphological filters and of the frozen verifier on LNDb. |
| 24 | `scripts/24_integrated_statistics.py` | Integrates all results: patient-level bootstrap confidence intervals and Holm correction across comparisons. |
| 25 | `scripts/25_generate_tables_and_figures.py` | Generates the tables and figures of the manuscript from the stage outputs. |
| 26 | `scripts/26_reproducibility_audit.py` | Reproducibility audit: arithmetic identities, leakage checks, mask alignment, seeds, frozen parameters and manuscript consistency. |
| 27 | `scripts/27_export_manuscript_package.py` | Assembles the manuscript results package (main and supplementary results, LaTeX tables, captions and figures). |

Stages 08–10 (nnU-Net preparation, training and inference) require a CUDA GPU. Training was executed with
`scripts/09_train_nnunet.py` (local GPUs) and `scripts/conduzir_colab_v4.py` / `notebooks/pulmonary_colab_training.ipynb` (Google Colab).
The trained checkpoints are provided in `models/`, so stages 10 onwards can be reproduced without retraining.

To reproduce the post-detection analysis only, place the archived stage outputs from `data/stage_outputs_10a_25/` in the folders
expected by each script (see the run manifests inside each stage folder) and run stages 11 to 25.
