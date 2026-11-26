# File-level documentation

Every code file, its function, the inputs and outputs detected in the code, and its provenance. `Executed version verified` = the SHA-256 of the source file equals the hash recorded by the final run of that stage.

## `patches/polylr.py`

- **Type:** Patch
- **Purpose:** Polynomial learning-rate scheduler patch applied to nnU-Net v2.8.1.
- **Inputs detected:** —
- **Outputs detected:** —
- **Third-party dependencies:** torch
- **Original SHA-256 (before re-annotation):** `fd3a8df4fa384d6a8d1912484805f72b936e46d6182bb93c368948efb89ce0a9`
- **Executed version verified:** not applicable / not recorded
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/00_check_environment.py`

- **Type:** Script
- **Purpose:** Records the execution environment (Python, CUDA, GPU, PyTorch, SimpleITK, nnU-Net, LungMask, statistical libraries, disk space and folder access) before any processing stage.
- **Inputs detected:** —
- **Outputs detected:** —
- **Third-party dependencies:** SimpleITK, lungmask, nnunetv2, torch
- **Original SHA-256 (before re-annotation):** `e8a95154258e106aacea4056fb01b1f96a2bb625c2f094e5fce89645ec50d8a1`
- **Executed version verified:** no
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/01_download_datasets.py`

- **Type:** Script
- **Purpose:** Downloads LIDC-IDRI and RIDER Lung CT (TCIA/NBIA API, series by series) and LNDb (Zenodo, file by file) with automatic resumption.
- **Inputs detected:** —
- **Outputs detected:** —
- **Third-party dependencies:** requests, tcia_utils
- **Original SHA-256 (before re-annotation):** `3732da339d5f368c485aaed640277c0b9afb6ebc3cd7649d9a6bbb423edf7b9f`
- **Executed version verified:** no
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/02_audit_raw.py`

- **Type:** Script
- **Purpose:** Audits the raw data: inventory and selection of patients, studies, series, slices and XML annotations, orientation and geometry checks.
- **Inputs detected:** `rider_pair_selection.json`
- **Outputs detected:** `benchmark.json`, `run_summary.json`
- **Third-party dependencies:** numpy, pydicom
- **Original SHA-256 (before re-annotation):** `7b4597013f3e451c9d156510a72618b15f5baa767f5f5b7bd5810728bd1f2ddb`
- **Executed version verified:** no
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/03_convert_dicom_to_nifti.py`

- **Type:** Script
- **Purpose:** Converts DICOM series (LIDC-IDRI, RIDER) and MetaImage volumes (LNDb) to NIfTI in Hounsfield units, with slice ordering by physical position.
- **Inputs detected:** `CURRENT_NIFTI_CONVERSION.json`
- **Outputs detected:** —
- **Third-party dependencies:** SimpleITK, numpy, pydicom
- **Original SHA-256 (before re-annotation):** `00341582e7a99b00fcfe207b6b2bb95482e940e87e6048b894508bb87d440427`
- **Executed version verified:** no
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/04_build_lidc_annotations.py`

- **Type:** Script
- **Purpose:** Builds the canonical LIDC-IDRI nodule annotations from the XML route (pylidc 0.2.1) on the canonical image grid.
- **Inputs detected:** `report_04.json`
- **Outputs detected:** —
- **Third-party dependencies:** SimpleITK, numpy, pylidc
- **Original SHA-256 (before re-annotation):** `aa96ca4900e58303445ce5b2ee0db287cefdc0bcdd6b867b1a2a4f44a1a22f28`
- **Executed version verified:** no
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/05_build_lndb_annotations.py`

- **Type:** Script
- **Purpose:** Builds the LNDb annotations from the official masks and challenge CSV files, with manifest-driven case discovery.
- **Inputs detected:** `lndb_alternate_label_set.json`
- **Outputs detected:** —
- **Third-party dependencies:** SimpleITK, numpy, scipy
- **Original SHA-256 (before re-annotation):** `180176c86c93533ac399aceb46c13cb2e73f1753df4bff56e99ac11a0ff423e6`
- **Executed version verified:** no
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/06_generate_lung_masks_and_crop.py`

- **Type:** Script
- **Purpose:** Generates lung masks (LungMask R231) and crops LIDC-IDRI, RIDER and LNDb volumes to the lung bounding box.
- **Inputs detected:** `crop_transform.json`, `INVALID.json`, `manifest.json`
- **Outputs detected:** `ct.nii.gz`, `lung_mask.nii.gz`, `lung_mask_crop.nii.gz`
- **Third-party dependencies:** SimpleITK, lungmask, numpy, torch
- **Original SHA-256 (before re-annotation):** `a1ecfd199daca0f20c570c0927073be996540d6889446166637a7e8966e82cdb`
- **Executed version verified:** no
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/07_draw_bd1_partition.py`

- **Type:** Script
- **Purpose:** Draws the BD1 partition into development and internal-validation sets and the five cross-validation folds.
- **Inputs detected:** —
- **Outputs detected:** —
- **Third-party dependencies:** sklearn
- **Original SHA-256 (before re-annotation):** `a69fcb6299f093b35a4bf26154f263286ca58ccf073dbaa264d3fced23c523ee`
- **Executed version verified:** no
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/08_prepare_nnunet.py`

- **Type:** Script
- **Purpose:** Prepares the nnU-Net v2 dataset tree for the canonical cohort under the adjudicated-or-negative-confirmed labelling policy.
- **Inputs detected:** `dataset.json`, `internal_pending.csv`, `INVALID.json`
- **Outputs detected:** `.stdout.log`, `.stderr.log`
- **Third-party dependencies:** SimpleITK
- **Original SHA-256 (before re-annotation):** `ef1ebe7d04e959a743e93580893839a278a1e29de26ff8c8074af83fec6e7463`
- **Executed version verified:** no
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/08a_calibrate_collapse_monitor.py`

- **Type:** Script
- **Purpose:** Calibrates the training-collapse monitor rule against archived training logs (no GPU, no image access).
- **Inputs detected:** —
- **Outputs detected:** —
- **Third-party dependencies:** —
- **Original SHA-256 (before re-annotation):** `d4c4bf6e57c573c975aee8599fa180d60f9338b65243b870aa01f06723852f9c`
- **Executed version verified:** no
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/08b_probe_gpu_fit.py`

- **Type:** Script
- **Purpose:** Probes GPU memory to verify that each planned configuration fits the available GPU with the planned batch and patch sizes.
- **Inputs detected:** `dataset.json`, `probe.json`
- **Outputs detected:** —
- **Third-party dependencies:** nnunetv2, torch
- **Original SHA-256 (before re-annotation):** `ad8873a122b6dff18132a08800c64df6932808c4e843d2f3279633d87f7641bc`
- **Executed version verified:** no
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/08c_freeze_training_config.py`

- **Type:** Script
- **Purpose:** Freezes, by introspection, the effective training configuration of the ten architectures into the manifest consumed by stage 09.
- **Inputs detected:** `dataset.json`
- **Outputs detected:** —
- **Third-party dependencies:** nnunetv2, torch
- **Original SHA-256 (before re-annotation):** `1e462b19f246e596badd28dc4b8b09ddf076b64783a56ac38f18ea42ce5dad5c`
- **Executed version verified:** no
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/09_train_nnunet.py`

- **Type:** Script
- **Purpose:** Trains the fifty definitive nnU-Net models (ten configurations, five folds) through the shared training core.
- **Inputs detected:** `registos_parciais.json`, `folds_ensaio.json`
- **Outputs detected:** —
- **Third-party dependencies:** —
- **Original SHA-256 (before re-annotation):** `89d2845db3eb71a93baa85d3493d1fafd155589adaf74e6c38999cc18bddb8a8`
- **Executed version verified:** no
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/10_infer_and_export_predictions.py`

- **Type:** Script
- **Purpose:** Runs nnU-Net v2 inference and exports binary masks, probability maps and connected components for BD1, RIDER and LNDb.
- **Inputs detected:** —
- **Outputs detected:** —
- **Third-party dependencies:** SimpleITK, numpy, scipy
- **Original SHA-256 (before re-annotation):** `5f1c5c09b7556b4be3ea582f5d1562b0b68f2eac3dc9277ad28a25480ddc8b9f`
- **Executed version verified:** no
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/10a_audit_prediction_outputs.py`

- **Type:** Script
- **Purpose:** Audits the completeness and integrity of the exported prediction products before the post-detection analyses.
- **Inputs detected:** `invalidate_downstream.txt`, `prediction_gate.json`
- **Outputs detected:** —
- **Third-party dependencies:** SimpleITK, numpy, scipy
- **Original SHA-256 (before re-annotation):** `57bb87cd92811a3bc6ce82e2a94dd37f9b3b3bf2e4ccf356b61fdb2fd0ad0d12`
- **Executed version verified:** yes
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/11_match_and_score_bd1.py`

- **Type:** Script
- **Purpose:** Performs one-to-one Hungarian matching (IoU) between predicted and reference components in BD1 and computes detection scores.
- **Inputs detected:** —
- **Outputs detected:** —
- **Third-party dependencies:** SimpleITK, numpy, scipy
- **Original SHA-256 (before re-annotation):** `0c47c9b408c1b57a388f0e993516333061ded01b2fda1006e44fcb11ff4d5ea3`
- **Executed version verified:** yes
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/11b_recompute_bd1_size_fields.py`

- **Type:** Script
- **Purpose:** Recomputes the size fields of the BD1 matched components used by the downstream stages.
- **Inputs detected:** `bd1_split.csv`, `bd1_object_matches.csv`, `bd1_segmentation_metrics.csv`, `gt_axial_diameters.csv`
- **Outputs detected:** `SIZE_REPAIR_GATE.json`
- **Third-party dependencies:** SimpleITK, numpy, scipy
- **Original SHA-256 (before re-annotation):** `417007a0a9674dadc2932b369b881afca85f65ed8419090271f05222501f5c1b`
- **Executed version verified:** yes
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/12_interobserver_analysis.py`

- **Type:** Script
- **Purpose:** Computes inter-observer agreement (pairwise Dice and mean surface distance) for LIDC-IDRI nodules and LNDb objects.
- **Inputs detected:** `report_12.json`
- **Outputs detected:** —
- **Third-party dependencies:** SimpleITK, numpy, scipy
- **Original SHA-256 (before re-annotation):** `ef002c1f9f557f9f1c4d274a151dc667e66b8f35b73d2fbfa5ccae016d913a52`
- **Executed version verified:** yes
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/13_prepare_filter_objects.py`

- **Type:** Script
- **Purpose:** Prepares predicted (TP/FP) and reference (TP/FN) objects of BD1 as regions of interest for the morphological-filter analysis.
- **Inputs detected:** `bd1_split.csv`, `bd1_folds.json`, `bd1_object_rois_manifest.json`
- **Outputs detected:** —
- **Third-party dependencies:** SimpleITK, numpy, pandas, scipy
- **Original SHA-256 (before re-annotation):** `18533b0b1d53c14d46ec08637b3629afdc9846f1dcb2d04630fb8cabcdc3762a`
- **Executed version verified:** yes
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/14_compute_filter_maps_gpu.py`

- **Type:** Script
- **Purpose:** Computes multiscale 3D morphological filter responses (LoG, Hessian blobness and vesselness, among others) on the BD1 regions of interest using the GPU.
- **Inputs detected:** —
- **Outputs detected:** `{name}.nii.gz`, `roi_failures{suf}.json`, `filter_features_manifest{suf}.json`
- **Third-party dependencies:** SimpleITK, numpy, pandas, scipy, torch
- **Original SHA-256 (before re-annotation):** `bbf4beb5d83cafd79d0c1263b892eb3e7ecdc72b848b3bd438b983f7e9dfadf6`
- **Executed version verified:** yes
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/15_extract_filter_features.py`

- **Type:** Script
- **Purpose:** Extracts per-component morphological filter features and the distance to the pleura for BD1, LNDb and RIDER.
- **Inputs detected:** —
- **Outputs detected:** `{args.dataset}_filter_features_manifest.json`
- **Third-party dependencies:** SimpleITK, numpy, pandas, scipy
- **Original SHA-256 (before re-annotation):** `d9adb7af9bf7c0b39c2020118abafd5e85ccf2befb595319e2e6a47267468497`
- **Executed version verified:** yes
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/16_analyze_filters_bd1.py`

- **Type:** Script
- **Purpose:** Statistical analysis of the filter features in BD1 (TP versus FP predictions; TP versus FN references).
- **Inputs detected:** —
- **Outputs detected:** `filter_bd1_analysis_manifest.json`
- **Third-party dependencies:** numpy, pandas, scipy, sklearn, statsmodels
- **Original SHA-256 (before re-annotation):** `8f02b14c81c5651ad2280b0f8518866d41539a13ef785f27180ebebe1251b59c`
- **Executed version verified:** yes
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/17_train_filter_verifier.py`

- **Type:** Script
- **Purpose:** Trains the morphological post-detection verifier on out-of-fold predictions and freezes its decision threshold.
- **Inputs detected:** —
- **Outputs detected:** `filter_verifier_threshold.json`, `filter_verifier_manifest.json`
- **Third-party dependencies:** numpy, pandas, sklearn
- **Original SHA-256 (before re-annotation):** `dee21d811bb8c277427958d3dfa4a1143b52e257ba3897eeb28052d9bffdb06a`
- **Executed version verified:** yes
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/17a_apply_filter_verifier_internal.py`

- **Type:** Script
- **Purpose:** Applies the frozen verifier to the internal-validation set without re-tuning.
- **Inputs detected:** `SHA256SUMS.txt`
- **Outputs detected:** `internal_verifier_metrics.csv`, `gates.json`, `manifest.json`
- **Third-party dependencies:** numpy, pandas
- **Original SHA-256 (before re-annotation):** `d955c91b6865e9c4a9efd870d353582162edc2dd0bc5e7a1997fe34c4308931e`
- **Executed version verified:** yes
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/18_prepare_rider_registration.py`

- **Type:** Script
- **Purpose:** Rigidly registers the second RIDER scan to the first scan per patient and architecture.
- **Inputs detected:** —
- **Outputs detected:** `input_manifest.csv`, `rider_registration_qc.csv`, `rider_registration_architecture_qc.csv`, `output_hashes.csv`, `manifest.json`
- **Third-party dependencies:** SimpleITK, numpy, pandas, scipy
- **Original SHA-256 (before re-annotation):** `d52a2202e7c9afa10a00a8bbf752f5f1e70caab38bd0872e29e41c380554c017`
- **Executed version verified:** yes
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/19_match_rider_hungarian_iou.py`

- **Type:** Script
- **Purpose:** Matches RIDER test-retest components by IoU with the Hungarian algorithm.
- **Inputs detected:** `output_hashes.csv`
- **Outputs detected:** `manifest.json`
- **Third-party dependencies:** SimpleITK, numpy, pandas, scipy
- **Original SHA-256 (before re-annotation):** `fe772431b8bda3323fc66badc0df8da86df82862b21c52d38a2629319851b965`
- **Executed version verified:** yes
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/20_extract_rider_morphometry_and_filters.py`

- **Type:** Script
- **Purpose:** Extracts morphometry and filter features for each matched RIDER test-retest pair.
- **Inputs detected:** `output_hashes.csv`
- **Outputs detected:** `input_manifest.csv`, `parity_report.json`, `gates.json`, `manifest.json`
- **Third-party dependencies:** SimpleITK, numpy, pandas, scipy
- **Original SHA-256 (before re-annotation):** `5954b870bf3c76ce5fa22b4aba8b1b26eef048f4b21a11e2e2241c81f7a3afcc`
- **Executed version verified:** yes
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/21_analyze_rider_repeatability.py`

- **Type:** Script
- **Purpose:** Repeatability analysis on RIDER (Bland-Altman, bootstrap confidence intervals, ICC(3,1), repeatability coefficient).
- **Inputs detected:** —
- **Outputs detected:** `gates.json`, `manifest.json`
- **Third-party dependencies:** numpy, pandas, scipy
- **Original SHA-256 (before re-annotation):** `ce238a9307d435d0a9e94acb3701effe695c93e65f0a4adf8325f6b020b4bb4f`
- **Executed version verified:** yes
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/22_evaluate_lndb.py`

- **Type:** Script
- **Purpose:** External evaluation of the nnU-Net configurations on LNDb without retraining (matching, TP, FP, FN and derived metrics).
- **Inputs detected:** —
- **Outputs detected:** `input_manifest.csv`, `gates.json`, `manifest.json`
- **Third-party dependencies:** SimpleITK, numpy, pandas, scipy
- **Original SHA-256 (before re-annotation):** `e5e2bb9288fb06ed09830f42ee59f5bfb7199e61434bea8ec8fdb72262f5c7fc`
- **Executed version verified:** yes
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/22a_prepare_lndb_filter_objects.py`

- **Type:** Script
- **Purpose:** Prepares LNDb predicted and reference objects for the external morphological-filter analysis.
- **Inputs detected:** `output_hashes.csv`
- **Outputs detected:** `failures.json`, `gates.json`, `manifest.json`
- **Third-party dependencies:** SimpleITK, numpy, pandas, scipy
- **Original SHA-256 (before re-annotation):** `ea4821d3f123f2356569b5375fc2337864f05906fa6ff7d94a0cb37cf0036174`
- **Executed version verified:** yes
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/22b_compute_lndb_filter_maps.py`

- **Type:** Script
- **Purpose:** Computes the morphological filter features on the LNDb objects with the same scales and parameters as BD1.
- **Inputs detected:** `output_hashes.csv`
- **Outputs detected:** `parity_report.json`, `gates.json`, `manifest.json`
- **Third-party dependencies:** SimpleITK, numpy, pandas, scipy
- **Original SHA-256 (before re-annotation):** `fb55b481c5939dbae41eb29adc9fc1e781d8514b41a24e838b42177cc25c934d`
- **Executed version verified:** yes
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/23_validate_filters_lndb.py`

- **Type:** Script
- **Purpose:** External validation of the morphological filters and of the frozen verifier on LNDb.
- **Inputs detected:** `output_hashes.csv`
- **Outputs detected:** `gates.json`, `manifest.json`
- **Third-party dependencies:** numpy, pandas, scipy, sklearn
- **Original SHA-256 (before re-annotation):** `e326337fc88ef7a9fd21f8220198eeb32d25ad190353631413fc4e2a78bafa3d`
- **Executed version verified:** yes
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/24_integrated_statistics.py`

- **Type:** Script
- **Purpose:** Integrates all results: patient-level bootstrap confidence intervals and Holm correction across comparisons.
- **Inputs detected:** `analysis/bd1_object_matches.csv`, `analysis/bd1_detection_metrics.csv`, `analysis/bd1_size_stratified_metrics.csv`, `figdata/filter_bd1_auc.csv`, `lndb_filter_group_comparisons.csv`, `lndb_external_metrics.csv`, `lndb_object_matches.csv`, `rider_repeatability_stats.csv`
- **Outputs detected:** `analysis/report_12.json`, `gates.json`, `manifest.json`
- **Third-party dependencies:** numpy, pandas, scipy
- **Original SHA-256 (before re-annotation):** `0b8a7582385d283b8ca0f4153b9c4d87fdcdbcd61511a1ffe763cdb75438d86a`
- **Executed version verified:** yes
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/25_generate_tables_and_figures.py`

- **Type:** Script
- **Purpose:** Generates the tables and figures of the manuscript from the stage outputs.
- **Inputs detected:** —
- **Outputs detected:** `manifest_figures.csv`, `manifest_tables.csv`, `input_manifest.csv`, `gates.json`, `manifest.json`
- **Third-party dependencies:** pandas
- **Original SHA-256 (before re-annotation):** `305646f8a6ff94f4564f19d2189a99b224f347d9a74161514576482756dcaed0`
- **Executed version verified:** yes
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/26_reproducibility_audit.py`

- **Type:** Script
- **Purpose:** Reproducibility audit: arithmetic identities, leakage checks, mask alignment, seeds, frozen parameters and manuscript consistency.
- **Inputs detected:** `rider_iou_audit.csv`, `lndb_scan_audit.csv`, `final_audit_report.md`
- **Outputs detected:** `reproducibility_checks.csv`, `input_manifest.csv`, `gates.json`, `manifest.json`
- **Third-party dependencies:** SimpleITK, numpy, pandas
- **Original SHA-256 (before re-annotation):** `b0da39830fdc1657b3a33d9c547bdc0b633b0a32e9787dd5ba9ae7e0bfe33895`
- **Executed version verified:** no
- **Logic unchanged (AST comparison after re-annotation):** yes


## `scripts/caminhos_canonicos.py`

- **Type:** Module
- **Purpose:** Shared module that resolves the canonical directory layout relative to the project base directory.
- **Inputs detected:** —
- **Outputs detected:** —
- **Third-party dependencies:** —
- **Original SHA-256 (before re-annotation):** `9a1becf4996e78cd41bd9c8c8372264a012a578ba7658898ab597338d8f424ff`
- **Executed version verified:** not applicable / not recorded
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/conduzir_colab_v4.py`

- **Type:** Script
- **Purpose:** Fold conductor for Google Colab: runs the folds of one architecture in order, advancing only after verifying the complete artefact list.
- **Inputs detected:** `{PRE}/{DS}/splits_final.json`
- **Outputs detected:** —
- **Third-party dependencies:** torch
- **Original SHA-256 (before re-annotation):** `f54f1e343eed5cf81beba9a030e31da1d4af6f67779a56b4b65b66c888057e7d`
- **Executed version verified:** not applicable / not recorded
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/lidc_grid.py`

- **Type:** Module
- **Purpose:** Shared utilities for the canonical LIDC-IDRI image grid.
- **Inputs detected:** —
- **Outputs detected:** —
- **Third-party dependencies:** SimpleITK, numpy
- **Original SHA-256 (before re-annotation):** `77b5c0cf087ba6179d6ed27b051f0245a90ce349d53766cd1793a08b6bbdcb8a`
- **Executed version verified:** not applicable / not recorded
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/pulmonary_base.py`

- **Type:** Module
- **Purpose:** Shared module that resolves the project base directory from the PULMONARY_BASE environment variable.
- **Inputs detected:** —
- **Outputs detected:** —
- **Third-party dependencies:** SimpleITK, numpy
- **Original SHA-256 (before re-annotation):** `dd3a227f7bdd9b1a1d9b2d98932f12956e301aa0d663ca2b1e4a50901de967f3`
- **Executed version verified:** not applicable / not recorded
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/treino_core.py`

- **Type:** Module
- **Purpose:** Shared training core: architecture registry, nnU-Net invocation, collapse monitor, provenance and per-fold state.
- **Inputs detected:** —
- **Outputs detected:** —
- **Third-party dependencies:** —
- **Original SHA-256 (before re-annotation):** `66ba05240c6f1715591185d38977f0ebb9b04cc979b54ee7bcb8dfaf93a42abb`
- **Executed version verified:** not applicable / not recorded
- **Logic unchanged (AST comparison after re-annotation):** yes

## `trainers/nnUNetTrainerCheckEvery10.py`

- **Type:** Custom nnU-Net trainer
- **Purpose:** Custom nnU-Net trainer that saves checkpoint_latest every 10 epochs.
- **Inputs detected:** —
- **Outputs detected:** —
- **Third-party dependencies:** nnunetv2, torch
- **Original SHA-256 (before re-annotation):** `dab893b268702b887192fb04da83b371781d623369a2a16b0f2515dbea75b6f7`
- **Executed version verified:** not applicable / not recorded
- **Logic unchanged (AST comparison after re-annotation):** yes

## `trainers/nnUNetTrainer_Foreground100.py`

- **Type:** Custom nnU-Net trainer
- **Purpose:** Custom nnU-Net trainer with foreground oversampling in 100% of the patches.
- **Inputs detected:** —
- **Outputs detected:** —
- **Third-party dependencies:** nnunetv2, torch
- **Original SHA-256 (before re-annotation):** `aa59ad18f964ff8f4cae0cf7ee0eced82c6890068e0aef2c71b090549e531a14`
- **Executed version verified:** not applicable / not recorded
- **Logic unchanged (AST comparison after re-annotation):** yes

## `trainers/nnUNetTrainer_Foreground100_LR5e3.py`

- **Type:** Custom nnU-Net trainer
- **Purpose:** Custom nnU-Net trainer with foreground oversampling in 100% of the patches and initial learning rate 5e-3.
- **Inputs detected:** —
- **Outputs detected:** —
- **Third-party dependencies:** nnunetv2, torch
- **Original SHA-256 (before re-annotation):** `522f862d8962ec2d0f48ee17abc7f395d3e286cf6f33ed36cbb9a4bcfe682718`
- **Executed version verified:** not applicable / not recorded
- **Logic unchanged (AST comparison after re-annotation):** yes

## `notebooks/16_feature_selection_rfecv_svr_kneedle.ipynb`

- **Type:** Notebook
- **Purpose:** Notebook used in the pipeline (see header cell).
- **Inputs detected:** —
- **Outputs detected:** —
- **Third-party dependencies:** —
- **Original SHA-256 (before re-annotation):** `f4c7860387d1eb5aa5eca18e2d6cc8b7fc1813bd1a165c2b7843e4002fb971ed`
- **Executed version verified:** not applicable / not recorded
- **Logic unchanged (AST comparison after re-annotation):** yes

## `notebooks/18_rider_inference_colab.ipynb`

- **Type:** Notebook
- **Purpose:** Notebook used in the pipeline (see header cell).
- **Inputs detected:** —
- **Outputs detected:** —
- **Third-party dependencies:** —
- **Original SHA-256 (before re-annotation):** `bbaaa43d1cffe709137d59dc84f1ff62c145e34cc601ae6b2bc584388482f2fb`
- **Executed version verified:** not applicable / not recorded
- **Logic unchanged (AST comparison after re-annotation):** only machine-specific paths changed

## `notebooks/pulmonary_colab_training.ipynb`

- **Type:** Notebook
- **Purpose:** Notebook used in the pipeline (see header cell).
- **Inputs detected:** —
- **Outputs detected:** —
- **Third-party dependencies:** —
- **Original SHA-256 (before re-annotation):** `5a04d76236f14aa45a4ce35842b2e0ce4362e655f0f56b9330ee8305800a125b`
- **Executed version verified:** not applicable / not recorded
- **Logic unchanged (AST comparison after re-annotation):** yes

## Stages 28–32 (v1.1.0)

Original SHA-256 = the file in the executed analysis folder, before anonymization and re-annotation.

## `scripts/28_lndb_reference_rectification/E1_associacao_geometria.py`

- **Type:** Script
- **Purpose:** One-to-one Hungarian matching of predicted objects and LNDb reference lesions per scan, configuration, reference and threshold, using the matching functions of stages 11 and 22. Pipeline position: Stage 28 (LNDb N1 reference rectification and re-evaluation).
- **Inputs detected:** input_manifest.csv, lndb_annotations_manifest.json, lndb_object_matches.csv, lndb_verifier_object_scores.csv, trainCTs.csv, {e}.mhd, {e}.nii.gz
- **Outputs detected:** E1_contagens_por_exame.csv, E1_objetos_tau010.csv, E1_portas.json, F_escores_pred13.csv, R2_contagens_por_exame.csv, SHA256_pred13.json
- **Third-party dependencies:** SimpleITK, numpy, pandas, scipy
- **Original SHA-256 (before anonymization and re-annotation):** `a914eee484c962d24d01601b1c143f4f9db0d3823fe13c15dd20a51cf664fdce`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/28_lndb_reference_rectification/E2_estatisticas.py`

- **Type:** Script
- **Purpose:** Recomputes the LNDb-dependent analyses with the N1 reference (211 scans) and the sensitivity cohorts (224 unique images, N2), with scan-level bootstrap. Pipeline position: Stage 28 (LNDb N1 reference rectification and re-evaluation).
- **Inputs detected:** bd1_filter_features.csv, c14_limiares_C.json, filter_verifier_threshold.json, lndb_filter_features.csv, lndb_verifier_object_scores.csv
- **Outputs detected:** E1_contagens_por_exame.csv, E1_objetos_tau010.csv, E2_M_ic.csv, E2_portas.json, F_escores_pred13.csv, F_respostas.csv
- **Third-party dependencies:** numpy, pandas, scipy, sklearn
- **Original SHA-256 (before anonymization and re-annotation):** `908bf21dcaa4b0952b14f018abe4cabcff58e33eae289776498896043c2dfd51`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/28_lndb_reference_rectification/E3_AUROC_A_C.py`

- **Type:** Script
- **Purpose:** Paired AUROC and average-precision differences between Models A and C on the LNDb candidates, with scan-level bootstrap. Pipeline position: Stage 28 (LNDb N1 reference rectification and re-evaluation).
- **Inputs detected:** Outputs of the preceding pipeline stages (see docs/post_v1_stages.md).
- **Outputs detected:** E3_AUROC_A_C.csv
- **Third-party dependencies:** numpy, pandas
- **Original SHA-256 (before anonymization and re-annotation):** `da2a75dd9af6435473b9ca28d1fcf6f2be6474da5c7f6de4055a6fd424cd8282`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/28_lndb_reference_rectification/E4_holm_conjunta.py`

- **Type:** Script
- **Purpose:** Joint Holm family combining LNDb and development p values of the sixteen responses for the external configurations. Pipeline position: Stage 28 (LNDb N1 reference rectification and re-evaluation).
- **Inputs detected:** 000_Supplementary_Material_ATIVO.tex, bd1_filter_features.csv, lndb_filter_group_comparisons.csv
- **Outputs detected:** E2_D.csv, E4_D1_por_configuracao.csv, E4_holm_conjunta_N1.csv, E4_porta.json
- **Third-party dependencies:** numpy, pandas, scipy
- **Original SHA-256 (before anonymization and re-annotation):** `e66be308d8f61c793fb72c87f923706021e87fe8befd8c68fda9bb0eec6a8146`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/28_lndb_reference_rectification/F1_respostas_referencia.py`

- **Type:** Script
- **Purpose:** Extracts the sixteen morphological responses of the N1 reference lesions of the 211 LNDb scans through the stage 13/14/15 modules. Pipeline position: Stage 28 (LNDb N1 reference rectification and re-evaluation).
- **Inputs detected:** F1_porta_paridade.json, F1_respostas_referencia.csv, ct.nii.gz, input_manifest.csv, lndb_filter_features.csv, lung_mask_crop.nii.gz
- **Outputs detected:** Tables written to the stage output directory (see docs/post_v1_stages.md).
- **Third-party dependencies:** SimpleITK, numpy, pandas, scipy
- **Original SHA-256 (before anonymization and re-annotation):** `ad4d56d0e637f8e71cd44da5334e0590ef211332e4a5fcf7ce30f53eee5a8bd5`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/28_lndb_reference_rectification/F_respostas.py`

- **Type:** Script
- **Purpose:** Extracts the sixteen morphological responses, the largest axial diameter and the frozen Model A score of the LNDb predicted objects through the stage 13/14/15 modules. Pipeline position: Stage 28 (LNDb N1 reference rectification and re-evaluation).
- **Inputs detected:** ct.nii.gz, filter_verifier.pkl, input_manifest.csv, lndb_filter_features.csv, lndb_verifier_object_scores.csv, lung_mask_crop.nii.gz, {e}.nii.gz
- **Outputs detected:** F_escores_pred13.csv, F_porta_paridade.json, F_respostas.csv
- **Third-party dependencies:** SimpleITK, numpy, pandas, scipy
- **Original SHA-256 (before anonymization and re-annotation):** `0c309e1b1002ed230f1bce011de68c486d0f64af1d93f289fcf08011bacfe8e2`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/28_lndb_reference_rectification/G1_geometria_s03_N1.py`

- **Type:** Script
- **Purpose:** Segmentation geometry (Dice, Jaccard, surface distances) of the N1 true-positive pairs, by size stratum and verifier decision, with the stage-32b geometry functions. Pipeline position: Stage 28 (LNDb N1 reference rectification and re-evaluation).
- **Inputs detected:** B1_tabelaC_geometria_conjunto_configuracao_estrato.csv, B2_tabelaD_geometria_tamanho_verificador.csv, B3_efeitos_verificador_aceites_rejeitados.csv, B5_comparacao_3DFREM_3DF_pareada.csv, _N1.csv, geometria_por_objeto.csv, gt_axial_diameters.csv, internal_verifier_object_decisions.csv, lndb_verifier_object_scores.csv
- **Outputs detected:** B0_geometria_por_objeto_N1.csv, E1_objetos_tau010.csv, F_respostas.csv, G1_porta.json
- **Third-party dependencies:** numpy, pandas
- **Original SHA-256 (before anonymization and re-annotation):** `308d36639595436ef1e57a14aff3e32691d9363336b282695800423d9c041ac2`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/28_lndb_reference_rectification/G2_figuras_lndb.py`

- **Type:** Script
- **Purpose:** Generates the LNDb detection and verification figures from the closed N1 tables. Pipeline position: Stage 28 (LNDb N1 reference rectification and re-evaluation).
- **Inputs detected:** identidade_visual_figuras.yaml, s17a_internal_verifier_metrics.csv, supl_s2_limiar_f1.pdf, supl_s2_limiar_fp_exam.pdf, supl_s2_limiar_precision.pdf, supl_s2_limiar_sensitivity.pdf
- **Outputs detected:** B0_geometria_por_objeto_N1.csv, E2_G.csv, E2_M.csv
- **Third-party dependencies:** matplotlib, numpy, pandas, yaml
- **Original SHA-256 (before anonymization and re-annotation):** `91cda7aa9751a0c551538fcba7a0961ca0c6736532294e2cea63e97ee6a018e8`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/28_lndb_reference_rectification/G3_hd_pares_N1.py`

- **Type:** Script
- **Purpose:** Surface Hausdorff distance of the N1 true-positive pairs, with a Dice parity gate. Pipeline position: Stage 28 (LNDb N1 reference rectification and re-evaluation).
- **Inputs detected:** Outputs of the preceding pipeline stages (see docs/post_v1_stages.md).
- **Outputs detected:** E1_objetos_tau010.csv, G3_hd_pares_N1.csv
- **Third-party dependencies:** numpy, pandas
- **Original SHA-256 (before anonymization and re-annotation):** `3f8b1766ce742947bb273c0358c4d8f5c258ec2b5a7166a1a2de276f6a200726`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/28_lndb_reference_rectification/P1_recorte_s06_replica.py`

- **Type:** Script
- **Purpose:** Strict reproduction of the lung-mask generation and cropping of stage 06 for the 13 LNDb scans outside the analytical cohort, with byte-level parity gates against the recorded outputs. Pipeline position: Stage 28 (LNDb N1 reference rectification and re-evaluation).
- **Inputs detected:** ct.nii.gz, lung_mask.nii.gz, lung_mask_crop.nii.gz
- **Outputs detected:** P1_porta.json
- **Third-party dependencies:** SimpleITK, lungmask, numpy, torch
- **Original SHA-256 (before anonymization and re-annotation):** `cd5800ce566137837679b5be48a08fbf028955754a7b8eef52816ad5a8755575`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/28_lndb_reference_rectification/P2_regenerar_mascaras_pulmonares_211.py`

- **Type:** Script
- **Purpose:** Regenerates the cropped lung masks of the 211 analytical LNDb scans with the validated stage-06 reproduction, accepting each mask only when its SHA-256 matches the recorded input manifest. Pipeline position: Stage 28 (LNDb N1 reference rectification and re-evaluation).
- **Inputs detected:** LNDB_INPUT_MANIFEST.csv, ct.nii.gz, lung_mask_crop.nii.gz
- **Outputs detected:** Tables written to the stage output directory (see docs/post_v1_stages.md).
- **Third-party dependencies:** Python standard library only
- **Original SHA-256 (before anonymization and re-annotation):** `da5301ce57e90226a19bb0f0ef7fa6eac6138d2d9d260bf140dcf418a90567f8`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/28_lndb_reference_rectification/R1_construir_referencias.py`

- **Type:** Script
- **Purpose:** Builds, from the official per-radiologist LNDb masks, the lesion-level references of the 236 scans: the N1 reference (union of the segmentations of each official lesion), the N2 reference and the regions to be ignored. Pipeline position: Stage 28 (LNDb N1 reference rectification and re-evaluation).
- **Inputs detected:** _rad*.mhd, trainCTs.csv, trainNodules.csv, trainNodules_gt.csv
- **Outputs detected:** exames_236.csv, lesoes_236.csv, resumo_R1.json
- **Third-party dependencies:** numpy
- **Original SHA-256 (before anonymization and re-annotation):** `526969cf9779f5a2a06c9de4aa2cc7534bb06b9ee97a1d8cc0805e556ec1f212`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/28_lndb_reference_rectification/R2_reavaliar_211.py`

- **Type:** Script
- **Purpose:** Re-evaluates the frozen LNDb predictions and the frozen Model A decisions against the N1 and N2 references, with a control that reproduces the previous reference. Pipeline position: Stage 28 (LNDb N1 reference rectification and re-evaluation).
- **Inputs detected:** input_manifest.csv, lndb_annotations_manifest.json, lndb_object_matches.csv, lndb_scan_audit.csv, lndb_verifier_object_scores.csv, trainCTs.csv, {e}.mhd, {e}.nii.gz
- **Outputs detected:** R2_contagens_por_exame.csv, R2_objetos_referencia_tau010.csv, R2_porta_referencia_antiga.json
- **Third-party dependencies:** SimpleITK, numpy, scipy
- **Original SHA-256 (before anonymization and re-annotation):** `ade3d1f4e7ce4904fda8e3092293773a3485aa93fc9c2a02c6feb3c81cfbc1f9`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/28_lndb_reference_rectification/R3_metricas_e_IC.py`

- **Type:** Script
- **Purpose:** Computes sensitivity, precision, F1 and FP per scan before and after verification for each reference, configuration and matching threshold, with scan-level bootstrap intervals. Pipeline position: Stage 28 (LNDb N1 reference rectification and re-evaluation).
- **Inputs detected:** V01_iou_pre_pos_por_configuracao.csv
- **Outputs detected:** R2_contagens_por_exame.csv, R3_controlo_historico.json, R3_ic_bootstrap.csv, R3_metricas.csv
- **Third-party dependencies:** numpy, pandas
- **Original SHA-256 (before anonymization and re-annotation):** `d59f5bb30062363bb088c4c518bc86b129e7bdf207cbf5b8df8d0371ceefce6c`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/28_lndb_reference_rectification/V_verificacao.py`

- **Type:** Script
- **Purpose:** Arithmetic verification of all LNDb counts, aggregates, metrics and intervals. Pipeline position: Stage 28 (LNDb N1 reference rectification and re-evaluation).
- **Inputs detected:** V_verificacao.json
- **Outputs detected:** E1_contagens_por_exame.csv, E1_objetos_tau010.csv, E2_M.csv, E2_M_ic.csv
- **Third-party dependencies:** numpy, pandas
- **Original SHA-256 (before anonymization and re-annotation):** `fc2372fd1eb7e25efb29f3961ff6006a012073eb400186e8498c86b6cd91d786`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/28_lndb_reference_rectification/protocolo_grafico.py`

- **Type:** Module
- **Purpose:** Shared plotting helpers (number formatting, figure registration) driven by the visual-identity YAML file. Pipeline position: Stage 28 (LNDb N1 reference rectification and re-evaluation).
- **Inputs detected:** .registro.json, identidade_visual_figuras.yaml
- **Outputs detected:** Tables written to the stage output directory (see docs/post_v1_stages.md).
- **Third-party dependencies:** matplotlib, yaml
- **Original SHA-256 (before anonymization and re-annotation):** `91a2f48819869e26d1b6561d312c4e45a09589c87ec703fde8d7b6762c10b638`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/28_lndb_reference_rectification/v03_auditoria_mascaras_oficiais.py`

- **Type:** Script
- **Purpose:** Compares the official per-radiologist LNDb masks with the previous local majority reference. Pipeline position: Stage 28 (LNDb N1 reference rectification and re-evaluation).
- **Inputs detected:** _rad*.mhd, input_manifest.csv, lndb_annotations_manifest.json, trainCTs.csv, trainNodules.csv, trainNodules_gt.csv
- **Outputs detected:** v03_duplicatas.csv, v03_nodulos_gt_por_exame.csv, v03_por_exame_236.csv, v03_resumo.json
- **Third-party dependencies:** numpy, scipy
- **Original SHA-256 (before anonymization and re-annotation):** `5401d701a48623bbaacdf6bfaa0682a84b6bfc26d080f009a62efcd09e28b52c`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/28_lndb_reference_rectification/v03_componentes_por_tipo_de_achado.py`

- **Type:** Script
- **Purpose:** Classifies each reference component by the type of official finding that covers it. Pipeline position: Stage 28 (LNDb N1 reference rectification and re-evaluation).
- **Inputs detected:** _rad*.mhd, input_manifest.csv, lndb_annotations_manifest.json, trainNodules.csv
- **Outputs detected:** v03_achados_por_leitor.csv, v03_componentes_referencia_local.csv, v03_componentes_resumo.json
- **Third-party dependencies:** numpy, scipy
- **Original SHA-256 (before anonymization and re-annotation):** `851ecbf8e81cd43ec5c1f5383fc89afdb37e6a44eab5dc147fe040c52b749ae4`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/28_lndb_reference_rectification/v03_impacto_classes_TP_FN.py`

- **Type:** Script
- **Purpose:** Cross-tabulates LNDb reference objects against the class of the official finding that originated them. Pipeline position: Stage 28 (LNDb N1 reference rectification and re-evaluation).
- **Inputs detected:** Outputs of the preceding pipeline stages (see docs/post_v1_stages.md).
- **Outputs detected:** v03_componentes_referencia_local.csv, v03_referencias_D4_por_classe.csv, v03_sensibilidade_por_classe_D4.csv
- **Third-party dependencies:** Python standard library only
- **Original SHA-256 (before anonymization and re-annotation):** `73392d417f847b2a3822966dfdfbe9049941ce0b0da2e2e0ffdcdaaa5decb776`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/29_selective_verification/E1b_estado_predicoes_por_tau.py`

- **Type:** Script
- **Purpose:** Records the state (TP, FP, ignored) of each LNDb prediction at IoU thresholds 0.10, 0.25 and 0.50 for the N1, N2 and alternative-representative references. Pipeline position: Stage 29 (selective verification).
- **Inputs detected:** E1_contagens_por_exame.csv, F_escores_pred13.csv, SHA256_pred13.json, input_manifest.csv, lndb_annotations_manifest.json, lndb_verifier_object_scores.csv, trainCTs.csv, {e}.mhd, {e}.nii.gz
- **Outputs detected:** E1b_predicoes_por_tau.csv
- **Third-party dependencies:** SimpleITK, numpy, pandas, scipy
- **Original SHA-256 (before anonymization and re-annotation):** `0789c8ad3dc0ade2775175593b7e98df685fb80a199c334ed43c97089eea99a8`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/29_selective_verification/S1_confianca_candidatos.py`

- **Type:** Script
- **Purpose:** Computes the detector confidence of each candidate (mean foreground probability over the component voxels). Pipeline position: Stage 29 (selective verification).
- **Inputs detected:** *.npz
- **Outputs detected:** S1_confianca_candidatos.csv, S1_paridade.csv, S1_resumo.json
- **Third-party dependencies:** SimpleITK, numpy, pandas, scipy
- **Original SHA-256 (before anonymization and re-annotation):** `1d4de1ccc0721ad007f03f867cfcac8001ea31820538b8db08fbadcddde2926d`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/29_selective_verification/S2_avaliacao_seletiva.py`

- **Type:** Script
- **Purpose:** Selective verification: direct acceptance of candidates above the confidence threshold and Model A for the remainder, compared with full verification. Pipeline position: Stage 29 (selective verification).
- **Inputs detected:** E1_contagens_por_exame.csv, E2_M.csv, F_escores_pred13.csv, V02_escores_oof_ABC.csv.gz, bd1_filter_features.csv, bd1_object_matches.csv, filter_verifier_threshold.json, internal_verifier_metrics.csv, internal_verifier_object_decisions.csv, lndb_annotations_manifest.json
- **Outputs detected:** E1b_predicoes_por_tau.csv, S1_confianca_candidatos.csv, S2_diferencas_C_B.csv, S2_fronteira_B.csv, S2_limiar_congelado.json, S2_metricas.csv, S2_portas.json
- **Third-party dependencies:** numpy, pandas
- **Original SHA-256 (before anonymization and re-annotation):** `9c6e4bb03aca1868b92454fc22bcb8dd88798a08fec60d8296cbfc9ae26676cb`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/29_selective_verification/S3_descritivo_confianca.py`

- **Type:** Script
- **Purpose:** Descriptive analysis of the detector confidence measure among accepted and rejected candidates. Pipeline position: Stage 29 (selective verification).
- **Inputs detected:** V02_escores_oof_ABC.csv.gz, bd1_filter_features.csv
- **Outputs detected:** S1_confianca_candidatos.csv, S3_descritivo_confianca.csv
- **Third-party dependencies:** numpy, pandas, sklearn
- **Original SHA-256 (before anonymization and re-annotation):** `78a63d4ea814b2eafd23db67d52621aee85e9807ea867d9376b5c939f37f4b9e`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/29_selective_verification/S4_curvas_operacionais.py`

- **Type:** Script
- **Purpose:** Descriptive comparison of the operating curves of selective and full verification. Pipeline position: Stage 29 (selective verification).
- **Inputs detected:** E1_contagens_por_exame.csv, internal_verifier_object_decisions.csv, lndb_verifier_object_scores.csv
- **Outputs detected:** E1b_predicoes_por_tau.csv, S4_curvas_operacionais.csv
- **Third-party dependencies:** numpy, pandas
- **Original SHA-256 (before anonymization and re-annotation):** `3ea24084becf51aa856d6cb1728d5ca559ec375b6e35906141c09de6dcedf195`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/30_radiological_assessment/c01_teste_alinhamento.py`

- **Type:** Script
- **Purpose:** Tests the voxel alignment between the CT used for the radiological markings and the CT of the dataset, with axis-reversal negative controls. Pipeline position: Stage 30 (radiological assessment linkage).
- **Inputs detected:** LIDC_0061_0000.nii.gz, ct.nii.gz
- **Outputs detected:** c01_alinhamento.json
- **Third-party dependencies:** SimpleITK, numpy
- **Original SHA-256 (before anonymization and re-annotation):** `eb6f3e782ebfdae3e48f2594ba36afd58e87cc876bdea99fb23b55de430f3694`
- **Anonymized:** yes
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/30_radiological_assessment/c01b_teste_nrrd.py`

- **Type:** Script
- **Purpose:** Tests the alignment of the radiologist's segmentation files with the current image grid, with inverted-content negative controls. Pipeline position: Stage 30 (radiological assessment linkage).
- **Inputs detected:** LIDC-IDRI-0061.nii.gz, ct.nii.gz
- **Outputs detected:** c01b_alinhamento_nrrd.json
- **Third-party dependencies:** SimpleITK, numpy
- **Original SHA-256 (before anonymization and re-annotation):** `94887a0340e26f460a808bb548aec41970d3cfb2de2e2d48a8b3ffc77576c042`
- **Anonymized:** yes
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/30_radiological_assessment/c01c_componentes_radiologist_0061.py`

- **Type:** Script
- **Purpose:** Describes the components of the radiologist's marking for one scan on the current image grid. Pipeline position: Stage 30 (radiological assessment linkage).
- **Inputs detected:** LIDC-IDRI-0061.nii.gz, ct.nii.gz
- **Outputs detected:** Tables written to the stage output directory (see docs/post_v1_stages.md).
- **Third-party dependencies:** SimpleITK, numpy, scipy
- **Original SHA-256 (before anonymization and re-annotation):** `1f9617d03d213d11ac67d77e3b21df39aecd31965fc5757af2d5966c5fc25959`
- **Anonymized:** yes
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/30_radiological_assessment/c02_correspondencia.py`

- **Type:** Script
- **Purpose:** Spatial correspondence between the radiologist's markers and the predicted and reference objects, without access to verifier decisions. Pipeline position: Stage 30 (radiological assessment linkage).
- **Inputs detected:** _cc.nii.gz, bd1_object_matches.csv, consensus.nii.gz, ct.nii.gz, lung_mask_crop.nii.gz
- **Outputs detected:** c02_esferas.csv, c02_identidade.json, c02_referencias.csv
- **Third-party dependencies:** SimpleITK, numpy, scipy
- **Original SHA-256 (before anonymization and re-annotation):** `57879b7d8b8a3969283338054d8202a4f70ad64e8c4fc4a51261d5d8b4a08832`
- **Anonymized:** yes
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/30_radiological_assessment/c02b_lateralidade.py`

- **Type:** Script
- **Purpose:** Robust laterality assignment by splitting the lung mask into the two lungs. Pipeline position: Stage 30 (radiological assessment linkage).
- **Inputs detected:** consensus.nii.gz, lung_mask_crop.nii.gz
- **Outputs detected:** c02_esferas.csv, c02_referencias.csv, c02b_esferas.csv, c02b_referencias.csv
- **Third-party dependencies:** SimpleITK, numpy, pandas, scipy
- **Original SHA-256 (before anonymization and re-annotation):** `86a03e76dd0a349a015b4a9454453d5815e8580196d4b1701f95035d8d5e8a8f`
- **Anonymized:** yes
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/30_radiological_assessment/c03_verificador.py`

- **Type:** Script
- **Purpose:** Frozen Model A decisions for the 3D-F and 3D-FREM objects of the assessed scans. Pipeline position: Stage 30 (radiological assessment linkage).
- **Inputs detected:** bd1_lung_mask_authority.csv, filter_verifier.pkl, filter_verifier_threshold.json, internal_verifier_object_decisions.csv, lung_mask_crop.nii.gz
- **Outputs detected:** c03_decisoes.csv, c03_validacao.json
- **Third-party dependencies:** SimpleITK, numpy, pandas, scipy
- **Original SHA-256 (before anonymization and re-annotation):** `8d3e1f0da5fba3cc52553a9aa21d86284125125b98a16d0994a48fa05f8dc411`
- **Anonymized:** yes
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/30_radiological_assessment/c04_juncao.py`

- **Type:** Script
- **Purpose:** Joins the frozen marker correspondence with the verifier decisions. Pipeline position: Stage 30 (radiological assessment linkage).
- **Inputs detected:** _cc.nii.gz
- **Outputs detected:** c02_esferas.csv, c02_referencias.csv, c03_decisoes.csv, c04_esferas_verificador.csv, c04_referencias_verificador.csv
- **Third-party dependencies:** SimpleITK, numpy, pandas, scipy
- **Original SHA-256 (before anonymization and re-annotation):** `ee6d460ca1744e784dfbed1b5c9c4eac6bbd6447aad20b42b88ebea6743fcaa3`
- **Anonymized:** yes
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/30_radiological_assessment/c05_proposta_ligacao.py`

- **Type:** Script
- **Purpose:** Proposes the link between the radiological records and the markers from the anatomical description. Pipeline position: Stage 30 (radiological assessment linkage).
- **Inputs detected:** c05_proposta_ligacao.md
- **Outputs detected:** c02b_referencias.csv, c04_esferas_verificador.csv, c04_referencias_verificador.csv, c05_proposta_ligacao.csv
- **Third-party dependencies:** pandas
- **Original SHA-256 (before anonymization and re-annotation):** `641b17991c799157fd66ec92961abac7c460c59fb2b2b5330e277755da490c4f`
- **Anonymized:** yes
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/30_radiological_assessment/c06_por_leitor_0061.py`

- **Type:** Script
- **Purpose:** Checks the per-reader annotations of one scan before assigning reference status to two records. Pipeline position: Stage 30 (radiological assessment linkage).
- **Inputs detected:** *.nii.gz, consensus.nii.gz, lung_mask_crop.nii.gz
- **Outputs detected:** Tables written to the stage output directory (see docs/post_v1_stages.md).
- **Third-party dependencies:** SimpleITK, numpy, scipy
- **Original SHA-256 (before anonymization and re-annotation):** `c96f5658b6b8091260d1a51b1e15a2e08138e7bc5a0249b87bf7af60282b70af`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/30_radiological_assessment/c07_confirmacao_cega.py`

- **Type:** Script
- **Purpose:** Builds the blinded confirmation table of the link between radiological records and markers. Pipeline position: Stage 30 (radiological assessment linkage).
- **Inputs detected:** c07_confirmacao_cega.md
- **Outputs detected:** c02b_esferas.csv, c02b_referencias.csv, c07_confirmacao_cega.csv
- **Third-party dependencies:** pandas
- **Original SHA-256 (before anonymization and re-annotation):** `f25862c9a68f0b1a6ad7ba9afd873b6d4ab4b4d6a39364ef843a5c5f2ae76861`
- **Anonymized:** yes
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/30_radiological_assessment/c08_oof_verificador.py`

- **Type:** Script
- **Purpose:** Reproduces the out-of-fold Model A score of each development object from the frozen stage-17 sources. Pipeline position: Stage 30 (radiological assessment linkage).
- **Inputs detected:** bd1_filter_features.csv, filter_verifier_cv.csv, filter_verifier_threshold.json
- **Outputs detected:** c08_oof_objetos_radiologist.csv, c08_validacao.json
- **Third-party dependencies:** numpy, pandas, sklearn
- **Original SHA-256 (before anonymization and re-annotation):** `dc143f0ccb84578e1e4544ac6ab69dd212874777bc2d78bcae0de818c7befe9a`
- **Anonymized:** yes
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/30_radiological_assessment/c08b_confere_publicado.py`

- **Type:** Script
- **Purpose:** Checks that the reproduced out-of-fold scores match the frozen stage-17 records. Pipeline position: Stage 30 (radiological assessment linkage).
- **Inputs detected:** bd1_filter_features.csv, filter_verifier_cv.csv, filter_verifier_threshold.json
- **Outputs detected:** c08b_conferencia_publicado.json
- **Third-party dependencies:** numpy, pandas, sklearn
- **Original SHA-256 (before anonymization and re-annotation):** `dd03a760229ed2584c2ba42f94168f45d6e377cc8c56fa819c428d82d008fa3d`
- **Anonymized:** yes
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/30_radiological_assessment/c09_congela_ligacao.py`

- **Type:** Script
- **Purpose:** Freezes the confirmed record-to-marker link with its SHA-256. Pipeline position: Stage 30 (radiological assessment linkage).
- **Inputs detected:**   c08_ligacao_congelada.csv
- **Outputs detected:** c07_confirmacao_cega_RESPONDIDA_20261109.csv, c08_ligacao_congelada.csv
- **Third-party dependencies:** pandas
- **Original SHA-256 (before anonymization and re-annotation):** `3a3bea33874f78df64ec14c6473a1186c239fcb925b55fda391b00c4707b4f89`
- **Anonymized:** yes
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/30_radiological_assessment/c10_desfechos.py`

- **Type:** Script
- **Purpose:** Joins the frozen link with the detection outcomes and verifier decisions. Pipeline position: Stage 30 (radiological assessment linkage).
- **Inputs detected:** c08_ligacao_congelada.SHA256, c10_tabelas.md
- **Outputs detected:** c03_decisoes.csv, c04_esferas_verificador.csv, c04_referencias_verificador.csv, c08_ligacao_congelada.csv, c08_oof_objetos_radiologist.csv, c10_objetos.csv, c10_opiniao_radiologista.csv, c10_registos.csv
- **Third-party dependencies:** numpy, pandas
- **Original SHA-256 (before anonymization and re-annotation):** `2f38acb5d9b4bbc103ff5000ea8c065c063ae1117d6c0b6012e4090753ca6bef`
- **Anonymized:** yes
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/30_radiological_assessment/c11_grafico_notas.py`

- **Type:** Script
- **Purpose:** Plots the delineation scores of the true-positive records with the verifier decision. Pipeline position: Stage 30 (radiological assessment linkage).
- **Inputs detected:** Outputs of the preceding pipeline stages (see docs/post_v1_stages.md).
- **Outputs detected:** c08_ligacao_congelada.csv, c10_objetos.csv, c11_notas_delineamento.csv
- **Third-party dependencies:** matplotlib, pandas
- **Original SHA-256 (before anonymization and re-annotation):** `e6d707fafd3dafdcc6d19d9a84ef7bfb7c7d8b41b5f016a308a970f890223430`
- **Anonymized:** yes
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/30_radiological_assessment/c12_geometria_tp.py`

- **Type:** Script
- **Purpose:** Dice and HD95 of the true-positive pairs linked to radiological records. Pipeline position: Stage 30 (radiological assessment linkage).
- **Inputs detected:** bd1_object_matches.csv, consensus.nii.gz, geometria_por_objeto.csv
- **Outputs detected:** c10_objetos.csv, c12_geometria_tp.csv
- **Third-party dependencies:** SimpleITK, numpy, pandas, scipy
- **Original SHA-256 (before anonymization and re-annotation):** `e03a40fd926b389852b9754fd58146ac377e7015b9e42e2b1a88d5c0551725e1`
- **Anonymized:** yes
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/30_radiological_assessment/c13_opiniao_planilha.py`

- **Type:** Script
- **Purpose:** Reads the radiologist's interpretation of each record from the primary spreadsheet. Pipeline position: Stage 30 (radiological assessment linkage).
- **Inputs detected:** Outputs of the preceding pipeline stages (see docs/post_v1_stages.md).
- **Outputs detected:** c08_ligacao_congelada.csv, c10_opiniao_radiologista.csv, c13_opiniao_planilha.csv
- **Third-party dependencies:** pandas
- **Original SHA-256 (before anonymization and re-annotation):** `208e5b63bd4206f53838cf402234b8828a8f21a362edd00ac50ecac8b7918608`
- **Anonymized:** yes
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/31_model_c_and_ablation/c14_modeloC_ablacoes.py`

- **Type:** Script
- **Purpose:** Fits Models B and C and the Model A ablations in the same five patient-grouped development folds, and applies Model C to the internal and external candidates. Pipeline position: Stage 31 (Model B, Model C and ablation in the development folds).
- **Inputs detected:** A1_quadro_candidatos_por_objeto.csv, B04_A_C_OPERACIONAL_CORRIGIDO.csv, bd1_filter_features.csv, filter_verifier_cv.csv, filter_verifier_threshold.json, geometria_por_objeto.csv, input_manifest.csv, manifesto_imagens_D2_D4.csv
- **Outputs detected:** c14_controlo_operacao.json, c14_ic_pareados_A_C_D2_D4.csv, c14_limiares_C.json, c14_modelos_D1.csv, c14_operacao_D2_D4.csv, c14_por_dobra.csv, c14_por_exame_D2_D4.csv, c14_porta_dobras.json
- **Third-party dependencies:** numpy, pandas, sklearn
- **Original SHA-256 (before anonymization and re-annotation):** `18f1fe395747596d960207d05cd0be400d0bdb1888919e4189baa8d6d6cfa062`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/32_supplementary_analyses/a_verifier_and_lndb_cohort/10_v01_iou_verificador.py`

- **Type:** Script
- **Purpose:** Sensitivity of the detector-plus-verifier procedure to the IoU matching threshold. Pipeline position: Stage 32 (supplementary analyses).
- **Inputs detected:** V01_FALHA.txt, V01_validacao_baseline.md
- **Outputs detected:** V01_contagens_por_exame.csv, V01_controlos.csv, V01_diferencas_bootstrap.csv, V01_iou_pre_pos_por_configuracao.csv
- **Third-party dependencies:** numpy, pandas
- **Original SHA-256 (before anonymization and re-annotation):** `2bd1c87040745664a56295126a76c2d9a37d457e12b669dcdf655d7977d2d08c`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/32_supplementary_analyses/a_verifier_and_lndb_cohort/20_v02_b_c_oof.py`

- **Type:** Script
- **Purpose:** Size-conditional discriminative increment (Model B versus Model C) on development out-of-fold predictions. Pipeline position: Stage 32 (supplementary analyses).
- **Inputs detected:** Outputs of the preceding pipeline stages (see docs/post_v1_stages.md).
- **Outputs detected:** V02_B_vs_C_oof.csv, V02_bootstrap_pareado.csv, V02_chaves_particoes_validacao.json, V02_controlo_c14.csv, V02_escores_oof_ABC.csv.gz
- **Third-party dependencies:** numpy, pandas, sklearn
- **Original SHA-256 (before anonymization and re-annotation):** `2dd10140b654195bd86e3004bf2c0c6eb1590ce43447512f313d6e0e280dc595`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/32_supplementary_analyses/a_verifier_and_lndb_cohort/30_v03_coorte_lndb.py`

- **Type:** Script
- **Purpose:** Documentary and quantitative audit of the LNDb cohort selection (236 to 211 scans). Pipeline position: Stage 32 (supplementary analyses).
- **Inputs detected:** *.csv, V03_FALHA.txt, V03_estado_referencias.md, inference_manifest_s10_20260918T114645_2953858.csv, inference_manifest_s10_20260918T194618_3177777.csv
- **Outputs detected:** V03_comparacao_metadados.csv, V03_exames_auditoria_236.csv, V03_exames_excluidos_auditoria.csv, V03_fluxo_coorte.csv, V03_pares_duplicatas.csv
- **Third-party dependencies:** numpy, pandas
- **Original SHA-256 (before anonymization and re-annotation):** `d10be77779c65ef888684d3f1277648b1eae53cfd2fe5024d9a05cd5f898d70f`
- **Anonymized:** yes
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/32_supplementary_analyses/a_verifier_and_lndb_cohort/40_v09_escores_limiar.py`

- **Type:** Script
- **Purpose:** Stability of the frozen threshold between out-of-fold and final Model A scores. Pipeline position: Stage 32 (supplementary analyses).
- **Inputs detected:** Outputs of the preceding pipeline stages (see docs/post_v1_stages.md).
- **Outputs detected:** V02_escores_oof_ABC.csv.gz, V09_D1_oof_vs_final.csv.gz, V09_distribuicao_escores.csv, V09_transicoes_theta.csv, V09_transicoes_theta_ic.csv, V09_validacao.json
- **Third-party dependencies:** numpy, pandas, sklearn
- **Original SHA-256 (before anonymization and re-annotation):** `47b4d0284b54f96375b7e6996fa8ff14959f1d91917c385da171fbe8e47590b2`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/32_supplementary_analyses/a_verifier_and_lndb_cohort/90_testes_integridade.py`

- **Type:** Script
- **Purpose:** Integrity tests of the inputs and outputs of the analyses V01, V02, V03 and V09. Pipeline position: Stage 32 (supplementary analyses).
- **Inputs detected:** MANIFESTO_SAIDAS.json, V03_sensibilidade_coorte_expandida.csv, inventario_entradas.json
- **Outputs detected:** V01_controlos.csv, V02_chaves_particoes_validacao.json, V02_controlo_c14.csv, V09_validacao.json
- **Third-party dependencies:** Python standard library only
- **Original SHA-256 (before anonymization and re-annotation):** `8d08065935581af8e725df5a9209b415be38a32407d7195b2763bdc486116ef1`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/32_supplementary_analyses/a_verifier_and_lndb_cohort/comum.py`

- **Type:** Module
- **Purpose:** Shared paths and helper functions of the supplementary analyses V01, V02, V03 and V09. Pipeline position: Stage 32 (supplementary analyses).
- **Inputs detected:** LNDB_INPUT_MANIFEST.csv, LNDB_RECONCILIACAO.csv, bd1_filter_features.csv, bd1_folds.json, bd1_object_matches.csv, bd1_split.csv, c14_modelos_D1.csv, filter_verifier.pkl, filter_verifier_B_plus_size.pkl, filter_verifier_cv.csv
- **Outputs detected:** Tables written to the stage output directory (see docs/post_v1_stages.md).
- **Third-party dependencies:** numpy, pandas, scipy, sklearn
- **Original SHA-256 (before anonymization and re-annotation):** `8616d6bae985834c8cea5b7ada6045eb7ad50777ea50cf5d644ecb7c1a311613`
- **Anonymized:** yes
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/32_supplementary_analyses/b_segmentation_geometry/s02_geometria_pares.py`

- **Type:** Script
- **Purpose:** Per-object segmentation geometry (Dice, Jaccard, surface distances, HD95) of the true-positive pairs in the internal and external datasets. Pipeline position: Stage 32 (supplementary analyses).
- **Inputs detected:** 02_PLANO_ANALITICO_CONGELADO.md, bd1_object_matches.csv, consensus.nii.gz, input_manifest.csv, lndb_object_matches.csv
- **Outputs detected:** Tables written to the stage output directory (see docs/post_v1_stages.md).
- **Third-party dependencies:** SimpleITK, numpy, scipy
- **Original SHA-256 (before anonymization and re-annotation):** `938a503564d6e932ad62ce397262d8f381bc4e9d1493ac451954c100bb359310`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/32_supplementary_analyses/b_segmentation_geometry/s03_geometria_analise.py`

- **Type:** Script
- **Purpose:** Geometry summaries by configuration, dataset, size stratum and verifier decision, with bootstrap intervals. Pipeline position: Stage 32 (supplementary analyses).
- **Inputs detected:** 02_PLANO_ANALITICO_CONGELADO.md, bd1_object_matches.csv, filter_verifier_threshold.json, gt_axial_diameters.csv, internal_verifier_object_decisions.csv, lndb_verifier_object_scores.csv
- **Outputs detected:** B0_geometria_por_objeto_com_verificador.csv, B1_tabelaC_geometria_conjunto_configuracao_estrato.csv, B2_tabelaD_geometria_tamanho_verificador.csv, B3_efeitos_verificador_aceites_rejeitados.csv, B4_rejeicao_por_tercil_dice.csv, B5_comparacao_3DFREM_3DF_pareada.csv, GATE_s02.json, GATE_s03.json, geometria_por_objeto.csv
- **Third-party dependencies:** numpy, pandas, scipy
- **Original SHA-256 (before anonymization and re-annotation):** `9104a71d6cb4a7364a0a16fdf9465f187e082c2a4fcba018daa6580483b9a1ff`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/32_supplementary_analyses/b_segmentation_geometry/s04_quadro_candidatos.py`

- **Type:** Script
- **Purpose:** Table of all predicted objects of 3D-F and 3D-FREM with original class, verifier score and decision. Pipeline position: Stage 32 (supplementary analyses).
- **Inputs detected:** 02_PLANO_ANALITICO_CONGELADO.md, bd1_filter_features.csv, bd1_object_matches.csv, ct.nii.gz, filter_verifier_threshold.json, internal_verifier_object_decisions.csv, lndb_filter_features.csv, lndb_object_matches.csv, lndb_verifier_object_scores.csv
- **Outputs detected:** A1_quadro_candidatos_por_objeto.csv, A1_quadro_candidatos_por_rid.csv, GATE_s04.json
- **Third-party dependencies:** SimpleITK, numpy, pandas, scipy
- **Original SHA-256 (before anonymization and re-annotation):** `b3d376fa22f2e16a7b3271e135f855bbbe84d99c2f5e951b791f523cc2bf0a5b`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/32_supplementary_analyses/c_reference_unit_and_fragments/G_fig6a_tamanho.py`

- **Type:** Script
- **Purpose:** Generates the figure of sensitivity by reference size in development. Pipeline position: Stage 32 (supplementary analyses).
- **Inputs detected:** identidade_visual_figuras.yaml, results_3_2_size_sensitivity.pdf
- **Outputs detected:** fig6a_dados.csv
- **Third-party dependencies:** matplotlib, numpy, pandas, yaml
- **Original SHA-256 (before anonymization and re-annotation):** `419d2d6b09789972f3218c67134a60a7f4b36167dbb2c1c6400fd158158af03b`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/32_supplementary_analyses/c_reference_unit_and_fragments/P01_11_diag45.py`

- **Type:** Script
- **Purpose:** Diagnosis of the patients whose reconstructed groupings differ from the consensus. Pipeline position: Stage 32 (supplementary analyses).
- **Inputs detected:** *.nii.gz, consensus.nii.gz, crop_transform.json
- **Outputs detected:** Tables written to the stage output directory (see docs/post_v1_stages.md).
- **Third-party dependencies:** SimpleITK, numpy
- **Original SHA-256 (before anonymization and re-annotation):** `30ccce8836e47bb7fb02f7a6b638583e20881f3f19c5f24cc5af20436dffd610`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/32_supplementary_analyses/c_reference_unit_and_fragments/P01_11_resumo.py`

- **Type:** Script
- **Purpose:** Summary of the reference-unit sensitivity analysis. Pipeline position: Stage 32 (supplementary analyses).
- **Inputs detected:** Outputs of the preceding pipeline stages (see docs/post_v1_stages.md).
- **Outputs detected:** P01_11_contagens.csv, P01_11_portas.csv, P01_11_preditos.csv, P01_11_referencias.csv, R_componentes_agrupamentos.csv, R_estratos.csv, R_metricas_cenarios.csv, R_mudanca_classe.csv, R_paridade_A_script11.csv
- **Third-party dependencies:** numpy, pandas
- **Original SHA-256 (before anonymization and re-annotation):** `942d0fed44a243ceed7a25f8815495a07213a5693a49ed0bc472eeb956be46e6`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/32_supplementary_analyses/c_reference_unit_and_fragments/P01_11_resumo_coerentes.py`

- **Type:** Script
- **Purpose:** Reference-unit comparison restricted to patients whose annotations reconstruct the consensus. Pipeline position: Stage 32 (supplementary analyses).
- **Inputs detected:** Outputs of the preceding pipeline stages (see docs/post_v1_stages.md).
- **Outputs detected:** P01_11_contagens.csv, P01_11_portas.csv, P01_11_preditos.csv, P01_11_referencias.csv, R_estratos_coerentes.csv, R_metricas_cenarios_coerentes.csv, R_mudanca_classe_coerentes.csv
- **Third-party dependencies:** numpy, pandas
- **Original SHA-256 (before anonymization and re-annotation):** `c7152e56e1bf7283a1d204cda17b099999b10373bdd1ffcb05dec64344b1da03`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/32_supplementary_analyses/c_reference_unit_and_fragments/P01_11_unidade_referencia.py`

- **Type:** Script
- **Purpose:** Sensitivity of the LIDC-IDRI reference unit: face-connected components versus reader groupings with merged fragments. Pipeline position: Stage 32 (supplementary analyses).
- **Inputs detected:** *.nii.gz, BLOCK_HASHES.csv, bd1_folds.json, bd1_split.csv, consensus.nii.gz, crop_transform.json, parcial.pkl
- **Outputs detected:** P01_11_contagens.csv, P01_11_portas.csv, P01_11_preditos.csv, P01_11_referencias.csv, consensus_thresholds.csv
- **Third-party dependencies:** SimpleITK, numpy, scipy
- **Original SHA-256 (before anonymization and re-annotation):** `689eb88ea726259a07fb3deb74d1b28e36a7a9b3c4623dbaeaf20e2d22229eeb`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/32_supplementary_analyses/c_reference_unit_and_fragments/P01_12_1_populacao.py`

- **Type:** Script
- **Purpose:** Compares the patients with reconstructable annotations with the remaining patients. Pipeline position: Stage 32 (supplementary analyses).
- **Inputs detected:** Outputs of the preceding pipeline stages (see docs/post_v1_stages.md).
- **Outputs detected:** P01_11_contagens.csv, P01_11_portas.csv, P01_11_referencias.csv, P01_12_1_composicao.csv, P01_12_1_metricas_A_831_45.csv
- **Third-party dependencies:** numpy, pandas
- **Original SHA-256 (before anonymization and re-annotation):** `95934f48c7deb6c973c7085d2af00cdf2ebc520635d7ef0fc7a5cc4bd796100a`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/32_supplementary_analyses/c_reference_unit_and_fragments/P01_12_lndb_unidade.py`

- **Type:** Script
- **Purpose:** Diagnosis of the reference unit in LNDb (N1). Pipeline position: Stage 32 (supplementary analyses).
- **Inputs detected:** Outputs of the preceding pipeline stages (see docs/post_v1_stages.md).
- **Outputs detected:** P01_12_lndb_contagens.csv, P01_12_lndb_lesoes.csv, P01_12_lndb_portas.csv
- **Third-party dependencies:** SimpleITK, numpy, pandas, scipy
- **Original SHA-256 (before anonymization and re-annotation):** `f2eeb2e213c09a42a98e564dd2670b6339edfbc01d7d5f72eae39e50d9e5bd48`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/32_supplementary_analyses/c_reference_unit_and_fragments/P01_13_auroc_fragmentos.py`

- **Type:** Script
- **Purpose:** AUROC of maximum blobness between detected and missed references after excluding fragments of larger groupings. Pipeline position: Stage 32 (supplementary analyses).
- **Inputs detected:** Outputs of the preceding pipeline stages (see docs/post_v1_stages.md).
- **Outputs detected:** P01_13_auroc_fragmentos.csv, P01_13_auroc_portas.json
- **Third-party dependencies:** numpy, pandas, scipy, sklearn
- **Original SHA-256 (before anonymization and re-annotation):** `67a5a8d5df3f46e8f8898dfb3d246d1c67882f92f79516aad456035533613ce8`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/32_supplementary_analyses/d_interobserver_matching_and_partition/G3_figura_rider_limiares_relativos.py`

- **Type:** Script
- **Purpose:** Generates the figure of RIDER pairs whose relative change exceeds the thresholds nu_1 and nu_2. Pipeline position: Stage 32 (supplementary analyses).
- **Inputs detected:** identidade_visual_figuras.yaml, results_3_6_rel_legend.pdf
- **Outputs detected:** N1_A2_limiares_relativos.csv
- **Third-party dependencies:** matplotlib, numpy, pandas, yaml
- **Original SHA-256 (before anonymization and re-annotation):** `46470e1c747de4fd8f59f52c1ce92601d01991da4f7b8a0f9d985d02ba08e434`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/32_supplementary_analyses/d_interobserver_matching_and_partition/N2_concordancia_leitores_LNDb.py`

- **Type:** Script
- **Purpose:** Inter-reader identification agreement in LNDb through the official finding groupings. Pipeline position: Stage 32 (supplementary analyses).
- **Inputs detected:** Outputs of the preceding pipeline stages (see docs/post_v1_stages.md).
- **Outputs detected:** N2_pares.csv, N2_resumo.csv, trainCTs.csv, trainNodules.csv, trainNodules_gt.csv
- **Third-party dependencies:** numpy, pandas
- **Original SHA-256 (before anonymization and re-annotation):** `43d8f88394bd2fee03f078f334b9088b85afdf3692962abdc47e266a63506c83`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/32_supplementary_analyses/d_interobserver_matching_and_partition/O02_N1_delineamento_lidc.py`

- **Type:** Script
- **Purpose:** Effect of the reference delineation rule (union versus consensus) in LNDb. Pipeline position: Stage 32 (supplementary analyses).
- **Inputs detected:** _lesoes.csv, _rad*.mhd, input_manifest.csv, lndb_verifier_object_scores.csv, {e}.mhd, {e}.nii.gz
- **Outputs detected:** trainNodules.csv, trainNodules_gt.csv
- **Third-party dependencies:** SimpleITK, numpy, scipy
- **Original SHA-256 (before anonymization and re-annotation):** `934a2523b17df3e6a4b260f7025d46c78b93ac45428be20f7658126a9f93700b`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/32_supplementary_analyses/d_interobserver_matching_and_partition/O03_auditoria_hungaro_lndb.py`

- **Type:** Script
- **Purpose:** Audit of the Hungarian matching rule (admissible-pair priority versus total-IoU maximization) in LNDb. Pipeline position: Stage 32 (supplementary analyses).
- **Inputs detected:** SHA256_pred13.json, input_manifest.csv, {e}.mhd, {e}.nii.gz
- **Outputs detected:** O03_lndb_AB.csv, trainCTs.csv
- **Third-party dependencies:** SimpleITK, numpy, scipy
- **Original SHA-256 (before anonymization and re-annotation):** `62b78c7111131c9e40fe106c0185226748d9baa4f87b8e8478c2411b40eff3a9`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/32_supplementary_analyses/d_interobserver_matching_and_partition/O09_pareamento_tamanho.py`

- **Type:** Script
- **Purpose:** RIDER test-retest matching rate by configuration and object size. Pipeline position: Stage 32 (supplementary analyses).
- **Inputs detected:** BLOCK_HASHES.csv, input_manifest.csv, rider_iou_audit.csv, rider_iou_unmatched.csv
- **Outputs detected:** O09_objetos.csv, O09_resumo.csv, rider_iou_pairs.csv, rider_pair_features.csv
- **Third-party dependencies:** SimpleITK, numpy, pandas, scipy
- **Original SHA-256 (before anonymization and re-annotation):** `9daa403870ea09bbd6da90d3d3d73df9a15b823a0fcf9dfb3291ccb141f4a308`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/32_supplementary_analyses/d_interobserver_matching_and_partition/P01_01_rider_AB.py`

- **Type:** Script
- **Purpose:** RIDER pairing under total-IoU priority versus pair-count priority. Pipeline position: Stage 32 (supplementary analyses).
- **Inputs detected:** BLOCK_HASHES.csv, _1.nii.gz, _2.nii.gz, input_manifest.csv, rider_iou_audit.csv, transform_hashes.csv
- **Outputs detected:** P01_01_pares_diferentes.csv, P01_01_rider_AB.csv, rider_iou_pairs.csv
- **Third-party dependencies:** SimpleITK, numpy, pandas, scipy
- **Original SHA-256 (before anonymization and re-annotation):** `cd054c068df2383a9f6c1827d1ffff4740e04e2a17e04bb7cd9ff07491077f0c`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/32_supplementary_analyses/d_interobserver_matching_and_partition/V01_reproduz_particao.py`

- **Type:** Script
- **Purpose:** Reproduces the D1/D2 partition from the recorded seed and compares it with the archived partition. Pipeline position: Stage 32 (supplementary analyses).
- **Inputs detected:** Outputs of the preceding pipeline stages (see docs/post_v1_stages.md).
- **Outputs detected:** Tables written to the stage output directory (see docs/post_v1_stages.md).
- **Third-party dependencies:** sklearn
- **Original SHA-256 (before anonymization and re-annotation):** `fb2db428697623958fdf4580162a5c739f7d06e59758431615e7c4f05b6e4131`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/32_supplementary_analyses/d_interobserver_matching_and_partition/V03_lesoes_multicomponente.py`

- **Type:** Script
- **Purpose:** Checks LNDb N1 lesions with more than one connected component and their response vectors. Pipeline position: Stage 32 (supplementary analyses).
- **Inputs detected:** F_respostas.csv, LNDb-*.npz
- **Outputs detected:** Tables written to the stage output directory (see docs/post_v1_stages.md).
- **Third-party dependencies:** numpy, pandas, scipy
- **Original SHA-256 (before anonymization and re-annotation):** `5dd622443417db20d904708d4518213363a796b837f47833b60a9849baf32ac4`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/32_supplementary_analyses/d_interobserver_matching_and_partition/V04_reconcilia_722_686.py`

- **Type:** Script
- **Purpose:** Reconciles the N1 lesion counts of the reference scan with those of the 224-image cohort. Pipeline position: Stage 32 (supplementary analyses).
- **Inputs detected:** F_respostas.csv, LNDb-*.npz, exames_236.csv, input_manifest.csv
- **Outputs detected:** Tables written to the stage output directory (see docs/post_v1_stages.md).
- **Third-party dependencies:** numpy, pandas, scipy
- **Original SHA-256 (before anonymization and re-annotation):** `ec4a0067be5d1ebe97cb3ff0deea313ecdac7b41790aa06baf05c43e8b96f693`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/32_supplementary_analyses/e_operating_intervals_and_scale/folds.py`

- **Type:** Script
- **Purpose:** Checks the equivalence between the verifier modeling folds and the archived development folds. Pipeline position: Stage 32 (supplementary analyses).
- **Inputs detected:** bd1_filter_features.csv, bd1_folds.json
- **Outputs detected:** Tables written to the stage output directory (see docs/post_v1_stages.md).
- **Third-party dependencies:** pandas, sklearn
- **Original SHA-256 (before anonymization and re-annotation):** `44ff3c9a3f4933d1d22efa5e689d09b274f8965cd835ed1d1a877e3f45d3da2d`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/32_supplementary_analyses/e_operating_intervals_and_scale/ic_oper.py`

- **Type:** Script
- **Purpose:** Scan-level bootstrap intervals of the operating metrics before and after verification. Pipeline position: Stage 32 (supplementary analyses).
- **Inputs detected:** c14_por_exame_D2_D4.csv, c15_ic_operacionais_por_exame_D2_D4.csv
- **Outputs detected:** Tables written to the stage output directory (see docs/post_v1_stages.md).
- **Third-party dependencies:** numpy, pandas
- **Original SHA-256 (before anonymization and re-annotation):** `8f320910c61993ee84f23a74d63588936b7a5bec466befccc99e873c276337a2`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes

## `scripts/32_supplementary_analyses/e_operating_intervals_and_scale/lndb_hu.py`

- **Type:** Script
- **Purpose:** Checks the LNDb intensity scale from headers and air, fat and soft-tissue histogram peaks in a systematic sample. Pipeline position: Stage 32 (supplementary analyses).
- **Inputs detected:** *_0000.nii.gz, LNDB_INPUT_MANIFEST.csv, c16_lndb_escala_intensidade.json
- **Outputs detected:** Tables written to the stage output directory (see docs/post_v1_stages.md).
- **Third-party dependencies:** numpy, scipy
- **Original SHA-256 (before anonymization and re-annotation):** `4bcb2c77289e83c495c6e1fa92bd7fbb532b2db7d5806d4e6ea5b1b3489dac50`
- **Anonymized:** no
- **Executed version verified:** yes (file taken from the executed analysis folder)
- **Logic unchanged (AST comparison after re-annotation):** yes
