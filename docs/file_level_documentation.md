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

## `scripts/27_export_manuscript_package.py`

- **Type:** Script
- **Purpose:** Assembles the manuscript results package (main and supplementary results, LaTeX tables, captions and figures).
- **Inputs detected:** —
- **Outputs detected:** —
- **Third-party dependencies:** —
- **Original SHA-256 (before re-annotation):** `6bda55898ca114233a554d2678ee77497acae64ac27bafdd6898c1f821b9be8e`
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
