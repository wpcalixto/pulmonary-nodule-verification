# Stages 28–32 (added in v1.1.0)

Stages 28–32 contain the analyses that complete the results reported in the article and were not part of v1.0.0. They read the frozen products of stages 00–26 (checkpoints, predictions, features, frozen verifier and thresholds) and do not retrain any model or change any threshold.

**Execution layout.** The scripts are archived as executed. Absolute path prefixes were removed, so every path is relative to the project directory used at execution (the working directory). In that layout, the outputs of stages 00–26 are under `resultados/` and the source collections under `dados/brutos/`. The folders in the last column of each table give the original output folder of each stage; their tables are archived in `data/stage_outputs_28_32/<stage>/` of the Zenodo record.

**Anonymization.** The radiologist is not identified. File names, paths and text strings that contained the radiologist's name were replaced by `radiologist`; the original marking files and spreadsheet are not included.

## 28 — LNDb N1 reference rectification and re-evaluation

LNDb distributes one mask per radiologist and no consensus mask. This stage builds the lesion-level N1 reference (union of the segmentations of the readers who segmented each official lesion), the N2 reference (lesions marked as nodule by at least two readers) and the ignored regions; reproduces the lung-mask cropping of stage 06 and the frozen inference for the 13 scans outside the analytical cohort; and re-evaluates the frozen predictions and the frozen Model A decisions (211 analytical scans; 224 unique images as sensitivity analysis). It supersedes the LNDb metrics of stages 22–25, which used a local majority reference.

**Article:** Sections 2.2, 3.1, 3.7, 3.8 and 3.10 (LNDb part); Table 3 and Table 4; LNDb rows of the supplementary tables and figures.

**Original output folders:** `resultados/retificacao_LNDb_20261111`; `resultados/investigacao_V03_LNDb_20261111`

| file | purpose |
|---|---|
| `scripts/28_lndb_reference_rectification/E1_associacao_geometria.py` | One-to-one Hungarian matching of predicted objects and LNDb reference lesions per scan, configuration, reference and threshold, using the matching functions of stages 11 and 22. |
| `scripts/28_lndb_reference_rectification/E2_estatisticas.py` | Recomputes the LNDb-dependent analyses with the N1 reference (211 scans) and the sensitivity cohorts (224 unique images, N2), with scan-level bootstrap. |
| `scripts/28_lndb_reference_rectification/E3_AUROC_A_C.py` | Paired AUROC and average-precision differences between Models A and C on the LNDb candidates, with scan-level bootstrap. |
| `scripts/28_lndb_reference_rectification/E4_holm_conjunta.py` | Joint Holm family combining LNDb and development p values of the sixteen responses for the external configurations. |
| `scripts/28_lndb_reference_rectification/F1_respostas_referencia.py` | Extracts the sixteen morphological responses of the N1 reference lesions of the 211 LNDb scans through the stage 13/14/15 modules. |
| `scripts/28_lndb_reference_rectification/F_respostas.py` | Extracts the sixteen morphological responses, the largest axial diameter and the frozen Model A score of the LNDb predicted objects through the stage 13/14/15 modules. |
| `scripts/28_lndb_reference_rectification/G1_geometria_s03_N1.py` | Segmentation geometry (Dice, Jaccard, surface distances) of the N1 true-positive pairs, by size stratum and verifier decision, with the stage-32b geometry functions. |
| `scripts/28_lndb_reference_rectification/G2_figuras_lndb.py` | Generates the LNDb detection and verification figures from the closed N1 tables. |
| `scripts/28_lndb_reference_rectification/G3_hd_pares_N1.py` | Surface Hausdorff distance of the N1 true-positive pairs, with a Dice parity gate. |
| `scripts/28_lndb_reference_rectification/P1_recorte_s06_replica.py` | Strict reproduction of the lung-mask generation and cropping of stage 06 for the 13 LNDb scans outside the analytical cohort, with byte-level parity gates against the recorded outputs. |
| `scripts/28_lndb_reference_rectification/P2_regenerar_mascaras_pulmonares_211.py` | Regenerates the cropped lung masks of the 211 analytical LNDb scans with the validated stage-06 reproduction, accepting each mask only when its SHA-256 matches the recorded input manifest. |
| `scripts/28_lndb_reference_rectification/R1_construir_referencias.py` | Builds, from the official per-radiologist LNDb masks, the lesion-level references of the 236 scans: the N1 reference (union of the segmentations of each official lesion), the N2 reference and the regions to be ignored. |
| `scripts/28_lndb_reference_rectification/R2_reavaliar_211.py` | Re-evaluates the frozen LNDb predictions and the frozen Model A decisions against the N1 and N2 references, with a control that reproduces the previous reference. |
| `scripts/28_lndb_reference_rectification/R3_metricas_e_IC.py` | Computes sensitivity, precision, F1 and FP per scan before and after verification for each reference, configuration and matching threshold, with scan-level bootstrap intervals. |
| `scripts/28_lndb_reference_rectification/V_verificacao.py` | Arithmetic verification of all LNDb counts, aggregates, metrics and intervals. |
| `scripts/28_lndb_reference_rectification/protocolo_grafico.py` | Shared plotting helpers (number formatting, figure registration) driven by the visual-identity YAML file. |
| `scripts/28_lndb_reference_rectification/v03_auditoria_mascaras_oficiais.py` | Compares the official per-radiologist LNDb masks with the previous local majority reference. |
| `scripts/28_lndb_reference_rectification/v03_componentes_por_tipo_de_achado.py` | Classifies each reference component by the type of official finding that covers it. |
| `scripts/28_lndb_reference_rectification/v03_impacto_classes_TP_FN.py` | Cross-tabulates LNDb reference objects against the class of the official finding that originated them. |

## 29 — Selective verification

Direct acceptance of candidates whose detector confidence reaches the routing threshold t_u, with Model A applied to the remaining candidates; t_u is chosen in D1 out-of-fold data under the false-positive constraint of full verification and applied, frozen, to D2 and D4.

**Article:** Section 3.8 (selective verification) and the selective-verification supplementary table.

**Original output folders:** `resultados/verificacao_seletiva_20261111`

| file | purpose |
|---|---|
| `scripts/29_selective_verification/E1b_estado_predicoes_por_tau.py` | Records the state (TP, FP, ignored) of each LNDb prediction at IoU thresholds 0.10, 0.25 and 0.50 for the N1, N2 and alternative-representative references. |
| `scripts/29_selective_verification/S1_confianca_candidatos.py` | Computes the detector confidence of each candidate (mean foreground probability over the component voxels). |
| `scripts/29_selective_verification/S2_avaliacao_seletiva.py` | Selective verification: direct acceptance of candidates above the confidence threshold and Model A for the remainder, compared with full verification. |
| `scripts/29_selective_verification/S3_descritivo_confianca.py` | Descriptive analysis of the detector confidence measure among accepted and rejected candidates. |
| `scripts/29_selective_verification/S4_curvas_operacionais.py` | Descriptive comparison of the operating curves of selective and full verification. |

## 30 — Radiological assessment linkage

Alignment of the CT used for the radiological markings with the dataset CT (with axis-reversal controls), spatial linkage of the radiologist's markers to predicted and reference objects without access to verifier decisions, blinded confirmation and freezing of the record-to-marker link, and junction with detection outcomes, verifier decisions and segmentation geometry.

**Article:** Section 3.9; Figure 10; radiological-record supplementary tables and figure.

**Original output folders:** `resultados/radiological_reanalysis` (folder renamed for anonymization)

| file | purpose |
|---|---|
| `scripts/30_radiological_assessment/c01_teste_alinhamento.py` | Tests the voxel alignment between the CT used for the radiological markings and the CT of the dataset, with axis-reversal negative controls. |
| `scripts/30_radiological_assessment/c01b_teste_nrrd.py` | Tests the alignment of the radiologist's segmentation files with the current image grid, with inverted-content negative controls. |
| `scripts/30_radiological_assessment/c01c_componentes_radiologist_0061.py` | Describes the components of the radiologist's marking for one scan on the current image grid. |
| `scripts/30_radiological_assessment/c02_correspondencia.py` | Spatial correspondence between the radiologist's markers and the predicted and reference objects, without access to verifier decisions. |
| `scripts/30_radiological_assessment/c02b_lateralidade.py` | Robust laterality assignment by splitting the lung mask into the two lungs. |
| `scripts/30_radiological_assessment/c03_verificador.py` | Frozen Model A decisions for the 3D-F and 3D-FREM objects of the assessed scans. |
| `scripts/30_radiological_assessment/c04_juncao.py` | Joins the frozen marker correspondence with the verifier decisions. |
| `scripts/30_radiological_assessment/c05_proposta_ligacao.py` | Proposes the link between the radiological records and the markers from the anatomical description. |
| `scripts/30_radiological_assessment/c06_por_leitor_0061.py` | Checks the per-reader annotations of one scan before assigning reference status to two records. |
| `scripts/30_radiological_assessment/c07_confirmacao_cega.py` | Builds the blinded confirmation table of the link between radiological records and markers. |
| `scripts/30_radiological_assessment/c08_oof_verificador.py` | Reproduces the out-of-fold Model A score of each development object from the frozen stage-17 sources. |
| `scripts/30_radiological_assessment/c08b_confere_publicado.py` | Checks that the reproduced out-of-fold scores match the frozen stage-17 records. |
| `scripts/30_radiological_assessment/c09_congela_ligacao.py` | Freezes the confirmed record-to-marker link with its SHA-256. |
| `scripts/30_radiological_assessment/c10_desfechos.py` | Joins the frozen link with the detection outcomes and verifier decisions. |
| `scripts/30_radiological_assessment/c11_grafico_notas.py` | Plots the delineation scores of the true-positive records with the verifier decision. |
| `scripts/30_radiological_assessment/c12_geometria_tp.py` | Dice and HD95 of the true-positive pairs linked to radiological records. |
| `scripts/30_radiological_assessment/c13_opiniao_planilha.py` | Reads the radiologist's interpretation of each record from the primary spreadsheet. |

## 31 — Model B, Model C and ablation

Models B and C and the Model A ablations fitted in the same five patient-grouped folds used to develop Model A, with Model C thresholds (Youden and target retention) frozen in D1 and applied to the internal and external candidates.

**Article:** Section 3.5; Table 2; Model A versus Model C comparisons in Sections 3.5 and 3.8 and the corresponding supplementary tables.

**Original output folders:** `resultados/modelo_c_ablacoes_dobras`

| file | purpose |
|---|---|
| `scripts/31_model_c_and_ablation/c14_modeloC_ablacoes.py` | Fits Models B and C and the Model A ablations in the same five patient-grouped development folds, and applies Model C to the internal and external candidates. |

## 32 — Supplementary analyses

a) verifier sensitivity to the IoU threshold, size-conditional increment (B versus C), LNDb cohort audit and threshold stability between out-of-fold and final scores; b) segmentation geometry of true-positive pairs by dataset, size and verifier decision; c) LIDC-IDRI reference-unit sensitivity and fragment-excluded AUROC; d) LNDb inter-reader agreement, Hungarian-rule audits, RIDER matching by size and pairing rule, partition reproduction and N1 multicomponent lesions; e) scan-level operating intervals, fold equivalence and LNDb intensity-scale check.

**Article:** Sections 3.1–3.6 and 3.10; Figure 6(a) and Figure 11; the corresponding supplementary tables and figures.

**Original output folders:** `resultados/analises_complementares_CMIG`; `resultados/analises_complementares_v8`; `resultados/ordem_P01_11_20261112`; `resultados/ordem_P01_12_20261112`; `resultados/ordem_P01_13_20261113`; `resultados/novos_resultados_20261112`; `resultados/ordem_01R_20261112`; `resultados/ordem_P03_01_20261113`; `resultados/ordem_P03_02_20261113`; `resultados/parecer_P01_20261112`; `resultados/fechamento_cientifico_20261111`; `codigo/auditoria_v4_20261109`

| file | purpose |
|---|---|
| `scripts/32_supplementary_analyses/a_verifier_and_lndb_cohort/10_v01_iou_verificador.py` | Sensitivity of the detector-plus-verifier procedure to the IoU matching threshold. |
| `scripts/32_supplementary_analyses/a_verifier_and_lndb_cohort/20_v02_b_c_oof.py` | Size-conditional discriminative increment (Model B versus Model C) on development out-of-fold predictions. |
| `scripts/32_supplementary_analyses/a_verifier_and_lndb_cohort/30_v03_coorte_lndb.py` | Documentary and quantitative audit of the LNDb cohort selection (236 to 211 scans). |
| `scripts/32_supplementary_analyses/a_verifier_and_lndb_cohort/40_v09_escores_limiar.py` | Stability of the frozen threshold between out-of-fold and final Model A scores. |
| `scripts/32_supplementary_analyses/a_verifier_and_lndb_cohort/90_testes_integridade.py` | Integrity tests of the inputs and outputs of the analyses V01, V02, V03 and V09. |
| `scripts/32_supplementary_analyses/a_verifier_and_lndb_cohort/comum.py` | Shared paths and helper functions of the supplementary analyses V01, V02, V03 and V09. |
| `scripts/32_supplementary_analyses/b_segmentation_geometry/s02_geometria_pares.py` | Per-object segmentation geometry (Dice, Jaccard, surface distances, HD95) of the true-positive pairs in the internal and external datasets. |
| `scripts/32_supplementary_analyses/b_segmentation_geometry/s03_geometria_analise.py` | Geometry summaries by configuration, dataset, size stratum and verifier decision, with bootstrap intervals. |
| `scripts/32_supplementary_analyses/b_segmentation_geometry/s04_quadro_candidatos.py` | Table of all predicted objects of 3D-F and 3D-FREM with original class, verifier score and decision. |
| `scripts/32_supplementary_analyses/c_reference_unit_and_fragments/G_fig6a_tamanho.py` | Generates the figure of sensitivity by reference size in development. |
| `scripts/32_supplementary_analyses/c_reference_unit_and_fragments/P01_11_diag45.py` | Diagnosis of the patients whose reconstructed groupings differ from the consensus. |
| `scripts/32_supplementary_analyses/c_reference_unit_and_fragments/P01_11_resumo.py` | Summary of the reference-unit sensitivity analysis. |
| `scripts/32_supplementary_analyses/c_reference_unit_and_fragments/P01_11_resumo_coerentes.py` | Reference-unit comparison restricted to patients whose annotations reconstruct the consensus. |
| `scripts/32_supplementary_analyses/c_reference_unit_and_fragments/P01_11_unidade_referencia.py` | Sensitivity of the LIDC-IDRI reference unit: face-connected components versus reader groupings with merged fragments. |
| `scripts/32_supplementary_analyses/c_reference_unit_and_fragments/P01_12_1_populacao.py` | Compares the patients with reconstructable annotations with the remaining patients. |
| `scripts/32_supplementary_analyses/c_reference_unit_and_fragments/P01_12_lndb_unidade.py` | Diagnosis of the reference unit in LNDb (N1). |
| `scripts/32_supplementary_analyses/c_reference_unit_and_fragments/P01_13_auroc_fragmentos.py` | AUROC of maximum blobness between detected and missed references after excluding fragments of larger groupings. |
| `scripts/32_supplementary_analyses/d_interobserver_matching_and_partition/G3_figura_rider_limiares_relativos.py` | Generates the figure of RIDER pairs whose relative change exceeds the thresholds nu_1 and nu_2. |
| `scripts/32_supplementary_analyses/d_interobserver_matching_and_partition/N2_concordancia_leitores_LNDb.py` | Inter-reader identification agreement in LNDb through the official finding groupings. |
| `scripts/32_supplementary_analyses/d_interobserver_matching_and_partition/O02_N1_delineamento_lidc.py` | Effect of the reference delineation rule (union versus consensus) in LNDb. |
| `scripts/32_supplementary_analyses/d_interobserver_matching_and_partition/O03_auditoria_hungaro_lndb.py` | Audit of the Hungarian matching rule (admissible-pair priority versus total-IoU maximization) in LNDb. |
| `scripts/32_supplementary_analyses/d_interobserver_matching_and_partition/O09_pareamento_tamanho.py` | RIDER test-retest matching rate by configuration and object size. |
| `scripts/32_supplementary_analyses/d_interobserver_matching_and_partition/P01_01_rider_AB.py` | RIDER pairing under total-IoU priority versus pair-count priority. |
| `scripts/32_supplementary_analyses/d_interobserver_matching_and_partition/V01_reproduz_particao.py` | Reproduces the D1/D2 partition from the recorded seed and compares it with the archived partition. |
| `scripts/32_supplementary_analyses/d_interobserver_matching_and_partition/V03_lesoes_multicomponente.py` | Checks LNDb N1 lesions with more than one connected component and their response vectors. |
| `scripts/32_supplementary_analyses/d_interobserver_matching_and_partition/V04_reconcilia_722_686.py` | Reconciles the N1 lesion counts of the reference scan with those of the 224-image cohort. |
| `scripts/32_supplementary_analyses/e_operating_intervals_and_scale/folds.py` | Checks the equivalence between the verifier modeling folds and the archived development folds. |
| `scripts/32_supplementary_analyses/e_operating_intervals_and_scale/ic_oper.py` | Scan-level bootstrap intervals of the operating metrics before and after verification. |
| `scripts/32_supplementary_analyses/e_operating_intervals_and_scale/lndb_hu.py` | Checks the LNDb intensity scale from headers and air, fat and soft-tissue histogram peaks in a systematic sample. |
