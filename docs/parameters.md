# Parameters

Command-line arguments and environment variables extracted directly from the code (help texts are reproduced verbatim from the source and may be in Portuguese).

## scripts/02_audit_raw.py

| argument | type | default | help |
|---|---|---|---|
| `--phases` |  | '' |  |
| `--benchmark` | 'store_true' |  |  |
| `--limit` | int | None | 'limite de ficheiros por coleccao; so para ensaio' |
| `--phases` |  | 'all' | 'lidc,rider,lndb,inventory,headers ou all (omissao: all)' |
| `--benchmark` | 'store_true' |  |  |
| `--limit` | int | None | 'limite de ficheiros por coleccao; so para ensaio' |

## scripts/03_convert_dicom_to_nifti.py

| argument | type | default | help |
|---|---|---|---|
| `--split-manifest` |  | '' | 'conversao dirigida: converte SO os blocos pedidos, sem repetir nenhum volume existente' |
| `--smoke` | int | 0 | 'ensaio: N casos por banco, para pasta descartavel, sem publicar ponteiro' |

## scripts/04_build_lidc_annotations.py

| argument | type | default | help |
|---|---|---|---|
| `--limit` | int | 0 | 'processar apenas as N primeiras series (ensaio)' |
| `--only` |  | '' | 'ensaio dirigido: lista de patient_id separada por virgula' |

## scripts/05_build_lndb_annotations.py

| argument | type | default | help |
|---|---|---|---|
| `--audit-only` | 'store_true' |  | 'mede e relata sem escrever qualquer produto' |

## scripts/06_generate_lung_masks_and_crop.py

| argument | type | default | help |
|---|---|---|---|
| `--only` |  | None | 'lista de case_id separada por virgula (ensaio)' |
| `--plan-only` | 'store_true' |  | 'valida manifestos e entradas; não executa LungMask nem escreve' |

## scripts/07_draw_bd1_partition.py

| argument | type | default | help |
|---|---|---|---|
| `--plan-only` | 'store_true' |  |  |
| `--allow-different` | 'store_true' |  | 'aceita uma particao que NAO reproduza a publicada; so com mudanca deliberada da coorte, e invalida tudo o que dependia da anterior' |
| `--supersede` | 'store_true' |  | 'substitui deliberadamente uma particao ja publicada; invalida tudo o que dela dependa' |

## scripts/08_prepare_nnunet.py

| argument | type | default | help |
|---|---|---|---|
| `--plan-only` | 'store_true' |  |  |
| `--skip-preprocess` | 'store_true' |  |  |
| `--deep` | 'store_true' |  |  |

## scripts/08a_calibrate_collapse_monitor.py

| argument | type | default | help |
|---|---|---|---|
| `--logs-root` |  | LOGS_ROOT_PADRAO |  |
| `--plan-only` | 'store_true' |  | 'mede e relata sem escrever nada' |
| `--supersede` | 'store_true' |  | 'substitui deliberadamente uma regra ja publicada' |

## scripts/08b_probe_gpu_fit.py

| argument | type | default | help |
|---|---|---|---|
| `--merge` |  |  | 'junta csv de varias placas e publica o ponteiro' |
| `--only` |  |  | 'sonda so estes pares, para ensaio' |
| `--compile` | 'store_true' |  | 'aplica torch.compile como o nnUNetTrainer faz; e o caminho REAL do treino em CUDA na 2.5.2' |
| `--time` | 'store_true' |  | 'cronometra o tempo de iteracao e deriva o tempo de epoca e de fold desta placa' |
| `--plans-dir` |  |  | 'pasta com dataset.json e *Plans.json, para sondar uma placa que ainda nao recebeu os dados preprocessados. O hash de cada plano fica no CSV e o modo --merge exige que todas as placas tenham sondado o |

## scripts/08c_freeze_training_config.py

| argument | type | default | help |
|---|---|---|---|
| `--fold` | int | 0 | 'fold usado apenas para instanciar o trainer; nenhum parametro auditado depende dele' |
| `--plan-only` | 'store_true' |  |  |
| `--supersede` | 'store_true' |  |  |

## scripts/09_train_nnunet.py

| argument | type | default | help |
|---|---|---|---|
| `--arch` |  |  | 'treina so estas arquiteturas (por omissao, as dez)' |
| `--folds` | int |  | 'treina so estes folds (por omissao, 0..4)' |
| `--resultados` |  | RESULTADOS_PADRAO | 'raiz do nnUNet_results' |
| `--python` |  | sys.executable | 'interpretador do venv com nnU-Net e os trainers' |
| `--continuar` | 'store_true' |  | 'retoma folds interrompidos a partir do checkpoint' |
| `--supersede` | 'store_true' |  | 'treina por cima de folds ja COMPLETED; invalida o que deles dependa' |
| `--parar-na-primeira` | 'store_true' |  | 'interrompe a serie no primeiro fold que nao complete' |
| `--plan-only` | 'store_true' |  | 'corre todos os portoes e nao treina nada' |
| `--emitir-ordens` |  |  | 'EMISSOR: resolve a cadeia e escreve uma ordem por fold em DIR, sem treinar. E o no que manda.' |
| `--ordem` |  |  | 'EXECUTOR: treina o fold que a ordem descreve. Nao resolve ponteiros nenhuns — verifica a assinatura da ordem e prova que os dados locais sao os que ela diz.' |
| `--preprocessed` |  |  | 'EXECUTOR: raiz do nnUNet_preprocessed local' |
| `--smoke` | int |  | 'ENSAIO: corre N epocas para exercitar o caminho de execucao. NAO e um treino: `num_epochs` define o escalonador de taxa, logo N epocas nao sao o inicio de 1000. Escreve em pasta descartavel e NAO pub |

## scripts/10a_audit_prediction_outputs.py

| argument | type | default | help |
|---|---|---|---|
| `--subset` | 'append' | None |  |
| `--arq` | 'append' | None |  |
| `--sem-recalcular-cc` | 'store_true' |  | 'salta o passo 6; use so em varrimento exploratorio' |

## scripts/13_prepare_filter_objects.py

| argument | type | default | help |
|---|---|---|---|
| `--matches` |  |  | 'bd1_object_matches.csv canónico (Script 11 + 11B)' |
| `--matches-sha256` |  |  |  |
| `--source-runs` |  | '' | 'run_ids do Script 11 e 11B, para o manifesto' |
| `--limit` | int | 0 | 'ENSAIO: só os N primeiros grupos (saída em features/smoke/)' |

## scripts/14_compute_filter_maps_gpu.py

| argument | type | default | help |
|---|---|---|---|
| `--predicted` |  |  |  |
| `--predicted-sha256` |  |  |  |
| `--reference` |  |  |  |
| `--reference-sha256` |  |  |  |
| `--source-run` |  | '' |  |
| `--backend` |  |  |  |
| `--c-global` | float | None | 'obrigatório em produção' |
| `--calibrate-c` | 'store_true' |  | 'mede max S no OOF e sai' |
| `--limit` | int | 0 |  |
| `--only-index` |  | '' | 'ENSAIO: índices separados por vírgula, sobre a ordem canónica' |
| `--out-suffix` |  | '' |  |
| `--examples-only` | 'store_true' |  | 'passagem dedicada: só os N_EXAMPLES primeiros TP_pred da ordem canónica' |
| `--workers` | int | 1 | 'processos paralelos (só backend=cpu; resultado independente do valor — gate)' |

## scripts/15_extract_filter_features.py

| argument | type | default | help |
|---|---|---|---|
| `--features` |  |  | 'filter_features_raw.parquet canónico (Script 14)' |
| `--features-sha256` |  |  |  |
| `--source-run` |  | '' |  |
| `--dataset` |  | 'bd1' |  |
| `--expected-rows` | int | 72430 |  |
| `--limit` | int | 0 |  |
| `--out-dir` |  | None | 'ENSAIO: escrever fora de figdata/' |

## scripts/16_analyze_filters_bd1.py

| argument | type | default | help |
|---|---|---|---|
| `--features` |  |  |  |
| `--features-sha256` |  |  |  |
| `--source-run` |  | '' |  |
| `--out-dir` |  | None | 'ENSAIO: escrever fora de figdata/analysis' |
| `--features-subset` |  | '' | 'ENSAIO: subconjunto de características' |
| `--skip-adjusted` | 'store_true' |  | 'ENSAIO' |
| `--n-boot` | int | N_BOOTSTRAP |  |

## scripts/17_train_filter_verifier.py

| argument | type | default | help |
|---|---|---|---|
| `--features` |  |  |  |
| `--features-sha256` |  |  |  |
| `--script16-manifest` |  |  |  |
| `--source-run` |  | '' |  |
| `--out-dir` |  | None | 'ENSAIO: escrever fora de models/config/analysis' |
| `--n-boot` | int | N_BOOTSTRAP |  |

## scripts/17a_apply_filter_verifier_internal.py

| argument | type | default | help |
|---|---|---|---|
| `--features` |  | os.path.join(BASE, 'figdata', 'bd1_filter_features.csv') |  |
| `--verifier` |  | os.path.join(BASE, 'models', 'filter_verifier_A.pkl') |  |
| `--verifier-sha256` |  | '' |  |
| `--script17-manifest` |  | os.path.join(ANALYSIS, 'filter_verifier_manifest.json') |  |
| `--threshold-config` |  | os.path.join(CONFIG, 'filter_verifier_threshold.json') |  |
| `--split` |  | os.path.join(CONFIG, 'bd1_split.csv') |  |
| `--script11-matches` |  |  | 'bd1_object_matches.csv do run canónico do Script 11B' |
| `--out-root` |  | os.path.join(ANALYSIS, 'internal_verifier_application') |  |

## scripts/18_prepare_rider_registration.py

| argument | type | default | help |
|---|---|---|---|
| `--limit-pairs` | int | 0 | 'ENSAIO: só os N primeiros pares' |
| `--out-root` |  | None | 'ENSAIO: raiz alternativa' |
| `--fixture-only` | 'store_true' |  |  |

## scripts/19_match_rider_hungarian_iou.py

| argument | type | default | help |
|---|---|---|---|
| `--s18-run` |  |  | 'pasta do run canónico do Script 18' |
| `--s18-manifest-sha256` |  |  |  |
| `--inspection` |  | '' | 'JSON com a classificação dos pares FLAG' |
| `--limit-pairs` | int | 0 |  |
| `--out-root` |  | None |  |

## scripts/20_extract_rider_morphometry_and_filters.py

| argument | type | default | help |
|---|---|---|---|
| `--s19-run` |  |  |  |
| `--s19-manifest-sha256` |  |  |  |
| `--pairs` |  | '' | 'por omissão, rider_iou_pairs.csv do run' |
| `--pairs-sha256` |  |  |  |
| `--c-calibration` |  | C_CALIB_DEFAULT |  |
| `--limit-patients` | int | 0 |  |
| `--patients` |  | '' | 'lista explícita, só para ensaio' |
| `--workers` | int | 1 |  |
| `--parity-sample` | int | 12 |  |
| `--parity-only` | 'store_true' |  |  |
| `--out-root` |  | None |  |

## scripts/21_analyze_rider_repeatability.py

| argument | type | default | help |
|---|---|---|---|
| `--features` |  |  |  |
| `--features-sha256` |  |  |  |
| `--script20-manifest` |  |  |  |
| `--script20-run-id` |  | S20_RUN_ESPERADO |  |
| `--n-boot` | int | B_VALID |  |
| `--out-root` |  | None |  |
| `--gates-only` | 'store_true' |  |  |

## scripts/22_evaluate_lndb.py

| argument | type | default | help |
|---|---|---|---|
| `--lndb-manifest` |  | MANIFESTO_LNDB |  |
| `--manifest-sha256` |  | '' |  |
| `--limit-scans` | int | 0 |  |
| `--out-root` |  | None |  |
| `--gates-only` | 'store_true' |  |  |

## scripts/22a_prepare_lndb_filter_objects.py

| argument | type | default | help |
|---|---|---|---|
| `--s22-run` |  |  |  |
| `--s22-manifest-sha256` |  |  |  |
| `--limit-scans` | int | 0 |  |
| `--out-root` |  | None |  |

## scripts/22b_compute_lndb_filter_maps.py

| argument | type | default | help |
|---|---|---|---|
| `--s22a-run` |  |  |  |
| `--s22a-manifest-sha256` |  |  |  |
| `--c-calibration` |  | C_CALIB |  |
| `--limit-scans` | int | 0 |  |
| `--workers` | int | 1 |  |
| `--parity-sample` | int | 10 |  |
| `--parity-only` | 'store_true' |  |  |
| `--out-root` |  | None |  |

## scripts/23_validate_filters_lndb.py

| argument | type | default | help |
|---|---|---|---|
| `--s22-run` |  |  |  |
| `--s22-manifest-sha256` |  |  |  |
| `--s22b-run` |  |  |  |
| `--s22b-manifest-sha256` |  |  |  |
| `--verifier` |  | os.path.join(BASE, 'models', 'filter_verifier_A.pkl') |  |
| `--verifier-sha256` |  | '' |  |
| `--s17-manifest` |  | os.path.join(ANALYSIS_DIR, 'filter_verifier_manifest.json') |  |
| `--bd1-features` |  | os.path.join(FIGDATA_DIR, 'bd1_filter_features.csv') |  |
| `--out-root` |  | None |  |

## scripts/24_integrated_statistics.py

| argument | type | default | help |
|---|---|---|---|
| `--s11b-run` |  | os.path.join(U, 'script11_final', 's11B_20260921T164539_27883') |  |
| `--s12-run` |  | os.path.join(U, 'script12_final', 's12_20260921T223557_82479') |  |
| `--s16-run` |  | os.path.join(U, 'script16_final', 's16_20260923T094823_494840') |  |
| `--s21-run` |  | os.path.join(ANALYSIS_DIR, 'rider_repeatability', 's21_20260923T215458_708846') |  |
| `--s22-run` |  | os.path.join(ANALYSIS_DIR, 'lndb_external', 's22_20260924T084718_826234') |  |
| `--s23-run` |  | os.path.join(ANALYSIS_DIR, 'lndb_filter_validation', 's23_20260924T153750_909396') |  |
| `--n-boot` | int | B_BOOT |  |
| `--out-root` |  | None |  |

## scripts/25_generate_tables_and_figures.py

| argument | type | default | help |
|---|---|---|---|
| `--out-root` |  | None |  |
| `--flowchart-spec` |  | '' |  |

## scripts/26_reproducibility_audit.py

| argument | type | default | help |
|---|---|---|---|
| `--manuscript` |  | '' |  |
| `--manuscript-sha256` |  | '' |  |
| `--geometry-limit` | int | 0 | 'só para ensaio; 0 = exaustivo' |
| `--out-root` |  | None |  |

## Environment variables

| variable | used by |
|---|---|
| `A` | 17a_apply_filter_verifier_internal.py, 23_validate_filters_lndb.py, 26_reproducibility_audit.py |
| `ANNOTATION_MERGE_STATUS` | 06_generate_lung_masks_and_crop.py |
| `BOOTSTRAP` | 26_reproducibility_audit.py |
| `COLLAPSED` | 08a_calibrate_collapse_monitor.py |
| `COMPLETED` | 09_train_nnunet.py |
| `CONFIDENCE_INTERVALS` | 26_reproducibility_audit.py |
| `CONSENSUS_RULE` | 26_reproducibility_audit.py |
| `CONVERSION_STATUS` | 04_build_lidc_annotations.py |
| `COVERAGE_STATUS` | 08_prepare_nnunet.py |
| `EXCLUDED_DUPLICATE_IMAGE_PRESERVED` | 26_reproducibility_audit.py |
| `EXCLUDED_REFERENCE_READER_DISCORDANCE` | 26_reproducibility_audit.py |
| `HEADERS_OK` | 02_audit_raw.py |
| `HEALTHY` | 08a_calibrate_collapse_monitor.py |
| `HYPOTHESIS_TESTS` | 26_reproducibility_audit.py |
| `INFERENCE` | 26_reproducibility_audit.py |
| `INSTALL_IN_CANONICAL_CLUSTER` | 00_check_environment.py |
| `LIDC_ANALYTIC_DENOMINATOR` | 02_audit_raw.py, 03_convert_dicom_to_nifti.py |
| `LNDB_ANALYTIC_N` | 23_validate_filters_lndb.py |
| `LNDB_DUPLICATE_GROUPS` | 05_build_lndb_annotations.py |
| `LNDB_MANIFEST_STATUS` | 05_build_lndb_annotations.py |
| `LNDB_NOMINAL_COUNT` | 05_build_lndb_annotations.py |
| `LNDB_PROVENANCE_STATUS` | 05_build_lndb_annotations.py |
| `MERGE_STATUS` | 03_convert_dicom_to_nifti.py |
| `NEVER_LEARNED` | 08a_calibrate_collapse_monitor.py |
| `NNUNET_VERSION` | 00_check_environment.py |
| `NULL_MODALITY` | 02_audit_raw.py |
| `OOF_COVERAGE_EXACT` | 08_prepare_nnunet.py |
| `PARTITION_FROZEN` | 08_prepare_nnunet.py |
| `PASS` | 01_download_datasets.py |
| `PASS_NO_EXPECTED_COUNT` | 01_download_datasets.py |
| `PRODUCT_STATUS` | 08_prepare_nnunet.py |
| `PULMONARY_ARCH_FILTER` | 10_infer_and_export_predictions.py |
| `PULMONARY_BASE` | 04_build_lidc_annotations.py, 07_draw_bd1_partition.py, 08a_calibrate_collapse_monitor.py, 08b_probe_gpu_fit.py, 08c_freeze_training_config.py, 17a_apply_filter_verifier_internal.py, 20_extract_rider_morphometry_and_filters.py, 21_analyze_rider_repeatability.py, 22_evaluate_lndb.py, 22a_prepare_lndb_filter_objects.py, 22b_compute_lndb_filter_maps.py, 23_validate_filters_lndb.py, 24_integrated_statistics.py, 25_generate_tables_and_figures.py, 26_reproducibility_audit.py, pulmonary_base.py |
| `PULMONARY_NNUNET_VERSION` | 00_check_environment.py |
| `PULMONARY_PREPROCESSED` | 09_train_nnunet.py, treino_core.py |
| `PULMONAR_ARQ` | conduzir_colab_v4.py |
| `PULMONAR_FOLDS` | conduzir_colab_v4.py |
| `RECALIBRATION_ON_LNDB` | 26_reproducibility_audit.py |
| `ROUTING_COVERAGE_COMPLETE` | 02_audit_raw.py |
| `SCRIPT04_OUTPUT_MANIFEST_FROZEN` | 08_prepare_nnunet.py, 26_reproducibility_audit.py |
| `SCRIPT04_OUTPUT_STATUS` | 06_generate_lung_masks_and_crop.py, 07_draw_bd1_partition.py, 08_prepare_nnunet.py, 26_reproducibility_audit.py |
| `SCRIPT05_ANALYTIC_COUNT` | 26_reproducibility_audit.py |
| `SCRIPT05_EXCLUDED_PRESERVED` | 26_reproducibility_audit.py |
| `SCRIPT05_NOMINAL_COUNT` | 26_reproducibility_audit.py |
| `SCRIPT06_OUTPUT_STATUS` | 08_prepare_nnunet.py |
| `SCRIPT06_REFERENCE_GUIDED_CROP` | 08_prepare_nnunet.py, caminhos_canonicos.py |
| `SCRIPT06_TRUNCATED_VOXELS` | 08_prepare_nnunet.py |
| `SCRIPT08_LABEL_OVERLAY` | 08_prepare_nnunet.py |
| `SELECTION_RULE` | 03_convert_dicom_to_nifti.py |
| `SHA256` | 00_check_environment.py |
| `TAU_LNDB_APPLIED` | 26_reproducibility_audit.py |
| `THRESHOLD_SOURCE` | 26_reproducibility_audit.py |
| `THRESHOLD_TUNING_ON_LNDB` | 26_reproducibility_audit.py |
| `TP_IOU_THRESHOLD_GATE` | 22a_prepare_lndb_filter_objects.py |
| `UNEXPECTED_MODALITY` | 02_audit_raw.py |
| `cuda` | 09_train_nnunet.py |
| `gpu` | 09_train_nnunet.py |
| `host` | 09_train_nnunet.py |
| `nnUNet_compile` | 09_train_nnunet.py |
| `nnUNet_preprocessed` | 08_prepare_nnunet.py, 08c_freeze_training_config.py, 10_infer_and_export_predictions.py |
| `nnUNet_raw` | 08_prepare_nnunet.py, 08c_freeze_training_config.py, 10_infer_and_export_predictions.py |
| `nnUNet_results` | 08_prepare_nnunet.py, 08c_freeze_training_config.py, 10_infer_and_export_predictions.py |
| `torch` | 09_train_nnunet.py |
