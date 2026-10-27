# Quality control report — 2026-10-26

| check | result |
|---|---|
| Code files re-annotated (header, comments removed, paths neutralised) | 48 |
| Executable logic identical (AST comparison) | 47 of 48 (the 2 Colab notebooks differ only in neutralised paths) |
| Stage scripts matched by SHA-256 to the final executed run | 20 |
| Machine-specific absolute paths in code/documentation | checked by `tests/test_smoke.py` |
| Text data files with paths/host names neutralised (copies) | see `metadata/file_manifest.csv`, status `original` vs copies |
| Parquet files with absolute paths | 0 |
| Tables described in `metadata/data_dictionary.csv` | 65 tables, 1085 variables |
| Hidden/temporary files (`.DS_Store`, `._*`, `__pycache__`) | removed |
| Test/temporary outputs (`TEST_*`) | excluded |

## Pending before publication (author curation)

1. Review the variable definitions in `metadata/data_dictionary.csv` (column `definition_source`: project dictionaries, authors from code, or naming rules).
2. Tables that could not be parsed automatically (0): none.
3. Confirm the nnU-Net version: the trainers, patches and model folder refer to v2.8.1, while `environment/training_lock.json` (12/08/2026) records nnunetv2 2.5.2.
4. Stages 00–10, 26 and 27 have no SHA-256 record of the executed script; confirm that the frozen v2.8.1 folder versions are the ones executed.
5. Auxiliary scripts recorded by stage 11 (`11_audit_hungarian_size_cc.py`, `11_patched_equivalence.py`) and alternative executed versions of stages 14 and 25 were not found on disk and are not included.
7. Confirm how predictions were obtained for the three folds aborted by the collapse monitor (no `checkpoint_final.pth`); `checkpoint_best.pth` of all 50 folds kept outside the record in `Zenodo_v281/checkpoints_best_v281_nao_enviados/`.
6. Image and mask file names keep the identifiers of the source datasets (upper case), against the lower-case naming rule, to preserve traceability.
