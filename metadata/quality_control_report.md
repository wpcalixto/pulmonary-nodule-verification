# Quality control report — 2026-11-26 (v1.1.0)

| check | result |
|---|---|
| Code files re-annotated (header, comments removed, paths neutralised) | 118 (47 from v1.0.0, 71 new in stages 28–32) |
| Executable logic identical (AST comparison) | v1.0.0 files: as in v1.0.0, header-only update verified; stages 28–32: 71 of 71 |
| Stage 28–32 files taken from the executed analysis folders (original SHA-256 in `docs/file_level_documentation.md`) | 71 |
| Radiologist anonymized (file names, paths, strings, data) | 18 code files, 1 data file; original markings not included |
| Machine-specific absolute paths and host names in code, documentation and data of stages 28–32 | 0 (3 data files neutralised) |
| Result tables of stages 28–32 | 191 CSV/JSON files, no image or mask |
| Smoke test (`python tests/test_smoke.py --checksums`) | OK |
| Manuscript-writing code (stage 27, LaTeX table formatters) | removed; only result-reproduction code is included |
| Hidden/temporary files (`.DS_Store`, `._*`, `__pycache__`) | removed |

## Pending before publication (author curation)

1. Review the variable definitions of the new tables in `metadata/data_dictionary.csv` (rows with `definition_source` = `pending curation (automatic profile)`).
2. Items 3–7 of the v1.0.0 report remain as recorded: nnU-Net version recorded in `environment/training_lock.json` (2.5.2, qualification arm) versus 2.8.1 (final stack); stages 00–10 and 26 without per-run SHA-256 record; auxiliary stage-11 scripts not found; predictions of the three folds stopped by the monitor; `checkpoint_best.pth` kept outside the record.
3. Stages 28–32 keep the folder layout of the original project (relative paths); see `docs/post_v1_stages.md`.
