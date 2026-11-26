# Changelog

## v1.1.0 — 2026-11-26
- Title revised to identify the record as the reproducibility package of the article *Multiscale morphology-informed post-detection verification of deep learning pulmonary nodule candidates in computed tomography: false-positive reduction, feature repeatability and external validation*.
- Manuscript-writing code removed (stage 27, manuscript package export, and the scripts that formatted LaTeX tables): the package contains only the code that reproduces the results.
- New stages 28–32 (71 files, `scripts/28_*` to `scripts/32_*`, documented in `docs/post_v1_stages.md`):
  LNDb N1 reference rectification and re-evaluation (28), selective verification (29), radiological assessment linkage (30),
  Model B, Model C and ablation in the development folds (31) and supplementary analyses (32).
- New result tables of stages 28–32 in `data/stage_outputs_28_32/` (191 CSV/JSON files; no image or mask).
- All LNDb results now use the N1 reference (union of the readers' segmentations of each official lesion; 211 analytical scans;
  224 unique images as sensitivity analysis). The LNDb metrics of stages 22–25 obtained with the earlier local majority reference are superseded.
- Radiologist anonymized in file names, paths and strings of stage 30; original marking files and spreadsheet not included.
- New files re-annotated with the standard header, original comments removed and absolute path prefixes made relative;
  executable logic verified unchanged by abstract-syntax-tree comparison (71/71). Headers of the v1.0.0 files updated (title, version, citation) without code changes.
- `tests/test_smoke.py` now scans sub-folders of `scripts/`. Documentation, metadata, file manifest, data dictionary and checksums updated.
- Trained models unchanged (same 50 checkpoints as v1.0.0).

## v1.0.0 — 2026-10-26
- First archived version: pipeline stages 00–27 (executed versions), ten nnU-Net v2.8.1 configurations × five folds (`checkpoint_final.pth`; `checkpoint_latest.pth` for three aborted folds),
  cohort splits, stage outputs, run records, documentation and checksums.
- Code re-annotated according to the repository protocol: original comments removed, standard headers inserted, machine-specific paths neutralised;
  executable logic verified unchanged by AST comparison.
