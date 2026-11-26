# Reproducibility

- **Executed versions.** Stages 10a–25 were matched to the exact files executed in the final runs through the SHA-256 recorded in each run manifest
  (`docs/file_level_documentation.md`). Stages 00–10 come from the frozen v2.8.1 pipeline used to train the fifty models.
- **Re-annotation.** Original comments were removed and the standard header was inserted; machine-specific absolute paths were replaced by
  relative paths. The abstract syntax tree of every Python file was compared before and after: the executable logic is identical (46/48 files;
  the two Colab notebooks differ only in the neutralised paths). Function docstrings were preserved verbatim.
- **Partition and seeds.** `data/nnunet_dataset/splits_final.json`, `data/run_records_00_10/` (BD1 partition draw, collapse rule, GPU fit, frozen training configuration).
- **Models.** `models/<configuration>/Dataset001_PulmonaryNodules/<trainer__plans__configuration>/fold_<k>/checkpoint_final.pth` — the checkpoint used by
  `nnUNetv2_predict` (default, no `-chk`) to produce every prediction analysed — with `plans.json`, `dataset.json` and `dataset_fingerprint.json`. Three folds (3D low-resolution, Foreground100 fold 4; ResEncM Foreground100 folds 0 and 1) were aborted
  by the frozen training-collapse monitor at epoch 300 (no learning) and are provided as `checkpoint_latest.pth` with the abort record (`estado_fold.json`).
  Validation probability maps (`.npz`) are not included.
- **Integrity.** `checksums/checksums_sha256.txt` and `metadata/file_manifest.csv`; `python tests/test_smoke.py --checksums`.
- **Stages 28–32 (v1.1.0).** Taken from the executed analysis folders, anonymized where the radiologist was named, re-annotated with the standard header, comments removed and absolute path prefixes made relative; the abstract syntax tree of each file was compared before and after (71/71 identical, with the path prefixes normalized). Their result tables are in `data/stage_outputs_28_32/`.
- **LNDb reference.** All LNDb results of the article use the N1 reference built in stage 28 (211 analytical scans; 224 unique images as sensitivity analysis). The LNDb metrics produced by stages 22–25 with the earlier local majority reference are superseded.
