# Reproducibility package for multiscale morphology-informed post-detection verification of deep learning pulmonary nodule candidates in computed tomography: code, trained nnU-Net models, cohort splits and analysis data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23674672.svg)](https://doi.org/10.5281/zenodo.23674672)

> **The dataset and the trained models are on Zenodo.** This GitHub repository contains **only the source code and documentation**.
> The dataset (cohort definitions, case lists and cross-validation splits, nnU-Net dataset definition and plans, frozen parameters, and the result tables of every stage) and the **trained nnU-Net models** (50 checkpoints, ~27 GB) are archived on Zenodo:
>
> **Zenodo (all versions): [https://doi.org/10.5281/zenodo.23674672](https://doi.org/10.5281/zenodo.23674672)**  
> Current version v1.1.0: [https://doi.org/10.5281/zenodo.24571683](https://doi.org/10.5281/zenodo.24571683)
>
> To reproduce the analyses, download the Zenodo record and place `data/` and `models/` next to the folders of this repository (see `docs/usage.md` and `docs/post_v1_stages.md`).
> Please cite the Zenodo record (see `CITATION.cff`).

## What is here and what is on Zenodo

| | GitHub (this repository) | Zenodo |
|---|---|---|
| Source code and notebooks (stages 00–32) | ✔ | ✔ |
| Documentation, metadata, data dictionary, environment, test | ✔ | ✔ |
| Dataset and result tables | — | ✔ |
| Trained models | — | ✔ |
| Checksums of the complete package | — | ✔ |

Apart from this README, the files are identical to those archived on Zenodo.

## Package documentation

### 1. Identification

- **Title:** Reproducibility package for multiscale morphology-informed post-detection verification of deep learning pulmonary nodule candidates in computed tomography: code, trained nnU-Net models, cohort splits and analysis data
- **Version:** v1.1.0 (2026-11-26)
- **Resource type:** Dataset and Software (combined reproducibility package)
- **DOI:** v1.1.0: [10.5281/zenodo.24571683](https://doi.org/10.5281/zenodo.24571683); concept DOI (all versions) [10.5281/zenodo.23674672](https://doi.org/10.5281/zenodo.23674672); v1.0.0: [10.5281/zenodo.23674673](https://doi.org/10.5281/zenodo.23674673)
- **Authors:**
  - Wesley Pacheco Calixto; ORCID 0000-0002-1928-4432; EMC/UFG; ISR/University of Coimbra; CPDT/IFG
  - Thiago Fellipe O. Camargo; ORCID 0000-0002-6917-8339; EMC/UFG; CPDT/IFG; Hospital Israelita Albert Einstein
  - Viviane M. Gomes Pacheco; ORCID 0000-0003-3693-4377; EMC/UFG; CPDT/IFG; PUC Goias
  - Marcella Scoczynski R. Martins; ORCID 0000-0002-5716-4968; UTFPR, Ponta Grossa
  - Antonio Paulo Coimbra; ORCID 0000-0003-3780-2405; ISR/University of Coimbra
- **Corresponding author:** Wesley Pacheco Calixto, wesley.pacheco@ufg.br
- **Language:** English (documentation and headers); Portuguese in verbatim docstrings, string literals and logs
- **License:** code under MIT; models, data and documentation under CC BY 4.0 (see `LICENSE`)

### 2. General description

This record contains the complete computational pipeline, the trained models and the derived data of the study *Multiscale morphology-informed post-detection verification of deep learning pulmonary nodule candidates in computed tomography: false-positive reduction, feature repeatability and external validation*.
It allows installation, minimal execution, traceability of inputs and outputs, reproduction of the analyses and citation by DOI.
Source CT images are public and are not redistributed.

### 3. Contents

- **Code** (`scripts/`, `trainers/`, `patches/`, `notebooks/`): stages 00–26 of the pipeline (detection, matching, morphological responses, verifier, RIDER repeatability, LNDb evaluation, tables and figures) and stages 28–32 (`docs/post_v1_stages.md`): LNDb N1 reference and re-evaluation, selective verification, radiological assessment linkage, Model B, Model C and ablation, and supplementary analyses.
- **Trained models** (`models/`): 50 nnU-Net v2.8.1 checkpoints (ten configurations × five folds; `checkpoint_final.pth`, and `checkpoint_latest.pth` for the three folds stopped by the training monitor) with plans and dataset definition. Unchanged since v1.0.0.
- **Cohort and splits** (`data/cohort_and_splits/`, `data/nnunet_dataset/`): case lists, cross-validation splits and nnU-Net plans.
- **Frozen parameters** (`data/run_records_00_10/`): partition, collapse rule, GPU fit and training configuration.
- **Results** (`data/stage_outputs_10a_25/`, `data/stage_outputs_28_32/`): result tables, statistics, figure and table data and the frozen verifier of each stage. All LNDb results of the article use the N1 reference of stage 28; the LNDb metrics of stages 22–25 obtained with the earlier local majority reference are superseded.
- **Metadata** (`metadata/`: dataset metadata, file manifest, data dictionary, quality-control report), documentation (`docs/`, `documentation/`), environment (`environment/`, `requirements.txt`), smoke test (`tests/`) and checksums (`checksums/`).
- Only what is necessary for reproduction is included; logs, superseded runs, manuscript-writing scripts and regenerable intermediate products (masks, filter maps, probability maps) are not.

### 4. Repository structure

```text
pulmonary_nodule_verification_v1.1.0/
├── .zenodo.json
├── CHANGELOG.md
├── CITATION.cff
├── LICENSE
├── README.md
├── checksums/  (1 file)
├── data/  (326 files)
├── docs/  (7 files)
├── documentation/  (4 files)
├── environment/  (3 files)
├── metadata/  (4 files)
├── models/  (54 files; archived as ten models_arq_<configuration>.zip files)
├── notebooks/  (3 files)
├── patches/  (2 files)
├── requirements.txt
├── scripts/  (111 files: stages 00–26 and sub-folders 28–32)
├── tests/  (1 file)
└── trainers/  (4 files)
```

### 5. Installation and minimal execution

See `docs/installation.md` and `docs/usage.md`. Minimal check: `python tests/test_smoke.py` (add `--checksums` to verify the SHA-256 of every file).

### 6. Source datasets

LIDC-IDRI (https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX), RIDER Lung CT (https://doi.org/10.7937/K9/TCIA.2015.U1X8A5NR) and
LNDb (https://doi.org/10.5281/zenodo.7153205). Obtain them from the original repositories and cite them.

### 7. Limitations

See `docs/limitations.md`.

### 8. How to cite

Calixto, W. P., Camargo, T. F. O., Pacheco, V. M. G., Martins, M. S. R., & Coimbra, A. P. (2026). Reproducibility package for multiscale morphology-informed post-detection verification of deep learning pulmonary nodule candidates in computed tomography: code, trained nnU-Net models, cohort splits and analysis data (v1.1.0) [Data set and software]. Zenodo. https://doi.org/10.5281/zenodo.23674672
