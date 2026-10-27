# Multiscale morphology-informed post-detection verification of pulmonary nodule candidates in CT: code, trained nnU-Net models, cohort splits and analysis data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23674673.svg)](https://doi.org/10.5281/zenodo.23674673)

> **The dataset is on Zenodo.** This GitHub repository contains **only the source code and documentation**.
> The **dataset** (cohort definitions, case lists and cross-validation splits, nnU-Net dataset definition and plans, frozen parameters, and the final result tables, statistics and figure data of every stage) and the **trained nnU-Net models** (50 checkpoints, ~27 GB) are archived on Zenodo:
>
> **Zenodo record: [https://doi.org/10.5281/zenodo.23674673](https://doi.org/10.5281/zenodo.23674673)** (DOI 10.5281/zenodo.23674673; all versions: 10.5281/zenodo.23674672)
>
> To reproduce the analyses, download the Zenodo record and place `data/` and `models/` next to the folders of this repository (see `docs/usage.md`).
> Please cite the Zenodo record (see `CITATION.cff`).

## What is here and what is on Zenodo

| | GitHub (this repository) | Zenodo (10.5281/zenodo.23674673) |
|---|---|---|
| Source code and notebooks | ✔ | ✔ |
| Documentation, metadata, data dictionary, environment, test | ✔ | ✔ |
| Dataset | — | ✔ |
| Trained models | — | ✔ |
| Results / figure and table data | — | ✔ |

This repository contains the complete computational pipeline (`scripts/`, stages 00–27 in the executed versions), the custom nnU-Net trainers (`trainers/`), the nnU-Net patch (`patches/`), the Colab notebooks (`notebooks/`), documentation (`docs/`, `documentation/`), metadata including the data dictionary (`metadata/`), the environment (`environment/`, `requirements.txt`) and a smoke test (`tests/`). Apart from this README, the files are identical to those archived on Zenodo; the integrity checksums of the complete package are in the Zenodo record.

## Package documentation (as archived on Zenodo)

### 1. Identification

- **Title:** Multiscale morphology-informed post-detection verification of pulmonary nodule candidates in CT: code, trained nnU-Net models, cohort splits and analysis data
- **Version:** v1.0.0
- **Resource type:** Dataset and Software (combined reproducibility package)
- **DOI:** [10.5281/zenodo.23674673](https://doi.org/10.5281/zenodo.23674673)
- **Authors:**
  - Wesley Pacheco Calixto; ORCID 0000-0002-1928-4432; EMC/UFG; ISR/University of Coimbra; CPDT/IFG
  - Thiago Fellipe O. Camargo; ORCID 0000-0002-6917-8339; EMC/UFG; CPDT/IFG; Hospital Israelita Albert Einstein
  - Viviane M. Gomes Pacheco; ORCID 0000-0003-3693-4377; EMC/UFG; CPDT/IFG; PUC Goias
  - Marcella Scoczynski R. Martins; ORCID 0000-0002-5716-4968; UTFPR, Ponta Grossa
  - Antonio Paulo Coimbra; ORCID 0000-0003-3780-2405; ISR/University of Coimbra
- **Corresponding author:** Wesley Pacheco Calixto, wesley.pacheco@ufg.br
- **Publication date:** to be set at publication
- **Language:** English (code comments and documentation); Portuguese in verbatim docstrings and logs
- **License:** code under MIT; models, data and documentation under CC BY 4.0 (see `LICENSE`)

### 2. General description

This record contains the complete computational pipeline, the trained models and the derived data of the study
*Multiscale morphology-informed post-detection verification of pulmonary nodule candidates in computed tomography*. It was prepared to allow installation, minimal execution, traceability of inputs and outputs, reproduction of the
analyses and citation by DOI. Source CT images are public and are not redistributed.

### 3. Contents

- Code (`scripts/`, `trainers/`, `patches/`, `notebooks/`): stages 00–27 of the pipeline, in the exact versions executed for the final results.
- Trained models (`models/`): 50 nnU-Net checkpoints (ten configurations × five folds; `checkpoint_final.pth`, the checkpoint used for inference, and `checkpoint_latest.pth` for three folds aborted by the collapse monitor) with plans and dataset definition.
- Cohort and splits (`data/cohort_and_splits/`, `data/nnunet_dataset/`): case lists, cross-validation splits and nnU-Net plans.
- Results (`data/stage_outputs_10a_25/`): final result tables, statistics, figure and table data, and the frozen verifier of the final run of each stage.
- Frozen parameters (`data/run_records_00_10/`): BD1 partition, collapse rule, GPU fit and training configuration consumed by stages 07–09.
- Only what is necessary for reproduction is included; logs, audit records, superseded runs and regenerable intermediate products are not.
- Metadata (`metadata/`), documentation (`docs/`, `documentation/`), environment (`environment/`, `requirements.txt`), test (`tests/`) and checksums (`checksums/`).

### 4. Repository structure

```text
pulmonary_nodule_verification_v1.0.0/
├── .zenodo.json
├── CHANGELOG.md
├── CITATION.cff
├── LICENSE
├── README.md
├── checksums/  (1 files)
├── data/  (135 files)
├── docs/  (6 files)
├── documentation/  (4 files)
├── environment/  (3 files)
├── metadata/  (4 files)
├── models/  (54 files)
├── notebooks/  (3 files)
├── patches/  (2 files)
├── requirements.txt
├── scripts/  (41 files)
├── tests/  (1 files)
├── trainers/  (4 files)
```

### 5. Installation and minimal execution

See `docs/installation.md` and `docs/usage.md`. Minimal check: `python tests/test_smoke.py`.

### 6. Source datasets

LIDC-IDRI (https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX), RIDER Lung CT (https://doi.org/10.7937/K9/TCIA.2015.U1X8A5NR) and
LNDb (https://doi.org/10.5281/zenodo.7153205). Obtain them from the original repositories and cite them.

### 7. Limitations

See `docs/limitations.md`.

### 8. How to cite

Calixto, W. P., Camargo, T. F. O., Pacheco, V. M. G., Martins, M. S. R., & Coimbra, A. P. (2026). Multiscale morphology-informed post-detection verification of pulmonary nodule candidates in CT: code, trained nnU-Net models, cohort splits and analysis data (v1.0.0) [Data set and software]. Zenodo. https://doi.org/10.5281/zenodo.23674673
