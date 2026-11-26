#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    comum.py

File type:
    Module

Purpose:
    Shared paths and helper functions of the supplementary analyses V01, V02, V03 and V09.
    Pipeline position: Stage 32 (supplementary analyses).

Repository:
    pulmonary_nodule_verification

Version:
    v1.1.0

Date:
    2026-11-26

Developer:
    Prof. Wesley Pacheco Calixto

Authors:
    Wesley Pacheco Calixto, 0000-0002-1928-4432, ORCID EMC/UFG; ISR/University of Coimbra; CPDT/IFG
    Thiago Fellipe O. Camargo, 0000-0002-6917-8339, ORCID EMC/UFG; CPDT/IFG; Hospital Israelita Albert Einstein
    Viviane M. Gomes Pacheco, 0000-0003-3693-4377, ORCID EMC/UFG; CPDT/IFG; PUC Goias
    Marcella Scoczynski R. Martins, 0000-0002-5716-4968, ORCID UTFPR, Ponta Grossa
    Antonio Paulo Coimbra, 0000-0003-3780-2405, ORCID ISR/University of Coimbra

Corresponding author:
    Wesley Pacheco Calixto, wesley.pacheco@ufg.br

License:
    MIT

Citation:
    If you use this code, please cite:
    Calixto, W. P., Camargo, T. F. O., Pacheco, V. M. G., Martins, M. S. R., & Coimbra, A. P. (2026). Reproducibility package for multiscale morphology-informed post-detection verification of deep learning pulmonary nodule candidates in computed tomography: code, trained nnU-Net models, cohort splits and analysis data (v1.1.0) [Data set and software]. Zenodo. https://doi.org/10.5281/zenodo.23674672

Related publication:
    Multiscale morphology-informed post-detection verification of deep learning pulmonary nodule candidates in computed tomography: false-positive reduction, feature repeatability and external validation (manuscript; DOI not yet available)

Inputs:
    - LNDB_INPUT_MANIFEST.csv
    - LNDB_RECONCILIACAO.csv
    - bd1_filter_features.csv
    - bd1_folds.json
    - bd1_object_matches.csv
    - bd1_split.csv
    - c14_modelos_D1.csv
    - filter_verifier.pkl
    - filter_verifier_B_plus_size.pkl
    - filter_verifier_cv.csv

Outputs:
    - Tables written to the stage output directory (see docs/post_v1_stages.md).

Dependencies:
    - numpy
    - pandas
    - scipy
    - sklearn

Execution:
    cd <project_dir> && python scripts/32_supplementary_analyses/a_verifier_and_lndb_cohort/comum.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import hashlib, json, os, platform
P = "."
R = P + "/resultados"
PP = R + "/pulmonar_pos_oof_prep"
OUT = R + "/analises_complementares_CMIG"
ART = {
 "s11_matches_D1_D2": PP + "/script11_final/s11B_20260921T164539_27883/analysis/bd1_object_matches.csv",
 "s17_threshold": PP + "/script17_final/s17_20260923T160316_563312/config/filter_verifier_threshold.json",
 "s17_cv": PP + "/script17_final/s17_20260923T160316_563312/analysis/filter_verifier_cv.csv",
 "s17_model_A": PP + "/script17_final/s17_20260923T160316_563312/models/filter_verifier.pkl",
 "s17_model_B": PP + "/script17_final/s17_20260923T160316_563312/models/filter_verifier_B_plus_size.pkl",
 "s17a_decisions_D2": PP + "/script17a_final/s17a_20260924T225525_1009246/internal_verifier_object_decisions.csv",
 "s17a_metrics_D2": PP + "/script17a_final/s17a_20260924T225525_1009246/internal_verifier_metrics.csv",
 "s22_matches_D4": PP + "/script22_final/s22_20260924T084718_826234/lndb_object_matches.csv",
 "s22_metrics_D4": PP + "/script22_final/s22_20260924T084718_826234/lndb_external_metrics.csv",
 "s22_input_manifest_D4": PP + "/script22_final/s22_20260924T084718_826234/input_manifest.csv",
 "s22_scan_audit_D4": PP + "/script22_final/s22_20260924T084718_826234/lndb_scan_audit.csv",
 "s23_scores_D4": PP + "/script23_final/s23_20260924T153750_909396/lndb_verifier_object_scores.csv",
 "s23_metrics_D4": PP + "/script23_final/s23_20260924T153750_909396/lndb_filter_verifier_metrics.csv",
 "features_D1_D2": P + "/artigos_relatorios/Artigos_14_10_2026/RESULTADOS_ADICIONAIS_CMIG/AUDITORIA_v4/fontes/bd1_filter_features.csv",
 "c14_modelos_D1": R + "/modelo_c_ablacoes_dobras/c14_modelos_D1.csv",
 "c14_script": P + "/codigo/radiological_reanalysis/c14_modeloC_ablacoes.py",
 "folds_nnunet": R + "/pulmonar_canonical_pre_oof/splits/bd1_folds.json",
 "split_bd1": R + "/pulmonar_canonical_pre_oof/splits/bd1_split.csv",
 "lndb_reconciliacao": PP + "/LNDB_RECONCILIACAO.csv",
 "lndb_input_manifest": PP + "/LNDB_INPUT_MANIFEST.csv",
 "lndb_annotations_manifest": R + "/pulmonar_canonical_pre_oof/cohort/lndb_annotations_manifest.json",
 "lndb_alternate_label_set": R + "/pulmonar_canonical_pre_oof/cohort/lndb_alternate_label_set.json",
 "lndb_trainNodules": P + "/dados/brutos/raw_5070_2026-08/lndb/trainNodules.csv",
 "lndb_trainNodules_gt": P + "/dados/brutos/raw_5070_2026-08/lndb/trainNodules_gt.csv",
}
LNDB_RAW = P + "/dados/brutos/raw_5070_2026-08/lndb/data"
SEED_P, SEED_E, NBOOT = 27, 270014, 2000
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def env():
    import numpy, pandas, sklearn, scipy
    return dict(python=platform.python_version(), numpy=numpy.__version__, pandas=pandas.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__, host=platform.node())
def out(*a):
    p = os.path.join(OUT, *a); os.makedirs(os.path.dirname(p), exist_ok=True); return p
def dump(obj, *a):
    json.dump(obj, open(out(*a), "w"), indent=1, ensure_ascii=False, default=float)
