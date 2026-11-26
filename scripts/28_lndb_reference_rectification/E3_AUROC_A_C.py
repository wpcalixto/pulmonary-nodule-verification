#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    E3_AUROC_A_C.py

File type:
    Script

Purpose:
    Paired AUROC and average-precision differences between Models A and C on the LNDb candidates, with scan-level bootstrap.
    Pipeline position: Stage 28 (LNDb N1 reference rectification and re-evaluation).

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
    - Outputs of the preceding pipeline stages (see docs/post_v1_stages.md).

Outputs:
    - E3_AUROC_A_C.csv

Dependencies:
    - numpy
    - pandas

Execution:
    cd <project_dir> && python scripts/28_lndb_reference_rectification/E3_AUROC_A_C.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import sys, numpy as np, pandas as pd
sys.argv = [sys.argv[0]]
exec(open(__file__.replace("E3_AUROC_A_C.py", "E2_estatisticas.py")).read().split("out = {k: [] for k in")[0])
rows = []
for coh, partes in COH.items():
    for a in ["3D_F", "3D_FREM"]:
        ob = OB[OB.coorte.isin(partes) & (OB.referencia == "N1") & (OB.configuracao == a) & (OB.fonte == "pred") & OB.status.isin(["TP", "FP"])].merge(PRED, on=["exame", "configuracao", "id"])
        ex = sorted(C[C.coorte.isin(partes) & (C.referencia == "N1") & (C.configuracao == a)].exame.unique()); g = {e: ob[ob.exame == e] for e in ex}
        y = (ob.status == "TP").astype(int); obs = (roc_auc_score(y, ob.p_tp) - roc_auc_score(y, ob.score_C), average_precision_score(y, ob.p_tp) - average_precision_score(y, ob.score_C))
        rng = np.random.default_rng(SEED); V = []
        for _ in range(NB):
            s = pd.concat([g[ex[i]] for i in rng.integers(0, len(ex), len(ex))]); yy = (s.status == "TP").astype(int)
            V.append((roc_auc_score(yy, s.p_tp) - roc_auc_score(yy, s.score_C), average_precision_score(yy, s.p_tp) - average_precision_score(yy, s.score_C)) if 0 < yy.sum() < len(yy) else (np.nan, np.nan))
        V = np.array(V)
        rows.append(dict(coorte=coh, configuracao=a, dAUROC=obs[0], dAUROC_lo=np.nanquantile(V[:, 0], .025), dAUROC_hi=np.nanquantile(V[:, 0], .975), dPM=obs[1], dPM_lo=np.nanquantile(V[:, 1], .025), dPM_hi=np.nanquantile(V[:, 1], .975)))
pd.DataFrame(rows).to_csv(O + "/saidas/E3_AUROC_A_C.csv", index=False); print(pd.DataFrame(rows).round(4).to_string())
