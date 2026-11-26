#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    S3_descritivo_confianca.py

File type:
    Script

Purpose:
    Descriptive analysis of the detector confidence measure among accepted and rejected candidates.
    Pipeline position: Stage 29 (selective verification).

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
    - V02_escores_oof_ABC.csv.gz
    - bd1_filter_features.csv

Outputs:
    - S1_confianca_candidatos.csv
    - S3_descritivo_confianca.csv

Dependencies:
    - numpy
    - pandas
    - sklearn

Execution:
    cd <project_dir> && python scripts/29_selective_verification/S3_descritivo_confianca.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
R = "resultados"; P = "."; O = R + "/verificacao_seletiva_20261111"
TH = 0.3417980765530837; Q = pd.read_csv(O + "/saidas/S1_confianca_candidatos.csv")
fe = pd.read_csv(P + "/artigos_relatorios/Artigos_14_10_2026/RESULTADOS_ADICIONAIS_CMIG/AUDITORIA_v4/fontes/bd1_filter_features.csv", low_memory=False)
fe = fe[(fe.subset == "oof") & fe.group.isin(["TP_pred", "FP_pred"]) & fe.architecture.isin(["3D_F", "3D_FREM"])]
sa = pd.read_csv(R + "/analises_complementares_CMIG/outputs/figdata/V02_escores_oof_ABC.csv.gz")
d = fe.merge(sa[["patient_id", "architecture", "component_id", "score_A"]], on=["patient_id", "architecture", "component_id"]).merge(
    Q[Q.conjunto == "D1"].rename(columns={"exame": "patient_id", "configuracao": "architecture", "id": "component_id"}), on=["patient_id", "architecture", "component_id"])
out = []
for a, g in d.groupby("architecture"):
    y = (g.group == "TP_pred").astype(int); rj = g[g.score_A < TH]; yr = (rj.group == "TP_pred").astype(int)
    out.append(dict(configuracao=a, n=len(g), auroc_q=roc_auc_score(y, g.q_media), auroc_A=roc_auc_score(y, g.score_A), spearman_q_A=g[["q_media", "score_A"]].corr("spearman").iloc[0, 1],
                    rejeitados_B=len(rj), TP_rejeitados_B=int(yr.sum()), auroc_q_entre_rejeitados=roc_auc_score(yr, rj.q_media) if 0 < yr.sum() < len(yr) else np.nan,
                    q_mediana_TP=float(g[y == 1].q_media.median()), q_mediana_FP=float(g[y == 0].q_media.median()),
                    frac_q_ge_099_TP=float((g[y == 1].q_media >= 0.99).mean()), frac_q_ge_099_FP=float((g[y == 0].q_media >= 0.99).mean()),
                    rejeitados_com_q_ge_099_TP=int(((rj.q_media >= 0.99) & (yr == 1)).sum()), rejeitados_com_q_ge_099_FP=int(((rj.q_media >= 0.99) & (yr == 0)).sum())))
pd.DataFrame(out).to_csv(O + "/saidas/S3_descritivo_confianca.csv", index=False); print(pd.DataFrame(out).round(4).T.to_string())
