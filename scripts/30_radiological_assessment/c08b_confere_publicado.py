#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    c08b_confere_publicado.py

File type:
    Script

Purpose:
    Checks that the reproduced out-of-fold scores match the frozen stage-17 records.
    Pipeline position: Stage 30 (radiological assessment linkage).

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
    - bd1_filter_features.csv
    - filter_verifier_cv.csv
    - filter_verifier_threshold.json

Outputs:
    - c08b_conferencia_publicado.json

Dependencies:
    - numpy
    - pandas
    - sklearn

Execution:
    cd <project_dir> && python scripts/30_radiological_assessment/c08b_confere_publicado.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import json, hashlib
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
P = "."
S17 = P + "/resultados/pulmonar_pos_oof_prep/script17_final/s17_20260923T160316_563312"
F = P + "/artigos_relatorios/Artigos_14_10_2026/RESULTADOS_ADICIONAIS_CMIG/AUDITORIA_v4/fontes/bd1_filter_features.csv"
O = P + "/resultados/radiological_reanalysis"
assert hashlib.sha256(open(F, "rb").read()).hexdigest().startswith("87fd83ce")
thr = json.load(open(S17 + "/config/filter_verifier_threshold.json"))["model_A"]; FEAT, th = thr["features"], thr["threshold"]
cv = pd.read_csv(S17 + "/analysis/filter_verifier_cv.csv"); cv = cv[cv.model == "A"].set_index("fold")
PUB = {"0": (143, 2752, 4845, .828, .716), "1": (144, 2778, 4819, .848, .747), "2": (144, 3023, 4577, .860, .796),
       "3": (143, 2961, 4634, .841, .771), "4": (144, 3159, 4439, .857, .800)}
df = pd.read_csv(F, low_memory=False); df = df[(df.subset == "oof") & df.group.isin(["TP_pred", "FP_pred"])].reset_index(drop=True)
y = (df.group == "TP_pred").astype(int).values; oof = np.full(len(df), np.nan); res = {}
for k, (tr, va) in enumerate(GroupKFold(n_splits=5).split(df, y, df.patient_id)):
    m = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=27))]).fit(df.loc[tr, FEAT], y[tr])
    oof[va] = m.predict_proba(df.loc[va, FEAT])[:, 1]
    calc = (df.loc[va, "patient_id"].nunique(), int(y[va].sum()), int((1 - y[va]).sum()), round(roc_auc_score(y[va], oof[va]), 3), round(average_precision_score(y[va], oof[va]), 3))
    res[str(k)] = dict(calculado=calc, publicado_S8=PUB[str(k)], igual=calc == PUB[str(k)])
a = oof >= th
cont = dict(tp=int((a & (y == 1)).sum()), fp=int((a & (y == 0)).sum()), fn=int((~a & (y == 1)).sum()), tn=int((~a & (y == 0)).sum()))
reg = {k: int(cv.loc["pooled_oof", k]) for k in cont}
ag = dict(auroc=round(roc_auc_score(y, oof), 3), pm=round(average_precision_score(y, oof), 3))
out = dict(dobras=res, agregado_calc=ag, agregado_publicado=dict(auroc=0.846, pm=0.764), contagens_calc=cont, contagens_registadas=reg,
           sens=round(cont["tp"] / (cont["tp"] + cont["fn"]), 3), prec=round(cont["tp"] / (cont["tp"] + cont["fp"]), 3), fp_exame=round(cont["fp"] / 718, 2), theta=th)
out["aprovado"] = bool(all(r["igual"] for r in res.values()) and ag == out["agregado_publicado"] and cont == reg)
json.dump(out, open(f"{O}/c08b_conferencia_publicado.json", "w"), indent=1, default=str)
print(json.dumps({k: out[k] for k in ("agregado_calc", "contagens_calc", "contagens_registadas", "sens", "prec", "fp_exame", "aprovado")}))
print({k: r["igual"] for k, r in res.items()})
