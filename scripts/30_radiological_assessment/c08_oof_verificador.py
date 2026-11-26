#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    c08_oof_verificador.py

File type:
    Script

Purpose:
    Reproduces the out-of-fold Model A score of each development object from the frozen stage-17 sources.
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
    - c08_oof_objetos_radiologist.csv
    - c08_validacao.json

Dependencies:
    - numpy
    - pandas
    - sklearn

Execution:
    cd <project_dir> && python scripts/30_radiological_assessment/c08_oof_verificador.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import json, hashlib, pickle, sys
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
P = "."
S17 = P + "/resultados/pulmonar_pos_oof_prep/script17_final/s17_20260923T160316_563312"
FEATCSV = P + "/artigos_relatorios/Artigos_14_10_2026/RESULTADOS_ADICIONAIS_CMIG/AUDITORIA_v4/fontes/bd1_filter_features.csv"
O = P + "/resultados/radiological_reanalysis"
assert hashlib.sha256(open(FEATCSV, "rb").read()).hexdigest().startswith("87fd83ce")
thr = json.load(open(S17 + "/config/filter_verifier_threshold.json"))["model_A"]
FEAT, theta = thr["features"], thr["threshold"]
cvref = pd.read_csv(S17 + "/analysis/filter_verifier_cv.csv"); cvref = cvref[(cvref.model == "A") & (cvref.fold != "pooled_oof")].assign(fold=lambda d: d.fold.astype(int)).set_index("fold")
df = pd.read_csv(FEATCSV, low_memory=False)
df = df[(df.subset == "oof") & df.group.isin(["TP_pred", "FP_pred"])].reset_index(drop=True)
assert len(df) == 37987 and df.patient_id.nunique() == 718
y = (df.group == "TP_pred").astype(int).values
oof = np.full(len(df), np.nan); folds = np.full(len(df), -1); res = {}
for k, (tr, va) in enumerate(GroupKFold(n_splits=5).split(df, y, df.patient_id)):
    m = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=27))])
    m.fit(df.loc[tr, FEAT], y[tr]); oof[va] = m.predict_proba(df.loc[va, FEAT])[:, 1]; folds[va] = k
    res[k] = dict(auroc=roc_auc_score(y[va], oof[va]), pm=average_precision_score(y[va], oof[va]),
                  dif_auroc=abs(roc_auc_score(y[va], oof[va]) - cvref.loc[k, "auroc_fold"]),
                  dif_pm=abs(average_precision_score(y[va], oof[va]) - cvref.loc[k, "average_precision_fold"]))
fpr, tpr, t = roc_curve(y, oof); J = tpr - fpr; th = t[J == J.max()].min()
val = dict(max_dif_auroc_dobra=max(r["dif_auroc"] for r in res.values()), max_dif_pm_dobra=max(r["dif_pm"] for r in res.values()),
           auroc_oof=roc_auc_score(y, oof), theta_reproduzido=float(th), theta_congelado=theta, dif_theta=abs(th - theta))
ok = val["max_dif_auroc_dobra"] < 1e-9 and val["max_dif_pm_dobra"] < 1e-9 and val["dif_theta"] < 1e-9
val["aprovado"] = bool(ok); print(val)
json.dump(val, open(f"{O}/c08_validacao.json", "w"), indent=1)
if not ok: sys.exit("FALHA: OOF nao reproduz o script 17; nada gravado")
df["p_oof"], df["dobra"] = oof, folds
ex = [f"LIDC-IDRI-{e}" for e in ("0061", "0080", "0235", "0620", "0879", "0922", "0925")]
s = df[df.patient_id.isin(ex) & df.architecture.isin(["3D_F", "3D_FREM"])][["patient_id", "architecture", "group", "component_id", "dobra", "p_oof"]].copy()
s["aceite_oof"] = s.p_oof >= theta
s.to_csv(f"{O}/c08_oof_objetos_radiologist.csv", index=False)
print(len(s), "objetos D1 gravados")
