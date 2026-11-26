#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    20_v02_b_c_oof.py

File type:
    Script

Purpose:
    Size-conditional discriminative increment (Model B versus Model C) on development out-of-fold predictions.
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
    - Outputs of the preceding pipeline stages (see docs/post_v1_stages.md).

Outputs:
    - V02_B_vs_C_oof.csv
    - V02_bootstrap_pareado.csv
    - V02_chaves_particoes_validacao.json
    - V02_controlo_c14.csv
    - V02_escores_oof_ABC.csv.gz

Dependencies:
    - numpy
    - pandas
    - sklearn

Execution:
    cd <project_dir> && python scripts/32_supplementary_analyses/a_verifier_and_lndb_cohort/20_v02_b_c_oof.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import sys, json, warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from comum import *
thr = json.load(open(ART["s17_threshold"]))["model_A"]; FEAT, THETA = thr["features"], thr["threshold"]
cv = pd.read_csv(ART["s17_cv"]); cv = cv[cv.fold != "pooled_oof"].assign(fold=lambda d: d.fold.astype(int))
df = pd.read_csv(ART["features_D1_D2"], low_memory=False)
df = df[(df.subset == "oof") & df.group.isin(["TP_pred", "FP_pred"])].reset_index(drop=True)
key = ["patient_id", "architecture", "component_id"]
dup = int(df.duplicated(key).sum())
assert len(df) == 37987 and df.patient_id.nunique() == 718 and dup == 0
y = (df.group == "TP_pred").astype(int).values; assert y.sum() == 14673 and (1 - y).sum() == 23314
df["size"] = np.log1p(df.diam_mm.astype(float))
MODELS = {"A": FEAT, "B": FEAT + ["size"], "C": ["size"]}
splits = list(GroupKFold(n_splits=5).split(df, y, df.patient_id))
pipe = lambda: Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=27))])
oof, fold, pf = {}, np.full(len(df), -1), []
for n, cols in MODELS.items():
    s = np.full(len(df), np.nan)
    for k, (tr, va) in enumerate(splits):
        s[va] = pipe().fit(df.loc[tr, cols], y[tr]).predict_proba(df.loc[va, cols])[:, 1]; fold[va] = k
        pf.append(dict(model=n, fold=k, auroc=roc_auc_score(y[va], s[va]), pm=average_precision_score(y[va], s[va])))
    oof[n] = s
pf = pd.DataFrame(pf); fail = []; gate = {}
for mdl in ["A", "B"]:
    ref = cv[cv.model == mdl].set_index("fold"); q = pf[pf.model == mdl].set_index("fold")
    gate[mdl] = dict(max_dif_auroc=float((q.auroc - ref.auroc_fold).abs().max()), max_dif_pm=float((q.pm - ref.average_precision_fold).abs().max()))
    if gate[mdl]["max_dif_auroc"] >= 1e-9 or gate[mdl]["max_dif_pm"] >= 1e-9: fail.append(f"dobras do Modelo {mdl}")
fpr, tpr, t = roc_curve(y, oof["A"]); j = tpr - fpr; tA = float(np.min(t[np.isclose(j, np.nanmax(j))]))
gate["theta_reproduzido"], gate["theta_congelado"] = tA, THETA
lo, hi = sorted([tA, THETA]); gate["escores_OOF_entre_theta_reproduzido_e_congelado_excluindo_extremos"] = int(((oof["A"] > lo) & (oof["A"] < hi)).sum())
gate["decisoes_iguais_theta_reproduzido_vs_congelado"] = bool(((oof["A"] >= tA) == (oof["A"] >= THETA)).all())
if abs(tA - THETA) >= 1e-9 or not gate["decisoes_iguais_theta_reproduzido_vs_congelado"]: fail.append("theta")
g = df.patient_id.astype(str).values; u = np.array(sorted(set(g))); ix = {p: np.flatnonzero(g == p) for p in u}
rng = np.random.default_rng(SEED_P); BS = {n: [] for n in MODELS}
for _ in range(NBOOT):
    q = np.concatenate([ix[p] for p in rng.choice(u, size=len(u), replace=True)])
    for n in MODELS: BS[n].append((roc_auc_score(y[q], oof[n][q]), average_precision_score(y[q], oof[n][q])))
BS = {n: np.asarray(v) for n, v in BS.items()}
obs = {n: (roc_auc_score(y, oof[n]), average_precision_score(y, oof[n])) for n in MODELS}
c14 = pd.read_csv(ART["c14_modelos_D1"]).set_index("model"); comp = []
for n in MODELS:
    for col, v in [("auroc_oof", obs[n][0]), ("pm_oof", obs[n][1]), ("auroc_ci_lo", np.quantile(BS[n][:, 0], .025)), ("auroc_ci_hi", np.quantile(BS[n][:, 0], .975)),
                   ("pm_ci_lo", np.quantile(BS[n][:, 1], .025)), ("pm_ci_hi", np.quantile(BS[n][:, 1], .975)),
                   ("delta_auroc_ci_lo", np.quantile(BS[n][:, 0] - BS["A"][:, 0], .025)), ("delta_pm_ci_hi", np.quantile(BS[n][:, 1] - BS["A"][:, 1], .975))]:
        d = abs(v - float(c14.loc[n, col])); comp.append(dict(modelo=n, campo=col, reproduzido=v, c14=float(c14.loc[n, col]), dif_abs=d))
        if d >= 1e-12: fail.append(f"c14 {n} {col}")
pd.DataFrame(comp).to_csv(out("outputs", "audits", "V02_controlo_c14.csv"), index=False)
gate["falhas"] = fail
dump(dict(porta=gate, chaves=dict(linhas=len(df), pacientes=int(df.patient_id.nunique()), TP=int(y.sum()), FP=int((1 - y).sum()), duplicados_chave=dup,
          chave=key, dobras=[dict(dobra=k, pacientes=int(df.loc[va, "patient_id"].nunique()), linhas=len(va)) for k, (tr, va) in enumerate(splits)],
          pacientes_em_mais_de_uma_dobra=int(pd.Series(fold).groupby(df.patient_id).nunique().gt(1).sum())), ambiente=env()),
     "outputs", "audits", "V02_chaves_particoes_validacao.json")
if fail: sys.exit("PORTA FALHOU: " + "; ".join(fail))
pd.DataFrame(dict(patient_id=df.patient_id, architecture=df.architecture, component_id=df.component_id, y=y, dobra=fold, score_A=oof["A"], score_B=oof["B"], score_C=oof["C"])).to_csv(
    out("outputs", "figdata", "V02_escores_oof_ABC.csv.gz"), index=False)
rows = []
for n in MODELS:
    rows.append(dict(contraste=n, auroc=obs[n][0], auroc_ic_lo=np.quantile(BS[n][:, 0], .025), auroc_ic_hi=np.quantile(BS[n][:, 0], .975),
                     pm=obs[n][1], pm_ic_lo=np.quantile(BS[n][:, 1], .025), pm_ic_hi=np.quantile(BS[n][:, 1], .975)))
for a, b in [("B", "C"), ("A", "C"), ("B", "A")]:
    d = BS[a] - BS[b]
    rows.append(dict(contraste=f"{a}-{b}", auroc=obs[a][0] - obs[b][0], auroc_ic_lo=np.quantile(d[:, 0], .025), auroc_ic_hi=np.quantile(d[:, 0], .975),
                     pm=obs[a][1] - obs[b][1], pm_ic_lo=np.quantile(d[:, 1], .025), pm_ic_hi=np.quantile(d[:, 1], .975)))
R = pd.DataFrame(rows).assign(n=len(y), pacientes=len(u), replicas=NBOOT, semente=SEED_P, unidade="paciente")
R.to_csv(out("outputs", "tables", "V02_B_vs_C_oof.csv"), index=False)
pd.DataFrame({"auroc_A": BS["A"][:, 0], "pm_A": BS["A"][:, 1], "auroc_B": BS["B"][:, 0], "pm_B": BS["B"][:, 1], "auroc_C": BS["C"][:, 0], "pm_C": BS["C"][:, 1]}).to_csv(
    out("outputs", "tables", "V02_bootstrap_pareado.csv"), index=False)
print(R.round(4).to_string())
