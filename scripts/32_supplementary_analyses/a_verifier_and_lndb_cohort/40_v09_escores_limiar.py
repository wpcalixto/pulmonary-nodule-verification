#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    40_v09_escores_limiar.py

File type:
    Script

Purpose:
    Stability of the frozen threshold between out-of-fold and final Model A scores.
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
    - V02_escores_oof_ABC.csv.gz
    - V09_D1_oof_vs_final.csv.gz
    - V09_distribuicao_escores.csv
    - V09_transicoes_theta.csv
    - V09_transicoes_theta_ic.csv
    - V09_validacao.json

Dependencies:
    - numpy
    - pandas
    - sklearn

Execution:
    cd <project_dir> && python scripts/32_supplementary_analyses/a_verifier_and_lndb_cohort/40_v09_escores_limiar.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import sys, json, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from comum import *
thr = json.load(open(ART["s17_threshold"]))["model_A"]; TH = thr["threshold"]
M = pickle.load(open(ART["s17_model_A"], "rb")); FEAT = M["features"]
assert FEAT == thr["features"] and M["threshold"] == TH
fail = []
F = pd.read_csv(ART["features_D1_D2"], low_memory=False); F = F[F.group.isin(["TP_pred", "FP_pred"])]
d2f = F[F.subset == "internal"].copy(); d2f["p_rec"] = M["model"].predict_proba(d2f[FEAT])[:, 1]
s17a = pd.read_csv(ART["s17a_decisions_D2"])
j = s17a.merge(d2f[["patient_id", "architecture", "component_id", "p_rec"]], on=["patient_id", "architecture", "component_id"], how="left", validate="1:1")
val = dict(D2_linhas=len(s17a), D2_sem_correspondencia=int(j.p_rec.isna().sum()), D2_max_dif_prob=float((j.p_rec - j.verifier_probability).abs().max()),
           D2_decisoes_iguais=bool(((j.p_rec >= TH) == (j.decision == "ACCEPT")).all()))
if val["D2_sem_correspondencia"] or val["D2_max_dif_prob"] >= 1e-12 or not val["D2_decisoes_iguais"]: fail.append("modelo congelado não reproduz D2")
oof = pd.read_csv(out("outputs", "figdata", "V02_escores_oof_ABC.csv.gz"))
d1f = F[F.subset == "oof"][["patient_id", "architecture", "component_id"] + FEAT]
d1 = oof.merge(d1f, on=["patient_id", "architecture", "component_id"], how="left", validate="1:1")
if d1[FEAT].isna().any().any() or len(d1) != 37987: fail.append("D1: junção OOF x respostas")
d1["score_final"] = M["model"].predict_proba(d1[FEAT])[:, 1]
d4 = pd.read_csv(ART["s23_scores_D4"]); val["D4_decisoes_coerentes_com_theta"] = bool(((d4.p_tp >= TH) == d4.retained).all())
if not val["D4_decisoes_coerentes_com_theta"]: fail.append("D4: retained != p>=theta")
val["theta"] = TH; val["ambiente"] = env()
if fail: dump(dict(validacao=val, falhas=fail), "outputs", "audits", "V09_validacao.json"); sys.exit("FALHA: " + "; ".join(fail))
def resumo(s, y, reg, ds, arch):
    s, y = np.asarray(s, float), np.asarray(y, int); r = dict(regime=reg, conjunto=ds, configuracao=arch, n=len(s), n_TP=int(y.sum()), n_FP=int((1 - y).sum()))
    for lab, v in [("todos", s), ("TP", s[y == 1]), ("FP", s[y == 0])]:
        q = np.quantile(v, [.25, .5, .75]) if len(v) else [np.nan] * 3
        r.update({f"{lab}_media": v.mean() if len(v) else np.nan, f"{lab}_dp": v.std(ddof=1) if len(v) > 1 else np.nan, f"{lab}_q1": q[0], f"{lab}_mediana": q[1], f"{lab}_q3": q[2],
                  f"{lab}_frac_acima_theta": (v >= TH).mean() if len(v) else np.nan})
    r["TP_aceitos"], r["FP_aceitos"] = int(((s >= TH) & (y == 1)).sum()), int(((s >= TH) & (y == 0)).sum())
    r["auroc"] = roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan
    r["brier_exploratorio"] = brier_score_loss(y, s) if len(y) else np.nan
    r["media_escore_menos_fracao_TP_exploratorio"] = s.mean() - y.mean() if len(y) else np.nan
    return r
R = []
for reg, col in [("D1 fora da dobra", "score_A"), ("D1 ajuste final (dentro da amostra)", "score_final")]:
    R.append(resumo(d1[col], d1.y, reg, "D1", "todas"))
    for a, q in d1.groupby("architecture"): R.append(resumo(q[col], q.y, reg, "D1", a))
s17a["y"] = (s17a.group == "TP_pred").astype(int)
R.append(resumo(s17a.verifier_probability, s17a.y, "D2 modelo final congelado", "D2", "todas"))
for a, q in s17a.groupby("architecture"): R.append(resumo(q.verifier_probability, q.y, "D2 modelo final congelado", "D2", a))
d4["y"] = (d4.group == "TP_pred").astype(int)
for a, q in d4.groupby("architecture"): R.append(resumo(q.p_tp, q.y, "D4 modelo final congelado", "D4", a))
pd.DataFrame(R).to_csv(out("outputs", "tables", "V09_distribuicao_escores.csv"), index=False)
d1["oof_ac"], d1["fin_ac"] = d1.score_A >= TH, d1.score_final >= TH
d1["delta"] = d1.score_final - d1.score_A
T = []
for (lab, q) in [("todos", d1), ("TP", d1[d1.y == 1]), ("FP", d1[d1.y == 0])] + [(f"config {a}", q) for a, q in d1.groupby("architecture")]:
    T.append(dict(estrato=lab, n=len(q), aceite_ambos=int((q.oof_ac & q.fin_ac).sum()), rejeitado_ambos=int((~q.oof_ac & ~q.fin_ac).sum()),
                  rejeitado_OOF_aceite_final=int((~q.oof_ac & q.fin_ac).sum()), aceite_OOF_rejeitado_final=int((q.oof_ac & ~q.fin_ac).sum()),
                  frac_cruza=float((q.oof_ac != q.fin_ac).mean()), delta_media=q.delta.mean(), delta_mediana=q.delta.median(),
                  delta_abs_q95=q.delta.abs().quantile(.95), delta_abs_max=q.delta.abs().max(), spearman=q[["score_A", "score_final"]].corr("spearman").iloc[0, 1],
                  pearson=q[["score_A", "score_final"]].corr().iloc[0, 1]))
pp = d1.groupby("patient_id").agg(n=("y", "size"), delta_media=("delta", "mean"), cruza=("oof_ac", lambda s: 0))
pp["cruza"] = d1.assign(c=d1.oof_ac != d1.fin_ac).groupby("patient_id").c.sum()
T.append(dict(estrato="por paciente (718)", n=len(pp), frac_cruza=float((pp.cruza > 0).mean()), delta_media=pp.delta_media.mean(), delta_mediana=pp.delta_media.median(),
              delta_abs_q95=pp.delta_media.abs().quantile(.95), delta_abs_max=pp.delta_media.abs().max()))
T = pd.DataFrame(T)
g = d1.patient_id.astype(str).values; u = np.array(sorted(set(g))); ix = {p: np.flatnonzero(g == p) for p in u}
rng = np.random.default_rng(SEED_P); B = []
yv, so, sf = d1.y.values, d1.score_A.values, d1.score_final.values
for _ in range(NBOOT):
    q = np.concatenate([ix[p] for p in rng.choice(u, size=len(u), replace=True)]); y_, a_, f_ = yv[q], so[q] >= TH, sf[q] >= TH
    B.append(dict(frac_cruza=(a_ != f_).mean(), delta_media=(sf[q] - so[q]).mean(),
                  d_sens=(f_ & (y_ == 1)).sum() / (y_ == 1).sum() - (a_ & (y_ == 1)).sum() / (y_ == 1).sum(),
                  d_frac_FP_aceite=(f_ & (y_ == 0)).sum() / (y_ == 0).sum() - (a_ & (y_ == 0)).sum() / (y_ == 0).sum()))
B = pd.DataFrame(B)
obs = dict(frac_cruza=float((d1.oof_ac != d1.fin_ac).mean()), delta_media=float(d1.delta.mean()),
           d_sens=float((d1.fin_ac & (d1.y == 1)).sum() / (d1.y == 1).sum() - (d1.oof_ac & (d1.y == 1)).sum() / (d1.y == 1).sum()),
           d_frac_FP_aceite=float((d1.fin_ac & (d1.y == 0)).sum() / (d1.y == 0).sum() - (d1.oof_ac & (d1.y == 0)).sum() / (d1.y == 0).sum()))
IC = pd.DataFrame([dict(medida=k, observado=v, ic_lo=B[k].quantile(.025), ic_hi=B[k].quantile(.975), replicas=NBOOT, semente=SEED_P, unidade="paciente") for k, v in obs.items()])
T.to_csv(out("outputs", "tables", "V09_transicoes_theta.csv"), index=False)
IC.to_csv(out("outputs", "tables", "V09_transicoes_theta_ic.csv"), index=False)
d1[["patient_id", "architecture", "component_id", "y", "score_A", "score_final"]].rename(columns={"score_A": "score_oof"}).to_csv(out("outputs", "figdata", "V09_D1_oof_vs_final.csv.gz"), index=False)
dump(dict(validacao=val, falhas=[]), "outputs", "audits", "V09_validacao.json")
print(json.dumps(val, default=str)); print(T.head(3).round(5).to_string()); print(IC.round(5).to_string())
print(pd.DataFrame(R)[["regime", "conjunto", "configuracao", "n", "todos_mediana", "TP_mediana", "FP_mediana", "todos_frac_acima_theta", "auroc", "brier_exploratorio"]].round(4).to_string())
