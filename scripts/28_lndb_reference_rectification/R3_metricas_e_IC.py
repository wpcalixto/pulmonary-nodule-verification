#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    R3_metricas_e_IC.py

File type:
    Script

Purpose:
    Computes sensitivity, precision, F1 and FP per scan before and after verification for each reference, configuration and matching threshold, with scan-level bootstrap intervals.
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
    - V01_iou_pre_pos_por_configuracao.csv

Outputs:
    - R2_contagens_por_exame.csv
    - R3_controlo_historico.json
    - R3_ic_bootstrap.csv
    - R3_metricas.csv

Dependencies:
    - numpy
    - pandas

Execution:
    cd <project_dir> && python scripts/28_lndb_reference_rectification/R3_metricas_e_IC.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import csv, json, sys
import numpy as np, pandas as pd
O = "resultados/retificacao_LNDb_20261111"
V01 = "resultados/analises_complementares_CMIG/outputs/tables/V01_iou_pre_pos_por_configuracao.csv"
c = pd.read_csv(O + "/saidas/R2_contagens_por_exame.csv")
def met(tp, fp, fn, nex):
    S = tp / (tp + fn) if tp + fn else np.nan; Pr = tp / (tp + fp) if tp + fp else np.nan
    F1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    return S, Pr, F1, fp / nex
rows = []
for (ref, a, t), q in c.groupby(["referencia", "configuracao", "tau"]):
    n = len(q); s = q[["TP", "FP", "FN", "TP_pos", "FP_pos", "FN_pos", "n_ref", "n_pred", "ignoradas", "ignoradas_aceites", "aceites"]].sum()
    pre = met(s.TP, s.FP, s.FN, n); pos = met(s.TP_pos, s.FP_pos, s.FN_pos, n)
    rows.append(dict(referencia=ref, configuracao=a, tau=t, exames=n, n_ref=int(s.n_ref), n_pred=int(s.n_pred), ignoradas=int(s.ignoradas), TP=int(s.TP), FP=int(s.FP), FN=int(s.FN),
                     TP_pos=int(s.TP_pos), FP_pos=int(s.FP_pos), FN_pos=int(s.FN_pos), S=pre[0], P=pre[1], F1=pre[2], FP_exame=pre[3], S_pos=pos[0], P_pos=pos[1], F1_pos=pos[2], FP_exame_pos=pos[3],
                     retencao_TP=s.TP_pos / s.TP if s.TP else np.nan, reducao_FP=1 - s.FP_pos / s.FP if s.FP else np.nan))
M = pd.DataFrame(rows); M.to_csv(O + "/saidas/R3_metricas.csv", index=False)
v = pd.read_csv(V01); v = v[v.dataset == "D4"]; falhas = []
for _, r in v.iterrows():
    m = M[(M.referencia == "antiga") & (M.configuracao == r.arch) & np.isclose(M.tau, r.tau)].iloc[0]
    if (m.TP, m.FP, m.FN, m.TP_pos, m.FP_pos, m.FN_pos) != (r.TP_pre, r.FP_pre, r.FN_pre, r.TP_pos, r.FP_pos, r.FN_pos): falhas.append(f"{r.arch} {r.tau}")
json.dump(dict(controlo_V01=falhas, aprovado=not falhas), open(O + "/saidas/R3_controlo_historico.json", "w"), indent=1)
if falhas: sys.exit("CONTROLO FALHOU " + str(falhas))
B = []
for (ref, a), q in c[c.tau.isin([0.10, 0.25, 0.50])].groupby(["referencia", "configuracao"]):
    ex = sorted(q.exame.unique()); n = len(ex); rng = np.random.default_rng(270014); idx = [rng.integers(0, n, n) for _ in range(2000)]
    for t in [0.10, 0.25, 0.50]:
        qt = q[np.isclose(q.tau, t)].set_index("exame").loc[ex]; A = qt[["TP", "FP", "FN", "TP_pos", "FP_pos", "FN_pos"]].to_numpy()
        vals = []
        for ii in idx:
            s = A[ii].sum(0); pre = met(*s[:3], n); pos = met(*s[3:], n); vals.append(list(pre) + list(pos) + [p2 - p1 for p1, p2 in zip(pre, pos)] + [s[3] / s[0] if s[0] else np.nan])
        V = np.array(vals, float); nomes = ["S", "P", "F1", "FP_exame", "S_pos", "P_pos", "F1_pos", "FP_exame_pos", "dS", "dP", "dF1", "dFP_exame", "retencao_TP"]
        m = M[(M.referencia == ref) & (M.configuracao == a) & np.isclose(M.tau, t)].iloc[0]
        obs = [m.S, m.P, m.F1, m.FP_exame, m.S_pos, m.P_pos, m.F1_pos, m.FP_exame_pos, m.S_pos - m.S, m.P_pos - m.P, m.F1_pos - m.F1, m.FP_exame_pos - m.FP_exame, m.retencao_TP]
        for k, nm in enumerate(nomes):
            x = V[:, k]; f = np.isfinite(x)
            B.append(dict(referencia=ref, configuracao=a, tau=t, medida=nm, observado=obs[k], ic_lo=np.quantile(x[f], .025) if f.any() else np.nan,
                          ic_hi=np.quantile(x[f], .975) if f.any() else np.nan, replicas_validas=int(f.sum()), exames=n))
pd.DataFrame(B).to_csv(O + "/saidas/R3_ic_bootstrap.csv", index=False)
p = M[np.isclose(M.tau, 0.10)][["referencia", "configuracao", "exames", "n_ref", "ignoradas", "TP", "FP", "FN", "S", "P", "F1", "FP_exame", "TP_pos", "FP_pos", "S_pos", "P_pos", "F1_pos", "FP_exame_pos", "retencao_TP"]]
print(p.round(4).to_string())
