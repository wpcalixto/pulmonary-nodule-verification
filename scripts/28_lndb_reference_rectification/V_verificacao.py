#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    V_verificacao.py

File type:
    Script

Purpose:
    Arithmetic verification of all LNDb counts, aggregates, metrics and intervals.
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
    - V_verificacao.json

Outputs:
    - E1_contagens_por_exame.csv
    - E1_objetos_tau010.csv
    - E2_M.csv
    - E2_M_ic.csv

Dependencies:
    - numpy
    - pandas

Execution:
    cd <project_dir> && python scripts/28_lndb_reference_rectification/V_verificacao.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import pandas as pd, numpy as np, json, sys
O = sys.argv[1]
C = pd.read_csv(O + "/saidas/E1_contagens_por_exame.csv", dtype={"coorte": str}); M = pd.read_csv(O + "/saidas/E2_M.csv", dtype={"coorte": str}); B = pd.read_csv(O + "/saidas/E2_M_ic.csv", dtype={"coorte": str})
OB = pd.read_csv(O + "/saidas/E1_objetos_tau010.csv", dtype={"coorte": str}, low_memory=False)
f = {}
f["TP+FN=Nref"] = int((C.TP + C.FN != C.n_ref).sum()); f["TP+FP+ign=Npred"] = int((C.TP + C.FP + C.ignoradas != C.n_pred).sum())
f["TP'<=TP"] = int((C.TP_pos > C.TP).sum()); f["FP'<=FP"] = int((C.FP_pos > C.FP).sum()); f["TP'+FP'+ign_ac=aceites"] = int((C.TP_pos + C.FP_pos + C.ignoradas_aceites != C.aceites).sum())
coh = {"211": ["211"], "224": ["211", "13"]}; agr = 0; mf = 0
for _, m in M.iterrows():
    c = C[C.coorte.isin(coh[m.coorte]) & (C.referencia == m.referencia) & (C.configuracao == m.configuracao) & np.isclose(C.tau, m.tau)]
    s = c[["TP", "FP", "FN", "TP_pos", "FP_pos"]].sum()
    agr += int((s.TP, s.FP, s.FN, s.TP_pos, s.FP_pos) != (m.TP, m.FP, m.FN, m.TP_pos, m.FP_pos)) + int(len(c) != m.exames)
    if m.TP + m.FN: mf += int(abs(m.S - m.TP / (m.TP + m.FN)) > 1e-12)
    if m.TP + m.FP: mf += int(abs(m.P - m.TP / (m.TP + m.FP)) > 1e-12)
    mf += int(abs(m.FP_exame - m.FP / m.exames) > 1e-12)
f["agregados_E2_vs_E1"] = agr; f["metricas_vs_contagens"] = mf
f["IC_nao_contem_observado"] = int(((B.observado < B.ic_lo - 1e-12) | (B.observado > B.ic_hi + 1e-12)).sum())
c1 = C[np.isclose(C.tau, 0.10)].set_index(["coorte", "referencia", "configuracao", "exame"])
oo = OB[OB.fonte == "pred"].groupby(["coorte", "referencia", "configuracao", "exame"]).status.value_counts().unstack(fill_value=0)
jj = c1[["TP", "FP", "ignoradas"]].join(oo, rsuffix="_obj", how="inner")
f["objetos_vs_contagens_pred"] = int(((jj.TP != jj.TP_obj) | (jj.FP != jj.FP_obj) | (jj.ignoradas != jj.get("IGN", 0))).sum())
f["linhas_contagens"] = len(C); f["linhas_metricas"] = len(M); f["intervalos"] = len(B)
f["aprovada"] = all(v == 0 for k, v in f.items() if k not in ("linhas_contagens", "linhas_metricas", "intervalos"))
json.dump(f, open(O + "/saidas/V_verificacao.json", "w"), indent=1); print(json.dumps(f))
