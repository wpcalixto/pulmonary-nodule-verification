#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    P01_11_resumo_coerentes.py

File type:
    Script

Purpose:
    Reference-unit comparison restricted to patients whose annotations reconstruct the consensus.
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
    - P01_11_contagens.csv
    - P01_11_portas.csv
    - P01_11_preditos.csv
    - P01_11_referencias.csv
    - R_estratos_coerentes.csv
    - R_metricas_cenarios_coerentes.csv
    - R_mudanca_classe_coerentes.csv

Dependencies:
    - numpy
    - pandas

Execution:
    cd <project_dir> && python scripts/32_supplementary_analyses/c_reference_unit_and_fragments/P01_11_resumo_coerentes.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import sys, numpy as np, pandas as pd
S = sys.argv[1]
C = pd.read_csv(S + "/P01_11_contagens.csv"); Rf = pd.read_csv(S + "/P01_11_referencias.csv", low_memory=False)
Pr = pd.read_csv(S + "/P01_11_preditos.csv"); Po = pd.read_csv(S + "/P01_11_portas.csv")
ok = set(Po.pid[Po.uniao_agrupamentos_igual_consenso == True]); NE = {"oof": 0, "internal": 0}
sub = C[C.pid.isin(ok)].drop_duplicates("pid")[["pid", "subset"]]
for s, n in sub.subset.value_counts().items(): NE[s] = n
M = C[C.pid.isin(ok)].groupby(["cen", "subset", "arch", "tau"])[["n_ref", "n_pred", "TP", "FP", "FN"]].sum().reset_index()
M["S"] = M.TP / M.n_ref; M["P"] = M.TP / (M.TP + M.FP); M["F1"] = 2 * M.TP / (2 * M.TP + M.FP + M.FN); M["FP_exame"] = M.FP / M.subset.map(NE)
W = M.pivot_table(index=["subset", "arch", "tau"], columns="cen", values=["n_ref", "TP", "FP", "FN", "S", "P", "F1", "FP_exame"])
for m in ["S", "P", "F1", "FP_exame", "n_ref", "TP", "FP"]:
    for c in "BC": W[(f"d_{m}", c)] = W[(m, c)] - W[(m, "A")]
for c in "BC": W[("dRel_S", c)] = W[("d_S", c)] / W[("S", "A")]
W.round(4).to_csv(S + "/R_metricas_cenarios_coerentes.csv")
print("pacientes coerentes:", NE)
t = W.xs(0.10, level="tau")
print(t[[("n_ref", "A"), ("n_ref", "B"), ("n_ref", "C"), ("S", "A"), ("S", "B"), ("S", "C"), ("P", "A"), ("P", "C"), ("F1", "A"), ("F1", "C"), ("FP_exame", "A"), ("FP_exame", "C")]].round(3).to_string())
for tau in (0.10, 0.25, 0.50):
    x = W.xs(tau, level="tau"); x = x[x.index.get_level_values("arch") != "F100_ResEncM"] if tau else x
    print(f"tau {tau}: dS B-A [{x[('d_S','B')].min():.3f},{x[('d_S','B')].max():.3f}] C-A [{x[('d_S','C')].min():.3f},{x[('d_S','C')].max():.3f}] dP C-A [{x[('d_P','C')].min():.3f},{x[('d_P','C')].max():.3f}] dF1 C-A [{x[('d_F1','C')].min():.3f},{x[('d_F1','C')].max():.3f}] dFP/ex C-A [{x[('d_FP_exame','C')].min():.3f},{x[('d_FP_exame','C')].max():.3f}]")
for tau in (0.10, 0.25, 0.50):
    x = W.xs(tau, level="tau")
    for s in ("oof", "internal"):
        y = x.xs(s, level="subset"); ra = y[("S", "A")].rank(); rc = y[("S", "C")].rank(); rb = y[("S", "B")].rank()
        print(f"ordem S tau {tau} {s}: A=B {bool((ra==rb).all())} A=C {bool((ra==rc).all())}; F1 A=C {bool((y[('F1','A')].rank()==y[('F1','C')].rank()).all())}")
R = Rf[(Rf.cen.isin(["A", "B", "C"])) & Rf.pid.isin(ok)].copy()
R["estrato"] = pd.cut(R.diam, [0, 3, 6, 15, np.inf], right=False, labels=["<3", "3-6", "6-15", ">=15"])
E = R.groupby(["cen", "subset", "arch", "estrato"], observed=False).agg(n=("detectado", "size"), det=("detectado", "sum")).reset_index(); E["S"] = E.det / E.n
E.pivot_table(index=["subset", "arch", "estrato"], columns="cen", values=["n", "S"], observed=False).round(4).to_csv(S + "/R_estratos_coerentes.csv")
P = Pr[Pr.pid.isin(ok)]
Z = P.assign(muda_B=P.A != P.B, muda_C=P.A != P.C, TP_FP=(P.A == "TP") & (P.C == "FP"), FP_TP=(P.A == "FP") & (P.C == "TP")).groupby(["subset", "arch"]).agg(n=("A", "size"), TP_A=("A", lambda v: (v == "TP").sum()), muda_B=("muda_B", "sum"), muda_C=("muda_C", "sum"), TP_FP=("TP_FP", "sum"), FP_TP=("FP_TP", "sum"))
Z.to_csv(S + "/R_mudanca_classe_coerentes.csv"); print(Z.groupby(level="subset").sum().to_string())
