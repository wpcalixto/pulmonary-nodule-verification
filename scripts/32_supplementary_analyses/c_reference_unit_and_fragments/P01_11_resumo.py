#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    P01_11_resumo.py

File type:
    Script

Purpose:
    Summary of the reference-unit sensitivity analysis.
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
    - R_componentes_agrupamentos.csv
    - R_estratos.csv
    - R_metricas_cenarios.csv
    - R_mudanca_classe.csv
    - R_paridade_A_script11.csv

Dependencies:
    - numpy
    - pandas

Execution:
    cd <project_dir> && python scripts/32_supplementary_analyses/c_reference_unit_and_fragments/P01_11_resumo.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import sys, numpy as np, pandas as pd
S, S11 = sys.argv[1], sys.argv[2]
C = pd.read_csv(S + "/P01_11_contagens.csv"); Rf = pd.read_csv(S + "/P01_11_referencias.csv", low_memory=False); Pr = pd.read_csv(S + "/P01_11_preditos.csv"); Po = pd.read_csv(S + "/P01_11_portas.csv")
NEX = {"oof": 718, "internal": 158}
print("PORTAS:", {c: Po[c].value_counts(dropna=False).to_dict() for c in Po.columns if c not in ("pid", "fontes")})
O = pd.read_csv(S11)
par = []
for (arch, sub), g in O[O.status.isin(["TP", "FN", "FP"])].groupby(["model", "subset"]):
    for t, col in ((0.10, "tp_010"), (0.25, "tp_025"), (0.50, "tp_050")):
        ref = g[g.status.isin(["TP", "FN"])]; n_ref = len(ref); tp = int(ref[col].sum())
        n_pred = int((g.status.isin(["TP", "FP"])).sum()); a = C[(C.cen == "A") & (C.arch == arch) & (C.subset == sub) & np.isclose(C.tau, t)][["n_ref", "n_pred", "TP"]].sum()
        par.append(dict(arch=arch, subset=sub, tau=t, n_ref_11=n_ref, n_ref_A=int(a.n_ref), TP_11=tp, TP_A=int(a.TP), n_pred_11=n_pred, n_pred_A=int(a.n_pred)))
P = pd.DataFrame(par); P["ok"] = (P.n_ref_11 == P.n_ref_A) & (P.TP_11 == P.TP_A) & (P.n_pred_11 == P.n_pred_A)
print("PARIDADE A x Script 11B:", int(P.ok.sum()), "de", len(P)); print(P[~P.ok].to_string() if (~P.ok).any() else "")
P.to_csv(S + "/R_paridade_A_script11.csv", index=False)
M = C.groupby(["cen", "subset", "arch", "tau"])[["n_ref", "n_pred", "TP", "FP", "FN"]].sum().reset_index()
M["S"] = M.TP / M.n_ref; M["P"] = np.where(M.n_pred > 0, M.TP / M.n_pred.replace(0, np.nan), np.nan); M["F1"] = 2 * M.S * M.P / (M.S + M.P); M["FP_exame"] = M.FP / M.subset.map(NEX)
W = M.pivot_table(index=["subset", "arch", "tau"], columns="cen", values=["n_ref", "TP", "FP", "S", "P", "F1", "FP_exame"])
for m in ("S", "P", "F1", "FP_exame"):
    for c in ("B", "C"): W[(f"d{m}", c)] = W[(m, c)] - W[(m, "A")]
W.to_csv(S + "/R_metricas_cenarios.csv");
x = W.xs(0.10, level="tau"); cols = [("n_ref", c) for c in "ABC"] + [("S", c) for c in "ABC"] + [("P", c) for c in "ABC"] + [("FP_exame", c) for c in "ABC"]
print(x[cols].round(3).to_string())
print("Variação de S (C-A), IoU 0,10: min", round(x[("dS", "C")].min(), 3), "max", round(x[("dS", "C")].max(), 3), "| (B-A):", round(x[("dS", "B")].min(), 3), round(x[("dS", "B")].max(), 3))
print("Variação de P (C-A):", round(x[("dP", "C")].min(), 3), round(x[("dP", "C")].max(), 3), "| FP/exame (C-A):", round(x[("dFP_exame", "C")].min(), 3), round(x[("dFP_exame", "C")].max(), 3))
R = Rf[Rf.cen.isin(["A", "B", "C"])].copy(); R["estrato"] = pd.cut(R.diam, [0, 3, 6, 15, 1e9], right=False, labels=["<3", "3-6", "6-15", ">=15"])
E = R.groupby(["cen", "subset", "arch", "estrato"], observed=False).agg(n=("detectado", "size"), det=("detectado", "sum")).reset_index(); E["S"] = E.det / E.n
E.to_csv(S + "/R_estratos.csv", index=False)
print(E[E.arch.isin(["3D_F", "3D_FREM"])].pivot_table(index=["subset", "arch", "estrato"], columns="cen", values=["n", "S"], observed=False).round(3).to_string())
for sub in ("oof", "internal"):
    q = Pr[Pr.subset == sub]
    print(sub, "objetos preditos", len(q), "| A->B mudam", int((q.A != q.B).sum()), "| A->C mudam", int((q.A != q.C).sum()), "| FP->TP em C", int(((q.A == "FP") & (q.C == "TP")).sum()), "| TP->FP em C", int(((q.A == "TP") & (q.C == "FP")).sum()))
Pr.assign(muda_B=Pr.A != Pr.B, muda_C=Pr.A != Pr.C).groupby(["subset", "arch"])[["muda_B", "muda_C"]].sum().to_csv(S + "/R_mudanca_classe.csv")
mp = Rf[Rf.cen == "A_mapa"][["pid", "id", "agrupamento", "n_agrupamentos_tocados", "vox"]]
ra = Rf[(Rf.cen == "A") & (Rf.arch == "3D_F")][["pid", "id", "diam", "subset"]].merge(mp, on=["pid", "id"])
mx = ra.groupby(["pid", "agrupamento"]).diam.transform("max")
peq = ra[ra.diam < 3]
print("componentes A:", len(ra), "| < 3 mm:", len(peq), "(oof", int((peq.subset == "oof").sum()), "int", int((peq.subset == "internal").sum()), ") | pertencem a agrupamento com componente >= 3 mm:", int((mx[peq.index] >= 3).sum()), "| agrupamento só com fragmentos < 3 mm:", int((mx[peq.index] < 3).sum()))
ra.assign(diam_max_agrupamento=mx).to_csv(S + "/R_componentes_agrupamentos.csv", index=False)
