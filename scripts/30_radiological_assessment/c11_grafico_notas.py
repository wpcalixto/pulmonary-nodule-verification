#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    c11_grafico_notas.py

File type:
    Script

Purpose:
    Plots the delineation scores of the true-positive records with the verifier decision.
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
    - Outputs of the preceding pipeline stages (see docs/post_v1_stages.md).

Outputs:
    - c08_ligacao_congelada.csv
    - c10_objetos.csv
    - c11_notas_delineamento.csv

Dependencies:
    - matplotlib
    - pandas

Execution:
    cd <project_dir> && python scripts/30_radiological_assessment/c11_grafico_notas.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import re
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
O = "resultados/radiological_reanalysis"
L = pd.read_csv(f"{O}/c08_ligacao_congelada.csv", dtype={"exame": str}); L = L[L.classe == "TP"].copy()
L["nota"] = L.descricao.str.extract(r"nota (\d+)").astype(int)
X = pd.read_csv(f"{O}/c10_objetos.csv", dtype={"exame": str}); X = X[X.configuracao == "3D_F"]
des = {r.n: (f"{r.estado} {r.decisao}".strip()) for r in X.itertuples()}
L["estado_atual"] = [des.get(n, "Sem referência atual") for n in L.n]
L["estado_atual"] = L.estado_atual.replace({"TP aceito": "TP, aceito", "TP rejeitado": "TP, rejeitado", "FN": "FN"})
L["rot"] = L.exame + "  " + L.registo.str.replace("Nódulo ", "N") + L.exame.isin(["0097", "1008"]).map({True: " (D2)", False: ""})
L = L.sort_values(["exame", "n"], ascending=[False, False]).reset_index(drop=True)
EST = {"TP, aceito": ("#2a78d6", "o"), "TP, rejeitado": ("#eb6834", "s"), "FN": ("#1baf7a", "^"), "Sem referência atual": ("#8a8a85", "D")}
def desenha(PAL, EDGE, SUF):
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(5.2, 5.6))
    prev = None
    for i, r in L.iterrows():
        if prev is not None and r.exame != prev: ax.axhline(i - 0.5, color="#d9d8d3", lw=0.6)
        prev = r.exame
        c, m = PAL[r.estado_atual]
        ax.plot([0, r.nota], [i, i], color="#e6e5e0", lw=1.2, zorder=1)
        ax.scatter(r.nota, i, s=42, color=c, marker=m, edgecolor=EDGE, linewidth=0.9, zorder=3)
    ax.set_yticks(range(len(L))); ax.set_yticklabels(L.rot, color="#3d3d3a")
    ax.set_xlim(-0.3, 10.6); ax.set_xticks(range(0, 11)); ax.set_ylim(-0.7, len(L) - 0.3)
    ax.set_xlabel("Nota de delineamento atribuída pela radiologista (0–10)", color="#3d3d3a")
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#b5b4ad"); ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", colors="#5f5e5a")
    ax.grid(axis="x", color="#efeeea", lw=0.6); ax.set_axisbelow(True)
    h = [plt.Line2D([], [], ls="", marker=m, color=c, markeredgecolor=EDGE, markersize=7, label=f"{k} (n={int((L.estado_atual==k).sum())})") for k, (c, m) in PAL.items() if (L.estado_atual == k).any()]
    ax.legend(handles=h, title="Estado atual na 3D-F e decisão do verificador", loc="upper left", bbox_to_anchor=(0, 1.11), ncol=2, frameon=False, fontsize=7.5, title_fontsize=7.5)
    fig.tight_layout()
    for ext in ("pdf", "png"): fig.savefig(f"{O}/c11_notas_delineamento{SUF}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
CINZA = {"TP, aceito": ("#1f1f1f", "o"), "TP, rejeitado": ("#8c8c8c", "s"), "FN": ("#bdbdbd", "^"), "Sem referência atual": ("#ffffff", "D")}
desenha(EST, "white", "")
desenha(CINZA, "#1f1f1f", "_cinza")
L[["n", "exame", "registo", "nota", "estado_atual"]].to_csv(f"{O}/c11_notas_delineamento.csv", index=False)
print(L.groupby("estado_atual").nota.agg(["count", "median", "min", "max"]).to_string())
