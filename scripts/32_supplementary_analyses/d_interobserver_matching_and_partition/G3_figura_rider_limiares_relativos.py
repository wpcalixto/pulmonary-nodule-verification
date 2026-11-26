#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    G3_figura_rider_limiares_relativos.py

File type:
    Script

Purpose:
    Generates the figure of RIDER pairs whose relative change exceeds the thresholds nu_1 and nu_2.
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
    - identidade_visual_figuras.yaml
    - results_3_6_rel_legend.pdf

Outputs:
    - N1_A2_limiares_relativos.csv

Dependencies:
    - matplotlib
    - numpy
    - pandas
    - yaml

Execution:
    cd <project_dir> && python scripts/32_supplementary_analyses/d_interobserver_matching_and_partition/G3_figura_rider_limiares_relativos.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import os, sys, hashlib, numpy as np, pandas as pd, yaml, matplotlib
matplotlib.use("pgf"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
AQUI = os.path.dirname(os.path.abspath(__file__)); B = os.path.dirname(AQUI)
ID = os.path.join(B, "..", "retificacao_LNDb_20261111", "scripts", "figuras"); sys.path.insert(0, ID)
from protocolo_grafico import formatador_virgula, marcar, registrar
ident = yaml.safe_load(open(ID + "/identidade_visual_figuras.yaml")); t = ident["typography"]; g = ident["geometry"]
plt.rcParams.update({"text.usetex": True, "font.family": t["font_family"], "font.size": t["axis_label_pt"], "axes.labelsize": t["axis_label_pt"],
                     "xtick.labelsize": t["tick_label_pt"], "ytick.labelsize": t["tick_label_pt"], "legend.fontsize": t["legend_pt"],
                     "pgf.texsystem": "pdflatex", "pgf.rcfonts": False, "pgf.preamble": r"\usepackage[T1]{fontenc}\usepackage{newtxtext,newtxmath}",
                     "savefig.bbox": "standard", "savefig.pad_inches": 0.0})
FIG = B + "/figuras"; os.makedirs(FIG, exist_ok=True)
MOD = sorted(ident["models"], key=lambda m: ident["models"][m]["order"])
COR = {m: ident["architecture_families"][ident["models"][m]["family"]]["color"] for m in MOD}
MK = {m: (ident["models"][m]["marker"] if isinstance(ident["models"][m]["marker"], str) else np.array(ident["models"][m]["marker"])) for m in MOD}
ROT = {m: m.replace("_", "-") for m in MOD}
D = pd.read_csv(B + "/saidas/N1_A2_limiares_relativos.csv")
assert set(D.architecture) <= set(MOD) and len(D) == 36 and D.proportion.between(0, 1).all()
assert (D.ci_lo <= D.proportion + 1e-12).all() and (D.proportion <= D.ci_hi + 1e-12).all()
def gravar(fig, nome, w, h):
    p = os.path.join(FIG, nome); fig.set_size_inches(w, h); fig.savefig(p, format="pdf"); registrar(fig, p, nome); plt.close(fig)
    print(nome, hashlib.sha256(open(p, "rb").read()).hexdigest()[:16])
XL = {"volume": r"Pairs with $|d_i|/m_i>\nu$ in volume [$-$]", "diameter": r"Pairs with $|d_i|/m_i>\nu$ in diameter [$-$]"}
for med in ["volume", "diameter"]:
    fig = plt.figure(); ax = fig.add_axes((0.30, 0.16, 0.66, 0.82)); n = len(MOD)
    for i, m in enumerate(MOD):
        y = n - 1 - i
        if i % 2 == 1: ax.axhspan(y - 0.5, y + 0.5, color="0.92", zorder=0)
        for lam, off, cheio in [(0.15, 0.17, False), (0.25, -0.17, True)]:
            r = D[(D.architecture == m) & (D.measure == med) & np.isclose(D.threshold, lam)]
            if r.empty: continue
            r = r.iloc[0]
            ax.plot([r.ci_lo, r.ci_hi], [y + off, y + off], color=COR[m], linewidth=1.8 if cheio else 1.0, zorder=2)
            marcar(ax.plot([r.proportion], [y + off], marker=MK[m], ms=10, mew=1.5, mfc=(COR[m] if cheio else "white"), mec=COR[m], ls="none", zorder=3), m)
    ax.set_yticks(range(n - 1, -1, -1)); ax.set_yticklabels([ROT[m] for m in MOD]); ax.set_ylim(-0.5, n - 0.5)
    ax.set_xlim(-0.03, 1.03); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.xaxis.set_major_formatter(formatador_virgula(2))
    ax.set_xlabel(XL[med]); ax.grid(True, axis="x", linewidth=0.5, alpha=0.35)
    gravar(fig, f"results_3_6_rider_rel_{med}.pdf", g["canonical_width_in"], 6.0)
fig = plt.figure(); h = []
for m in MOD: h.append(Line2D([], [], marker=MK[m], color=COR[m], mfc=COR[m], mec=COR[m], ms=10, ls="none", label=ROT[m]))
h.append(Line2D([], [], marker="o", color="black", mfc="white", mec="black", ms=10, mew=1.5, ls="none", label=r"$\nu_1$"))
h.append(Line2D([], [], marker="o", color="black", mfc="black", mec="black", ms=10, ls="none", label=r"$\nu_2$"))
h.append(Line2D([], [], color="black", linewidth=1.4, label=r"95\% CI"))
for l in h[:len(MOD)]: marcar(l, l.get_label().replace("-", "_"))
fig.legend(handles=h, loc="upper left", frameon=False, handletextpad=0.3, borderaxespad=0.0, labelspacing=0.25, bbox_to_anchor=(0, 1))
gravar(fig, "results_3_6_rel_legend.pdf", 2.856, 4.70)
