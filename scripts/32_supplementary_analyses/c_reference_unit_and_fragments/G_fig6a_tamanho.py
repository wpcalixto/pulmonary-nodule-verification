#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    G_fig6a_tamanho.py

File type:
    Script

Purpose:
    Generates the figure of sensitivity by reference size in development.
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
    - results_3_2_size_sensitivity.pdf

Outputs:
    - fig6a_dados.csv

Dependencies:
    - matplotlib
    - numpy
    - pandas
    - yaml

Execution:
    cd <project_dir> && python scripts/32_supplementary_analyses/c_reference_unit_and_fragments/G_fig6a_tamanho.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import os, sys, json, hashlib, numpy as np, pandas as pd, yaml, matplotlib
matplotlib.use("pgf"); import matplotlib.pyplot as plt
AQUI = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, AQUI + "/figuras")
from protocolo_grafico import formatador_virgula, marcar, registrar
REF, OUT = sys.argv[1], sys.argv[2]
ident = yaml.safe_load(open(AQUI + "/figuras/identidade_visual_figuras.yaml")); t = ident["typography"]; g = ident["geometry"]
plt.rcParams.update({"text.usetex": True, "font.family": t["font_family"], "font.size": t["axis_label_pt"], "axes.labelsize": t["axis_label_pt"],
                     "xtick.labelsize": t["tick_label_pt"], "ytick.labelsize": t["tick_label_pt"], "legend.fontsize": t["legend_pt"],
                     "pgf.texsystem": "pdflatex", "pgf.rcfonts": False, "pgf.preamble": r"\usepackage[T1]{fontenc}\usepackage{newtxtext,newtxmath}",
                     "savefig.bbox": "standard", "savefig.pad_inches": 0.0})
R = pd.read_csv(REF, low_memory=False); R = R[(R.cen == "A") & (R.subset == "oof")]
R["estrato"] = pd.cut(R.diam, [3, 6, 15, np.inf], right=False, labels=["3-6", "6-15", ">=15"])
E = R.dropna(subset=["estrato"]).groupby(["arch", "estrato"], observed=True).detectado.agg(["size", "mean"]).reset_index()
E.to_csv(OUT + "/fig6a_dados.csv", index=False)
fig = plt.figure(); fig.set_size_inches(g["canonical_width_in"], 7.2)
ax = fig.add_axes((0.13451, 0.30486, 0.84543, 0.66563))
for c, m in ident["models"].items():
    cor = ident["architecture_families"][m["family"]]["color"]; mk = m["marker"]
    mk = mk if isinstance(mk, str) else np.array(mk)
    y = E[E.arch == c].set_index("estrato")["mean"].reindex(["3-6", "6-15", ">=15"]).values
    marcar(ax.plot([0, 1, 2], y, color=cor, linewidth=1.5, marker=mk, ms=10, mew=1.5, mfc="white", mec=cor, zorder=3), c)
ax.set_xticks([0, 1, 2]); ax.set_xticklabels([r"3--6 mm", r"6--15 mm", r"$\geq$15 mm"]); ax.set_xlim(-0.25, 2.25)
ax.set_ylim(0, 1); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.yaxis.set_major_formatter(formatador_virgula(1))
ax.set_ylabel(ident["labels"]["sensitivity"]); ax.set_xlabel(ident["labels_emenda_P01_13"]["reference_size_stratum"])
ax.grid(True, axis="y", linewidth=0.5, alpha=0.35)
p = OUT + "/results_3_2_size_sensitivity.pdf"; fig.savefig(p, format="pdf"); registrar(fig, p, "Figura 6(a)"); plt.close(fig)
print(p, hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]); print(E.pivot(index="arch", columns="estrato", values="mean").round(3))
