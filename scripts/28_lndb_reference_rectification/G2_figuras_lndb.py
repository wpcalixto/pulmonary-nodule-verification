#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    G2_figuras_lndb.py

File type:
    Script

Purpose:
    Generates the LNDb detection and verification figures from the closed N1 tables.
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
    - identidade_visual_figuras.yaml
    - s17a_internal_verifier_metrics.csv
    - supl_s2_limiar_f1.pdf
    - supl_s2_limiar_fp_exam.pdf
    - supl_s2_limiar_precision.pdf
    - supl_s2_limiar_sensitivity.pdf

Outputs:
    - B0_geometria_por_objeto_N1.csv
    - E2_G.csv
    - E2_M.csv

Dependencies:
    - matplotlib
    - numpy
    - pandas
    - yaml

Execution:
    cd <project_dir> && python scripts/28_lndb_reference_rectification/G2_figuras_lndb.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import os, sys, csv, json, hashlib
import numpy as np, pandas as pd, yaml, matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
AQUI = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, AQUI)
from protocolo_grafico import formatador_virgula, marcar, registrar
B = os.path.dirname(os.path.dirname(AQUI)); S = B + "/saidas"; FIG = B + "/figuras"
ident = yaml.safe_load(open(AQUI + "/identidade_visual_figuras.yaml")); t = ident["typography"]; g = ident["geometry"]
plt.rcParams.update({"text.usetex": True, "font.family": t["font_family"], "font.size": t["axis_label_pt"], "axes.labelsize": t["axis_label_pt"],
                     "xtick.labelsize": t["tick_label_pt"], "ytick.labelsize": t["tick_label_pt"], "legend.fontsize": t["legend_pt"],
                     "pgf.texsystem": "pdflatex", "pgf.rcfonts": False, "pgf.preamble": r"\usepackage[T1]{fontenc}\usepackage{newtxtext,newtxmath}",
                     "savefig.bbox": "standard", "savefig.pad_inches": 0.0})
COR = ident["architecture_families"]["3D"]["color"]; MK = {c: ident["models"][c]["marker"] for c in ["3D_F", "3D_FREM"]}; CF = ["3D_F", "3D_FREM"]
LAB = ident["labels"]
def gravar(fig, nome, altura):
    p = os.path.join(FIG, nome); fig.set_size_inches(g["canonical_width_in"], altura); fig.savefig(p, format="pdf"); registrar(fig, p, nome); plt.close(fig)
    print(nome, hashlib.sha256(open(p, "rb").read()).hexdigest()[:16])
M = pd.read_csv(S + "/E2_M.csv", dtype={"coorte": str}); M = M[(M.coorte == "211") & (M.referencia == "N1") & np.isclose(M.tau, 0.10)].set_index("configuracao")
I = pd.read_csv(S + "/s17a_internal_verifier_metrics.csv"); I = I[(I.detector == "original")].set_index("architecture")
MET = [("sensitivity", "S", "S_pos", "sensitivity", (0, 1)), ("precision", "P", "P_pos", "precision", (0, 1)), ("f1", "F1", "F1_pos", "f1", (0, 1)), ("fp_exam", "FP_exame", "FP_exame_pos", "fp_per_exam", (0, 3))]
def pontos(nome, vals_aberto, vals_cheio, rot, lim, altura=4.2):
    fig = plt.figure(); ax = fig.add_axes((0.27, 0.30, 0.70, 0.67))
    ax.axhspan(-0.5, 0.5, color="0.92", zorder=0)
    for i, c in enumerate(CF):
        y = 1 - i; a, b = vals_aberto[c], vals_cheio[c]
        ax.plot([a, b], [y + 0.08, y - 0.08], color=COR, linewidth=1.0, zorder=2)
        marcar(ax.plot([a], [y + 0.08], marker=MK[c], ms=10, mew=1.5, mfc="white", mec=COR, ls="none", zorder=3), c)
        marcar(ax.plot([b], [y - 0.08], marker=MK[c], ms=10, mew=1.5, mfc=COR, mec=COR, ls="none", zorder=3), c)
        assert lim[0] <= a <= lim[1] and lim[0] <= b <= lim[1]
    ax.set_yticks([1, 0]); ax.set_yticklabels(["3D-F", "3D-FREM"]); ax.set_ylim(-0.5, 1.5)
    ax.set_xlim(lim[0] - 0.03 * (lim[1] - lim[0]), lim[1] + 0.03 * (lim[1] - lim[0]))
    if lim[1] <= 1: ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.xaxis.set_major_formatter(formatador_virgula(2))
    else: ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel(LAB[rot]); ax.grid(True, axis="x", linewidth=0.5, alpha=0.35); gravar(fig, nome, altura)
for nm, k1, k2, rot, lim in MET:
    ik = {"S": "sensitivity", "P": "precision", "F1": "f1", "FP_exame": "fp_per_exam"}[k1]
    pontos(f"results_3_7_lndb_{nm}.pdf", {c: float(I.loc[c, ik]) for c in CF}, {c: float(M.loc[c, k1]) for c in CF}, rot, lim)
    pontos(f"results_3_8_lndb_verifier_{nm}.pdf", {c: float(M.loc[c, k1]) for c in CF}, {c: float(M.loc[c, k2]) for c in CF}, rot, lim)
G = pd.read_csv(S + "/geometria_N1/B0_geometria_por_objeto_N1.csv", low_memory=False); G = G[G.arch.isin(CF) & (G.status == "TP")]
G["aceite"] = G.aceite.astype(str) == "True"
def q(v): v = np.asarray(v, float); v = v[np.isfinite(v)]; return (np.median(v), np.percentile(v, 25), np.percentile(v, 75), len(v))
def painel(nome, linhas, xl, xlim, xt, cheio_aberto):
    fig = plt.figure(); ax = fig.add_axes((0.34, 0.14, 0.46, 0.84)); n = len(linhas); ys = list(range(n - 1, -1, -1))
    ax.axhspan(-0.5, n / 2 - 0.5, color="0.92", zorder=0)
    for y, (rot, items, ntxt) in zip(ys, linhas):
        for off, (c, val, fill) in zip([0.15, -0.15], items):
            m, lo, hi, _ = val; ax.plot([lo, hi], [y + off, y + off], color=COR, linewidth=1.8 if fill else 1.0, zorder=2)
            marcar(ax.plot([m], [y + off], marker=MK[c], ms=10, mew=1.5, mfc=(COR if fill else "white"), mec=COR, ls="none", zorder=3), c)
        ax.text(1.02, y, ntxt, transform=ax.get_yaxis_transform(), va="center", ha="left")
    ax.set_yticks(ys); ax.set_yticklabels([l[0] for l in linhas]); ax.set_ylim(-0.5, n - 0.5)
    ax.set_xlim(*xlim); ax.set_xticks(xt)
    if xlim[1] <= 1.05: ax.xaxis.set_major_formatter(formatador_virgula(2))
    ax.set_xlabel(xl); ax.grid(True, axis="x", linewidth=0.5, alpha=0.35)
    ax.text(-0.52, 0.75, LAB["conjunto_interno"], transform=ax.transAxes, rotation=90, va="center", ha="center")
    ax.text(-0.52, 0.25, LAB["conjunto_lndb"], transform=ax.transAxes, rotation=90, va="center", ha="center")
    gravar(fig, nome, 5.6)
for met, xl, xlim, xt, suf in [("dice", "Dice similarity coefficient", (-0.03, 1.03), [0, 0.25, 0.5, 0.75, 1.0], "dice"), ("hd95", r"HD$_{95}$ [mm]", None, None, "hd95")]:
    lin_t, lin_v = [], []
    for cj in ["D2", "D4"]:
        for est, rot in [("<6", "$<$6 mm"), ("6-15", "6--15 mm"), (">=15", r"$\geq$15 mm")]:
            vals = [(c, q(G[(G.conjunto == cj) & (G.arch == c) & (G.estrato_com == est)][met]), True) for c in CF]
            lin_t.append((rot, vals, f"$n$ = {vals[0][1][3]}/{vals[1][1][3]}"))
        for c in CF:
            ga = G[(G.conjunto == cj) & (G.arch == c)]
            a, r = q(ga[ga.aceite][met]), q(ga[~ga.aceite][met]); lin_v.append((c.replace("_", "-"), [(c, a, True), (c, r, False)], f"$n$ = {a[3]}/{r[3]}"))
    if xlim is None:
        mx = max(max(v[1][2] for v in l[1]) for l in lin_t + lin_v); top = float(np.ceil(mx / 2) * 2); xlim, xt = (-0.03 * top, top * 1.03), list(np.arange(0, top + 0.1, 2 if top > 4 else 1))
    painel(f"results_compl_geometria_tamanho_{suf}.pdf", lin_t, xl, xlim, xt, True)
    painel(f"results_compl_geometria_verificador_{suf}.pdf", lin_v, xl, xlim, xt, True)
GR = pd.read_csv(S + "/E2_G.csv", dtype={"coorte": str}); GR = GR[(GR.coorte == "211") & (GR.referencia == "N1")]
TH = 0.3417980765530837
for met, nome, rot, ylim, leg in [("S", "supl_s2_limiar_sensitivity.pdf", "sensitivity", (0, 1), True), ("P", "supl_s2_limiar_precision.pdf", "precision", (0, 1), False),
                                  ("F1", "supl_s2_limiar_f1.pdf", "f1", (0, 1), False), ("FP_exame", "supl_s2_limiar_fp_exam.pdf", "fp_per_exam", (0, 3), False)]:
    fig = plt.figure(); ax = fig.add_axes((0.175, 0.37, 0.785, 0.60))
    for c in CF:
        q_ = GR[GR.configuracao == c]; grade = q_[~np.isclose(q_.limiar, TH)].sort_values("limiar"); assert len(grade) == 19
        ax.plot(grade.limiar, grade[met], color=COR, linewidth=1.0, zorder=2)
        marcar(ax.plot(grade.limiar, grade[met], marker=MK[c], ms=7, mew=1.3, mfc="white", mec=COR, ls="none", zorder=3), c)
        fz = q_[np.isclose(q_.limiar, TH)].iloc[0]; marcar(ax.plot([TH], [fz[met]], marker=MK[c], ms=9, mew=1.3, mfc=COR, mec=COR, ls="none", zorder=4), c)
        assert ylim[0] <= grade[met].min() and grade[met].max() <= ylim[1]
    ax.axvline(TH, color="0.45", linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.set_xlim(0, 1); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.xaxis.set_major_formatter(formatador_virgula(2))
    f_ = 0.03 * (ylim[1] - ylim[0]); ax.set_ylim(ylim[0] - f_, ylim[1] + f_)
    if ylim[1] <= 1: ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.yaxis.set_major_formatter(formatador_virgula(2))
    else: ax.set_yticks([0, 1, 2, 3])
    ax.set_xlabel(LAB["limiar_verificador"]); ax.set_ylabel(LAB[rot]); ax.grid(True, linewidth=0.5, alpha=0.35)
    if leg:
        h = [Line2D([], [], marker=MK[c], ls="none", ms=8, mew=1.3, mfc="white", mec=COR, label=c.replace("_", "-")) for c in CF]
        h.append(Line2D([], [], color="0.45", linewidth=1.0, linestyle=(0, (4, 3)), label=LAB["limiar_congelado"]))
        fig.legend(handles=h, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.01), frameon=False, handlelength=1.6, columnspacing=1.6)
    gravar(fig, nome, 4.6)
