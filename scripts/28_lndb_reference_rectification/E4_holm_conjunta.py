#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    E4_holm_conjunta.py

File type:
    Script

Purpose:
    Joint Holm family combining LNDb and development p values of the sixteen responses for the external configurations.
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
    - 000_Supplementary_Material_ATIVO.tex
    - bd1_filter_features.csv
    - lndb_filter_group_comparisons.csv

Outputs:
    - E2_D.csv
    - E4_D1_por_configuracao.csv
    - E4_holm_conjunta_N1.csv
    - E4_porta.json

Dependencies:
    - numpy
    - pandas
    - scipy

Execution:
    cd <project_dir> && python scripts/28_lndb_reference_rectification/E4_holm_conjunta.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import ast, re, json, sys, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
P = "."; R = P + "/resultados"; O = R + "/retificacao_LNDb_20261111"
src = open(P + "/codigo/scripts_e_relatorios/script14_run/23_validate_filters_lndb.py").read(); ns = {"np": np, "mannwhitneyu": mannwhitneyu}
for n in ["auroc", "mw", "holm"]: exec(ast.get_source_segment(src, [x for x in ast.parse(src).body if isinstance(x, ast.FunctionDef) and x.name == n][0]), ns)
F16 = ["log_max", "log_mean", "log_p95", "log_scale_mm", "blob_max", "blob_mean", "blob_p95", "blob_scale_mm", "vessel_max", "vessel_mean", "vessel_p95", "vessel_scale_mm", "contrast_hu", "centroid_peak_dist_mm", "pleura_min_dist_mm", "pleura_centroid_dist_mm"]
fe = pd.read_csv(P + "/artigos_relatorios/Artigos_14_10_2026/RESULTADOS_ADICIONAIS_CMIG/AUDITORIA_v4/fontes/bd1_filter_features.csv", low_memory=False); fe = fe[fe.subset == "oof"]
L = []
for comp, pg, ng in [("A", "TP_pred", "FP_pred"), ("B", "TP_gt", "FN_gt")]:
    for a in ["3D_F", "3D_FREM"]:
        for f in F16:
            x = fe[(fe.architecture == a) & (fe.group == pg)][f].dropna().to_numpy(float); y = fe[(fe.architecture == a) & (fe.group == ng)][f].dropna().to_numpy(float)
            L.append(dict(comparacao=comp, configuracao=a, resposta=f, conjunto="D1", auroc=ns["auroc"](x, y), p_raw=ns["mw"](x, y)[1], n_pos=len(x), n_neg=len(y)))
D1 = pd.DataFrame(L)
old = pd.read_csv(R + "/pulmonar_pos_oof_prep/script23_final/s23_20260924T153750_909396/lndb_filter_group_comparisons.csv")
old = old.assign(comparacao=old.comparison.str[0], configuracao=old.architecture, resposta=old.feature, conjunto="D4")[["comparacao", "configuracao", "resposta", "conjunto", "auroc", "p_raw"]]
new = pd.read_csv(O + "/saidas/E2_D.csv", dtype={"coorte": str}); new = new[(new.coorte == "211") & (new.referencia == "N1")]
new = new.assign(comparacao=new.comparacao.str[0], conjunto="D4")[["comparacao", "configuracao", "resposta", "conjunto", "auroc", "p_raw", "p_holm"]].rename(columns={"p_holm": "p_holm32"})
def conj(d4):
    T = pd.concat([d4, D1], ignore_index=True); T["p_holm64"] = np.nan
    for c, q in T.groupby("comparacao"): T.loc[q.index, "p_holm64"] = ns["holm"](q.p_raw.to_numpy())
    return T
def fp(p):
    if p >= 0.001: return f"{p:.3f}".replace(".", ",")
    m, e = f"{p:.1e}".split("e"); return f"${m.replace('.', '{,}')}\\times10^{{{int(e)}}}$"
To = conj(old.reset_index(drop=True))
tex = open(O + "/saidas/000_Supplementary_Material_ATIVO.tex", encoding="utf-8").read()
ROT = {"log_max": "LoG máximo", "blob_max": "Blobness máximo", "vessel_max": "Vesselness máximo"}
falhas = 0; verif = 0
for lab, comp in [("tab_s_lndb_a", "A"), ("tab_s_lndb_b", "B")]:
    i = tex.index("\\label{" + lab + "}"); e = tex.index("\\end{longtable}", i); blk = tex[i:e]
    for a, cab in [("3D_F", "3D-F ---"), ("3D_FREM", "3D-FREM ---")]:
        k = blk.index(cab); seg = blk[k:]
        for f, rot in ROT.items():
            row = [r for r in seg.split("\\\\") if r.strip().startswith("\\quad " + rot)][0]
            cells = [c.strip() for c in row.split("&")]
            d4 = To[(To.comparacao == comp) & (To.configuracao == a) & (To.resposta == f) & (To.conjunto == "D4")].iloc[0]
            d1 = To[(To.comparacao == comp) & (To.configuracao == a) & (To.resposta == f) & (To.conjunto == "D1")].iloc[0]
            exp = [fp(d4.p_holm64), f"{d1.auroc:.3f}".replace(".", ","), fp(d1.p_holm64)]
            got = [cells[3], cells[4], cells[5]]; verif += 1
            if exp != got: falhas += 1; print("DIF", lab, a, f, exp, got)
Tn = conj(new.drop(columns="p_holm32").reset_index(drop=True)).merge(new[["comparacao", "configuracao", "resposta", "conjunto", "p_holm32"]], how="left", on=["comparacao", "configuracao", "resposta", "conjunto"])
Tn.to_csv(O + "/saidas/E4_holm_conjunta_N1.csv", index=False); D1.to_csv(O + "/saidas/E4_D1_por_configuracao.csv", index=False)
porta = dict(celulas_verificadas=verif, falhas=falhas, aprovada=falhas == 0); json.dump(porta, open(O + "/saidas/E4_porta.json", "w")); print(porta)
if falhas: sys.exit("PORTA E4 FALHOU")
