#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    v03_impacto_classes_TP_FN.py

File type:
    Script

Purpose:
    Cross-tabulates LNDb reference objects against the class of the official finding that originated them.
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
    - Outputs of the preceding pipeline stages (see docs/post_v1_stages.md).

Outputs:
    - v03_componentes_referencia_local.csv
    - v03_referencias_D4_por_classe.csv
    - v03_sensibilidade_por_classe_D4.csv

Dependencies:
    - Python standard library only

Execution:
    cd <project_dir> && python scripts/28_lndb_reference_rectification/v03_impacto_classes_TP_FN.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import csv, sys, os
from collections import defaultdict, Counter
B = os.path.dirname(os.path.abspath(__file__)) + "/.."
S22 = sys.argv[1]
c = [x for x in csv.DictReader(open(B + "/saidas/v03_componentes_referencia_local.csv")) if x["grupo"] == "incluido_211" and x["mascara"] == "referencia_local"]
mine = defaultdict(dict)
for x in c: mine[x["lndb_id"]][int(x["componente"])] = x
rows, bad = [], 0
for x in csv.DictReader(open(S22)):
    if x["status"] not in ("TP", "FN"): continue
    y = mine[x["scan_id"]].get(int(float(x["gt_component_id"])))
    if not y or int(y["voxels"]) != int(x["n_voxels_gt"]): bad += 1; continue
    rows.append(dict(exame=x["scan_id"], configuracao=x["architecture"], gt_component_id=x["gt_component_id"], status_artigo=x["status"], iou=x["iou"], voxels=y["voxels"], classe_achado_oficial=y["classe"], achados=y["achados"], n_leitores=y["n_leitores"]))
assert bad == 0 and len(rows) == 1700
with open(B + "/saidas/v03_referencias_D4_por_classe.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
L = ["configuracao,classe,TP,FN,total,sensibilidade_na_classe"]
for a in ["3D_F", "3D_FREM"]:
    cnt = Counter((r["classe_achado_oficial"], r["status_artigo"]) for r in rows if r["configuracao"] == a)
    for cl in ["nodulo_segmentado", "nodulo_marca_pontual", "nao_nodulo"]:
        tp, fn = cnt[(cl, "TP")], cnt[(cl, "FN")]; L.append(f"{a},{cl},{tp},{fn},{tp+fn},{tp/(tp+fn):.4f}")
open(B + "/saidas/v03_sensibilidade_por_classe_D4.csv", "w").write("\n".join(L) + "\n"); print("\n".join(L))
