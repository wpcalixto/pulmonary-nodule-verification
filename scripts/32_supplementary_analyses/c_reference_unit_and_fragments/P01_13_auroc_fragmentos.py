#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    P01_13_auroc_fragmentos.py

File type:
    Script

Purpose:
    AUROC of maximum blobness between detected and missed references after excluding fragments of larger groupings.
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
    - P01_13_auroc_fragmentos.csv
    - P01_13_auroc_portas.json

Dependencies:
    - numpy
    - pandas
    - scipy
    - sklearn

Execution:
    cd <project_dir> && python scripts/32_supplementary_analyses/c_reference_unit_and_fragments/P01_13_auroc_fragmentos.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import sys, ast, hashlib, json, warnings, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, average_precision_score
FEAT, S16, MAT, REF, OUT = sys.argv[1:6]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
assert sha(FEAT).startswith("87fd83ce96096d24") and sha(S16).startswith("8f02b14c81c5651a")
src = open(S16).read(); ns = {"np": np, "pd": pd, "warnings": warnings, "mannwhitneyu": mannwhitneyu, "roc_auc_score": roc_auc_score,
                               "average_precision_score": average_precision_score, "N_BOOTSTRAP": 2000, "SEED": 27}
for n in ("point_stats", "bootstrap_patient"):
    exec(ast.get_source_segment(src, [x for x in ast.parse(src).body if isinstance(x, ast.FunctionDef) and x.name == n][0]), ns)
F = pd.read_csv(FEAT, usecols=["patient_id", "architecture", "subset", "group", "component_id", "diam_mm", "blob_max"])
F = F[(F.subset == "oof") & F.group.isin(["TP_gt", "FN_gt"])].copy()
F["size_cat"] = pd.cut(F.diam_mm, [0.0, 3.0, 6.0, 15.0, np.inf], right=False, labels=["lt3mm", "3to6mm", "6to15mm", "ge15mm"])
M = pd.read_csv(MAT); R = pd.read_csv(REF, low_memory=False); R = R[(R.cen == "A") & (R.subset == "oof")]
X = F.merge(M[["pid", "id", "agrupamento", "tipo", "diam"]], left_on=["patient_id", "component_id"], right_on=["pid", "id"], how="left")
X = X.merge(R[["pid", "id", "arch", "detectado"]], left_on=["patient_id", "component_id", "architecture"], right_on=["pid", "id", "arch"], how="left", suffixes=("", "_r"))
portas = {"linhas_B_oof": len(X), "sem_correspondencia": int(X.tipo.isna().sum()),
          "classe_concorda": int(((X.group == "TP_gt") == (X.detectado == True)).sum()),
          "diam_concorda_1e-3": int((np.abs(X.diam_mm - X.diam) < 1e-3).sum()),
          "estrato_lt3_concorda": int(((X.diam_mm < 3) == (X.diam < 3)).sum())}
print(json.dumps(portas)); assert portas["sem_correspondencia"] == 0 and portas["classe_concorda"] == len(X) and portas["estrato_lt3_concorda"] == len(X)
def linha(nome, sub):
    st = ns["point_stats"](sub.loc[sub.group == "TP_gt", "blob_max"].to_numpy(), sub.loc[sub.group == "FN_gt", "blob_max"].to_numpy())
    ci = ns["bootstrap_patient"](sub, "blob_max", "TP_gt", "FN_gt")
    u = sub.drop_duplicates(["patient_id", "component_id"])
    return dict(analise=nome, ocorrencias=len(sub), TP=st["n_pos"], FN=st["n_neg"], pacientes=sub.patient_id.nunique(),
                componentes=len(u), agrupamentos=u.dropna(subset=["agrupamento"]).groupby(["patient_id", "agrupamento"]).ngroups,
                auroc=st["auroc"], ic_lo=ci.get("auroc_ci_lo"), ic_hi=ci.get("auroc_ci_hi"), PM=st["average_precision"],
                mediana_TP=st["median_pos"], mediana_FN=st["median_neg"], replicas=ci.get("bootstrap_replicates_used"))
L = X[X.size_cat == "lt3mm"]; frag = X.tipo.isin(["fragmento_de_lesao_ge3", "sem_agrupamento"])
res = [linha("lt3_original", L), linha("lt3_sem_fragmentos_e_sem_agrupamento", L[~frag.loc[L.index]]),
       linha("lt3_so_fragmentos_FN_vs_TP_isolados", L[(L.group == "TP_gt") | (L.tipo == "fragmento_de_lesao_ge3")]),
       linha("agregada_original", X), linha("agregada_sem_fragmentos_e_sem_agrupamento", X[~frag])]
T = pd.DataFrame(res); T.to_csv(OUT + "/P01_13_auroc_fragmentos.csv", index=False); print(T.round(4).to_string())
print(L.groupby(["tipo", "group"]).size().to_string())
json.dump(portas, open(OUT + "/P01_13_auroc_portas.json", "w"), indent=1)
