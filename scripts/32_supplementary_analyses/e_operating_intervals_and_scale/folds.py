#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    folds.py

File type:
    Script

Purpose:
    Checks the equivalence between the verifier modeling folds and the archived development folds.
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
    - bd1_filter_features.csv
    - bd1_folds.json

Outputs:
    - Tables written to the stage output directory (see docs/post_v1_stages.md).

Dependencies:
    - pandas
    - sklearn

Execution:
    cd <project_dir> && python scripts/32_supplementary_analyses/e_operating_intervals_and_scale/folds.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import sys,json,itertools,pandas as pd
from sklearn.model_selection import GroupKFold
from collections import Counter
B=""
df=pd.read_csv(B+"artigos_relatorios/Artigos_14_10_2026/RESULTADOS_ADICIONAIS_CMIG/AUDITORIA_v4/fontes/bd1_filter_features.csv")
print(list(df.columns)[:8])
if "subset" in df: df=df[df.subset=="oof"]
df=df[df.group.isin(["TP_pred","FP_pred"])]
print("linhas",len(df),"pacientes",df.patient_id.nunique())
g={}
for k,(tr,va) in enumerate(GroupKFold(5).split(df,groups=df.patient_id)):
    for p in df.patient_id.iloc[va].unique(): g[p]=k
d=json.load(open(B+"resultados/pulmonar_canonical_pre_oof/splits/bd1_folds.json"))["folds"]
key=[k for k in d["fold_0"] if k!="train"][0]
n={p:int(k.split("_")[1]) for k,v in d.items() for p in v[key]}
c=Counter((n.get(p),g[p]) for p in g)
best=max(itertools.permutations(range(5)),key=lambda P:sum(v for (a,b),v in c.items() if a is not None and P[a]==b))
print("sem dobra nnU-Net:",sum(v for (a,b),v in c.items() if a is None))
print("melhor correspondencia",best,"pacientes coincidentes",sum(v for (a,b),v in c.items() if a is not None and best[a]==b),"de",len(g))
print("tamanhos GroupKFold",Counter(g.values()),"nnU-Net",Counter(n.values()))
