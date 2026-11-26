#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    P01_12_1_populacao.py

File type:
    Script

Purpose:
    Compares the patients with reconstructable annotations with the remaining patients.
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
    - P01_11_referencias.csv
    - P01_12_1_composicao.csv
    - P01_12_1_metricas_A_831_45.csv

Dependencies:
    - numpy
    - pandas

Execution:
    cd <project_dir> && python scripts/32_supplementary_analyses/c_reference_unit_and_fragments/P01_12_1_populacao.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import sys, pandas as pd, numpy as np
S, TH, O = sys.argv[1:4]
C=pd.read_csv(S+"/P01_11_contagens.csv"); r=pd.read_csv(S+"/P01_11_referencias.csv",low_memory=False); po=pd.read_csv(S+"/P01_11_portas.csv"); th=pd.read_csv(TH)
bad=set(po.pid[po.uniao_agrupamentos_igual_consenso==False]); G=lambda s: np.where(s.isin(bad),"45","831")
C["grupo"]=G(C.pid); A=r[(r.cen=="A")&(r.arch=="3D_F")].copy(); A["grupo"]=G(A.pid)
pp=C.drop_duplicates("pid")[["pid","subset","grupo"]]
print("Pacientes:\n",pd.crosstab(pp.subset,pp.grupo,margins=True))
t=th.merge(pp,left_on="patient_id",right_on="pid")
a=t.groupby("grupo").agg(agrupamentos=("cluster","size"),leitores_medio=("R_readers","mean"),frac_1_leitor=("R_readers",lambda v:(v==1).mean()),frac_4_leitores=("R_readers",lambda v:(v==4).mean()))
a["agr_por_paciente"]=a.agrupamentos/pp.groupby("grupo").size(); print(a.round(3))
q=A.groupby(["subset","grupo"]).agg(n_ref=("diam","size"),diam_mediano=("diam","median"),frac_lt3=("diam",lambda v:(v<3).mean()),frac_lt6=("diam",lambda v:(v<6).mean()),frac_ge15=("diam",lambda v:(v>=15).mean()))
q["ref_por_paciente"]=q.n_ref/pp.groupby(["subset","grupo"]).size(); print(q.round(3)); q.round(4).to_csv(O+"/P01_12_1_composicao.csv")
A10=C[(C.cen=="A")&(C.tau==0.10)]
def met(g): tp,fp,n=g.TP.sum(),g.FP.sum(),g.n_ref.sum(); return pd.Series(dict(n_ref=n,S=tp/n,P=tp/(tp+fp) if tp+fp else np.nan,FP_exame=fp/g.pid.nunique()))
out=pd.concat({k:v.groupby(["subset","arch"]).apply(met) for k,v in (("876",A10),("831",A10[A10.grupo=="831"]),("45",A10[A10.grupo=="45"]))},axis=1)
out.round(4).to_csv(O+"/P01_12_1_metricas_A_831_45.csv"); print(out.round(3).to_string())
x=out.drop(index="F100_ResEncM",level=1)
for m in ("S","P","FP_exame"): d=x[("831",m)]-x[("876",m)]; print(f"d{m} 831-876: {d.min():.3f} a {d.max():.3f}")
for s in ("oof","internal"):
    y=out.xs(s,level=0); print(s,"ordenação S 831 = 876:",bool((y[("831","S")].rank()==y[("876","S")].rank()).all()))
