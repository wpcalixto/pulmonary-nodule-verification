#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    ic_oper.py

File type:
    Script

Purpose:
    Scan-level bootstrap intervals of the operating metrics before and after verification.
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
    - c14_por_exame_D2_D4.csv
    - c15_ic_operacionais_por_exame_D2_D4.csv

Outputs:
    - Tables written to the stage output directory (see docs/post_v1_stages.md).

Dependencies:
    - numpy
    - pandas

Execution:
    cd <project_dir> && python scripts/32_supplementary_analyses/e_operating_intervals_and_scale/ic_oper.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import numpy as np,pandas as pd
B="resultados/modelo_c_ablacoes_dobras/"
df=pd.read_csv(B+"c14_por_exame_D2_D4.csv")
def met(tp,fp,nref,nex):
    S=tp/nref; Pr=tp/(tp+fp) if tp+fp else np.nan; F1=2*tp/(2*tp+fp+(nref-tp)); return S,Pr,F1,fp/nex
rows=[]
for (ds,ar),g in df.groupby(["dataset","arch"]):
    a=g[["n_ref","cand_tp","cand_fp","A_tp","A_fp"]].to_numpy(float); n=len(a)
    def allm(x):
        s=x.sum(0); pre=met(s[1],s[2],s[0],len(x)); pos=met(s[3],s[4],s[0],len(x))
        return np.array(pre+pos+tuple(np.subtract(pos,pre)))
    obs=allm(a); rng=np.random.default_rng(270014)
    bs=np.array([allm(a[rng.integers(0,n,n)]) for _ in range(2000)])
    lo,hi=np.nanpercentile(bs,2.5,0),np.nanpercentile(bs,97.5,0)
    names=[f"{w}_{m}" for w in ("antes","depois","dif") for m in ("S","Pr","F1","FPexame")]
    for k,nm in enumerate(names): rows.append([ds,ar,nm,obs[k],lo[k],hi[k],n,2000,270014])
out=pd.DataFrame(rows,columns=["conjunto","config","metrica","estimativa","ic_inf","ic_sup","n_exames","replicas","semente"])
out.to_csv(B+"c15_ic_operacionais_por_exame_D2_D4.csv",index=False)
pd.set_option("display.width",200)
print(out.assign(t=lambda d:d.apply(lambda r:f"{r.estimativa:.4f} ({r.ic_inf:.4f}-{r.ic_sup:.4f})",axis=1)).pivot_table(index="metrica",columns=["conjunto","config"],values="t",aggfunc="first").to_string())
print("exames unicos D2:",df[df.dataset=="D2"].exam_id.nunique())
