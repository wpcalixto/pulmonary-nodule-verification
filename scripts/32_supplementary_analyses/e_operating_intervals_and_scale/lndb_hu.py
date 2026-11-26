#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    lndb_hu.py

File type:
    Script

Purpose:
    Checks the LNDb intensity scale from headers and air, fat and soft-tissue histogram peaks in a systematic sample.
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
    - *_0000.nii.gz
    - LNDB_INPUT_MANIFEST.csv
    - c16_lndb_escala_intensidade.json

Outputs:
    - Tables written to the stage output directory (see docs/post_v1_stages.md).

Dependencies:
    - numpy
    - scipy

Execution:
    cd <project_dir> && python scripts/32_supplementary_analyses/e_operating_intervals_and_scale/lndb_hu.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import numpy as np, csv, gzip, struct, sys, json
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
P=""
LD=P+"dados/brutos/raw_5070_2026-08/lndb/data/"
ids=[r["case_id"] for r in csv.DictReader(open(P+"resultados/pulmonar_pos_oof_prep/LNDB_INPUT_MANIFEST.csv"))]
amostra=ids[::10]
def mhd(i):
    h=dict(l.split(" = ",1) for l in open(LD+i+".mhd").read().strip().splitlines())
    d=[int(x) for x in h["DimSize"].split()]
    return np.fromfile(LD+i+".raw",dtype="<i2").reshape(d[::-1]),h
def nii(p):
    b=gzip.open(p).read(); dim=struct.unpack("<8h",b[40:56]); dt=struct.unpack("<h",b[70:72])[0]
    off=int(struct.unpack("<f",b[108:112])[0]); sl,it=struct.unpack("<ff",b[112:120])
    t={4:"<i2",16:"<f4",64:"<f8",512:"<u2",8:"<i4"}[dt]; n=dim[1]*dim[2]*dim[3]
    a=np.frombuffer(b[off:off+n*np.dtype(t).itemsize],dtype=t).reshape(dim[3],dim[2],dim[1]).astype(np.float32)
    if sl not in (0,1) or it!=0: a=a*(sl or 1)+it
    return a
def medir(a):
    a=a.astype(np.float32); mn=float(a.min())
    v=a[a>mn+1].ravel()[::7]
    lo,hi=np.percentile(v,[0.5,99.5]); bins=np.arange(lo,hi+4,4); h,_=np.histogram(v,bins); h=gaussian_filter1d(h.astype(float),2)
    pk,pr=find_peaks(h,prominence=h.max()*0.01); c=(bins[pk]+2)
    def pico(a0,a1):
        s=[(h[k],c[j]) for j,k in enumerate(pk) if a0<=c[j]<=a1]; return round(max(s)[1]) if s else None
    z=a.shape[0]//2; ar=a[z-5:z+5,8:40,a.shape[2]//2-20:a.shape[2]//2+20]
    return dict(min=mn,ar_ant_med=float(np.median(ar)),pico_ar=pico(-1100,-900),pico_pulmao=pico(-900,-700),pico_gordura=pico(-150,-50),pico_tecido=pico(0,80),
                todos_picos=[round(x) for x in c[np.argsort(-h[pk])][:5]])
out=[]
for i in amostra:
    a,h=mhd(i); m=medir(a); m.update(conjunto="LNDb",exame=i,ElementType=h["ElementType"]); out.append(m)
import glob
for p in sorted(glob.glob(P+"dados/brutos/nnUNet_raw_s08_20260811/Dataset001_PulmonaryNodules/imagesTr/*_0000.nii.gz"))[:5]:
    m=medir(nii(p)); m.update(conjunto="LIDC (controlo, HU)",exame=p.split("/")[-1][:14],ElementType="nii"); out.append(m)
json.dump(out,open(P+"resultados/modelo_c_ablacoes_dobras/c16_lndb_escala_intensidade.json","w"),indent=1)
for m in out: print(m["conjunto"][:4],m["exame"],"min",int(m["min"]),"ar_ant",int(m["ar_ant_med"]),"ar",m["pico_ar"],"pulm",m["pico_pulmao"],"gord",m["pico_gordura"],"tec",m["pico_tecido"],"| picos",m["todos_picos"])
