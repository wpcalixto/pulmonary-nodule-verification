#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    S1_confianca_candidatos.py

File type:
    Script

Purpose:
    Computes the detector confidence of each candidate (mean foreground probability over the component voxels).
    Pipeline position: Stage 29 (selective verification).

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
    - *.npz

Outputs:
    - S1_confianca_candidatos.csv
    - S1_paridade.csv
    - S1_resumo.json

Dependencies:
    - SimpleITK
    - numpy
    - pandas
    - scipy

Execution:
    cd <project_dir> && python scripts/29_selective_verification/S1_confianca_candidatos.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import os, glob, sys, json
import numpy as np, pandas as pd, scipy.ndimage as nd, SimpleITK as sitk
from multiprocessing import Pool
R = "resultados"; PP = R + "/Pulmonary_Project"; O = R + "/verificacao_seletiva_20261111"
ST = nd.generate_binary_structure(3, 1)
T = []
for a in ["3D_F", "3D_FREM"]:
    for f in range(5):
        for p in sorted(glob.glob(f"{PP}/Script10_canonical_oof/blocks/lidc_oof__{a}__fold_{f}/*.npz")): T.append(("D1", a, os.path.basename(p)[:-4], p))
    for p in sorted(glob.glob(f"{PP}/Script10_canonical/blocks/lidc_internal__{a}/*.npz")): T.append(("D2", a, os.path.basename(p)[:-4], p))
    for p in sorted(glob.glob(f"{PP}/Script10_canonical_lndb/blocks/lndb__{a}/*.npz")): T.append(("D4", a, os.path.basename(p)[:-4], p))
    for p in sorted(glob.glob(f"{R}/retificacao_LNDb_20261111/pred13/{a}/*.npz")): T.append(("D4", a, os.path.basename(p)[:-4], p))
def um(t):
    conj, a, e, p = t
    pr = np.load(p)["probabilities"]; seg = pr.argmax(0) == 1; pf = pr[1]
    nii = p[:-4] + ".nii.gz"; par = None
    if os.path.exists(nii): par = bool((sitk.GetArrayFromImage(sitk.ReadImage(nii)) > 0).__eq__(seg).all())
    lab, n = nd.label(seg, structure=ST); out = []
    if n:
        idx = np.arange(1, n + 1)
        mean = nd.mean(pf, lab, idx); mx = nd.maximum(pf, lab, idx); vox = nd.sum(np.ones_like(pf, dtype=np.int32), lab, idx)
        for i in range(n): out.append(dict(conjunto=conj, configuracao=a, exame=e, id=i + 1, voxels=int(vox[i]), q_media=float(mean[i]), q_max=float(mx[i])))
    return out, (conj, a, e, par, n)
if __name__ == "__main__":
    with Pool(7) as pool: RES = pool.map(um, T, chunksize=4)
    df = pd.DataFrame([x for r, _ in RES for x in r]); au = pd.DataFrame([dict(conjunto=c, configuracao=a, exame=e, paridade_nii=p, componentes=n) for _, (c, a, e, p, n) in RES])
    df.to_csv(O + "/saidas/S1_confianca_candidatos.csv", index=False); au.to_csv(O + "/saidas/S1_paridade.csv", index=False)
    res = dict(ficheiros=len(T), paridade_falhas=int((au.paridade_nii == False).sum()), sem_nii=int(au.paridade_nii.isna().sum()), candidatos=len(df),
               por_conjunto=df.groupby(["conjunto", "configuracao"]).size().to_dict())
    json.dump({k: (str(v) if isinstance(v, dict) else v) for k, v in res.items()}, open(O + "/saidas/S1_resumo.json", "w"), indent=1); print(res)
