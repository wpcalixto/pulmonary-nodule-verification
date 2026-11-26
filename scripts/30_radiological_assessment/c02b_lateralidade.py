#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    c02b_lateralidade.py

File type:
    Script

Purpose:
    Robust laterality assignment by splitting the lung mask into the two lungs.
    Pipeline position: Stage 30 (radiological assessment linkage).

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
    - consensus.nii.gz
    - lung_mask_crop.nii.gz

Outputs:
    - c02_esferas.csv
    - c02_referencias.csv
    - c02b_esferas.csv
    - c02b_referencias.csv

Dependencies:
    - SimpleITK
    - numpy
    - pandas
    - scipy

Execution:
    cd <project_dir> && python scripts/30_radiological_assessment/c02b_lateralidade.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import numpy as np, pandas as pd, SimpleITK as sitk
from scipy import ndimage as nd

P = "."
LUNG = P + "/arquivo/versoes_mais_novas_que_a_oficial/Pulmonar_v281_2026-09-01/08_DADOS_INTERMEDIOS/s06_20260810T183307_96979/lung_masks/lidc_idri"
O = P + "/resultados/radiological_reanalysis"
S6 = nd.generate_binary_structure(3, 1)
esf = pd.read_csv(f"{O}/c02_esferas.csv", dtype={"exame": str})
ref = pd.read_csv(f"{O}/c02_referencias.csv", dtype={"exame": str})
info = {}
for e in sorted(set(esf.exame) | set(ref.exame)):
    img = sitk.ReadImage(f"{LUNG}/LIDC-IDRI-{e}/lung_mask_crop.nii.gz")
    L = sitk.GetArrayFromImage(img) > 0
    k, A = 0, L.copy()
    while True:
        lab, n = nd.label(A, structure=S6)
        sz = np.bincount(lab.ravel())[1:]
        big = np.argsort(sz)[::-1][:2] + 1
        if n >= 2 and sz[big[1] - 1] > 0.2 * sz[big[0] - 1]:
            break
        A = nd.binary_erosion(A, structure=S6); k += 1
        assert k < 30, f"nao separou {e}"
    xs = {}
    for b in big:
        z, y, x = nd.center_of_mass(lab == b)
        xs[b] = img.TransformContinuousIndexToPhysicalPoint([x, y, z])[0]
    esq = max(xs, key=xs.get)
    seeds = np.where(np.isin(lab, big), lab, 0)
    _, ind = nd.distance_transform_edt(seeds == 0, return_indices=True)
    lado = np.where(seeds[tuple(ind)] == esq, "esquerdo", "direito")
    xmid = (xs[big[0]] + xs[big[1]]) / 2
    info[e] = (img, lado, xmid, k)
    print(e, "erosoes:", k, "x pulmoes:", {int(b): round(v, 1) for b, v in xs.items()})

def aplica(df):
    lp, dm = [], []
    for _, r in df.iterrows():
        img, lado, xmid, _ = info[r.exame]
        i = img.TransformPhysicalPointToIndex((float(r.x_mm), float(r.y_mm), float(r.z_mm))) if "x_mm" in r else None
        lp.append(lado[i[2], i[1], i[0]]); dm.append(round(abs(r.x_mm - xmid), 1))
    df["lado_pulmao"], df["dist_plano_medio_mm"] = lp, dm
    return df

aplica(esf).to_csv(f"{O}/c02b_esferas.csv", index=False)
CONS = P + "/arquivo/Ultima_Versao/Resultados/evidencia_nvme/do_5070/annotations/consensus/lidc"
xyz = []
for _, r in ref.iterrows():
    img = info[r.exame][0]
    c = sitk.Resample(sitk.ReadImage(f"{CONS}/LIDC-IDRI-{r.exame}/consensus.nii.gz"), img, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
    lab, _ = nd.label(sitk.GetArrayFromImage(c) > 0, structure=S6)
    z, y, x = nd.center_of_mass(lab == int(r.ref_obj))
    xyz.append(img.TransformContinuousIndexToPhysicalPoint([x, y, z]))
ref["x_mm"], ref["y_mm"], ref["z_mm"] = [round(p[0], 1) for p in xyz], [round(p[1], 1) for p in xyz], [round(p[2], 1) for p in xyz]
aplica(ref).to_csv(f"{O}/c02b_referencias.csv", index=False)
d1 = (esf.lado.str[:3] != esf.lado_pulmao.str[:3]).sum(); d2 = (ref.lado.str[:3] != ref.lado_pulmao.str[:3]).sum()
print("discordancias regra antiga vs nova: esferas", d1, "referencias", d2)
print(ref[ref.lado.str[:3] != ref.lado_pulmao.str[:3]][["exame", "ref_obj", "diam_mm", "lado", "lado_pulmao", "dist_plano_medio_mm"]])
print(esf[esf.lado.str[:3] != esf.lado_pulmao.str[:3]][["exame", "esfera", "lado", "lado_pulmao", "dist_plano_medio_mm"]])
