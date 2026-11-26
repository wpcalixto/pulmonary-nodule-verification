#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    c06_por_leitor_0061.py

File type:
    Script

Purpose:
    Checks the per-reader annotations of one scan before assigning reference status to two records.
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
    - *.nii.gz
    - consensus.nii.gz
    - lung_mask_crop.nii.gz

Outputs:
    - Tables written to the stage output directory (see docs/post_v1_stages.md).

Dependencies:
    - SimpleITK
    - numpy
    - scipy

Execution:
    cd <project_dir> && python scripts/30_radiological_assessment/c06_por_leitor_0061.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import glob, os, re
import numpy as np, SimpleITK as sitk
from scipy import ndimage as nd
P = "."
A = P + "/arquivo/Ultima_Versao/Resultados/evidencia_nvme/do_5070/annotations"
LUNG = P + "/arquivo/versoes_mais_novas_que_a_oficial/Pulmonar_v281_2026-09-01/08_DADOS_INTERMEDIOS/s06_20260810T183307_96979/lung_masks/lidc_idri"
S6 = nd.generate_binary_structure(3, 1)
e = "LIDC-IDRI-0061"
img = sitk.ReadImage(f"{LUNG}/{e}/lung_mask_crop.nii.gz"); L = sitk.GetArrayFromImage(img) > 0
A_ = L.copy()
while True:
    lab, n = nd.label(A_, structure=S6); sz = np.bincount(lab.ravel())[1:]; big = np.argsort(sz)[::-1][:2] + 1
    if n >= 2 and sz[big[1]-1] > 0.2*sz[big[0]-1]: break
    A_ = nd.binary_erosion(A_, structure=S6)
xs = {b: img.TransformContinuousIndexToPhysicalPoint(list(nd.center_of_mass(lab == b))[::-1])[0] for b in big}
esq = max(xs, key=xs.get); seeds = np.where(np.isin(lab, big), lab, 0)
_, ind = nd.distance_transform_edt(seeds == 0, return_indices=True); lado = seeds[tuple(ind)]
zz = np.where(L.any(axis=(1, 2)))[0]
cons = sitk.Resample(sitk.ReadImage(f"{A}/consensus/lidc/{e}/consensus.nii.gz"), img, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
clab, nc = nd.label(sitk.GetArrayFromImage(cons) > 0, structure=S6)
print("objetos de consenso:", nc)
for f in sorted(glob.glob(f"{A}/per_reader/lidc/{e}/*.nii.gz")):
    m = sitk.Resample(sitk.ReadImage(f), img, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
    a = sitk.GetArrayFromImage(m) > 0
    if not a.any(): print(os.path.basename(f), "fora da grade"); continue
    z, y, x = nd.center_of_mass(a); i = (int(round(z)), int(round(y)), int(round(x)))
    s = "esq" if lado[i] == esq else "dir"
    h = (z - zz.min()) / (zz.max() - zz.min())
    ov = sorted(set(np.unique(clab[a])) - {0})
    d = 2 * (a.sum() * np.prod(img.GetSpacing()) * 3 / (4 * np.pi)) ** (1/3)
    print(os.path.basename(f).split("_ann")[0], s, f"alt={h:.2f}", f"diam~{d:.1f}mm", "consenso:", ov)
