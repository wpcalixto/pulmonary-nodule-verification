#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    c01c_componentes_radiologist_0061.py

File type:
    Script

Purpose:
    Describes the components of the radiologist's marking for one scan on the current image grid.
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
    - LIDC-IDRI-0061.nii.gz
    - ct.nii.gz

Outputs:
    - Tables written to the stage output directory (see docs/post_v1_stages.md).

Dependencies:
    - SimpleITK
    - numpy
    - scipy

Execution:
    cd <project_dir> && python scripts/30_radiological_assessment/c01c_componentes_radiologist_0061.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import os, json
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as nd

BASE = "."
CAR = os.path.join(BASE, "artigos_relatorios/radiologist_extracted/radiologist")
CT = os.path.join(BASE, "resultados/pulmonar_canonical_pre_oof/recrops/images/LIDC-IDRI-0061/ct.nii.gz")
PRED = os.path.join(BASE, "resultados/Pulmonary_Project/Script10_canonical_oof/blocks/lidc_oof__3D_F__fold_3/LIDC-IDRI-0061.nii.gz")
F = {"modelo": "LIDC-0061/LIDC-0061/LIDC_0061_mascara_modelo-segmentation.seg.nrrd",
     "original": "LIDC-0061/LIDC-0061/LIDC_0061_mascara_original-segmentation.seg.nrrd",
     "radiologist": "wpp/Pasta A/LIDC-0061/LIDC-0061-Radiologist.nrrd"}

hdr = open(os.path.join(CAR, F["radiologist"]), "rb").read(1500).split(b"\n\n")[0].decode("latin-1")
print("--- cabecalho Radiologist 0061 ---\n" + "\n".join(l for l in hdr.splitlines() if not l.startswith("Segment")))
ct = sitk.ReadImage(CT, sitk.sitkFloat32)
ref = sitk.ReadImage(PRED)
H = sitk.GetArrayFromImage(ct)
sp = np.array(ref.GetSpacing())[::-1]

def na_grade(rel):
    m = sitk.ReadImage(os.path.join(CAR, rel))
    if m.GetNumberOfComponentsPerPixel() > 1:
        m = sitk.VectorIndexSelectionCast(m, 0)
    m = sitk.Cast(m != 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(sitk.Resample(m, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)).astype(bool)

A = {k: na_grade(v) for k, v in F.items()}
A["pred"] = sitk.GetArrayFromImage(ref) != 0
D = {k: nd.distance_transform_edt(~A[k], sampling=sp) for k in ("modelo", "original", "pred")}
lab, n = nd.label(A["radiologist"])
print(f"--- {n} componentes da Radiologist ---")
print("id nvox  HU   sobre_modelo sobre_original sobre_pred  dmin_modelo dmin_original dmin_pred (mm)  centro_idx(z,y,x)")
for i in range(1, n + 1):
    c = lab == i
    z, y, x = [int(round(v)) for v in nd.center_of_mass(c)]
    print(i, int(c.sum()), round(float(H[c].mean())), *[round(float((c & A[k]).sum() / c.sum()), 2) for k in ("modelo", "original", "pred")],
          *[round(float(D[k][c].min()), 1) for k in ("modelo", "original", "pred")], (z, y, x))
for k in ("modelo", "original", "pred"):
    l2, n2 = nd.label(A[k]); print(k, "componentes:", n2, "centros:", [tuple(int(v) for v in nd.center_of_mass(l2 == j)) for j in range(1, min(n2, 8) + 1)])
