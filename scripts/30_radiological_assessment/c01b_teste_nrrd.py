#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    c01b_teste_nrrd.py

File type:
    Script

Purpose:
    Tests the alignment of the radiologist's segmentation files with the current image grid, with inverted-content negative controls.
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
    - c01b_alinhamento_nrrd.json

Dependencies:
    - SimpleITK
    - numpy

Execution:
    cd <project_dir> && python scripts/30_radiological_assessment/c01b_teste_nrrd.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import json, os, itertools
import numpy as np
import SimpleITK as sitk

BASE = "."
CAR = os.path.join(BASE, "artigos_relatorios/radiologist_extracted/radiologist")
CT = os.path.join(BASE, "resultados/pulmonar_canonical_pre_oof/recrops/images/LIDC-IDRI-0061/ct.nii.gz")
PRED = os.path.join(BASE, "resultados/Pulmonary_Project/Script10_canonical_oof/blocks/lidc_oof__3D_F__fold_3/LIDC-IDRI-0061.nii.gz")
OUT = os.path.join(BASE, "resultados/radiological_reanalysis/c01b_alinhamento_nrrd.json")
M = {"modelo_antigo": "LIDC-0061/LIDC-0061/LIDC_0061_mascara_modelo-segmentation.seg.nrrd",
     "original_antiga": "LIDC-0061/LIDC-0061/LIDC_0061_mascara_original-segmentation.seg.nrrd",
     "radiologist": "wpp/Pasta A/LIDC-0061/LIDC-0061-Radiologist.nrrd"}

def binar(img):
    if img.GetNumberOfComponentsPerPixel() > 1:
        img = sitk.VectorIndexSelectionCast(img, 0)
    return sitk.Cast(img != 0, sitk.sitkUInt8)

def flipped(img, axes):
    if not any(axes):
        return img
    a = np.flip(sitk.GetArrayFromImage(img), axis=tuple(2 - k for k, f in enumerate(axes) if f)).copy()
    o = sitk.GetImageFromArray(a); o.CopyInformation(img)
    return o

ct = sitk.ReadImage(CT, sitk.sitkFloat32)
pred = sitk.Cast(sitk.ReadImage(PRED) != 0, sitk.sitkUInt8)
P = sitk.GetArrayFromImage(pred).astype(bool)
H = sitk.GetArrayFromImage(ct)
res = {"n_vox_pred_atual": int(P.sum())}
for nome, rel in M.items():
    raw = sitk.ReadImage(os.path.join(CAR, rel))
    m = binar(raw)
    res[nome] = {"size": list(m.GetSize()), "origin": [round(x, 2) for x in m.GetOrigin()],
                 "direction": [round(x, 3) for x in m.GetDirection()], "n_vox": int(sitk.GetArrayFromImage(m).sum())}
    for ax in itertools.product([False, True], repeat=3):
        r = sitk.Resample(flipped(m, ax), pred, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        A = sitk.GetArrayFromImage(r).astype(bool)
        k = "flip_" + "".join("1" if a else "0" for a in ax)
        res[nome][k] = {"n_vox_na_grade": int(A.sum()), "frac_sobre_pred": round(float((A & P).sum() / max(A.sum(), 1)), 3),
                        "hu_medio": round(float(H[A].mean()), 1) if A.any() else None}
json.dump(res, open(OUT, "w"), indent=1)
for nome in M:
    print(nome, res[nome]["size"], "nvox", res[nome]["n_vox"])
    print("   ", {k[5:]: (v["frac_sobre_pred"], v["hu_medio"]) for k, v in res[nome].items() if k.startswith("flip")})
