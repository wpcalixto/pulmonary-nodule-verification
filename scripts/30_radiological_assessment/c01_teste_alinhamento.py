#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    c01_teste_alinhamento.py

File type:
    Script

Purpose:
    Tests the voxel alignment between the CT used for the radiological markings and the CT of the dataset, with axis-reversal negative controls.
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
    - LIDC_0061_0000.nii.gz
    - ct.nii.gz

Outputs:
    - c01_alinhamento.json

Dependencies:
    - SimpleITK
    - numpy

Execution:
    cd <project_dir> && python scripts/30_radiological_assessment/c01_teste_alinhamento.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import json, os, sys, itertools
import numpy as np
import SimpleITK as sitk

BASE = "."
CAR = os.path.join(BASE, "artigos_relatorios/radiologist_extracted/radiologist")
REC = os.path.join(BASE, "resultados/pulmonar_canonical_pre_oof/recrops/images")
OUT = os.path.join(BASE, "resultados/radiological_reanalysis/c01_alinhamento.json")
EX = ["0061", "0080", "0097", "0235", "0879", "0922", "0925", "1008"]

def info(img):
    return dict(size=list(img.GetSize()), spacing=[round(x, 4) for x in img.GetSpacing()],
                origin=[round(x, 2) for x in img.GetOrigin()],
                direction=[round(x, 3) for x in img.GetDirection()])

def flipped(img, axes):
    if not any(axes):
        return img
    a = sitk.GetArrayFromImage(img)
    a = np.flip(a, axis=tuple(2 - k for k, f in enumerate(axes) if f)).copy()
    o = sitk.GetImageFromArray(a)
    o.CopyInformation(img)
    return o

def corr_on(cur, old):
    r = sitk.Resample(old, cur, sitk.Transform(), sitk.sitkLinear, -5000.0, sitk.sitkFloat32)
    a = sitk.GetArrayFromImage(cur).astype(np.float64).ravel()
    b = sitk.GetArrayFromImage(r).astype(np.float64).ravel()
    m = b > -4000
    if m.sum() < 1000:
        return None, int(m.sum())
    return float(np.corrcoef(a[m], b[m])[0, 1]), int(m.sum())

res = {"teste_ct_0061": {}, "marcacoes": {}}
cur = sitk.ReadImage(os.path.join(REC, "LIDC-IDRI-0061", "ct.nii.gz"), sitk.sitkFloat32)
old = sitk.ReadImage(os.path.join(CAR, "LIDC-0061/LIDC-0061/LIDC_0061_0000.nii.gz"), sitk.sitkFloat32)
res["teste_ct_0061"]["atual"] = info(cur)
res["teste_ct_0061"]["antiga"] = info(old)
for ax in itertools.product([False, True], repeat=3):
    c, n = corr_on(cur, flipped(old, ax))
    res["teste_ct_0061"]["flip_" + "".join("1" if a else "0" for a in ax)] = {"r": c, "n_vox_sobrepostos": n}

for e in EX:
    p = os.path.join(CAR, "wpp/Pasta A", f"LIDC-{e}", f"LIDC-{e}-Radiologist.nrrd")
    m = sitk.ReadImage(p)
    d = {"cabecalho": info(m), "n_comp": m.GetNumberOfComponentsPerPixel()}
    arr = sitk.GetArrayFromImage(m)
    d["valores"] = {str(int(v)): int(c) for v, c in zip(*np.unique(arr, return_counts=True))}
    lab = sitk.ConnectedComponent(sitk.Cast(m > 0, sitk.sitkUInt8)) if m.GetNumberOfComponentsPerPixel() == 1 else None
    if lab is not None:
        st = sitk.LabelShapeStatisticsImageFilter(); st.Execute(lab)
        d["componentes"] = [{"id": int(l), "nvox": int(st.GetNumberOfPixels(l)),
                             "centroide_mm_LPS": [round(x, 1) for x in st.GetCentroid(l)]} for l in st.GetLabels()]
    rp = os.path.join(REC, f"LIDC-IDRI-{e}", "ct.nii.gz")
    if os.path.exists(rp):
        rc = sitk.ReadImage(rp)
        d["atual"] = info(rc)
        if lab is not None:
            dentro = []
            for c in d["componentes"]:
                idx = rc.TransformPhysicalPointToContinuousIndex(c["centroide_mm_LPS"])
                dentro.append(all(-0.5 <= idx[k] <= rc.GetSize()[k] - 0.5 for k in range(3)))
            d["centroides_dentro_do_volume_atual"] = dentro
    res["marcacoes"][e] = d

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)
t = res["teste_ct_0061"]
print("CT 0061 r por inversao:", {k: (round(v["r"], 3) if v["r"] is not None else None) for k, v in t.items() if k.startswith("flip")})
for e, d in res["marcacoes"].items():
    print(e, "comp:", len(d.get("componentes", [])), "valores:", d["valores"] if len(d["valores"]) < 6 else len(d["valores"]),
          "dentro:", d.get("centroides_dentro_do_volume_atual"), "esp:", d["cabecalho"]["spacing"], "atual:", d.get("atual", {}).get("spacing"))
