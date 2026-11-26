#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    c02_correspondencia.py

File type:
    Script

Purpose:
    Spatial correspondence between the radiologist's markers and the predicted and reference objects, without access to verifier decisions.
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
    - _cc.nii.gz
    - bd1_object_matches.csv
    - consensus.nii.gz
    - ct.nii.gz
    - lung_mask_crop.nii.gz

Outputs:
    - c02_esferas.csv
    - c02_identidade.json
    - c02_referencias.csv

Dependencies:
    - SimpleITK
    - numpy
    - scipy

Execution:
    cd <project_dir> && python scripts/30_radiological_assessment/c02_correspondencia.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import csv, json, os
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as nd

P = "."
R = P + "/resultados"
CAR = P + "/artigos_relatorios/radiologist_extracted/radiologist/wpp/Pasta A"
CONS_ARQ = P + "/arquivo/Ultima_Versao/Resultados/evidencia_nvme/do_5070/annotations/consensus/lidc"
CONS_REC = R + "/pulmonar_canonical_pre_oof/recrops/labels"
LUNG = R + "/Pulmonary_Project/lung_masks/lidc_idri"
OOF = R + "/Pulmonary_Project/Script10_canonical_oof/blocks"
INT = R + "/Pulmonary_Project/Script10_canonical/blocks"
MATCH = R + "/pulmonar_pos_oof_prep/script11_final/s11B_20260921T164539_27883/analysis/bd1_object_matches.csv"
OUT = R + "/radiological_reanalysis"
EX = {"0061": ("D1", 3), "0080": ("D1", 0), "0097": ("D2", None), "0235": ("D1", 1), "0620": ("D1", 2),
      "0879": ("D1", 2), "0922": ("D1", 2), "0925": ("D1", 2), "1008": ("D2", None)}
ARCH = ["3D_F", "3D_FREM"]
S6 = nd.generate_binary_structure(3, 1)
RAIO = 5.0

matches = {}
for r in csv.DictReader(open(MATCH)):
    if r["model"] in ARCH and r["patient_id"][-4:] in EX:
        matches.setdefault((r["patient_id"][-4:], r["model"]), []).append(r)

def pred_path(e, a):
    cj, f = EX[e]
    d = f"{OOF}/lidc_oof__{a}__fold_{f}" if cj == "D1" else f"{INT}/lidc_internal__{a}"
    return f"{d}/LIDC-IDRI-{e}_cc.nii.gz"

def na_grade(img, ref):
    return sitk.GetArrayFromImage(sitk.Resample(img, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, img.GetPixelID()))

ident = {"consenso_vs_recrop": {}, "iou": {}}
esf_rows, ref_rows = [], []
for e, (cj, fold) in EX.items():
    pimg = {a: sitk.ReadImage(pred_path(e, a)) for a in ARCH}
    grid = pimg["3D_F"]
    sp = np.array(grid.GetSpacing())[::-1]
    PL = {a: sitk.GetArrayFromImage(pimg[a]).astype(np.int32) for a in ARCH}
    cons = na_grade(sitk.ReadImage(f"{CONS_ARQ}/LIDC-IDRI-{e}/consensus.nii.gz"), grid) > 0
    if os.path.exists(f"{CONS_REC}/LIDC-IDRI-{e}/consensus.nii.gz"):
        rc = sitk.GetArrayFromImage(sitk.ReadImage(f"{CONS_REC}/LIDC-IDRI-{e}/consensus.nii.gz")) > 0
        ident["consenso_vs_recrop"][e] = {"mesma_forma": rc.shape == cons.shape,
                                          "voxeis_diferentes": int((rc != cons).sum()) if rc.shape == cons.shape else None,
                                          "n_vox": int(rc.sum())}
    GL, ng = nd.label(cons, structure=S6)
    st = {}
    for a in ARCH:
        difs = []
        for m in matches[(e, a)]:
            if m["status"] == "TP":
                G = GL == int(m["gt_component_id"]); Pm = PL[a] == int(m["pred_component_id"])
                difs.append(abs((G & Pm).sum() / (G | Pm).sum() - float(m["iou"])))
        ngt_reg = len({m["gt_component_id"] for m in matches[(e, a)] if m["gt_component_id"]})
        npr_reg = len({m["pred_component_id"] for m in matches[(e, a)] if m["pred_component_id"]})
        ident["iou"][f"{e}_{a}"] = {"n_tp": len(difs), "max_dif_iou": round(max(difs), 5) if difs else None,
                                    "gt_reg/obt": f"{ngt_reg}/{ng}", "pred_reg/obt": f"{npr_reg}/{int(PL[a].max())}"}
        st[a] = {("P", int(m["pred_component_id"])): m["status"] for m in matches[(e, a)] if m["pred_component_id"]}
        st[a].update({("G", int(m["gt_component_id"])): (m["status"], m["pred_component_id"]) for m in matches[(e, a)] if m["gt_component_id"]})
    lung = na_grade(sitk.ReadImage(f"{LUNG}/LIDC-IDRI-{e}/lung_mask_crop.nii.gz"), grid)
    lz = np.where((lung > 0).any(axis=(1, 2)))[0]
    cxi = nd.center_of_mass(lung > 0)[2]
    XMID = grid.TransformContinuousIndexToPhysicalPoint([float(cxi), 0.0, 0.0])[0]
    d_pleura = nd.distance_transform_edt(lung > 0, sampling=sp)
    ct = None
    def dist_obj(lab, n):
        return [nd.distance_transform_edt(lab != i, sampling=sp) for i in range(1, n + 1)]
    DP = {a: dist_obj(PL[a], int(PL[a].max())) for a in ARCH}
    DG = dist_obj(GL, ng)

    def local(c):
        z, y, x = c
        zi = int(round(z)); yi = int(round(y)); xi = int(round(x))
        px = grid.TransformContinuousIndexToPhysicalPoint([float(x), float(y), float(z)])[0]
        lado = "esquerdo" if px > XMID else "direito"
        alt = (zi - lz.min()) / max(lz.max() - lz.min(), 1)
        return lado, round(float(alt), 2), round(float(d_pleura[zi, yi, xi]), 1)

    def nearest(c):
        zi, yi, xi = [int(round(v)) for v in c]
        out = {}
        for a in ARCH:
            ds = [d[zi, yi, xi] for d in DP[a]]
            if ds:
                j = int(np.argmin(ds)); out[a] = (j + 1, round(float(ds[j]), 1), st[a].get(("P", j + 1), "?"))
            else:
                out[a] = (None, None, None)
        ds = [d[zi, yi, xi] for d in DG]
        out["ref"] = (int(np.argmin(ds)) + 1, round(float(min(ds)), 1)) if ds else (None, None)
        return out

    if e != "0620":
        m = sitk.ReadImage(f"{CAR}/LIDC-{e}/LIDC-{e}-Radiologist.nrrd")
        M = na_grade(sitk.Cast(m != 0, sitk.sitkUInt8), grid) > 0
        H = sitk.GetArrayFromImage(sitk.ReadImage(f"{R}/pulmonar_canonical_pre_oof/recrops/images/LIDC-IDRI-{e}/ct.nii.gz"))
        SL, ns = nd.label(M, structure=S6)
        for i in range(1, ns + 1):
            c = nd.center_of_mass(SL == i)
            lado, alt, dpl = local(c)
            nb = nearest(c)
            phys = grid.TransformContinuousIndexToPhysicalPoint([float(c[2]), float(c[1]), float(c[0])])
            row = dict(exame=e, conjunto=cj, esfera=i, nvox=int((SL == i).sum()), hu_medio=int(H[SL == i].mean()),
                       x_mm=round(phys[0], 1), y_mm=round(phys[1], 1), z_mm=round(phys[2], 1),
                       lado=lado, altura_0base_1apice=alt, dist_pleura_mm=dpl,
                       usar="nao (radiologista nao declara marcacao)" if e == "0061" else "sim")
            for a in ARCH:
                j, d, s = nb[a]
                row.update({f"{a}_obj": j, f"{a}_dist_mm": d, f"{a}_estado": s, f"{a}_ate5mm": (d is not None and d <= RAIO)})
            gi, gd = nb["ref"]
            row.update(ref_obj=gi, ref_dist_mm=gd, ref_ate5mm=(gd is not None and gd <= RAIO),
                       ref_estado_3D_F=st["3D_F"].get(("G", gi), ("?",))[0] if gi else None)
            esf_rows.append(row)
    else:
        SL, ns = None, 0
    for g in range(1, ng + 1):
        c = nd.center_of_mass(GL == g)
        lado, alt, dpl = local(c)
        row = dict(exame=e, conjunto=cj, ref_obj=g, nvox=int((GL == g).sum()), lado=lado, altura_0base_1apice=alt, dist_pleura_mm=dpl)
        for a in ARCH:
            s, pid = st[a].get(("G", g), ("?", ""))
            row.update({f"{a}_estado": s, f"{a}_pred": pid})
        diam = [m["diam_mm_gt"] for m in matches[(e, "3D_F")] if m["gt_component_id"] == str(g)]
        row["diam_mm"] = diam[0] if diam else ""
        if ns:
            dd = DG[g - 1]
            row["esfera_mais_proxima_mm"] = round(float(min(dd[SL == i].min() for i in range(1, ns + 1))), 1)
        ref_rows.append(row)

os.makedirs(OUT, exist_ok=True)
json.dump(ident, open(f"{OUT}/c02_identidade.json", "w"), indent=1)
for nome, rows in (("c02_esferas.csv", esf_rows), ("c02_referencias.csv", ref_rows)):
    ks = list(dict.fromkeys(k for r in rows for k in r))
    w = csv.DictWriter(open(f"{OUT}/{nome}", "w", newline=""), fieldnames=ks); w.writeheader(); w.writerows(rows)
print(json.dumps(ident["consenso_vs_recrop"]))
print("IoU:", {k: (v["max_dif_iou"], v["gt_reg/obt"], v["pred_reg/obt"]) for k, v in ident["iou"].items()})
print("esferas:", len(esf_rows), "referencias:", len(ref_rows))
