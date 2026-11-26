#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    O03_auditoria_hungaro_lndb.py

File type:
    Script

Purpose:
    Audit of the Hungarian matching rule (admissible-pair priority versus total-IoU maximization) in LNDb.
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
    - SHA256_pred13.json
    - input_manifest.csv
    - {e}.mhd
    - {e}.nii.gz

Outputs:
    - O03_lndb_AB.csv
    - trainCTs.csv

Dependencies:
    - SimpleITK
    - numpy
    - scipy

Execution:
    cd <project_dir> && python scripts/32_supplementary_analyses/d_interobserver_matching_and_partition/O03_auditoria_hungaro_lndb.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import csv, json, hashlib, os, sys, numpy as np, scipy.ndimage as nd, SimpleITK as sitk
from scipy.optimize import linear_sum_assignment
from collections import Counter
P = "."; R = P + "/resultados"; O = R + "/retificacao_LNDb_20261111"
S22D = R + "/pulmonar_pos_oof_prep/script22_final/s22_20260924T084718_826234"
PRED211 = R + "/Pulmonary_Project/Script10_canonical_lndb/blocks/lndb__{a}/{e}.nii.gz"; PRED13 = O + "/pred13/{a}/{e}.nii.gz"
CT = P + "/dados/brutos/raw_5070_2026-08/lndb/data/{e}.mhd"; OUT = sys.argv[1]
LIM = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75]; ARCH = ["3D_F", "3D_FREM"]; S6 = nd.generate_binary_structure(3, 1)
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def iou(gl, pl, ng, npr):
    if ng == 0 or npr == 0: return np.zeros((ng, npr))
    g, p = gl.ravel(), pl.ravel(); gs = np.bincount(g, minlength=ng + 1)[1:]; ps = np.bincount(p, minlength=npr + 1)[1:]
    m = np.zeros((ng, npr)); ov = (g > 0) & (p > 0)
    for (gi, pi), it in Counter(zip(g[ov].tolist(), p[ov].tolist())).items(): m[gi - 1, pi - 1] = it / (gs[gi - 1] + ps[pi - 1] - it)
    return m
def A(m, t):
    if m.size == 0: return set()
    r, c = linear_sum_assignment(-m); return {(int(i), int(j)) for i, j in zip(r, c) if m[i, j] >= t}
def B(m, t):
    if m.size == 0: return set()
    w = np.where(m >= t, 1000.0 + m, 0.0); r, c = linear_sum_assignment(-w); return {(int(i), int(j)) for i, j in zip(r, c) if m[i, j] >= t}
man = {r["scan_id"]: r for r in csv.DictReader(open(S22D + "/input_manifest.csv"))}; EX211 = sorted(man)
P13 = json.load(open(O + "/pred13/SHA256_pred13.json")); EX13 = sorted({k.split("/")[1] for k in P13})
radn = {f"LNDb-{int(r['LNDbID']):04d}": int(r["RadN"]) for r in csv.DictReader(open(P + "/dados/brutos/raw_5070_2026-08/lndb/trainCTs.csv"))}
rows = []
for e in EX211 + EX13:
    rd = sitk.ImageFileReader(); rd.SetFileName(CT.format(e=e)); rd.ReadImageInformation()
    Z = np.load(f"{O}/referencias/{e}.npz")
    for a in ARCH:
        if e in man:
            p = PRED211.format(a=a, e=e)
            if not os.path.exists(p): p = PRED211.format(a=[x for x in ARCH if x != a][0], e=e)
            assert sha(p) == man[e][f"prediction_{a}_sha256"]
        else:
            p = PRED13.format(a=a, e=e); assert sha(p) == P13[f"{a}/{e}"]
        pr = sitk.ReadImage(p); arr = sitk.GetArrayFromImage(pr); lab, npred = nd.label(arr > 0, structure=S6)
        ci = np.round(np.array(rd.GetOrigin() and sitk.Image([1,1,1], sitk.sitkUInt8).TransformPhysicalPointToContinuousIndex(pr.GetOrigin())) if False else 0)
        img = sitk.Image([1, 1, 1], sitk.sitkUInt8); img.SetOrigin(rd.GetOrigin()); img.SetSpacing(rd.GetSpacing()); img.SetDirection(rd.GetDirection())
        o = np.round(np.array(img.TransformPhysicalPointToContinuousIndex(pr.GetOrigin()))).astype(int)
        pl = np.zeros(Z["N1"].shape, np.int32); Zs, Ys, Xs = lab.shape; pl[o[2]:o[2] + Zs, o[1]:o[1] + Ys, o[0]:o[0] + Xs] = lab
        for ref in ["N1", "N2"]:
            if ref == "N2" and radn[e] < 2: continue
            gl = Z[ref].astype(np.int32); ng = int(gl.max()); m = iou(gl, pl, ng, npred)
            multi_g = int(((m > 0).sum(1) > 1).sum()) if m.size else 0; multi_p = int(((m > 0).sum(0) > 1).sum()) if m.size else 0
            for t in LIM:
                a_, b_ = A(m, t), B(m, t)
                rows.append(dict(coorte="211" if e in man else "13", exame=e, config=a, ref=ref, tau=t, n_ref=ng, n_pred=npred, TP_A=len(a_), TP_B=len(b_), pares_iguais=int(a_ == b_), ref_multiplos_pred=multi_g, pred_multiplas_ref=multi_p))
    print(e, flush=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("FIM", len(rows))
