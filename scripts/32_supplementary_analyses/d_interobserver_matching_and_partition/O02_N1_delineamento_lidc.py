#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    O02_N1_delineamento_lidc.py

File type:
    Script

Purpose:
    Effect of the reference delineation rule (union versus consensus) in LNDb.
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
    - _lesoes.csv
    - _rad*.mhd
    - input_manifest.csv
    - lndb_verifier_object_scores.csv
    - {e}.mhd
    - {e}.nii.gz

Outputs:
    - trainNodules.csv
    - trainNodules_gt.csv

Dependencies:
    - SimpleITK
    - numpy
    - scipy

Execution:
    cd <project_dir> && python scripts/32_supplementary_analyses/d_interobserver_matching_and_partition/O02_N1_delineamento_lidc.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import csv, json, hashlib, os, sys, math, numpy as np, scipy.ndimage as nd, SimpleITK as sitk, re, glob
from scipy.optimize import linear_sum_assignment
from collections import Counter
P = "."; R = P + "/resultados"; O = R + "/retificacao_LNDb_20261111"; RAW = P + "/dados/brutos/raw_5070_2026-08/lndb"
S22D = R + "/pulmonar_pos_oof_prep/script22_final/s22_20260924T084718_826234"
PRED = R + "/Pulmonary_Project/Script10_canonical_lndb/blocks/lndb__{a}/{e}.nii.gz"; CT = RAW + "/data/{e}.mhd"
MASK, OUT = sys.argv[1], sys.argv[2]; PL = 0.60; TAUS = [0.10, 0.25, 0.50]; ARCH = ["3D_F", "3D_FREM"]; S6 = nd.generate_binary_structure(3, 1)
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def iou(gl, pl, ng, npr):
    if ng == 0 or npr == 0: return np.zeros((ng, npr))
    g, p = gl.ravel(), pl.ravel(); gs = np.bincount(g, minlength=ng + 1)[1:]; ps = np.bincount(p, minlength=npr + 1)[1:]
    m = np.zeros((ng, npr)); ov = (g > 0) & (p > 0)
    for (gi, pi), it in Counter(zip(g[ov].tolist(), p[ov].tolist())).items(): m[gi - 1, pi - 1] = it / (gs[gi - 1] + ps[pi - 1] - it)
    return m
def B(m, t):
    if m.size == 0: return set()
    w = np.where(m >= t, 1000.0 + m, 0.0); r, c = linear_sum_assignment(-w); return {(int(i), int(j)) for i, j in zip(r, c) if m[i, j] >= t}
man = {r["scan_id"]: r for r in csv.DictReader(open(S22D + "/input_manifest.csv"))}; EX = sorted(man); assert len(EX) == 211
TN = {(int(r["LNDbID"]), int(r["RadID"]), int(r["FindingID"])): int(r["Nodule"]) for r in csv.DictReader(open(RAW + "/trainNodules.csv"))}
GT = {}
for r in csv.DictReader(open(RAW + "/trainNodules_gt.csv")): GT.setdefault(int(r["LNDbID"]), []).append(r)
ret = {(r["architecture"], r["scan_id"], int(r["object_id"])): r["retained"] == "True" for r in csv.DictReader(open(R + "/pulmonar_pos_oof_prep/script23_final/s23_20260924T153750_909396/lndb_verifier_object_scores.csv"))}
rows, les_rows = [], []
for e in EX:
    i = int(e.split("-")[1]); Z = np.load(f"{O}/referencias/{e}.npz")
    labs = {}
    for f in sorted(glob.glob(f"{MASK}/{e}_rad*.mhd")):
        dim = [int(x) for x in re.search(r"DimSize = (.*)", open(f).read()).group(1).split()]
        labs[int(re.search(r"_rad(\d)", f).group(1))] = np.fromfile(f.replace(".mhd", ".raw"), dtype=np.uint8).reshape(dim[::-1])
    shp = Z["N1"].shape; n1 = Z["N1"].astype(np.int32); ign_extra = np.zeros(shp, bool)
    cons = np.zeros(shp, np.int32); nl = 0; uni_chk = np.zeros(shp, np.int32); nu = 0
    for g in GT.get(i, []):
        rads = [int(x) for x in g["RadID"].split(",")]; rfs = [int(float(x)) for x in g["RadFindingID"].split(",")]
        ms = [labs[r] == f for r, f in zip(rads, rfs) if TN[(i, r, f)] == 1 and int((labs[r] == f).sum()) > 1]
        if not ms: continue
        k = len(ms); acc = np.sum(ms, axis=0); need = math.ceil(PL * k - 1e-9)
        u = acc >= 1; nu += 1; uni_chk[u & (uni_chk == 0)] = nu
        c = acc >= need
        if c.any(): nl += 1; cons[c & (cons == 0)] = nl
        else: ign_extra |= u
        les_rows.append(dict(exame=e, FindingID=g["FindingID"], k=k, need=need, vox_uniao=int(u.sum()), vox_consenso=int(c.sum())))
    assert nu == int(n1.max()), (e, nu, int(n1.max()))
    assert np.array_equal(uni_chk > 0, n1 > 0), e
    rd = sitk.ImageFileReader(); rd.SetFileName(CT.format(e=e)); rd.ReadImageInformation()
    img = sitk.Image([1, 1, 1], sitk.sitkUInt8); img.SetOrigin(rd.GetOrigin()); img.SetSpacing(rd.GetSpacing()); img.SetDirection(rd.GetDirection())
    for a in ARCH:
        p = PRED.format(a=a, e=e)
        if not os.path.exists(p): p = PRED.format(a=[x for x in ARCH if x != a][0], e=e)
        assert sha(p) == man[e][f"prediction_{a}_sha256"]
        pr = sitk.ReadImage(p); lab, npred = nd.label(sitk.GetArrayFromImage(pr) > 0, structure=S6)
        o = np.round(np.array(img.TransformPhysicalPointToContinuousIndex(pr.GetOrigin()))).astype(int)
        pl = np.zeros(shp, np.int32); Zs, Ys, Xs = lab.shape; pl[o[2]:o[2] + Zs, o[1]:o[1] + Ys, o[0]:o[0] + Xs] = lab
        accd = {j for j in range(npred) if ret[(a, e, j + 1)]}
        for ref, gl, ign in [("N1", n1, Z["ignorar_N1"]), ("N1_LIDC", cons, Z["ignorar_N1"] | ign_extra)]:
            ng = int(gl.max()); m = iou(gl, pl, ng, npred); toca = set(int(v) - 1 for v in np.unique(pl[ign]) if v)
            for t in TAUS:
                pares = B(m, t); pm = {c for _, c in pares}; ig = {j for j in range(npred) if j not in pm and j in toca}
                tpp = sum(1 for _, c in pares if c in accd)
                rows.append(dict(exame=e, config=a, ref=ref, tau=t, n_ref=ng, n_pred=npred, ign=len(ig), TP=len(pares), FP=npred - len(pares) - len(ig), FN=ng - len(pares),
                                 TP_pos=tpp, FP_pos=len(accd - pm - ig), FN_pos=ng - tpp, iou_med_TP=float(np.median([m[r_, c_] for r_, c_ in pares])) if pares else ""))
    print(e, flush=True)
for nome, X in [(OUT, rows), (OUT.replace(".csv", "_lesoes.csv"), les_rows)]:
    with open(nome, "w", newline="") as f: w = csv.DictWriter(f, fieldnames=list(X[0])); w.writeheader(); w.writerows(X)
print("FIM", len(rows))
