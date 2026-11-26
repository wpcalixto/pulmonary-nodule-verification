#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    P01_01_rider_AB.py

File type:
    Script

Purpose:
    RIDER pairing under total-IoU priority versus pair-count priority.
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
    - BLOCK_HASHES.csv
    - _1.nii.gz
    - _2.nii.gz
    - input_manifest.csv
    - rider_iou_audit.csv
    - transform_hashes.csv

Outputs:
    - P01_01_pares_diferentes.csv
    - P01_01_rider_AB.csv
    - rider_iou_pairs.csv

Dependencies:
    - SimpleITK
    - numpy
    - pandas
    - scipy

Execution:
    cd <project_dir> && python scripts/32_supplementary_analyses/d_interobserver_matching_and_partition/P01_01_rider_AB.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import os, sys, csv, glob, hashlib, numpy as np, pandas as pd, scipy.ndimage as nd, SimpleITK as sitk
from scipy.optimize import linear_sum_assignment
R = "resultados"; PP = R + "/pulmonar_pos_oof_prep"; BASE = R + "/Pulmonary_Project"; OUT = sys.argv[1]
S6 = nd.generate_binary_structure(3, 1); S18 = glob.glob(PP + "/script18_final/s18_20260923T183100_*")[0]
aud = pd.read_csv(glob.glob(PP + "/script19_final/*/rider_iou_audit.csv")[0]); prs = pd.read_csv(glob.glob(PP + "/script19_final/*/rider_iou_pairs.csv")[0])
man = pd.read_csv(glob.glob(PP + "/script20_final/*/input_manifest.csv")[0]); CT = {r.patient_id: R + "/Pulmonar_v281_2026-09-01/08_DADOS_INTERMEDIOS/" + r.ct_path.split("runs/", 1)[1] for r in man[man.acquisition == 1].itertuples()}
SHACT = {r.patient_id: r.ct_sha256 for r in man[man.acquisition == 1].itertuples()}
th = pd.read_csv(S18 + "/transform_hashes.csv") if os.path.exists(S18 + "/transform_hashes.csv") else None
BLK = BASE + "/Script10_canonical_rider/blocks"; BH = {}; POR = {}
for d in glob.glob(BLK + "/rider__*"):
    for h in csv.DictReader(open(d + "/BLOCK_HASHES.csv")):
        BH[(os.path.basename(d)[7:], h["file"])] = h["sha256"]
        if os.path.exists(d + "/" + h["file"]): POR.setdefault(h["sha256"], d + "/" + h["file"])
def pred(arch, nome):
    p = f"{BLK}/rider__{arch}/{nome}"; s = BH[(arch, nome)]
    if not os.path.exists(p): p = POR[s]
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == s, p
    return p
def iou(l1, n1, l2, n2):
    m = np.zeros((n1, n2))
    if n1 == 0 or n2 == 0: return m
    a, b = l1.ravel(), l2.ravel(); sa = np.bincount(a, minlength=n1 + 1)[1:]; sb = np.bincount(b, minlength=n2 + 1)[1:]
    ov = (a > 0) & (b > 0); k = np.bincount(a[ov].astype(np.int64) * (n2 + 1) + b[ov]);
    for q in np.nonzero(k)[0]:
        i, j = int(q // (n2 + 1)) - 1, int(q % (n2 + 1)) - 1; m[i, j] = k[q] / (sa[i] + sb[j] - k[q])
    return m
linhas, dif = [], []
for r in aud.itertuples():
    pid, arch = r.patient_id, r.architecture
    if r.n_ct1_native == 0 or r.n_ct2_native == 0:
        linhas.append(dict(patient_id=pid, architecture=arch, n1=r.n_ct1_native, n2=r.n_ct2_native, pares_A=0, pares_B=0, iguais=True, paridade_s19=True, comp_multiplos=0)); continue
    assert hashlib.sha256(open(CT[pid], "rb").read()).hexdigest() == SHACT[pid], CT[pid]; ct1 = sitk.ReadImage(CT[pid]); tx = sitk.ReadTransform(f"{S18}/transforms/{pid}.tfm")
    i1 = sitk.ReadImage(pred(arch, f"{pid}_1.nii.gz")); l1, n1 = nd.label(sitk.GetArrayFromImage(i1) > 0, structure=S6)
    i2 = sitk.ReadImage(pred(arch, f"{pid}_2.nii.gz")); l2, n2 = nd.label(sitk.GetArrayFromImage(i2) > 0, structure=S6)
    assert (n1, n2) == (r.n_ct1_native, r.n_ct2_native), (pid, arch, n1, n2)
    li = sitk.GetImageFromArray(l2.astype(np.int32)); li.CopyInformation(i2)
    l2r = sitk.GetArrayFromImage(sitk.Resample(sitk.Cast(li, sitk.sitkInt32), ct1, tx, sitk.sitkNearestNeighbor, 0, sitk.sitkInt32)).astype(np.int64)
    m = iou(l1.astype(np.int64), n1, l2r, n2)
    ra, ca = linear_sum_assignment(-m); A = {(int(a) + 1, int(b) + 1) for a, b in zip(ra, ca) if m[a, b] > 0}
    w = np.where(m > 0, 1000.0 + m, 0.0); rb, cb = linear_sum_assignment(-w); B = {(int(a) + 1, int(b) + 1) for a, b in zip(rb, cb) if m[a, b] > 0}
    s19 = set(zip(prs[(prs.patient_id == pid) & (prs.architecture == arch)].component_id_ct1, prs[(prs.patient_id == pid) & (prs.architecture == arch)].component_id_ct2_native))
    multi = int(((m > 0).sum(1) > 1).sum() + ((m > 0).sum(0) > 1).sum())
    linhas.append(dict(patient_id=pid, architecture=arch, n1=n1, n2=n2, pares_A=len(A), pares_B=len(B), iguais=A == B, paridade_s19=A == s19, comp_multiplos=multi))
    for x in sorted(A ^ B): dif.append(dict(patient_id=pid, architecture=arch, ct1=x[0], ct2=x[1], em_A=x in A, em_B=x in B, iou=float(m[x[0] - 1, x[1] - 1])))
    print(pid, arch, len(A), len(B), A == B, A == s19, flush=True)
pd.DataFrame(linhas).to_csv(OUT + "/P01_01_rider_AB.csv", index=False); pd.DataFrame(dif, columns=["patient_id", "architecture", "ct1", "ct2", "em_A", "em_B", "iou"]).to_csv(OUT + "/P01_01_pares_diferentes.csv", index=False)
L = pd.DataFrame(linhas); print("FIM pares A", L.pares_A.sum(), "B", L.pares_B.sum(), "combinações diferentes", (~L.iguais).sum(), "paridade S19 falhas", (~L.paridade_s19).sum())
