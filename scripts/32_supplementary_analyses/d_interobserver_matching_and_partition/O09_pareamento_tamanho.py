#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    O09_pareamento_tamanho.py

File type:
    Script

Purpose:
    RIDER test-retest matching rate by configuration and object size.
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
    - input_manifest.csv
    - rider_iou_audit.csv
    - rider_iou_unmatched.csv

Outputs:
    - O09_objetos.csv
    - O09_resumo.csv
    - rider_iou_pairs.csv
    - rider_pair_features.csv

Dependencies:
    - SimpleITK
    - numpy
    - pandas
    - scipy

Execution:
    cd <project_dir> && python scripts/32_supplementary_analyses/d_interobserver_matching_and_partition/O09_pareamento_tamanho.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import os, csv, glob, hashlib, sys, numpy as np, scipy.ndimage as nd, SimpleITK as sitk, pandas as pd
R = "resultados"; PP = R + "/pulmonar_pos_oof_prep"; BASE = R + "/Pulmonary_Project"
OUT = sys.argv[1]; S6 = nd.generate_binary_structure(3, 1)
man = pd.read_csv(glob.glob(PP + "/script20_final/*/input_manifest.csv")[0])
pairs = pd.read_csv(glob.glob(PP + "/script19_final/*/rider_iou_pairs.csv")[0]); un = pd.read_csv(glob.glob(PP + "/script19_final/*/rider_iou_unmatched.csv")[0])
feat = pd.read_csv(glob.glob(PP + "/script20_final/*/rider_pair_features.csv")[0])
rows = []
aud = pd.read_csv(glob.glob(PP + "/script19_final/*/rider_iou_audit.csv")[0])
BLK = BASE + "/Script10_canonical_rider/blocks"; BH = {}; POR_SHA = {}
for d in sorted(glob.glob(BLK + "/rider__*")):
    for h in csv.DictReader(open(d + "/BLOCK_HASHES.csv")):
        BH[(os.path.basename(d)[7:], h["file"])] = h["sha256"]
        if os.path.exists(d + "/" + h["file"]): POR_SHA.setdefault(h["sha256"], d + "/" + h["file"])
SUBST = []
for r in aud.itertuples():
    for q in (1, 2):
        n_esp = r.n_ct1_native if q == 1 else r.n_ct2_native
        if n_esp == 0: continue
        nome = f"{r.patient_id}_{q}.nii.gz"; sha_esp = BH[(r.architecture, nome)]; p = f"{BLK}/rider__{r.architecture}/{nome}"
        if not os.path.exists(p): p = POR_SHA[sha_esp]; SUBST.append((r.patient_id, r.architecture, q, p))
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == sha_esp, p
        img = sitk.ReadImage(p); arr = sitk.GetArrayFromImage(img) > 0; sx, sy, sz = img.GetSpacing()
        lab, n = nd.label(arr, structure=S6); assert n == n_esp, (p, n, n_esp)
        for k, sl in enumerate(nd.find_objects(lab), start=1):
            m = lab[sl] == k; best = 0.0
            for z in range(m.shape[0]):
                ys, xs = np.nonzero(m[z])
                if ys.size: best = max(best, (ys.max() - ys.min() + 1) * sy, (xs.max() - xs.min() + 1) * sx)
            rows.append(dict(patient_id=r.patient_id, architecture=r.architecture, acquisition=q, component_id=k, n_vox=int(m.sum()), vol_mm3=float(m.sum() * sx * sy * sz), diam_mm=best))
print("substituições por ficheiro de SHA idêntico:", SUBST)
O = pd.DataFrame(rows)
p1 = set(zip(pairs.patient_id, pairs.architecture, pairs.component_id_ct1)); p2 = set(zip(pairs.patient_id, pairs.architecture, pairs.component_id_ct2_native))
O["pareado"] = [(a, b, c) in (p1 if q == 1 else p2) for a, b, c, q in zip(O.patient_id, O.architecture, O.component_id, O.acquisition)]
chk = O[(O.acquisition == 1) & O.pareado].merge(feat, left_on=["patient_id", "architecture", "component_id"], right_on=["patient_id", "architecture", "component_id_ct1"])
dif = float((chk.vol_mm3 - chk.volume_ct1_mm3).abs().max()); n_un = len(un); n_obj_unp = int((~O.pareado).sum())
print("paridade:", len(chk), len(pairs), dif, n_un, n_obj_unp)
O["estrato"] = pd.cut(O.diam_mm, [0, 3, 6, 15, 1e9], right=False, labels=["<3", "3-6", "6-15", ">=15"])
O.to_csv(OUT + "/O09_objetos.csv", index=False)
S = O.groupby(["architecture", "acquisition", "estrato"], observed=False).agg(n=("pareado", "size"), pareados=("pareado", "sum")).reset_index(); S["taxa"] = S.pareados / S.n
T = O.groupby(["acquisition", "estrato"], observed=False).agg(n=("pareado", "size"), pareados=("pareado", "sum")).reset_index(); T["architecture"] = "TODAS"; T["taxa"] = T.pareados / T.n
pd.concat([S, T]).to_csv(OUT + "/O09_resumo.csv", index=False); print(T.round(3).to_string())
