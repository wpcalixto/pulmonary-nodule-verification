#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    P01_11_diag45.py

File type:
    Script

Purpose:
    Diagnosis of the patients whose reconstructed groupings differ from the consensus.
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
    - *.nii.gz
    - consensus.nii.gz
    - crop_transform.json

Outputs:
    - Tables written to the stage output directory (see docs/post_v1_stages.md).

Dependencies:
    - SimpleITK
    - numpy

Execution:
    cd <project_dir> && python scripts/32_supplementary_analyses/c_reference_unit_and_fragments/P01_11_diag45.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import sys, csv, glob, re, json, collections, numpy as np, SimpleITK as sitk
sys.argv = [sys.argv[0], "tmp/_p01_11_x"] + sys.argv[1:]
exec(open(__file__.replace("P01_11_diag45.py", "P01_11_unidade_referencia.py")).read().split("def um(pid):")[0])
pids = [r["pid"] for r in csv.DictReader(open(sys.argv[2])) if r["uniao_agrupamentos_igual_consenso"] == "False"]
w = csv.writer(open(sys.argv[3], "w")); w.writerow(["pid", "cluster", "limiar", "n_vox_esperado", "n_vox_obtido", "n_sessoes", "cons_vox", "uniao_vox", "so_consenso", "so_uniao"])
for pid in pids:
    ct = json.load(open(f"{R}/pulmonar_canonical_pre_oof/recrops/images/{pid}/crop_transform.json")); z0, z1, y0, y1, x0, x1 = ct["crop_bbox"]
    cons = sitk.GetArrayFromImage(sitk.ReadImage(f"{ARQ}/consensus/lidc/{pid}/consensus.nii.gz")) > 0
    por = collections.defaultdict(dict)
    for f in sorted(glob.glob(f"{ARQ}/per_reader/lidc/{pid}/*.nii.gz")):
        k = int(re.search(r"nodule(\d+)", f).group(1)); s = re.search(r"(rec\w+_session\d+)", f).group(1)
        a = sitk.GetArrayFromImage(sitk.ReadImage(f)) > 0; por[k][s] = a | por[k].get(s, a)
    uni = np.zeros(cons.shape, bool); linhas = []
    for k, (t, nv) in TH[pid].items():
        ms = list(por[k].values()); c = (np.sum(ms, axis=0) >= t) if ms else np.zeros(cons.shape, bool); uni |= c
        linhas.append([pid, k, t, nv, int(c.sum()), len(ms)])
    a, b = cons[z0:z1, y0:y1, x0:x1], uni[z0:z1, y0:y1, x0:x1]
    for l in linhas: w.writerow(l + [int(a.sum()), int(b.sum()), int((a & ~b).sum()), int((b & ~a).sum())])
    print(pid, int(a.sum()), int(b.sum()), int((a & ~b).sum()), int((b & ~a).sum()), flush=True)
