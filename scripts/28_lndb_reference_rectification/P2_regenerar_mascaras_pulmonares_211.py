#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    P2_regenerar_mascaras_pulmonares_211.py

File type:
    Script

Purpose:
    Regenerates the cropped lung masks of the 211 analytical LNDb scans with the validated stage-06 reproduction, accepting each mask only when its SHA-256 matches the recorded input manifest.
    Pipeline position: Stage 28 (LNDb N1 reference rectification and re-evaluation).

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
    - LNDB_INPUT_MANIFEST.csv
    - ct.nii.gz
    - lung_mask_crop.nii.gz

Outputs:
    - Tables written to the stage output directory (see docs/post_v1_stages.md).

Dependencies:
    - Python standard library only

Execution:
    cd <project_dir> && python scripts/28_lndb_reference_rectification/P2_regenerar_mascaras_pulmonares_211.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import os, sys, csv, json, subprocess, shutil, importlib.util
sp = importlib.util.spec_from_file_location("p1", os.path.join(os.path.dirname(os.path.abspath(__file__)), "P1_recorte_s06_replica.py")); P1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(P1)
SRV = "resultados"
P1.IN, P1.OUT = "tmp/s06rep/in2", "tmp/s06rep/out2"
man = list(csv.DictReader(open("tmp/s06rep/LNDB_INPUT_MANIFEST.csv")))
log = open("tmp/s06rep/P2.log", "a"); res = []
for r in man:
    e = r["case_id"]
    if os.path.exists(f"tmp/s06rep/P2_ok/{e}"): continue
    import time, hashlib
    src = f"tmp/s06rep/in_all/{e}/ct.nii.gz"
    while not (os.path.exists(src) and hashlib.sha256(open(src, "rb").read()).hexdigest() == r["sha256_ct_source"]): time.sleep(5)
    os.makedirs(f"{P1.IN}/{e}", exist_ok=True); shutil.copy(src, f"{P1.IN}/{e}/ct.nii.gz")
    x = P1.caso(e)
    ok = (x["sha_ct_fonte"] == r["sha256_ct_source"], x["sha_ct_crop"] == r["sha256_ct_crop"], x["sha_lung_mask_crop"] == r["sha256_lung_mask_crop"])
    x["ok"] = ok; res.append(x); log.write(json.dumps(x) + "\n"); log.flush()
    if all(ok):
        os.makedirs(f"tmp/s06rep/aceites/lung_masks/lndb/{e}", exist_ok=True)
        shutil.copy(f"{P1.OUT}/lung_masks/lndb/{e}/lung_mask_crop.nii.gz", f"tmp/s06rep/aceites/lung_masks/lndb/{e}/lung_mask_crop.nii.gz")
        os.makedirs("tmp/s06rep/P2_ok", exist_ok=True); open(f"tmp/s06rep/P2_ok/{e}", "w").write("ok")
    shutil.rmtree(f"{P1.IN}/{e}", ignore_errors=True); shutil.rmtree(f"{P1.OUT}/lndb/images/{e}", ignore_errors=True); shutil.rmtree(f"{P1.OUT}/lung_masks/lndb/{e}", ignore_errors=True)
    os.remove(src)
    print(e, ok, flush=True)
subprocess.run(["rsync", "-a", "tmp/s06rep/aceites/", f"{SRV}/retificacao_LNDb_20261111/s06_mascaras/"], check=True)
print("FIM")
