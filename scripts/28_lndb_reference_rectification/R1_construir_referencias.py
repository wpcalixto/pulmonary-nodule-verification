#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    R1_construir_referencias.py

File type:
    Script

Purpose:
    Builds, from the official per-radiologist LNDb masks, the lesion-level references of the 236 scans: the N1 reference (union of the segmentations of each official lesion), the N2 reference and the regions to be ignored.
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
    - _rad*.mhd
    - trainCTs.csv
    - trainNodules.csv
    - trainNodules_gt.csv

Outputs:
    - exames_236.csv
    - lesoes_236.csv
    - resumo_R1.json

Dependencies:
    - numpy

Execution:
    cd <project_dir> && python scripts/28_lndb_reference_rectification/R1_construir_referencias.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import csv, json, glob, re, os
import numpy as np
from multiprocessing import Pool
M = "tmp/lndb_inv/masks"; P = "."; RAW = P + "/dados/brutos/raw_5070_2026-08/lndb"
O = P + "/resultados/retificacao_LNDb_20261111/referencias"; os.makedirs(O, exist_ok=True)
radn = {f"LNDb-{int(r['LNDbID']):04d}": int(r["RadN"]) for r in csv.DictReader(open(RAW + "/trainCTs.csv"))}
TN = {(f"LNDb-{int(r['LNDbID']):04d}", int(r["RadID"]), int(r["FindingID"])): int(r["Nodule"]) for r in csv.DictReader(open(RAW + "/trainNodules.csv"))}
GT = {}
for r in csv.DictReader(open(RAW + "/trainNodules_gt.csv")):
    GT.setdefault(f"LNDb-{int(r['LNDbID']):04d}", []).append(r)
def um(e):
    fs = sorted(glob.glob(f"{M}/{e}_rad*.mhd")); labs = {}
    for f in fs:
        dim = [int(x) for x in re.search(r"DimSize = (.*)", open(f).read()).group(1).split()]
        labs[int(re.search(r"_rad(\d)", f).group(1))] = np.fromfile(f.replace(".mhd", ".raw"), dtype=np.uint8).reshape(dim[::-1])
    N = len(labs); shp = next(iter(labs.values())).shape
    acc = sum((a > 0).astype(np.uint8) for a in labs.values())
    ref_antiga = acc >= max(1, round(0.5 * N))
    les, usados = [], set()
    for g in GT.get(e, []):
        rads = [int(x) for x in str(g["RadID"]).split(",")]; rfs = [int(float(x)) for x in str(g["RadFindingID"]).split(",")]
        vox_l, nod_l, seg_l = [], 0, 0
        for r, f in zip(rads, rfs):
            usados.add((r, f)); m = labs[r] == f; n = int(m.sum()); vox_l.append(n)
            if TN[(e, r, f)] == 1:
                nod_l += 1; seg_l += int(n > 1)
        les.append(dict(lndb_id=e, FindingID=int(g["FindingID"]), AgrLevel=int(g["AgrLevel"]), Nodule_gt=int(g["Nodule"]), leitores=len(rads),
                        leitores_nodulo=nod_l, leitores_segmentacao=seg_l, voxels_por_leitor="|".join(map(str, vox_l)),
                        classe=("nodulo_segmentado" if nod_l and seg_l else ("nodulo_marca_pontual" if nod_l else "nao_nodulo")),
                        rads=rads, rfs=rfs))
    orf = sorted({(r, int(v)) for r, a in labs.items() for v in np.unique(a) if v} - usados)
    def mapa(sel):
        lab = np.zeros(shp, np.int16); cov = np.zeros(shp, np.uint8); sobre = 0
        for i, L in enumerate(sel, 1):
            m = np.zeros(shp, bool); c = np.zeros(shp, np.uint8)
            for r, f in zip(L["rads"], L["rfs"]):
                mm = labs[r] == f; m |= mm; c += mm
            conf = m & (lab > 0); sobre += int(conf.sum())
            ganha = m & ((lab == 0) | (c > cov))
            lab[ganha] = i; cov[ganha] = c[ganha]
        return lab, sobre
    n1 = [L for L in les if L["classe"] == "nodulo_segmentado"]
    n2 = [L for L in n1 if L["leitores_nodulo"] >= 2]
    lab1, s1 = mapa(n1); lab2, s2 = mapa(n2) if N >= 2 else (np.zeros(shp, np.int16), 0)
    ig1 = np.zeros(shp, bool)
    for L in les:
        if L["classe"] == "nodulo_marca_pontual":
            for r, f in zip(L["rads"], L["rfs"]): ig1 |= labs[r] == f
    ig2 = ig1.copy()
    for L in n1:
        if L["leitores_nodulo"] < 2:
            for r, f in zip(L["rads"], L["rfs"]): ig2 |= labs[r] == f
    np.savez_compressed(f"{O}/{e}.npz", ref_antiga=ref_antiga, N1=lab1, N2=lab2, ignorar_N1=ig1, ignorar_N2=ig2,
                        N1_findings=np.array([L["FindingID"] for L in n1], np.int32), N2_findings=np.array([L["FindingID"] for L in n2], np.int32))
    for L in les: L["em_N1"] = L in n1; L["em_N2"] = L in n2 and N >= 2; L.pop("rads"); L.pop("rfs")
    res = dict(lndb_id=e, leitores=N, radN=radn.get(e), elegivel_N2=N >= 2, vox_ref_antiga=int(ref_antiga.sum()), lesoes_N1=len(n1), lesoes_N2=len(n2) if N >= 2 else 0,
               marcas_nodulo=sum(L["classe"] == "nodulo_marca_pontual" for L in les), nao_nodulos=sum(L["classe"] == "nao_nodulo" for L in les),
               voxels_sobrepostos_N1=s1, voxels_sobrepostos_N2=s2, achados_leitor_sem_agrupamento=len(orf),
               nodule_gt_discorda=sum((L["Nodule_gt"] == 1) != (L["leitores_nodulo"] >= 1) for L in les))
    print(e, json.dumps(res), flush=True)
    return res, les
with Pool(6) as p: R = p.map(um, sorted(radn), chunksize=2)
E = [a for a, b in R]; L = [x for a, b in R for x in b]
def w(p, X):
    with open(p, "w", newline="") as f: d = csv.DictWriter(f, fieldnames=list(X[0])); d.writeheader(); d.writerows(X)
w(O + "/exames_236.csv", E); w(O + "/lesoes_236.csv", L)
tot = {k: sum(x[k] for x in E) for k in ["lesoes_N1", "lesoes_N2", "marcas_nodulo", "nao_nodulos", "voxels_sobrepostos_N1", "voxels_sobrepostos_N2", "achados_leitor_sem_agrupamento", "nodule_gt_discorda"]}
tot["exames"] = len(E); tot["elegiveis_N2"] = sum(x["elegivel_N2"] for x in E)
json.dump(tot, open(O + "/resumo_R1.json", "w"), indent=1); print(json.dumps(tot, indent=1))
