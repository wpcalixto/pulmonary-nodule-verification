#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    v03_componentes_por_tipo_de_achado.py

File type:
    Script

Purpose:
    Classifies each reference component by the type of official finding that covers it.
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
    - input_manifest.csv
    - lndb_annotations_manifest.json
    - trainNodules.csv

Outputs:
    - v03_achados_por_leitor.csv
    - v03_componentes_referencia_local.csv
    - v03_componentes_resumo.json

Dependencies:
    - numpy
    - scipy

Execution:
    cd <project_dir> && python scripts/28_lndb_reference_rectification/v03_componentes_por_tipo_de_achado.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import csv, json, os, glob, re
import numpy as np
from scipy import ndimage
from multiprocessing import Pool
M = "tmp/lndb_inv/masks"; P = "."; RAW = P + "/dados/brutos/raw_5070_2026-08/lndb"
O = P + "/resultados/investigacao_V03_LNDb_20261111/saidas"
s22 = {r["scan_id"] for r in csv.DictReader(open(P + "/resultados/pulmonar_pos_oof_prep/script22_final/s22_20260924T084718_826234/input_manifest.csv"))}
man = json.load(open(P + "/resultados/pulmonar_canonical_pre_oof/cohort/lndb_annotations_manifest.json")); R = {r["lndb_id"]: r for r in man["rows"]}
TN = {}
for r in csv.DictReader(open(RAW + "/trainNodules.csv")):
    TN[(f"LNDb-{int(r['LNDbID']):04d}", int(r["RadID"]), int(r["FindingID"]))] = (int(r["Nodule"]), float(r["Volume"]))
def grupo(e): return "incluido_211" if e in s22 else ("referencia_vazia_13" if R[e]["analytic_status"] == "EXCLUDED_REFERENCE_READER_DISCORDANCE" else "duplicata_nao_representante_12")
def um(e):
    fs = sorted(glob.glob(f"{M}/{e}_rad*.mhd")); N = len(fs); labs = {}
    for f in fs:
        dim = [int(x) for x in re.search(r"DimSize = (.*)", open(f).read()).group(1).split()]
        labs[int(re.search(r"_rad(\d)", f).group(1))] = np.fromfile(f.replace(".mhd", ".raw"), dtype=np.uint8).reshape(dim[::-1])
    acc = sum((a > 0).astype(np.uint8) for a in labs.values())
    ach = []
    for r, a in labs.items():
        ids, cnt = np.unique(a[a > 0], return_counts=True)
        for i, c in zip(ids, cnt):
            nod, vol = TN.get((e, r, int(i)), (None, None))
            ach.append(dict(lndb_id=e, grupo=grupo(e), rad=r, FindingID=int(i), Nodule=nod, Volume=vol, voxels=int(c), tipo=("segmentacao" if c > 1 else "marca_pontual")))
    seg = {(a["rad"], a["FindingID"]): a for a in ach}
    comps = []
    for nome, mask in [("referencia_local", acc >= max(1, round(0.5 * N))), ("uniao", acc >= 1)]:
        lab, n = ndimage.label(mask, ndimage.generate_binary_structure(3, 1))
        if n == 0: continue
        objs = ndimage.find_objects(lab)
        for k, sl in enumerate(objs, 1):
            m = lab[sl] == k; cov = set()
            for r, a in labs.items():
                for i in np.unique(a[sl][m]):
                    if i: cov.add((r, int(i)))
            nods = [seg[c] for c in cov if seg[c]["Nodule"] == 1]; nn = [seg[c] for c in cov if seg[c]["Nodule"] == 0]
            if any(x["tipo"] == "segmentacao" for x in nods): cl = "nodulo_segmentado"
            elif nods: cl = "nodulo_marca_pontual"
            elif nn: cl = "nao_nodulo"
            else: cl = "sem_achado"
            comps.append(dict(lndb_id=e, grupo=grupo(e), mascara=nome, n_leitores=N, componente=k, voxels=int(m.sum()), achados=";".join(f"rad{r}:F{i}" for r, i in sorted(cov)), classe=cl))
    return comps, ach
with Pool(6) as p: RES = p.map(um, sorted(R), chunksize=2)
C = [x for a, b in RES for x in a]; A = [x for a, b in RES for x in b]
def w(p, L):
    with open(p, "w", newline="") as f: d = csv.DictWriter(f, fieldnames=list(L[0])); d.writeheader(); d.writerows(L)
w(O + "/v03_componentes_referencia_local.csv", C); w(O + "/v03_achados_por_leitor.csv", A)
from collections import Counter
res = {}
for g in ["incluido_211", "referencia_vazia_13", "duplicata_nao_representante_12"]:
    for m in ["referencia_local", "uniao"]:
        q = [c for c in C if c["grupo"] == g and c["mascara"] == m]
        res[f"{g}|{m}"] = dict(componentes=len(q), por_classe=dict(Counter(c["classe"] for c in q)),
                               por_classe_e_leitores={f"{k[0]}|N={k[1]}": v for k, v in Counter((c["classe"], c["n_leitores"]) for c in q).items()},
                               componentes_1_voxel=sum(c["voxels"] == 1 for c in q))
res["achados_por_tipo"] = {f"{k[0]}|Nodule={k[1]}|{k[2]}": v for k, v in Counter((a["grupo"], a["Nodule"], a["tipo"]) for a in A).items()}
res["achados_sem_linha_em_trainNodules"] = sum(a["Nodule"] is None for a in A)
json.dump(res, open(O + "/v03_componentes_resumo.json", "w"), indent=1); print(json.dumps(res, indent=1))
