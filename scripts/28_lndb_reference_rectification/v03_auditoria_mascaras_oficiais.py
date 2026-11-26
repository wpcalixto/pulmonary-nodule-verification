#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    v03_auditoria_mascaras_oficiais.py

File type:
    Script

Purpose:
    Compares the official per-radiologist LNDb masks with the previous local majority reference.
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
    - trainCTs.csv
    - trainNodules.csv
    - trainNodules_gt.csv

Outputs:
    - v03_duplicatas.csv
    - v03_nodulos_gt_por_exame.csv
    - v03_por_exame_236.csv
    - v03_resumo.json

Dependencies:
    - numpy
    - scipy

Execution:
    cd <project_dir> && python scripts/28_lndb_reference_rectification/v03_auditoria_mascaras_oficiais.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import csv, json, os, hashlib, glob, re, sys
import numpy as np
from scipy import ndimage
M = "tmp/lndb_inv/masks"; P = "."
RAW = P + "/dados/brutos/raw_5070_2026-08/lndb"; O = P + "/resultados/investigacao_V03_LNDb_20261111/saidas"; os.makedirs(O, exist_ok=True)
man = json.load(open(P + "/resultados/pulmonar_canonical_pre_oof/cohort/lndb_annotations_manifest.json"))
rows = man["rows"]; R = {r["lndb_id"]: r for r in rows}
s22 = {r["scan_id"] for r in csv.DictReader(open(P + "/resultados/pulmonar_pos_oof_prep/script22_final/s22_20260924T084718_826234/input_manifest.csv"))}
radn = {f"LNDb-{int(r['LNDbID']):04d}": int(r["RadN"]) for r in csv.DictReader(open(RAW + "/trainCTs.csv"))}
tn = list(csv.DictReader(open(RAW + "/trainNodules.csv"))); tg = list(csv.DictReader(open(RAW + "/trainNodules_gt.csv")))
def hdr(p):
    h = {}
    for l in open(p):
        if "=" in l: k, v = l.split("=", 1); h[k.strip()] = v.strip()
    return h
def geom(h): return (h["DimSize"], h["ElementSpacing"], h["Offset"], h["TransformMatrix"])
def md5(p):
    m = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): m.update(b)
    return m.hexdigest()
def um_exame(e):
    hc = hdr(f"{RAW}data/{e}.mhd"); files = sorted(glob.glob(f"{M}/{e}_rad*.mhd")); N = len(files)
    acc = None; per = []
    for f in files:
        h = hdr(f); dim = [int(x) for x in h["DimSize"].split()]
        a = np.fromfile(f.replace(".mhd", ".raw"), dtype=np.uint8).reshape(dim[::-1])
        per.append(dict(r=int(re.search(r"_rad(\d)", f).group(1)), vox=int((a > 0).sum()), labels=sorted(int(x) for x in np.unique(a) if x), geom_ok=geom(h) == geom(hc), lab=a))
        acc = (a > 0).astype(np.uint8) if acc is None else acc + (a > 0)
    thr = max(1, round(0.5 * N)) if N else None
    maj = (acc >= thr) if N else None
    st6 = ndimage.generate_binary_structure(3, 1)
    r = dict(lndb_id=e, grupo=("incluido_211" if e in s22 else ("referencia_vazia_13" if R[e]["analytic_status"] == "EXCLUDED_REFERENCE_READER_DISCORDANCE" else "duplicata_nao_representante_12")),
             radN_trainCTs=radn.get(e), mascaras_oficiais=N, readers_documented_local=R[e]["n_readers_documented"], geometria_mascaras_igual_CT=all(p["geom_ok"] for p in per),
             vox_por_leitor=";".join(f"rad{p['r']}:{p['vox']}" for p in per), labels_por_leitor=";".join(f"rad{p['r']}:{'|'.join(map(str, p['labels']))}" for p in per),
             vox_uniao=int((acc >= 1).sum()) if N else 0, vox_ge2=int((acc >= 2).sum()) if N else 0, vox_ge3=int((acc >= 3).sum()) if N else 0,
             limiar_regra_local=thr, vox_regra_local=int(maj.sum()) if N else 0, comp_regra_local=int(ndimage.label(maj, st6)[1]) if N else 0,
             comp_uniao=int(ndimage.label(acc >= 1, st6)[1]) if N else 0, n_vox_consensus_artigo=int(R[e]["n_vox_consensus"]), n_comp_consensus_artigo=int(R[e]["n_components_consensus"]))
    r["regra_local_reproduz_referencia_artigo"] = r["vox_regra_local"] == r["n_vox_consensus_artigo"]
    r["uniao_igual_referencia_artigo"] = r["vox_uniao"] == r["n_vox_consensus_artigo"]
    nn = []
    for g in [g for g in tg if f"LNDb-{int(g['LNDbID']):04d}" == e]:
        rads = [int(x) for x in str(g["RadID"]).split(",")]; rf = [int(float(x)) for x in str(g["RadFindingID"]).split(",")]
        pres = []
        for rr, ff in zip(rads, rf):
            p = [q for q in per if q["r"] == rr]
            pres.append(f"rad{rr}:F{ff}:{int((p[0]['lab'] == ff).sum()) if p else 'sem_mascara'}")
        dq = 2 * (3 * float(g["Volume"]) / (4 * np.pi)) ** (1 / 3)
        x, y, z = float(g["x"]), float(g["y"]), float(g["z"]); off = [float(v) for v in hc["Offset"].split()]; sp = [float(v) for v in hc["ElementSpacing"].split()]
        i, j, k = [int(round((c - o) / s)) for c, o, s in zip((x, y, z), off, sp)]
        ok = N and 0 <= k < acc.shape[0] and 0 <= j < acc.shape[1] and 0 <= i < acc.shape[2]
        nn.append(dict(lndb_id=e, grupo=r["grupo"], FindingID=g["FindingID"], AgrLevel=g["AgrLevel"], Nodule=g["Nodule"], Volume=g["Volume"], d_eq_mm=round(dq, 2),
                       voxels_por_leitor=";".join(pres), centroide_na_uniao=bool(acc[k, j, i] >= 1) if ok else None, centroide_na_referencia_local=bool(maj[k, j, i]) if ok else None))
    print(e, r["grupo"], N, r["vox_uniao"], r["vox_regra_local"], r["n_vox_consensus_artigo"], r["regra_local_reproduz_referencia_artigo"], flush=True)
    return r, nn
from multiprocessing import Pool
with Pool(6) as pool: RES = pool.map(um_exame, sorted(R), chunksize=2)
out = [a for a, b in RES]; nods = [x for a, b in RES for x in b]
def wcsv(p, L):
    with open(p, "w", newline="") as f: w = csv.DictWriter(f, fieldnames=list(L[0])); w.writeheader(); w.writerows(L)
wcsv(O + "/v03_por_exame_236.csv", out); wcsv(O + "/v03_nodulos_gt_por_exame.csv", nods)
ip = man["intrapair"]; D = []
for q in ip:
    a, b = q["member_a"], q["member_b"]
    D.append(dict(par=q["duplicate_group_id"], a=a, b=b, representante=q["representative"], criterio=q["deciding_criterion"],
                  md5_raw_a=md5(f"{RAW}data/{a}.raw"), md5_raw_b=md5(f"{RAW}data/{b}.raw"), geometria_igual=geom(hdr(f"{RAW}data/{a}.mhd")) == geom(hdr(f"{RAW}data/{b}.mhd")),
                  radN_a=radn.get(a), radN_b=radn.get(b), vox_uniao_a=[x for x in out if x["lndb_id"] == a][0]["vox_uniao"], vox_uniao_b=[x for x in out if x["lndb_id"] == b][0]["vox_uniao"],
                  vox_ref_artigo_a=R[a]["n_vox_consensus"], vox_ref_artigo_b=R[b]["n_vox_consensus"]))
    D[-1]["imagem_identica"] = D[-1]["md5_raw_a"] == D[-1]["md5_raw_b"]
wcsv(O + "/v03_duplicatas.csv", D)
res = dict(exames=len(out), mascaras=sum(x["mascaras_oficiais"] for x in out), regra_local_reproduz_236=sum(x["regra_local_reproduz_referencia_artigo"] for x in out),
           uniao_igual_referencia=sum(x["uniao_igual_referencia_artigo"] for x in out), geometria_ok=sum(x["geometria_mascaras_igual_CT"] for x in out),
           radN_igual_mascaras=sum(x["radN_trainCTs"] == x["mascaras_oficiais"] for x in out))
for g in ["incluido_211", "referencia_vazia_13", "duplicata_nao_representante_12"]:
    q = [x for x in out if x["grupo"] == g]
    res[g] = dict(n=len(q), uniao_vazia=sum(x["vox_uniao"] == 0 for x in q), regra_local_vazia=sum(x["vox_regra_local"] == 0 for x in q),
                  por_n_leitores={n: sum(x["mascaras_oficiais"] == n for x in q) for n in (0, 1, 2, 3)},
                  vox_uniao=sum(x["vox_uniao"] for x in q), vox_regra_local=sum(x["vox_regra_local"] for x in q), comp_uniao=sum(x["comp_uniao"] for x in q), comp_regra_local=sum(x["comp_regra_local"] for x in q))
json.dump(res, open(O + "/v03_resumo.json", "w"), indent=1); print(json.dumps(res, indent=1))
