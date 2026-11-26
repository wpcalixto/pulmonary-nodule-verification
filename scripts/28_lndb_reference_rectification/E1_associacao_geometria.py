#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    E1_associacao_geometria.py

File type:
    Script

Purpose:
    One-to-one Hungarian matching of predicted objects and LNDb reference lesions per scan, configuration, reference and threshold, using the matching functions of stages 11 and 22.
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
    - input_manifest.csv
    - lndb_annotations_manifest.json
    - lndb_object_matches.csv
    - lndb_verifier_object_scores.csv
    - trainCTs.csv
    - {e}.mhd
    - {e}.nii.gz

Outputs:
    - E1_contagens_por_exame.csv
    - E1_objetos_tau010.csv
    - E1_portas.json
    - F_escores_pred13.csv
    - R2_contagens_por_exame.csv
    - SHA256_pred13.json

Dependencies:
    - SimpleITK
    - numpy
    - pandas
    - scipy

Execution:
    cd <project_dir> && python scripts/28_lndb_reference_rectification/E1_associacao_geometria.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import csv, json, sys, os, hashlib, ast, math, types
import numpy as np, scipy.ndimage as nd, SimpleITK as sitk, pandas as pd
from scipy.optimize import linear_sum_assignment
from collections import Counter
from multiprocessing import Pool
P = "."; R = P + "/resultados"; O = R + "/retificacao_LNDb_20261111"
S22D = R + "/pulmonar_pos_oof_prep/script22_final/s22_20260924T084718_826234"
PRED211 = R + "/Pulmonary_Project/Script10_canonical_lndb/blocks/lndb__{a}/{e}.nii.gz"; PRED13 = O + "/pred13/{a}/{e}.nii.gz"
CT = P + "/dados/brutos/raw_5070_2026-08/lndb/data/{e}.mhd"
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def extrai(path, nomes, ns):
    src = open(path).read()
    for n in nomes: exec(ast.get_source_segment(src, [x for x in ast.parse(src).body if isinstance(x, ast.FunctionDef) and x.name == n][0]), ns)
    return ns
S11F = O + "/sandbox/scripts/11_match_and_score_bd1.py"; assert sha(S11F).startswith("438c50d3")
NS = extrai(S11F, ["build_iou_matrix"], {"np": np, "Counter": Counter})
S02F = R + "/analises_complementares_v8/00_scripts/s02_geometria_pares.py"; assert sha(S02F).startswith("938a5035")
STRUCT_6 = nd.generate_binary_structure(3, 1)
G = extrai(S02F, ["recorte", "superficie", "metricas_par"], {"np": np, "nd": nd, "math": math, "PAD": 2, "STRUCT_6": STRUCT_6})
LIM = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75]; ARCH = ["3D_F", "3D_FREM"]
def emparelhar_por_limiar(mat, n_gt, n_pred, theta):
    if n_gt == 0 or n_pred == 0: return set()
    w = np.where(mat >= theta, 1000.0 + mat, 0.0); rows, cols = linear_sum_assignment(-w)
    return {(int(r), int(c)) for r, c in zip(rows, cols) if mat[r, c] >= theta}
man = {r["scan_id"]: r for r in csv.DictReader(open(S22D + "/input_manifest.csv"))}
EX211 = sorted(man); EX13 = ["LNDb-0005", "LNDb-0007", "LNDb-0020", "LNDb-0037", "LNDb-0043", "LNDb-0044", "LNDb-0049", "LNDb-0109", "LNDb-0123", "LNDb-0124", "LNDb-0126", "LNDb-0171", "LNDb-0179"]
ret = {}
for r in csv.DictReader(open(R + "/pulmonar_pos_oof_prep/script23_final/s23_20260924T153750_909396/lndb_verifier_object_scores.csv")):
    ret[(r["architecture"], r["scan_id"], int(r["object_id"]))] = r["retained"] == "True"
for r in csv.DictReader(open(O + "/saidas/F_escores_pred13.csv")):
    ret[(r["configuracao"], r["exame"], int(r["id"]))] = r["retained"] == "True"
ann = json.load(open(R + "/pulmonar_canonical_pre_oof/cohort/lndb_annotations_manifest.json"))
ALT = {q["representative"]: q["excluded"] for q in ann["intrapair"]}
radn = {f"LNDb-{int(r['LNDbID']):04d}": int(r["RadN"]) for r in csv.DictReader(open(P + "/dados/brutos/raw_5070_2026-08/lndb/trainCTs.csv"))}
PRED13_SHA = json.load(open(O + "/pred13/SHA256_pred13.json"))
def geo(e):
    rd = sitk.ImageFileReader(); rd.SetFileName(CT.format(e=e)); rd.ReadImageInformation()
    img = sitk.Image([1, 1, 1], sitk.sitkUInt8); img.SetOrigin(rd.GetOrigin()); img.SetSpacing(rd.GetSpacing()); img.SetDirection(rd.GetDirection())
    return img, rd.GetSize(), rd.GetSpacing()
def encaixe(pr, oi, size):
    assert np.allclose(pr.GetSpacing(), oi.GetSpacing(), atol=1e-6) and np.allclose(pr.GetDirection(), oi.GetDirection(), atol=1e-6)
    ci = np.array(oi.TransformPhysicalPointToContinuousIndex(pr.GetOrigin())); assert np.abs(ci - np.round(ci)).max() <= 1e-3
    o = np.round(ci).astype(int); assert (o >= 0).all() and (o + np.array(pr.GetSize()) <= np.array(size)).all()
    return int(o[2]), int(o[1]), int(o[0])
def avaliar(gt_lab, n_gt, pr_lab, n_pred, ignorar, a, e_pred, rot, e_ref, coorte, sp_zyx, detalhe):
    mat = NS["build_iou_matrix"](gt_lab, pr_lab, n_gt, n_pred)
    toca = set(int(v) - 1 for v in np.unique(pr_lab[ignorar]) if v) if ignorar is not None else set()
    acc = {j for j in range(n_pred) if ret[(a, e_pred, j + 1)]}
    C, OBJ = [], []
    for t in LIM:
        pares = emparelhar_por_limiar(mat, n_gt, n_pred, t); pm = {c for _, c in pares}
        ign = {j for j in range(n_pred) if j not in pm and j in toca}
        tp = len(pares); tpp = sum(1 for _, c in pares if c in acc)
        C.append(dict(coorte=coorte, referencia=rot, configuracao=a, exame=e_ref, tau=t, n_ref=n_gt, n_pred=n_pred, ignoradas=len(ign), ignoradas_aceites=len(ign & acc),
                      TP=tp, FP=n_pred - tp - len(ign), FN=n_gt - tp, TP_pos=tpp, FP_pos=len(acc - pm - ign), FN_pos=n_gt - tpp, aceites=len(acc)))
        if t == 0.10 and detalhe:
            gp = {r_: c for r_, c in pares}; pg = {c: r_ for r_, c in pares}
            for j in range(n_pred):
                OBJ.append(dict(coorte=coorte, referencia=rot, configuracao=a, exame=e_ref, fonte="pred", id=j + 1, status="TP" if j in pg else ("IGN" if j in ign else "FP"),
                                par=(pg[j] + 1) if j in pg else "", iou=float(mat[pg[j], j]) if j in pg else 0.0, aceite=j in acc))
            for gi in range(n_gt):
                c = gp.get(gi); d = dict(coorte=coorte, referencia=rot, configuracao=a, exame=e_ref, fonte="ref", id=gi + 1, status="TP" if c is not None else "FN",
                                         par=(c + 1) if c is not None else "", iou=float(mat[gi, c]) if c is not None else 0.0, aceite=(c in acc) if c is not None else "")
                if c is not None:
                    Pm, Gm = pr_lab == c + 1, gt_lab == gi + 1; sl = G["recorte"](Pm, Gm); m = G["metricas_par"](Pm[sl], Gm[sl], sp_zyx)
                    d.update(dice=m["dice"], hd95=m["hd95"], dsm=m["dsm"])
                OBJ.append(d)
    return C, OBJ
def um(e):
    oi, size, spx = geo(e); sp_zyx = (spx[2], spx[1], spx[0]); Z0 = np.load(f"{O}/referencias/{e}.npz"); C, OB = [], []
    coorte = "211" if e in EX211 else "13"
    for a in ARCH:
        if e in EX211:
            p = PRED211.format(a=a, e=e)
            if not os.path.exists(p):
                alt_p = PRED211.format(a=[x for x in ARCH if x != a][0], e=e)
                assert sha(alt_p) == man[e][f"prediction_{a}_sha256"]; p = alt_p
            assert sha(p) == man[e][f"prediction_{a}_sha256"]
        else:
            p = PRED13.format(a=a, e=e); assert sha(p) == PRED13_SHA[f"{a}/{e}"]
        pr = sitk.ReadImage(p); arr = sitk.GetArrayFromImage(pr); assert np.isin(arr, (0, 1)).all()
        lab_c, n_pred = nd.label(arr > 0, structure=STRUCT_6); z0, y0, x0 = encaixe(pr, oi, size); Zs, Ys, Xs = lab_c.shape
        pr_lab = np.zeros(Z0["N1"].shape, np.int32); pr_lab[z0:z0 + Zs, y0:y0 + Ys, x0:x0 + Xs] = lab_c
        if e in EX211:
            old, n_old = nd.label(Z0["ref_antiga"], structure=STRUCT_6); c, ob = avaliar(old, n_old, pr_lab, n_pred, None, a, e, "antiga", e, coorte, sp_zyx, True); C += c; OB += ob
        c, ob = avaliar(Z0["N1"].astype(np.int32), int(Z0["N1"].max()), pr_lab, n_pred, Z0["ignorar_N1"], a, e, "N1", e, coorte, sp_zyx, True); C += c; OB += ob
        if radn[e] >= 2:
            c, ob = avaliar(Z0["N2"].astype(np.int32), int(Z0["N2"].max()), pr_lab, n_pred, Z0["ignorar_N2"], a, e, "N2", e, coorte, sp_zyx, True); C += c; OB += ob
        alt = ALT.get(e, e); Za = Z0 if alt == e else np.load(f"{O}/referencias/{alt}.npz")
        c, _ = avaliar(Za["N1"].astype(np.int32), int(Za["N1"].max()), pr_lab, n_pred, Za["ignorar_N1"], a, e, "N1_alt", alt, coorte, sp_zyx, False); C += c
        if radn[alt] >= 2:
            c, _ = avaliar(Za["N2"].astype(np.int32), int(Za["N2"].max()), pr_lab, n_pred, Za["ignorar_N2"], a, e, "N2_alt", alt, coorte, sp_zyx, False); C += c
    print(e, flush=True); return C, OB
if __name__ == "__main__":
    with Pool(7) as pool: RES = pool.map(um, EX211 + EX13, chunksize=1)
    C = pd.DataFrame([x for a, b in RES for x in a]); OB = pd.DataFrame([x for a, b in RES for x in b])
    C.to_csv(O + "/saidas/E1_contagens_por_exame.csv", index=False); OB.to_csv(O + "/saidas/E1_objetos_tau010.csv", index=False)
    s22 = pd.read_csv(S22D + "/lndb_object_matches.csv"); s22 = s22[s22.status.isin(["TP", "FN"])]
    mine = OB[(OB.referencia == "antiga") & (OB.fonte == "ref")].set_index(["configuracao", "exame", "id"])
    f1 = 0
    for _, r in s22.iterrows():
        k = (r.architecture, r.scan_id, int(r.gt_component_id))
        if k not in mine.index or mine.loc[k, "status"] != r.status or (r.status == "TP" and (int(mine.loc[k, "par"]) != int(r.pred_component_id) or abs(mine.loc[k, "iou"] - r.iou) > 1e-12)): f1 += 1
    r2 = pd.read_csv(O + "/saidas/R2_contagens_por_exame.csv"); e1 = C[C.coorte == "211"]
    cols = ["referencia", "configuracao", "exame", "tau", "n_ref", "n_pred", "ignoradas", "TP", "FP", "FN", "TP_pos", "FP_pos", "FN_pos"]
    mm = r2[cols].merge(e1[cols], on=["referencia", "configuracao", "exame", "tau"], suffixes=("_r2", "_e1"))
    f2 = int(sum((mm[c + "_r2"] != mm[c + "_e1"]).sum() for c in cols[4:])) + abs(len(mm) - len(r2))
    porta = dict(s22_objetos=len(s22), falhas_s22=f1, linhas_R2=len(r2), linhas_cruzadas=len(mm), falhas_R2=f2, aprovada=f1 == 0 and f2 == 0)
    json.dump(porta, open(O + "/saidas/E1_portas.json", "w"), indent=1); print(json.dumps(porta))
    if not porta["aprovada"]: sys.exit("PORTA E1 FALHOU")
