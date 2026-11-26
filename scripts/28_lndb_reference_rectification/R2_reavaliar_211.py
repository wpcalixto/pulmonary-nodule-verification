#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    R2_reavaliar_211.py

File type:
    Script

Purpose:
    Re-evaluates the frozen LNDb predictions and the frozen Model A decisions against the N1 and N2 references, with a control that reproduces the previous reference.
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
    - lndb_scan_audit.csv
    - lndb_verifier_object_scores.csv
    - trainCTs.csv
    - {e}.mhd
    - {e}.nii.gz

Outputs:
    - R2_contagens_por_exame.csv
    - R2_objetos_referencia_tau010.csv
    - R2_porta_referencia_antiga.json

Dependencies:
    - SimpleITK
    - numpy
    - scipy

Execution:
    cd <project_dir> && python scripts/28_lndb_reference_rectification/R2_reavaliar_211.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import csv, json, sys, os, hashlib, importlib.util
import numpy as np, scipy.ndimage as nd, SimpleITK as sitk
from scipy.optimize import linear_sum_assignment
from multiprocessing import Pool
P = "."; R = P + "/resultados"; O = R + "/retificacao_LNDb_20261111"
S22D = R + "/pulmonar_pos_oof_prep/script22_final/s22_20260924T084718_826234"
PRED = R + "/Pulmonary_Project/Script10_canonical_lndb/blocks/lndb__{a}/{e}.nii.gz"
CT = P + "/dados/brutos/raw_5070_2026-08/lndb/data/{e}.mhd"
sp = importlib.util.spec_from_file_location("s11", P + "/arquivo/Ultima_Versao/Scripts/novos/11_match_and_score_bd1.py")
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
assert sha(sp.origin) .startswith("438c50d3"), "Script 11 canónico divergente"
import ast, types
from collections import Counter
_src = open(sp.origin).read(); _fn = [x for x in ast.parse(_src).body if isinstance(x, ast.FunctionDef) and x.name == "build_iou_matrix"][0]
S11 = types.SimpleNamespace(); _ns = {"np": np, "Counter": Counter}; exec(ast.get_source_segment(_src, _fn), _ns); S11.build_iou_matrix = _ns["build_iou_matrix"]
STRUCT_6 = nd.generate_binary_structure(3, 1)
LIM = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75]; ARCH = ["3D_F", "3D_FREM"]
def emparelhar_por_limiar(mat, n_gt, n_pred, theta):
    if n_gt == 0 or n_pred == 0: return set()
    w = np.where(mat >= theta, 1000.0 + mat, 0.0); rows, cols = linear_sum_assignment(-w)
    return {(int(r), int(c)) for r, c in zip(rows, cols) if mat[r, c] >= theta}
man = {r["scan_id"]: r for r in csv.DictReader(open(S22D + "/input_manifest.csv"))}
EX = sorted(man)
ret = {}
for r in csv.DictReader(open(R + "/pulmonar_pos_oof_prep/script23_final/s23_20260924T153750_909396/lndb_verifier_object_scores.csv")):
    ret[(r["architecture"], r["scan_id"], int(r["object_id"]))] = r["retained"] == "True"
ann = json.load(open(R + "/pulmonar_canonical_pre_oof/cohort/lndb_annotations_manifest.json"))
ALT = {q["representative"]: q["excluded"] for q in ann["intrapair"] if q["representative"] in man}
radn = {f"LNDb-{int(r['LNDbID']):04d}": int(r["RadN"]) for r in csv.DictReader(open(P + "/dados/brutos/raw_5070_2026-08/lndb/trainCTs.csv"))}
def geo(e):
    rd = sitk.ImageFileReader(); rd.SetFileName(CT.format(e=e)); rd.ReadImageInformation()
    img = sitk.Image([1, 1, 1], sitk.sitkUInt8); img.SetOrigin(rd.GetOrigin()); img.SetSpacing(rd.GetSpacing()); img.SetDirection(rd.GetDirection())
    return img, rd.GetSize()
def encaixe(pr, origin_img, size):
    if not np.allclose(pr.GetSpacing(), origin_img.GetSpacing(), atol=1e-6) or not np.allclose(pr.GetDirection(), origin_img.GetDirection(), atol=1e-6): raise SystemExit("geometria")
    ci = np.array(origin_img.TransformPhysicalPointToContinuousIndex(pr.GetOrigin()))
    if np.abs(ci - np.round(ci)).max() > 1e-3: raise SystemExit("deslocamento não inteiro")
    o = np.round(ci).astype(int); sz = np.array(pr.GetSize())
    if not ((o >= 0).all() and (o + sz <= np.array(size)).all()): raise SystemExit("recorte fora")
    return int(o[2]), int(o[1]), int(o[0])
def avaliar(gt_lab, n_gt, pr_lab, n_pred, ignorar, a, e_pred, rot, e_ref):
    mat = S11.build_iou_matrix(gt_lab, pr_lab, n_gt, n_pred)
    toca = set(int(v) - 1 for v in np.unique(pr_lab[ignorar]) if v) if ignorar is not None else set()
    out, objs = [], []
    for t in LIM:
        pares = emparelhar_por_limiar(mat, n_gt, n_pred, t); pm = {c for _, c in pares}
        ign = {j for j in range(n_pred) if j not in pm and j in toca}
        acc = {j for j in range(n_pred) if ret[(a, e_pred, j + 1)]}
        tp = len(pares); fp = n_pred - tp - len(ign)
        tpp = sum(1 for _, c in pares if c in acc); fpp = len(acc - pm - ign)
        out.append(dict(referencia=rot, configuracao=a, exame=e_ref, tau=t, n_ref=n_gt, n_pred=n_pred, ignoradas=len(ign), ignoradas_aceites=len(ign & acc),
                        TP=tp, FP=fp, FN=n_gt - tp, TP_pos=tpp, FP_pos=fpp, FN_pos=n_gt - tpp, aceites=len(acc)))
        if t == 0.10:
            gp = {r_: c for r_, c in pares}
            for gi in range(n_gt):
                c = gp.get(gi); objs.append(dict(referencia=rot, configuracao=a, exame=e_ref, gt_id=gi + 1, n_voxels_gt=int((gt_lab == gi + 1).sum()),
                                                 status="TP" if c is not None else "FN", pred_id=(c + 1) if c is not None else "", iou=float(mat[gi, c]) if c is not None else 0.0,
                                                 aceite=ret[(a, e_pred, c + 1)] if c is not None else ""))
    return out, objs
SUBST = []
def um(e):
    origin_img, size = geo(e); Z0 = np.load(f"{O}/referencias/{e}.npz"); C, Ob, ctl = [], [], []
    for a in ARCH:
        p = PRED.format(a=a, e=e)
        if not os.path.exists(p):
            alt_p = PRED.format(a=[x for x in ARCH if x != a][0], e=e)
            if os.path.exists(alt_p) and sha(alt_p) == man[e][f"prediction_{a}_sha256"]: p = alt_p; SUBST.append((a, e))
            else: raise SystemExit(f"predição ausente {a}/{e}")
        if sha(p) != man[e][f"prediction_{a}_sha256"]: raise SystemExit(f"SHA predição {a}/{e}")
        pr = sitk.ReadImage(p); arr = sitk.GetArrayFromImage(pr); assert np.isin(arr, (0, 1)).all()
        lab_c, n_pred = nd.label(arr > 0, structure=STRUCT_6)
        z0, y0, x0 = encaixe(pr, origin_img, size); Zs, Ys, Xs = lab_c.shape
        pr_lab = np.zeros(Z0["N1"].shape, np.int32); pr_lab[z0:z0 + Zs, y0:y0 + Ys, x0:x0 + Xs] = lab_c
        old, n_old = nd.label(Z0["ref_antiga"], structure=STRUCT_6)
        o, ob = avaliar(old, n_old, pr_lab, n_pred, None, a, e, "antiga", e); C += o; Ob += ob
        n1 = int(Z0["N1"].max()); o, ob = avaliar(Z0["N1"].astype(np.int32), n1, pr_lab, n_pred, Z0["ignorar_N1"], a, e, "N1", e); C += o; Ob += ob
        if radn[e] >= 2:
            n2 = int(Z0["N2"].max()); o, ob = avaliar(Z0["N2"].astype(np.int32), n2, pr_lab, n_pred, Z0["ignorar_N2"], a, e, "N2", e); C += o; Ob += ob
        alt = ALT.get(e, e); Za = Z0 if alt == e else np.load(f"{O}/referencias/{alt}.npz")
        n1a = int(Za["N1"].max()); o, _ = avaliar(Za["N1"].astype(np.int32), n1a, pr_lab, n_pred, Za["ignorar_N1"], a, e, "N1_alt", alt); C += o
        if radn[alt] >= 2:
            n2a = int(Za["N2"].max()); o, _ = avaliar(Za["N2"].astype(np.int32), n2a, pr_lab, n_pred, Za["ignorar_N2"], a, e, "N2_alt", alt); C += o
    if SUBST: print("SUBSTITUICAO_SHA_IDENTICO", SUBST, flush=True)
    print(e, flush=True); return C, Ob
if __name__ == "__main__":
    with Pool(6) as pool: RES = pool.map(um, EX, chunksize=2)
    C = [x for a, b in RES for x in a]; Ob = [x for a, b in RES for x in b]
    def w(p, X):
        with open(p, "w", newline="") as f: d = csv.DictWriter(f, fieldnames=list(X[0])); d.writeheader(); d.writerows(X)
    w(O + "/saidas/R2_contagens_por_exame.csv", C); w(O + "/saidas/R2_objetos_referencia_tau010.csv", Ob)
    s22 = [r for r in csv.DictReader(open(S22D + "/lndb_object_matches.csv")) if r["status"] in ("TP", "FN")]
    mine = {(o["configuracao"], o["exame"], o["gt_id"]): o for o in Ob if o["referencia"] == "antiga"}
    falhas = 0
    for r in s22:
        o = mine.get((r["architecture"], r["scan_id"], int(float(r["gt_component_id"]))))
        if not o or o["status"] != r["status"] or o["n_voxels_gt"] != int(r["n_voxels_gt"]) or (r["status"] == "TP" and (int(float(r["pred_component_id"])) != o["pred_id"] or abs(float(r["iou"]) - o["iou"]) > 1e-12)): falhas += 1
    aud = {(r["architecture"], r["scan_id"]): r for r in csv.DictReader(open(S22D + "/lndb_scan_audit.csv"))}
    for c in C:
        if c["referencia"] == "antiga" and c["tau"] == 0.10:
            q = aud[(c["configuracao"], c["exame"])]
            if (c["TP"], c["FN"], c["FP"]) != (int(q["tp"]), int(q["fn"]), int(q["fp"])): falhas += 1
    porta = dict(objetos_s22=len(s22), objetos_reproduzidos=len(mine), falhas=falhas, aprovada=falhas == 0 and len(s22) == len(mine))
    json.dump(porta, open(O + "/saidas/R2_porta_referencia_antiga.json", "w"), indent=1); print(json.dumps(porta))
    if not porta["aprovada"]: sys.exit("PORTA FALHOU")
