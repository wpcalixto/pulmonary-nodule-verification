#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    s02_geometria_pares.py

File type:
    Script

Purpose:
    Per-object segmentation geometry (Dice, Jaccard, surface distances, HD95) of the true-positive pairs in the internal and external datasets.
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
    - 02_PLANO_ANALITICO_CONGELADO.md
    - bd1_object_matches.csv
    - consensus.nii.gz
    - input_manifest.csv
    - lndb_object_matches.csv

Outputs:
    - Tables written to the stage output directory (see docs/post_v1_stages.md).

Dependencies:
    - SimpleITK
    - numpy
    - scipy

Execution:
    cd <project_dir> && python scripts/32_supplementary_analyses/b_segmentation_geometry/s02_geometria_pares.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import datetime
from multiprocessing import Pool

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as nd

ROOT = "."
AC = os.path.join(ROOT, "analises_complementares_v8")
OUT = os.path.join(AC, "frenteB")
PLANO = os.path.join(AC, "02_PLANO_ANALITICO_CONGELADO.md")
PLANO_SHA = "d519c9e76a6d8710254a5ed8d028ac5b935ded869067c5977e3a8565e6c1a5f1"

PP = os.path.join(ROOT, "pulmonar_pos_oof_prep")
S11B_MATCH = os.path.join(PP, "script11_final", "s11B_20260921T164539_27883", "analysis", "bd1_object_matches.csv")
S22_MATCH = os.path.join(PP, "script22_final", "s22_20260924T084718_826234", "lndb_object_matches.csv")
S22A_MAN = os.path.join(PP, "script22a_final", "s22a_20260924T125735_874698", "input_manifest.csv")
S06 = os.path.join(ROOT, "Pulmonar_v281_2026-09-01", "08_DADOS_INTERMEDIOS", "s06_20260810T183307_96979")
REF_D4 = os.path.join(ROOT, "Pulmonar_v281_2026-09-01", "04_EVIDENCIA", "annotations", "consensus", "lndb")
BLK_INT = os.path.join(ROOT, "Pulmonary_Project", "Script10_canonical", "blocks")
BLK_LNDB = os.path.join(ROOT, "Pulmonary_Project", "Script10_canonical_lndb", "blocks")

ARCHS_D2 = ["2D", "2D_REM", "2D_REL", "3D_F", "3D_FREM", "3D_CF", "3D_CFREM",
            "F100_lowres", "F100_ResEncM", "F100_fullres"]
ARCHS_D4 = ["3D_F", "3D_FREM"]
STRUCT_6 = nd.generate_binary_structure(3, 1)
TOL_IOU = {"D2": 5e-5, "D4": 1e-9}
TOL_HD_S11 = 0.005 + 1e-9
PAD = 2

class Falha(Exception):
    pass

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()

def le(path):
    if not os.path.isfile(path):
        raise Falha(f"FAIL_INPUT ausente {path}")
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    if not np.isin(arr, (0, 1)).all():
        raise Falha(f"FAIL_INPUT não binária {path}")
    b = arr > 0
    lab, n = nd.label(b, structure=STRUCT_6)
    return img, b, lab, int(n)

def mesma_grade(a, b):
    return (a.GetSize() == b.GetSize()
            and np.allclose(a.GetOrigin(), b.GetOrigin(), rtol=0, atol=1e-6)
            and np.allclose(a.GetSpacing(), b.GetSpacing(), rtol=0, atol=1e-6)
            and np.allclose(a.GetDirection(), b.GetDirection(), rtol=0, atol=1e-6))

def recorte(*masks):
    idx = np.argwhere(np.logical_or.reduce(masks))
    lo = np.maximum(idx.min(0) - PAD, 0)
    hi = idx.max(0) + PAD + 1
    return tuple(slice(l, h) for l, h in zip(lo, hi))

def superficie(m):
    mp = np.pad(m, PAD, constant_values=False)
    er = nd.binary_erosion(mp, structure=STRUCT_6)
    s = mp & ~er
    return s[PAD:-PAD, PAD:-PAD, PAD:-PAD]

def metricas_par(P, G, sp_zyx):
    """P, G: booleanos na mesma grade (já recortados). sp_zyx: espaçamento (z, y, x)."""
    nP, nG = int(P.sum()), int(G.sum())
    inter = int(np.logical_and(P, G).sum())
    uni = nP + nG - inter
    r = {"n_vox_pred": nP, "n_vox_ref": nG, "n_inter": inter,
         "dice": 2.0 * inter / (nP + nG) if (nP + nG) > 0 else math.nan,
         "jaccard": inter / uni if uni > 0 else math.nan}
    sP, sG = superficie(P), superficie(G)
    if sP.any() and sG.any():
        Pp = np.pad(sP, PAD); Gp = np.pad(sG, PAD)
        dG = nd.distance_transform_edt(~Gp, sampling=sp_zyx)
        dP = nd.distance_transform_edt(~Pp, sampling=sp_zyx)
        d_pg = dG[Pp]
        d_gp = dP[Gp]
        r["hd95"] = float(max(np.percentile(d_pg, 95, method="linear"),
                              np.percentile(d_gp, 95, method="linear")))
        r["dsm"] = float(0.5 * (d_pg.mean() + d_gp.mean()))
        r["hd100_sup"] = float(max(d_pg.max(), d_gp.max()))
        r["n_sup_pred"], r["n_sup_ref"] = int(sP.sum()), int(sG.sum())
    else:
        r.update(hd95=math.nan, dsm=math.nan, hd100_sup=math.nan,
                 n_sup_pred=int(sP.sum()), n_sup_ref=int(sG.sum()))
    Pp = np.pad(P, PAD); Gp = np.pad(G, PAD)
    d_gt = nd.distance_transform_edt(~Gp, sampling=sp_zyx)
    d_pr = nd.distance_transform_edt(~Pp, sampling=sp_zyx)
    r["hd_s11_reproduzido"] = float(max(d_pr[Gp].max(), d_gt[Pp].max()))
    return r

def processa(tarefa):
    conj, ident, linhas = tarefa
    saida, erros = [], []
    try:
        if conj == "D2":
            ref_img, ref_b, ref_lab, n_ref = le(os.path.join(S06, "lidc_idri", "labels", ident, "consensus.nii.gz"))
        else:
            ref_img, ref_b, ref_lab, n_ref = le(os.path.join(REF_D4, ident, "consensus.nii.gz"))
        sp = ref_img.GetSpacing()
        sp_zyx = (sp[2], sp[1], sp[0])
        por_arch = {}
        for r in linhas:
            por_arch.setdefault(r["arch"], []).append(r)
        for arch, rs in por_arch.items():
            if conj == "D2":
                pimg, pb, plab, n_pred = le(os.path.join(BLK_INT, f"lidc_internal__{arch}", f"{ident}.nii.gz"))
                if not mesma_grade(pimg, ref_img):
                    raise Falha(f"FAIL_GEOMETRY {conj} {ident} {arch}")
            else:
                pimg, pb_c, plab_c, n_pred = le(os.path.join(BLK_LNDB, f"lndb__{arch}", f"{ident}.nii.gz"))
                if not (np.allclose(pimg.GetSpacing(), ref_img.GetSpacing(), rtol=0, atol=1e-6)
                        and np.allclose(pimg.GetDirection(), ref_img.GetDirection(), rtol=0, atol=1e-6)):
                    raise Falha(f"FAIL_GEOMETRY espaçamento/direção {ident} {arch}")
                ci = np.array(ref_img.TransformPhysicalPointToContinuousIndex(pimg.GetOrigin()))
                if np.abs(ci - np.round(ci)).max() > 1e-3:
                    raise Falha(f"FAIL_GEOMETRY deslocamento não inteiro {ident} {arch}")
                o = np.round(ci).astype(int)[::-1]
                esperado = rs[0]["crop_offset_zyx"]
                if list(o) != esperado:
                    raise Falha(f"FAIL_GEOMETRY offset {list(o)} != manifesto {esperado} {ident} {arch}")
                Z, Y, X = plab_c.shape
                plab = np.zeros(ref_lab.shape, np.int32)
                plab[o[0]:o[0] + Z, o[1]:o[1] + Y, o[2]:o[2] + X] = plab_c
            n_ref_reg = len({r["gt_id"] for r in rs if r["gt_id"] is not None})
            n_pred_reg = len({r["pred_id"] for r in rs if r["pred_id"] is not None})
            if n_ref_reg != n_ref or n_pred_reg != n_pred:
                raise Falha(f"FAIL_IDENTITY contagem {conj} {ident} {arch}: ref {n_ref_reg}/{n_ref} pred {n_pred_reg}/{n_pred}")
            for r in rs:
                base = {k: r[k] for k in ("conjunto", "id", "arch", "status", "gt_id", "pred_id",
                                          "diam_mm_ref", "iou_registada", "hd_s11_registado")}
                if r["status"] != "TP":
                    saida.append(base)
                    continue
                G = ref_lab == r["gt_id"]
                P = plab == r["pred_id"]
                sl = recorte(G, P)
                m = metricas_par(P[sl], G[sl], sp_zyx)
                if abs(m["jaccard"] - r["iou_registada"]) > TOL_IOU[conj]:
                    raise Falha(f"FAIL_IDENTITY IoU {conj} {ident} {arch} g{r['gt_id']} p{r['pred_id']}: "
                                f"{m['jaccard']:.6f} vs {r['iou_registada']}")
                if conj == "D4" and (m["n_vox_ref"] != r["nvox_gt_reg"] or m["n_vox_pred"] != r["nvox_pred_reg"]):
                    raise Falha(f"FAIL_IDENTITY voxels {ident} {arch} g{r['gt_id']}")
                if conj == "D2" and abs(m["hd_s11_reproduzido"] - r["hd_s11_registado"]) > TOL_HD_S11:
                    raise Falha(f"FAIL_IDENTITY HD_S11 {ident} {arch} g{r['gt_id']}: "
                                f"{m['hd_s11_reproduzido']:.4f} vs {r['hd_s11_registado']}")
                base.update(m)
                base["spacing_xyz"] = "|".join(f"{v:.6g}" for v in sp)
                saida.append(base)
    except Falha as e:
        erros.append(str(e))
    except Exception as e:
        erros.append(f"FAIL_UNEXPECTED {conj} {ident}: {type(e).__name__}: {e}")
    return saida, erros

def tarefas():
    t = {}
    with open(S11B_MATCH) as f:
        for r in csv.DictReader(f):
            if r["subset"] != "internal":
                continue
            st = r["status"]
            t.setdefault(("D2", r["patient_id"]), []).append({
                "conjunto": "D2", "id": r["patient_id"], "arch": r["model"], "status": st,
                "gt_id": int(r["gt_component_id"]) if r["gt_component_id"] else None,
                "pred_id": int(r["pred_component_id"]) if r["pred_component_id"] else None,
                "diam_mm_ref": float(r["diam_mm_gt"]) if r["gt_component_id"] and r["diam_mm_gt"] else None,
                "iou_registada": float(r["iou"]) if st == "TP" else None,
                "hd_s11_registado": float(r["hausdorff_mm"]) if st == "TP" else None})
    offs = {}
    with open(S22A_MAN) as f:
        for r in csv.DictReader(f):
            offs[r["scan_id"]] = json.loads(r["crop_offset_zyx"])
    with open(S22_MATCH) as f:
        for r in csv.DictReader(f):
            st = r["status"]
            gid = int(r["gt_component_id"]) if r["gt_component_id"] else None
            t.setdefault(("D4", r["scan_id"]), []).append({
                "conjunto": "D4", "id": r["scan_id"], "arch": r["architecture"], "status": st,
                "gt_id": gid, "pred_id": int(r["pred_component_id"]) if r["pred_component_id"] else None,
                "diam_mm_ref": float(r["diam_mm_gt"]) if gid is not None else None,
                "iou_registada": float(r["iou"]) if st == "TP" else None,
                "hd_s11_registado": None,
                "nvox_gt_reg": int(r["n_voxels_gt"]) if gid is not None else None,
                "nvox_pred_reg": int(r["n_voxels_pred"]) if r["pred_component_id"] else None,
                "crop_offset_zyx": offs[r["scan_id"]]})
    return [(k[0], k[1], v) for k, v in sorted(t.items())]

COLS = ["conjunto", "id", "arch", "status", "gt_id", "pred_id", "diam_mm_ref", "n_vox_pred", "n_vox_ref",
        "n_inter", "dice", "jaccard", "hd95", "dsm", "hd100_sup", "n_sup_pred", "n_sup_ref",
        "iou_registada", "hd_s11_registado", "hd_s11_reproduzido", "spacing_xyz"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if sha256(PLANO) != PLANO_SHA:
        sys.exit("FAIL_PLAN: o plano congelado mudou")
    inicio = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    ts = tarefas()
    if a.smoke:
        d2 = [x for x in ts if x[0] == "D2"][:a.smoke]
        d4 = [x for x in ts if x[0] == "D4"][:a.smoke]
        ts = d2 + d4
    linhas, erros = [], []
    with Pool(a.workers) as pool:
        for s, e in pool.imap_unordered(processa, ts, chunksize=1):
            linhas.extend(s)
            erros.extend(e)
    sufixo = "_smoke" if a.smoke else ""
    os.makedirs(OUT, exist_ok=True)
    gate = {"script": os.path.basename(__file__), "script_sha256": sha256(__file__),
            "plano_sha256": PLANO_SHA, "inicio": inicio,
            "fim": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "n_tarefas": len(ts), "n_linhas": len(linhas), "erros": erros,
            "estado": "PASS" if not erros else "FAIL",
            "entradas_sha256": {p: sha256(p) for p in (S11B_MATCH, S22_MATCH, S22A_MAN)},
            "versoes": {"numpy": np.__version__, "SimpleITK": sitk.Version_VersionString(),
                        "scipy": __import__("scipy").__version__, "python": sys.version.split()[0]}}
    if erros:
        with open(os.path.join(OUT, f"GATE_s02{sufixo}.json"), "w") as f:
            json.dump(gate, f, indent=2, ensure_ascii=False)
        print(json.dumps(gate, indent=2, ensure_ascii=False)[:4000])
        sys.exit(1)
    linhas.sort(key=lambda r: (r["conjunto"], r["arch"], r["id"], r["gt_id"] or 10**9, r["pred_id"] or 10**9))
    p = os.path.join(OUT, f"geometria_por_objeto{sufixo}.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in linhas:
            w.writerow({k: ("" if r.get(k) is None or (isinstance(r.get(k), float) and math.isnan(r[k]))
                            else r.get(k)) for k in COLS})
    gate["saida"] = {os.path.basename(p): sha256(p)}
    with open(os.path.join(OUT, f"GATE_s02{sufixo}.json"), "w") as f:
        json.dump(gate, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: gate[k] for k in ("estado", "n_tarefas", "n_linhas", "inicio", "fim")}, ensure_ascii=False))

if __name__ == "__main__":
    main()
