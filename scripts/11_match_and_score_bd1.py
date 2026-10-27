#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    11_match_and_score_bd1.py

File type:
    Script

Purpose:
    Performs one-to-one Hungarian matching (IoU) between predicted and reference components in BD1 and computes detection scores.

Repository:
    pulmonary_nodule_verification

Version:
    v1.0.0

Date:
    2026-10-26

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
    Calixto, W. P., Camargo, T. F. O., Pacheco, V. M. G., Martins, M. S. R., & Coimbra, A. P. (2026). Multiscale morphology-informed post-detection verification of pulmonary nodule candidates in CT: code, trained nnU-Net models, cohort splits and analysis data (v1.0.0) [Data set and software]. Zenodo. https://doi.org/10.5281/zenodo.23674673

Related publication:
    Multiscale morphology-informed post-detection verification of pulmonary nodule candidates in computed tomography (manuscript; DOI not yet available)

Inputs:
    - Outputs of the preceding pipeline stage (see docs/usage.md).

Outputs:
    - See docs/file_level_documentation.md.

Dependencies:
    - SimpleITK
    - numpy
    - scipy

Execution:
    PULMONARY_BASE=<project_dir> python scripts/11_match_and_score_bd1.py

Portability notes:
    All paths are resolved relative to the project base directory given by the PULMONARY_BASE environment variable. No machine-specific path is required.

Reproducibility notes:
    Environment pinned in requirements.txt and environment/. Seeds and frozen parameters are recorded in the run manifests under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) are not included and must be obtained from their original repositories.
"""

import csv
import json
import math
import logging
import os
import numpy as np
import SimpleITK as sitk
from collections import Counter
from scipy import ndimage as nd
from scipy.optimize import linear_sum_assignment

from pulmonary_base import BASE
from caminhos_canonicos import raiz_recortes

LABELS_DIR   = os.path.join(raiz_recortes(), "lidc_idri", "labels")
PRED_OOF_DIR = os.path.join(BASE, "predictions", "bd1_oof")
PRED_INT_DIR = os.path.join(BASE, "predictions", "bd1_internal")
CONFIG_DIR   = os.path.join(BASE, "config")
ANALYSIS_DIR = os.path.join(BASE, "analysis")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

ARCH_NAMES  = ["2D", "2D_REM", "2D_REL", "3D_F", "3D_FREM", "3D_CF", "3D_CFREM",
               "F100_lowres", "F100_ResEncM", "F100_fullres"]
THRESHOLDS  = [0.10, 0.25, 0.50]
SIZE_BINS   = [3.0, 6.0, 15.0, float("inf")]
SIZE_LABELS = ["small", "medium", "large"]
N_BOOT      = 2000
SEED        = 27

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_split():
    """Retorna (dev_ids, internal_ids) de config/bd1_split.csv."""
    path = os.path.join(CONFIG_DIR, "bd1_split.csv")
    dev, internal = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            pid = row["patient_id"].strip()
            (dev if row["split"].strip().lower() == "development" else internal).append(pid)
    return sorted(dev), sorted(internal)


def load_pid_to_fold():
    """Retorna dict patient_id → fold_idx a partir de config/bd1_folds.json."""
    path = os.path.join(CONFIG_DIR, "bd1_folds.json")
    with open(path) as f:
        data = json.load(f)
    m = {}
    for i in range(5):
        for pid in data["folds"][f"fold_{i}"]["val"]:
            m[pid] = i
    return m


def load_binary_cc(nii_path):
    """
    Lê NIfTI binária e computa componentes conectados.
    Retorna (binary, labeled, n_components, spacing_xyz).
    """
    img    = sitk.ReadImage(nii_path)
    binary = sitk.GetArrayFromImage(img).astype(np.uint8)
    labeled, n = nd.label(binary)
    return binary, labeled, n, img.GetSpacing()


def build_iou_matrix(gt_lab, pred_lab, n_gt, n_pred):
    """
    Constrói matriz IoU (n_gt × n_pred) com método esparso
    (considera apenas voxels com sobreposição).
    """
    if n_gt == 0 or n_pred == 0:
        return np.zeros((n_gt, n_pred))

    gt_flat   = gt_lab.ravel()
    pred_flat = pred_lab.ravel()
    gt_sz     = np.bincount(gt_flat,   minlength=n_gt   + 1)[1:]
    pred_sz   = np.bincount(pred_flat, minlength=n_pred + 1)[1:]

    mat = np.zeros((n_gt, n_pred))
    ov  = (gt_flat > 0) & (pred_flat > 0)
    if ov.any():
        for (gi, pi), inter in Counter(
            zip(gt_flat[ov].tolist(), pred_flat[ov].tolist())
        ).items():
            i, j = gi - 1, pi - 1
            u = gt_sz[i] + pred_sz[j] - inter
            mat[i, j] = inter / u if u > 0 else 0.0
    return mat


def dice_from_iou(j):
    return 2.0 * j / (1.0 + j) if j > 0 else 0.0


def global_dice(gt_bin, pred_bin):
    inter = int(np.logical_and(gt_bin, pred_bin).sum())
    denom = int(gt_bin.sum()) + int(pred_bin.sum())
    return 2.0 * inter / denom if denom > 0 else 1.0


def hausdorff_mm(gt_bin, pred_bin, sp_xyz):
    """Hausdorff simétrico em mm. sp_xyz = SimpleITK (x, y, z)."""
    if not gt_bin.any() or not pred_bin.any():
        return float("nan")
    sp_zyx = (sp_xyz[2], sp_xyz[1], sp_xyz[0])
    d_gt   = nd.distance_transform_edt(~gt_bin.astype(bool), sampling=sp_zyx)
    d_pred = nd.distance_transform_edt(~pred_bin.astype(bool), sampling=sp_zyx)
    return float(max(float(d_pred[gt_bin.astype(bool)].max()),
                     float(d_gt[pred_bin.astype(bool)].max())))


def diam_mm(comp_bin, sp_xyz):
    """Diâmetro de esfera equivalente em mm."""
    vol = float(comp_bin.sum()) * sp_xyz[0] * sp_xyz[1] * sp_xyz[2]
    return (6.0 * vol / math.pi) ** (1.0 / 3.0) if vol > 0 else 0.0


def size_cat(d):
    """Retorna categoria de tamanho (small/medium/large) ou '' para nódulos <3 mm."""
    for i in range(len(SIZE_BINS) - 1):
        if SIZE_BINS[i] <= d < SIZE_BINS[i + 1]:
            return SIZE_LABELS[i]
    return ""


def bootstrap_ci(per_pat, pids, theta):
    """Reamostrar todos os pacientes com reposição → IC 95% para sens, prec, F1."""
    rng  = np.random.default_rng(SEED)
    pats = list(pids)
    n    = len(pats)
    sb, pb, fb = [], [], []
    for _ in range(N_BOOT):
        samp = [pats[i] for i in rng.integers(0, n, size=n)]
        tp   = sum(per_pat.get(p, {}).get(theta, (0, 0, 0))[0] for p in samp)
        fn   = sum(per_pat.get(p, {}).get(theta, (0, 0, 0))[1] for p in samp)
        fp   = sum(per_pat.get(p, {}).get(theta, (0, 0, 0))[2] for p in samp)
        tot  = tp + fn
        s    = tp / tot       if tot       > 0 else 0.0
        pr   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f    = 2*pr*s/(pr+s)  if (pr + s)  > 0 else 0.0
        sb.append(s); pb.append(pr); fb.append(f)
    ci = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return ci(sb), ci(pb), ci(fb)


def process_patient(pid, arch, pred_path):
    """
    Carrega GT e predição, constrói matriz IoU, aplica algoritmo Húngaro
    e classifica componentes para cada threshold.

    Retorna:
      obj_rows  — linhas para bd1_object_matches.csv (GT + FP)
      seg_rows  — linhas para bd1_segmentation_metrics.csv (GT × threshold)
      per_theta — {theta: (tp, fn, fp)}
      dice_g    — Dice global do paciente (float)
    """
    gt_path = os.path.join(LABELS_DIR, pid, "consensus.nii.gz")

    if not os.path.isfile(gt_path):
        log.error(f"    GT ausente: {gt_path}")
        raise SystemExit("FAIL_INPUT")

    if not os.path.isfile(pred_path):
        log.error(f"    Predição ausente: {pred_path}")
        raise SystemExit("FAIL_INPUT")

    gt_bin, gt_lab, n_gt, sp = load_binary_cc(gt_path)
    pr_bin, pr_lab, n_pred, _ = load_binary_cc(pred_path)
    dice_g = global_dice(gt_bin, pr_bin)

    mat = build_iou_matrix(gt_lab, pr_lab, n_gt, n_pred)
    if n_gt > 0 and n_pred > 0:
        rows, cols = linear_sum_assignment(-mat)
    else:
        rows, cols = np.array([], int), np.array([], int)

    tp_by_theta = {
        t: {(int(r), int(c)) for r, c in zip(rows, cols) if mat[r, c] >= t}
        for t in THRESHOLDS
    }
    per_theta = {}
    for t in THRESHOLDS:
        tp = len(tp_by_theta[t])
        per_theta[t] = (tp, n_gt - tp, n_pred - tp)

    obj_rows, seg_rows = [], []
    matched_pred_010 = {c for _, c in tp_by_theta[0.10]}

    for gi in range(n_gt):
        gc  = (gt_lab == gi + 1).astype(np.uint8)
        d   = diam_mm(gc, sp)
        cat = size_cat(d)

        match_010 = next(
            ((gi, c) for (r, c) in tp_by_theta[0.10] if r == gi), None
        )

        if match_010:
            _, pj    = match_010
            iou_val  = float(mat[gi, pj])
            pc       = (pr_lab == pj + 1).astype(np.uint8)
            hd       = hausdorff_mm(gc, pc, sp)
            dice_tp  = dice_from_iou(iou_val)
            obj_rows.append({
                "patient_id":        pid,
                "model":             arch,
                "gt_component_id":   gi + 1,
                "pred_component_id": pj + 1,
                "iou":               round(iou_val, 4),
                "dice_matched":      round(dice_tp, 4),
                "hausdorff_mm":      round(hd, 2) if not math.isnan(hd) else "",
                "dice_global":       round(dice_g, 4),
                "diam_mm_gt":        round(d, 2),
                "size_category":     cat,
                "status":            "TP",
                "tp_010": 1,
                "tp_025": int((gi, pj) in tp_by_theta[0.25]),
                "tp_050": int((gi, pj) in tp_by_theta[0.50]),
            })
            for t in THRESHOLDS:
                is_tp = (gi, pj) in tp_by_theta[t]
                seg_rows.append({
                    "patient_id":        pid,
                    "model":             arch,
                    "threshold":         t,
                    "gt_component_id":   gi + 1,
                    "pred_component_id": pj + 1 if is_tp else "",
                    "status":            "TP" if is_tp else "FN",
                    "iou":               round(iou_val, 4) if is_tp else 0.0,
                    "dice":              round(dice_tp, 4) if is_tp else 0.0,
                    "hausdorff_mm":      round(hd, 2)
                                         if (is_tp and not math.isnan(hd)) else "",
                    "diam_mm_gt":        round(d, 2),
                    "size_category":     cat,
                })
        else:
            obj_rows.append(_fn_row(pid, arch, gi + 1, round(d, 2), cat, dice_g))
            for t in THRESHOLDS:
                seg_rows.append(_seg_fn_row(pid, arch, t, gi + 1, round(d, 2), cat))

    for pj in range(n_pred):
        if pj not in matched_pred_010:
            obj_rows.append({
                "patient_id":        pid,
                "model":             arch,
                "gt_component_id":   "",
                "pred_component_id": pj + 1,
                "iou":               0.0,
                "dice_matched":      0.0,
                "hausdorff_mm":      "",
                "dice_global":       round(dice_g, 4),
                "diam_mm_gt":        "",
                "size_category":     "",
                "status":            "FP",
                "tp_010": 0, "tp_025": 0, "tp_050": 0,
            })

    return obj_rows, seg_rows, per_theta, dice_g


def _fn_row(pid, arch, gi, d, cat, dice_g):
    return {
        "patient_id": pid, "model": arch,
        "gt_component_id": gi, "pred_component_id": "",
        "iou": 0.0, "dice_matched": 0.0, "hausdorff_mm": "",
        "dice_global": round(dice_g, 4) if not math.isnan(dice_g) else "",
        "diam_mm_gt": d, "size_category": cat,
        "status": "FN", "tp_010": 0, "tp_025": 0, "tp_050": 0,
    }


def _seg_fn_row(pid, arch, t, gi, d, cat):
    return {
        "patient_id": pid, "model": arch, "threshold": t,
        "gt_component_id": gi, "pred_component_id": "",
        "status": "FN", "iou": 0.0, "dice": 0.0, "hausdorff_mm": "",
        "diam_mm_gt": d, "size_category": cat,
    }


def aggregate_detection(arch, subset, per_pat, pids, pat_dice_g):
    """Linhas para bd1_detection_metrics.csv."""
    rows = []
    for t in THRESHOLDS:
        tp  = sum(per_pat.get(p, {}).get(t, (0, 0, 0))[0] for p in pids)
        fn  = sum(per_pat.get(p, {}).get(t, (0, 0, 0))[1] for p in pids)
        fp  = sum(per_pat.get(p, {}).get(t, (0, 0, 0))[2] for p in pids)
        tot = tp + fn
        s   = tp / tot       if tot       > 0 else 0.0
        pr  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1  = 2*pr*s/(pr+s)  if (pr + s)  > 0 else 0.0

        ci_s, ci_p, ci_f = bootstrap_ci(per_pat, pids, t)

        dg_vals = [v for v in pat_dice_g.values() if not math.isnan(v)]
        dice_g_mean = float(np.mean(dg_vals)) if dg_vals else ""

        rows.append({
            "model": arch, "subset": subset, "threshold": t,
            "TP": tp, "FN": fn, "FP": fp,
            "sensitivity": round(s, 4),
            "sens_lo":     round(ci_s[0], 4), "sens_hi": round(ci_s[1], 4),
            "precision":   round(pr, 4),
            "prec_lo":     round(ci_p[0], 4), "prec_hi": round(ci_p[1], 4),
            "f1":          round(f1, 4),
            "f1_lo":       round(ci_f[0], 4), "f1_hi": round(ci_f[1], 4),
            "dice_global": round(dice_g_mean, 4) if isinstance(dice_g_mean, float) else "",
        })
    return rows


def aggregate_size(arch, subset, seg_rows):
    """Linhas para bd1_size_stratified_metrics.csv."""
    rows = []
    for cat in SIZE_LABELS:
        for t in THRESHOLDS:
            sub = [r for r in seg_rows if r["size_category"] == cat and r["threshold"] == t]
            tp  = sum(1 for r in sub if r["status"] == "TP")
            fn  = sum(1 for r in sub if r["status"] == "FN")
            n_gt = tp + fn
            rows.append({
                "model": arch, "subset": subset,
                "size_category": cat, "threshold": t,
                "n_gt": n_gt, "TP": tp, "FN": fn,
                "sensitivity": round(tp / n_gt, 4) if n_gt > 0 else "",
            })
    return rows


def compute_froc(all_det, n_scans_by_subset):
    """
    Gera pontos FROC discretos a partir das métricas de detecção.

    Para cada (model, subset, threshold) já calculados em all_det, derive:
      fp_per_scan = FP / n_scans
      sensitivity = TP / (TP + FN)

    Sem mapas de probabilidade disponíveis (predições binárias), os três limiares
    de IoU {0.10, 0.25, 0.50} funcionam como pontos discretos da curva.
    """
    rows = []
    for r in all_det:
        n_scans = n_scans_by_subset.get(r["subset"], 1)
        tp, fn, fp = r["TP"], r["FN"], r["FP"]
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fps  = fp / n_scans   if n_scans  > 0 else 0.0
        rows.append({
            "model":       r["model"],
            "subset":      r["subset"],
            "threshold":   r["threshold"],
            "fp_per_scan": round(fps, 4),
            "sensitivity": round(sens, 4),
        })
    return rows


def save_csv(path, rows, fieldnames):
    if not rows:
        log.warning(f"Nenhuma linha para {os.path.basename(path)}")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    log.info(f"Salvo: {os.path.relpath(path, BASE)} ({len(rows)} linhas)")


def main():
    dev_ids, internal_ids = load_split()
    pid_to_fold           = load_pid_to_fold()

    all_obj  = []
    all_seg  = []
    all_det  = []
    all_size = []

    for arch in ARCH_NAMES:
        log.info(f"=== Arquitetura: {arch} ===")

        def fold_of(pid):
            if pid not in pid_to_fold:
                log.error(f"    Fold desconhecido para {pid}")
                raise SystemExit("FAIL_INPUT")
            return pid_to_fold[pid]

        for subset_name, pids, pred_fn in [
            ("oof",      dev_ids,       lambda pid, a=arch:
                os.path.join(PRED_OOF_DIR, a,
                             f"fold_{fold_of(pid)}", f"{pid}.nii.gz")),
            ("internal", internal_ids, lambda pid, a=arch:
                os.path.join(PRED_INT_DIR, a, f"{pid}.nii.gz")),
        ]:
            log.info(f"  {subset_name}: {len(pids)} pacientes …")
            obj_sub, seg_sub, per_pat, pat_dice_g = [], [], {}, {}

            for pid in pids:
                o, s, pp, dg = process_patient(pid, arch, pred_fn(pid))
                for r in o:
                    r["subset"] = subset_name
                for r in s:
                    r["subset"] = subset_name
                obj_sub.extend(o)
                seg_sub.extend(s)
                per_pat[pid]    = pp
                pat_dice_g[pid] = dg

            all_obj.extend(obj_sub)
            all_seg.extend(seg_sub)
            all_det.extend(aggregate_detection(arch, subset_name, per_pat,
                                               pids, pat_dice_g))
            all_size.extend(aggregate_size(arch, subset_name, seg_sub))

    save_csv(
        os.path.join(ANALYSIS_DIR, "bd1_object_matches.csv"), all_obj,
        ["subset", "patient_id", "model", "gt_component_id", "pred_component_id",
         "iou", "dice_matched", "hausdorff_mm", "dice_global",
         "diam_mm_gt", "size_category", "status", "tp_010", "tp_025", "tp_050"],
    )
    save_csv(
        os.path.join(ANALYSIS_DIR, "bd1_detection_metrics.csv"), all_det,
        ["model", "subset", "threshold", "TP", "FN", "FP",
         "sensitivity", "sens_lo", "sens_hi",
         "precision", "prec_lo", "prec_hi",
         "f1", "f1_lo", "f1_hi", "dice_global"],
    )
    save_csv(
        os.path.join(ANALYSIS_DIR, "bd1_segmentation_metrics.csv"), all_seg,
        ["subset", "patient_id", "model", "threshold",
         "gt_component_id", "pred_component_id", "status",
         "iou", "dice", "hausdorff_mm", "diam_mm_gt", "size_category"],
    )
    save_csv(
        os.path.join(ANALYSIS_DIR, "bd1_size_stratified_metrics.csv"), all_size,
        ["model", "subset", "size_category", "threshold", "n_gt", "TP", "FN", "sensitivity"],
    )

    n_scans_by_subset = {"oof": len(dev_ids), "internal": len(internal_ids)}
    froc_rows = compute_froc(all_det, n_scans_by_subset)
    save_csv(
        os.path.join(ANALYSIS_DIR, "bd1_froc.csv"), froc_rows,
        ["model", "subset", "threshold", "fp_per_scan", "sensitivity"],
    )

    log.info("Script 11 concluído.")


if __name__ == "__main__":
    main()
