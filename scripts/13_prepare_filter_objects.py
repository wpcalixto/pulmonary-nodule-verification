#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    13_prepare_filter_objects.py

File type:
    Script

Purpose:
    Prepares predicted (TP/FP) and reference (TP/FN) objects of BD1 as regions of interest for the morphological-filter analysis.

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
    - bd1_split.csv
    - bd1_folds.json
    - bd1_object_rois_manifest.json

Outputs:
    - See docs/file_level_documentation.md.

Dependencies:
    - SimpleITK
    - numpy
    - pandas
    - scipy

Execution:
    PULMONARY_BASE=<project_dir> python scripts/13_prepare_filter_objects.py

Portability notes:
    All paths are resolved relative to the project base directory given by the PULMONARY_BASE environment variable. No machine-specific path is required.

Reproducibility notes:
    Environment pinned in requirements.txt and environment/. Seeds and frozen parameters are recorded in the run manifests under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) are not included and must be obtained from their original repositories.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import logging
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage as nd

from pulmonary_base import BASE
from caminhos_canonicos import raiz_recortes, ct_de, label_de, FalhaEntrada

PRED_OOF_DIR = os.path.join(BASE, "predictions", "bd1_oof")
PRED_INT_DIR = os.path.join(BASE, "predictions", "bd1_internal")
CONFIG_DIR   = os.path.join(BASE, "config")
FEATURES_DIR = os.path.join(BASE, "features")
LOGS_DIR     = os.path.join(BASE, "logs")

MARGIN_MM = 15.0
STRUCT_6  = nd.generate_binary_structure(3, 1)
IOU_CANON = 0.10
EXPECTED_ARCHS = ["2D", "2D_REM", "2D_REL", "3D_F", "3D_FREM", "3D_CF", "3D_CFREM",
                  "F100_lowres", "F100_ResEncM", "F100_fullres"]

for _d in [FEATURES_DIR, LOGS_DIR]:
    os.makedirs(_d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "prepare_filter_objects.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def fail_input(msg):
    log.error(msg)
    raise SystemExit("FAIL_INPUT")


def fail_output(msg):
    log.error(msg)
    raise SystemExit("FAIL_OUTPUT")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_geometry(a: sitk.Image, b: sitk.Image, rotulo: str) -> None:
    ok = (a.GetSize() == b.GetSize()
          and np.allclose(a.GetSpacing(), b.GetSpacing(), rtol=0, atol=1e-6)
          and np.allclose(a.GetOrigin(), b.GetOrigin(), rtol=0, atol=1e-6)
          and np.allclose(a.GetDirection(), b.GetDirection(), rtol=0, atol=1e-6))
    if not ok:
        fail_input(f"GEOMETRY_GATE: grelhas incompatíveis — {rotulo}")


def load_labeled(path: str):
    """Máscara binária → (imagem, rotulado 6-conn, n) — a MESMA rotulagem do Script 11."""
    if not os.path.isfile(path):
        fail_input(f"máscara ausente: {path}")
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    if not np.isin(arr, (0, 1)).all():
        fail_input(f"máscara não binária: {path}")
    labeled, n = nd.label(arr.astype(np.uint8), structure=STRUCT_6)
    return img, labeled, int(n)


def diam_axial_bbox_mm(coords_zyx: np.ndarray, sp_zyx: tuple) -> float:
    """Maior diâmetro axial (bbox por corte) — idêntico a `diam_mm` do Script 11
    corrigido e ao Script 11B; CANONICAL_SIZE_DIAMETER."""
    sy, sx = sp_zyx[1], sp_zyx[2]
    best = 0.0
    for z in np.unique(coords_zyx[:, 0]):
        sc = coords_zyx[coords_zyx[:, 0] == z]
        best = max(best, (sc[:, 1].max() - sc[:, 1].min() + 1) * sy,
                   (sc[:, 2].max() - sc[:, 2].min() + 1) * sx)
    return float(best)


def object_record(labeled: np.ndarray, cid: int, sp_zyx: tuple, shape: tuple, rotulo: str) -> dict:
    coords = np.argwhere(labeled == cid)
    if coords.size == 0:
        fail_input(f"COMPONENT_IDENTITY: component_id {cid} irresolúvel em {rotulo}")
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)
    sz, sy, sx = (float(s) for s in sp_zyx)
    mz, my, mx = (int(np.ceil(MARGIN_MM / s)) for s in (sz, sy, sx))
    D, H, W = shape
    z0, z1 = int(zmin) - mz, int(zmax) + 1 + mz
    y0, y1 = int(ymin) - my, int(ymax) + 1 + my
    x0, x1 = int(xmin) - mx, int(xmax) + 1 + mx
    clipped = z0 < 0 or y0 < 0 or x0 < 0 or z1 > D or y1 > H or x1 > W
    z0, y0, x0 = max(0, z0), max(0, y0), max(0, x0)
    z1, y1, x1 = min(D, z1), min(H, y1), min(W, x1)
    if not (0 <= zmin <= zmax < D and 0 <= ymin <= ymax < H and 0 <= xmin <= xmax < W):
        fail_output(f"componente fora do volume declarado em {rotulo}")
    if not (0 <= z0 < z1 <= D and 0 <= y0 < y1 <= H and 0 <= x0 < x1 <= W):
        fail_output(f"ROI fora do volume em {rotulo}")
    n_vox = int(len(coords))
    vol = n_vox * sz * sy * sx
    diam = diam_axial_bbox_mm(coords, sp_zyx)
    if n_vox <= 0 or vol <= 0 or diam <= 0:
        fail_output(f"morfometria não positiva em {rotulo}")
    return {
        "component_id": int(cid), "voxel_count": n_vox,
        "volume_mm3": round(vol, 3), "diam_mm": round(diam, 3),
        "bbox_orig_z0": int(zmin), "bbox_orig_z1": int(zmax) + 1,
        "bbox_orig_y0": int(ymin), "bbox_orig_y1": int(ymax) + 1,
        "bbox_orig_x0": int(xmin), "bbox_orig_x1": int(xmax) + 1,
        "bbox_z0": z0, "bbox_z1": z1, "bbox_y0": y0, "bbox_y1": y1, "bbox_x0": x0, "bbox_x1": x1,
        "margin_vox_z": mz, "margin_vox_y": my, "margin_vox_x": mx,
        "roi_clipped_at_image_boundary": int(clipped),
    }


def rel(path: str) -> str:
    r = os.path.relpath(path, BASE)
    if os.path.isabs(r) or r.startswith(".."):
        fail_output(f"PATH_PORTABILITY: caminho não relativo a PULMONARY_BASE: {path}")
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", required=True, help="bd1_object_matches.csv canónico (Script 11 + 11B)")
    ap.add_argument("--matches-sha256", required=True)
    ap.add_argument("--source-runs", default="", help="run_ids do Script 11 e 11B, para o manifesto")
    ap.add_argument("--limit", type=int, default=0, help="ENSAIO: só os N primeiros grupos (saída em features/smoke/)")
    args = ap.parse_args()
    global FEATURES_DIR
    if args.limit:
        FEATURES_DIR = os.path.join(FEATURES_DIR, "smoke"); os.makedirs(FEATURES_DIR, exist_ok=True)
    t0 = datetime.datetime.now().astimezone()

    if not os.path.isfile(args.matches):
        fail_input(f"fonte canónica ausente: {args.matches}")
    h = sha256(args.matches)
    if h != args.matches_sha256:
        fail_input(f"SOURCE_AUTHORITY: SHA da fonte {h[:12]} != declarado {args.matches_sha256[:12]}")
    df = pd.read_csv(args.matches, dtype={"gt_component_id": "Int64", "pred_component_id": "Int64"})
    for c in ("subset", "patient_id", "model", "gt_component_id", "pred_component_id", "status", "diam_mm_gt"):
        if c not in df.columns:
            fail_input(f"fonte sem coluna `{c}`")
    if set(df["status"].unique()) - {"TP", "FP", "FN"}:
        fail_input(f"status inesperado na fonte: {set(df['status'].unique())}")

    dev, internal = [], []
    with open(os.path.join(CONFIG_DIR, "bd1_split.csv"), newline="") as f:
        for r in csv.DictReader(f):
            (dev if r["split"].strip().lower() == "development" else internal).append(r["patient_id"].strip())
    with open(os.path.join(CONFIG_DIR, "bd1_folds.json")) as f:
        fj = json.load(f)
    pid_to_fold = {p: int(k.split("_")[1]) for k, info in fj["folds"].items() for p in info["val"]}
    subset_of = {p: "oof" for p in dev}
    subset_of.update({p: "internal" for p in internal})
    src_pids = set(df["patient_id"].unique())
    if src_pids != set(subset_of):
        fail_input(f"pacientes da fonte ({len(src_pids)}) != split canónico ({len(subset_of)})")
    archs = sorted(df["model"].unique())
    if archs != sorted(EXPECTED_ARCHS):
        fail_input(f"arquitecturas da fonte {archs} != esperadas")

    raiz = raiz_recortes()
    present = set(map(tuple, df[["subset", "patient_id", "model"]].drop_duplicates().values.tolist()))
    absent = [(subset_of[p], p, a) for p in sorted(subset_of) for a in EXPECTED_ARCHS if (subset_of[p], p, a) not in present]
    if not args.limit:
        for s_, p_, a_ in absent:
            gp = label_de(raiz, "lidc_idri", p_)
            pp = (os.path.join(PRED_OOF_DIR, a_, f"fold_{pid_to_fold[p_]}", f"{p_}.nii.gz") if s_ == "oof"
                  else os.path.join(PRED_INT_DIR, a_, f"{p_}.nii.gz"))
            if not os.path.isfile(pp):
                fail_input(f"predição ausente: {pp}")
            if sitk.GetArrayFromImage(sitk.ReadImage(gp)).any() or sitk.GetArrayFromImage(sitk.ReadImage(pp)).any():
                fail_input(f"COVERAGE: grupo {s_}/{p_}/{a_} ausente da fonte mas com objectos")
        log.info(f"COVERAGE: {len(present)} grupos na fonte + {len(absent)} ausentes legítimos (GT e predição vazias) = {len(present) + len(absent)}")
    pred_rois, ref_rois = [], []
    identities_checked = 0
    groups = df.groupby(["subset", "patient_id", "model"], sort=True)
    log.info(f"Grupos (subset × paciente × arquitectura): {len(groups)}")
    if args.limit:
        keys = list(groups.groups)[: args.limit]
        df = df[df.set_index(["subset", "patient_id", "model"]).index.isin(keys)]
        groups = df.groupby(["subset", "patient_id", "model"], sort=True)
        log.info(f"ENSAIO --limit: {len(groups)} grupos")

    last_pid, gt_cache = None, None
    for idx, ((subset, pid, arch), grp) in enumerate(groups):
        if idx % 500 == 0:
            log.info(f"  {idx}/{len(groups)} …")
        pid = str(pid)
        if subset_of.get(pid) != subset:
            fail_input(f"{pid}: subset da fonte {subset} != split canónico {subset_of.get(pid)}")
        if subset == "oof":
            if pid not in pid_to_fold:
                fail_input(f"{pid}: fold desconhecido")
            p_path = os.path.join(PRED_OOF_DIR, arch, f"fold_{pid_to_fold[pid]}", f"{pid}.nii.gz")
        else:
            p_path = os.path.join(PRED_INT_DIR, arch, f"{pid}.nii.gz")

        if pid != last_pid:
            gt_path = label_de(raiz, "lidc_idri", pid)
            ct_path = ct_de(raiz, "lidc_idri", pid)
            gt_img, gt_lab, n_gt = load_labeled(gt_path)
            ct_img = sitk.ReadImage(ct_path)
            check_geometry(ct_img, gt_img, f"CT vs GT {pid}")
            gt_cache = (gt_path, ct_path, gt_img, gt_lab, n_gt, ct_img)
            last_pid = pid
        gt_path, ct_path, gt_img, gt_lab, n_gt, ct_img = gt_cache
        pr_img, pr_lab, n_pred = load_labeled(p_path)
        check_geometry(ct_img, pr_img, f"CT vs predição {pid}/{arch}")
        sp_xyz = gt_img.GetSpacing(); sp_zyx = (sp_xyz[2], sp_xyz[1], sp_xyz[0])
        shape = gt_lab.shape

        tp = grp[grp["status"] == "TP"]; fp = grp[grp["status"] == "FP"]; fn = grp[grp["status"] == "FN"]
        if len(tp) + len(fp) != n_pred:
            fail_output(f"{pid}/{arch}/{subset}: N_pred {n_pred} != TP+FP {len(tp) + len(fp)}")
        if len(tp) + len(fn) != n_gt:
            fail_output(f"{pid}/{arch}/{subset}: N_gt {n_gt} != TP+FN {len(tp) + len(fn)}")
        gt_ids = pd.concat([tp["gt_component_id"], fn["gt_component_id"]]).astype(int).tolist()
        pr_ids = pd.concat([tp["pred_component_id"], fp["pred_component_id"]]).astype(int).tolist()
        if sorted(gt_ids) != list(range(1, n_gt + 1)) or sorted(pr_ids) != list(range(1, n_pred + 1)):
            fail_input(f"{pid}/{arch}/{subset}: ids do 11 não reproduzem a rotulagem 6-conn "
                       f"(gt {sorted(gt_ids)} vs 1..{n_gt}; pred {sorted(pr_ids)} vs 1..{n_pred})")
        identities_checked += 1

        base = {"patient_id": pid, "architecture": arch, "subset": subset,
                "ct_path": rel(ct_path),
                "spacing_z": round(sp_zyx[0], 4), "spacing_y": round(sp_zyx[1], 4), "spacing_x": round(sp_zyx[2], 4)}
        for _, row in grp.iterrows():
            st = row["status"]
            if st in ("TP", "FP"):
                cid = int(row["pred_component_id"])
                rec = object_record(pr_lab, cid, sp_zyx, shape, f"pred {pid}/{arch}#{cid}")
                pred_rois.append({**base, "group": "TP_pred" if st == "TP" else "FP_pred",
                                  "source": "pred", "mask_path": rel(p_path), **rec})
            if st in ("TP", "FN"):
                cid = int(row["gt_component_id"])
                rec = object_record(gt_lab, cid, sp_zyx, shape, f"gt {pid}#{cid}")
                if abs(round(rec["diam_mm"], 2) - float(row["diam_mm_gt"])) > 0.011:
                    fail_output(f"{pid}#{cid}: diam_mm {rec['diam_mm']:.2f} != diam_mm_gt canónico {row['diam_mm_gt']}")
                ref_rois.append({**base, "group": "TP_gt" if st == "TP" else "FN_gt",
                                 "source": "gt", "mask_path": rel(gt_path), **rec})

    pdf = pd.DataFrame(pred_rois); rdf = pd.DataFrame(ref_rois)
    key = ["subset", "patient_id", "architecture", "group", "component_id"]
    for name, d in (("predicted", pdf), ("reference", rdf)):
        if d.duplicated(key).any():
            fail_output(f"{name}: chave duplicada")
        k2 = ["subset", "patient_id", "architecture", "component_id"]
        if d.duplicated(k2).any():
            fail_output(f"{name}: GROUP_EXCLUSIVITY — componente em dois grupos")
        if (d["volume_mm3"] <= 0).any() or (d["diam_mm"] <= 0).any() or (d["voxel_count"] <= 0).any():
            fail_output(f"{name}: morfometria não positiva")
        for c in ("ct_path", "mask_path"):
            if d[c].map(os.path.isabs).any():
                fail_output(f"{name}: caminho absoluto em {c}")
    n = Counter(pdf["group"]) + Counter(rdf["group"])
    if n["TP_pred"] != n["TP_gt"]:
        fail_output(f"TP_pred {n['TP_pred']} != TP_gt {n['TP_gt']} (matching um-para-um)")
    src = Counter(df["status"])
    if n["TP_pred"] != src["TP"] or n["FP_pred"] != src["FP"] or n["FN_gt"] != src["FN"]:
        fail_output(f"contagens vs fonte: {dict(n)} vs {dict(src)}")

    pred_out = os.path.join(FEATURES_DIR, "bd1_predicted_object_rois.parquet")
    ref_out = os.path.join(FEATURES_DIR, "bd1_reference_object_rois.parquet")
    pdf.to_parquet(pred_out, index=False); rdf.to_parquet(ref_out, index=False)

    by = defaultdict(int)
    for d in (pdf, rdf):
        for (a, s, g), c in d.groupby(["architecture", "subset", "group"]).size().items():
            by[f"{a}|{s}|{g}"] = int(c)
    manifest = {
        "script": "13_prepare_filter_objects.py", "script_sha256": sha256(os.path.abspath(__file__)),
        "run_at": t0.isoformat(), "PULMONARY_BASE": BASE,
        "source": {"matches": args.matches, "matches_sha256": h, "source_runs": args.source_runs,
                   "classification": f"Script 11 canonical, IoU = {IOU_CANON}, NOT recomputed"},
        "connectivity": "SCIPY 6 (generate_binary_structure(3,1)) — same as Script 11",
        "size_diameter": "MAXIMUM_AXIAL_DIAMETER_MM (axial slice bbox) — verified == diam_mm_gt of 11B for all GT objects",
        "margin_mm": MARGIN_MM,
        "counts": {"TP_pred": n["TP_pred"], "FP_pred": n["FP_pred"], "TP_gt": n["TP_gt"], "FN_gt": n["FN_gt"],
                   "predicted_rows": len(pdf), "reference_rows": len(rdf),
                   "roi_clipped_pred": int(pdf["roi_clipped_at_image_boundary"].sum()),
                   "roi_clipped_ref": int(rdf["roi_clipped_at_image_boundary"].sum())},
        "counts_by_arch_subset_group": dict(sorted(by.items())),
        "coverage": {"groups_in_source": len(present), "groups_absent_verified_empty": len(absent),
                     "total_patient_arch": len(present) + len(absent)},
        "gates": {"SOURCE_AUTHORITY": "PASS", "COMPONENT_IDENTITY": f"PASS ({identities_checked} groups)",
                  "COVERAGE": "PASS" if not args.limit else "NOT_RUN (ensaio)",
                  "GROUP_EXCLUSIVITY": "PASS", "GEOMETRY": "PASS", "ROI_15MM": "PASS",
                  "PATH_PORTABILITY": "PASS", "COUNT_IDENTITIES": "PASS"},
        "outputs_sha256": {os.path.basename(p): sha256(p) for p in (pred_out, ref_out)},
    }
    with open(os.path.join(FEATURES_DIR, "bd1_object_rois_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"TP_pred={n['TP_pred']}  FP_pred={n['FP_pred']}  TP_gt={n['TP_gt']}  FN_gt={n['FN_gt']}")
    log.info("Script 13 concluído.")


if __name__ == "__main__":
    try:
        main()
    except FalhaEntrada as e:
        log.error(str(e))
        raise SystemExit("FAIL_INPUT")
