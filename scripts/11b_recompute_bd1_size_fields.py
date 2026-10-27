#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    11b_recompute_bd1_size_fields.py

File type:
    Script

Purpose:
    Recomputes the size fields of the BD1 matched components used by the downstream stages.

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
    - bd1_object_matches.csv
    - bd1_segmentation_metrics.csv
    - gt_axial_diameters.csv

Outputs:
    - SIZE_REPAIR_GATE.json

Dependencies:
    - SimpleITK
    - numpy
    - scipy

Execution:
    PULMONARY_BASE=<project_dir> python scripts/11b_recompute_bd1_size_fields.py

Portability notes:
    All paths are resolved relative to the project base directory given by the PULMONARY_BASE environment variable. No machine-specific path is required.

Reproducibility notes:
    Environment pinned in requirements.txt and environment/. Seeds and frozen parameters are recorded in the run manifests under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) are not included and must be obtained from their original repositories.
"""

import csv
import hashlib
import json
import math
import os
import shutil
import sys
import datetime
from collections import Counter, defaultdict

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as nd

from pulmonary_base import BASE
from caminhos_canonicos import raiz_recortes

ARCHIVE   = sys.argv[1]
REFERENCE = sys.argv[2]
OUT_DIR   = sys.argv[3]

THRESHOLDS  = [0.10, 0.25, 0.50]
SIZE_BINS   = [3.0, 6.0, 15.0, float("inf")]
SIZE_LABELS = ["small", "medium", "large"]
STRUCT_6    = nd.generate_binary_structure(3, 1)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def size_cat(d):
    """Igual ao Script 11: '' para < 3 mm (fora da estratificação)."""
    for i in range(len(SIZE_BINS) - 1):
        if SIZE_BINS[i] <= d < SIZE_BINS[i + 1]:
            return SIZE_LABELS[i]
    return ""


def cat_audit(d):
    return "lt3" if d < 3 else size_cat(d)


def diam_axial_bbox_mm(coords_zyx, sp_zyx):
    """Mesma lógica geométrica de `volume_and_diam_axial` (Script 20), aplicada a
    um componente JÁ isolado com 6-conectividade — não reutiliza o label() do 20."""
    sy, sx = sp_zyx[1], sp_zyx[2]
    best = 0.0
    for z in np.unique(coords_zyx[:, 0]):
        sc = coords_zyx[coords_zyx[:, 0] == z]
        y_range = (sc[:, 1].max() - sc[:, 1].min() + 1) * sy
        x_range = (sc[:, 2].max() - sc[:, 2].min() + 1) * sx
        best = max(best, y_range, x_range)
    return float(best)


def diam_sphere_mm(n_vox, sp_xyz):
    vol = float(n_vox) * sp_xyz[0] * sp_xyz[1] * sp_xyz[2]
    return (6.0 * vol / math.pi) ** (1.0 / 3.0) if vol > 0 else 0.0


def read_csv(p):
    with open(p, newline="") as f:
        r = csv.DictReader(f)
        return list(r), r.fieldnames


def write_csv(p, rows, fields):
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    if os.path.exists(OUT_DIR):
        raise SystemExit("FAIL_OUTPUT: OUT_DIR já existe — nunca sobrescrever")
    os.makedirs(os.path.join(OUT_DIR, "analysis"))
    os.makedirs(os.path.join(OUT_DIR, "gates"))
    an = os.path.join(ARCHIVE, "analysis")
    in_sha = {f: sha256(os.path.join(an, f)) for f in sorted(os.listdir(an))}

    dev, internal = [], []
    with open(os.path.join(BASE, "config", "bd1_split.csv"), newline="") as f:
        for r in csv.DictReader(f):
            (dev if r["split"].strip().lower() == "development" else internal).append(r["patient_id"].strip())
    subset_of = {p: "oof" for p in dev}
    subset_of.update({p: "internal" for p in internal})
    labels = os.path.join(raiz_recortes(), "lidc_idri", "labels")

    axial, sphere = {}, {}
    n_comp = 0
    for pid in sorted(subset_of):
        gp = os.path.join(labels, pid, "consensus.nii.gz")
        if not os.path.isfile(gp):
            raise SystemExit(f"FAIL_INPUT: GT ausente {gp}")
        img = sitk.ReadImage(gp)
        b = sitk.GetArrayFromImage(img).astype(np.uint8)
        sp = img.GetSpacing(); sp_zyx = (sp[2], sp[1], sp[0])
        lab, n = nd.label(b, structure=STRUCT_6)
        for gi in range(1, n + 1):
            co = np.argwhere(lab == gi)
            key = (subset_of[pid], pid, gi)
            axial[key]  = diam_axial_bbox_mm(co, sp_zyx)
            sphere[key] = diam_sphere_mm(len(co), sp)
            n_comp += 1
    if n_comp != 2887:
        raise SystemExit(f"FAIL_INPUT: GT components 6-conn = {n_comp}, esperado 2887")

    trans = Counter((cat_audit(sphere[k]), cat_audit(axial[k])) for k in axial)
    computed = {f"{a}->{b}": v for (a, b), v in sorted(trans.items())}
    ref = json.load(open(REFERENCE))["sphere_to_bbox"]
    if computed != ref["matrix"]:
        raise SystemExit(f"FAIL_METHOD: matriz de transição difere da congelada\n{computed}\n{ref['matrix']}")
    changed = sum(v for (a, b), v in trans.items() if a != b)
    if changed != ref["changed"]:
        raise SystemExit("FAIL_METHOD: changed != referência")

    obj, f_obj = read_csv(os.path.join(an, "bd1_object_matches.csv"))
    seen = Counter(); touched_obj = 0
    for r in obj:
        if r["gt_component_id"] == "":
            if r["diam_mm_gt"] != "" or r["size_category"] != "":
                raise SystemExit("FAIL_INPUT: linha FP com campos de tamanho preenchidos")
            continue
        key = (r["subset"], r["patient_id"], int(r["gt_component_id"]))
        if key not in axial:
            raise SystemExit(f"FAIL_INPUT: chave GT sem diâmetro canónico {key}")
        seen[(r["model"],) + key] += 1
        d = axial[key]
        r["diam_mm_gt"] = round(d, 2); r["size_category"] = size_cat(d); touched_obj += 1
    dup = sum(1 for v in seen.values() if v > 1)
    keys_obj = {k[1:] for k in seen}
    missing = set(axial) - keys_obj
    extra = keys_obj - set(axial)
    if dup or missing or extra:
        raise SystemExit(f"FAIL_INPUT: dup={dup} missing={len(missing)} extra={len(extra)}")
    if len(seen) != 2887 * 10:
        raise SystemExit(f"FAIL_INPUT: linhas GT×modelo = {len(seen)}, esperado 28870")

    seg, f_seg = read_csv(os.path.join(an, "bd1_segmentation_metrics.csv"))
    touched_seg = 0
    for r in seg:
        key = (r["subset"], r["patient_id"], int(r["gt_component_id"]))
        if key not in axial:
            raise SystemExit(f"FAIL_INPUT: chave GT sem diâmetro canónico {key}")
        d = axial[key]
        r["diam_mm_gt"] = round(d, 2); r["size_category"] = size_cat(d); touched_seg += 1
    if touched_seg != 2887 * 10 * 3:
        raise SystemExit(f"FAIL_INPUT: linhas seg = {touched_seg}, esperado 86610")

    archs = []
    for r in seg:
        if r["model"] not in archs: archs.append(r["model"])
    size_rows = []
    by = defaultdict(list)
    for r in seg: by[(r["model"], r["subset"])].append(r)
    for arch in archs:
        for subset in ("oof", "internal"):
            sub_all = by[(arch, subset)]
            for cat in SIZE_LABELS:
                for t in THRESHOLDS:
                    sub = [r for r in sub_all if r["size_category"] == cat and float(r["threshold"]) == t]
                    tp = sum(1 for r in sub if r["status"] == "TP")
                    fn = sum(1 for r in sub if r["status"] == "FN")
                    n_gt = tp + fn
                    size_rows.append({"model": arch, "subset": subset, "size_category": cat, "threshold": t,
                                      "n_gt": n_gt, "TP": tp, "FN": fn,
                                      "sensitivity": round(tp / n_gt, 4) if n_gt > 0 else ""})
    if len(size_rows) != 180:
        raise SystemExit("FAIL_OUTPUT: size rows != 180")

    obj0, _ = read_csv(os.path.join(an, "bd1_object_matches.csv"))
    seg0, _ = read_csv(os.path.join(an, "bd1_segmentation_metrics.csv"))
    inv = ("diam_mm_gt", "size_category")
    for a, b in zip(obj0, obj):
        if any(a[c] != str(b[c]) for c in f_obj if c not in inv):
            raise SystemExit("FAIL_OUTPUT: coluna não-tamanho alterada em object_matches")
    for a, b in zip(seg0, seg):
        if any(a[c] != str(b[c]) for c in f_seg if c not in inv):
            raise SystemExit("FAIL_OUTPUT: coluna não-tamanho alterada em segmentation_metrics")

    oa = os.path.join(OUT_DIR, "analysis")
    write_csv(os.path.join(oa, "bd1_object_matches.csv"), obj, f_obj)
    write_csv(os.path.join(oa, "bd1_segmentation_metrics.csv"), seg, f_seg)
    write_csv(os.path.join(oa, "bd1_size_stratified_metrics.csv"), size_rows,
              ["model", "subset", "size_category", "threshold", "n_gt", "TP", "FN", "sensitivity"])
    for f in ("bd1_detection_metrics.csv", "bd1_froc.csv"):
        shutil.copyfile(os.path.join(an, f), os.path.join(oa, f))
    with open(os.path.join(OUT_DIR, "gt_axial_diameters.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["subset", "patient_id", "gt_component_id", "d_axial_bbox_mm", "d_sphere_mm", "size_category"])
        for k in sorted(axial): w.writerow([k[0], k[1], k[2], round(axial[k], 3), round(sphere[k], 3), size_cat(axial[k])])
    out_sha = {f: sha256(os.path.join(oa, f)) for f in sorted(os.listdir(oa))}
    gate = {
        "gate": "SIZE_REPAIR_GATE", "status": "PASS",
        "timestamp": datetime.datetime.now().astimezone().isoformat(),
        "script": "11b_recompute_bd1_size_fields.py", "script_sha256": sha256(os.path.abspath(__file__)),
        "archive_in": ARCHIVE, "inputs_sha256": in_sha, "reference": REFERENCE, "reference_sha256": sha256(REFERENCE),
        "CANONICAL_SIZE_DIAMETER": "MAXIMUM_AXIAL_DIAMETER_MM", "implementation_semantics": "AXIAL_SLICE_BOUNDING_BOX",
        "CONNECTED_COMPONENT_CONNECTIVITY": 6,
        "gt_components": n_comp, "changed_category_total": changed, "transition_matrix": computed,
        "rows_touched": {"object_matches_gt_rows": touched_obj, "segmentation_rows": touched_seg, "size_rows": len(size_rows)},
        "unchanged_byte_copies": {f: in_sha[f] == out_sha[f] for f in ("bd1_detection_metrics.csv", "bd1_froc.csv")},
        "outputs_sha256": out_sha,
    }
    json.dump(gate, open(os.path.join(OUT_DIR, "gates", "SIZE_REPAIR_GATE.json"), "w"), indent=2)
    print(json.dumps({k: gate[k] for k in ("status", "gt_components", "changed_category_total", "rows_touched", "unchanged_byte_copies")}, indent=1))


if __name__ == "__main__":
    main()
