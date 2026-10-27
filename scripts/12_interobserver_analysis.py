#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    12_interobserver_analysis.py

File type:
    Script

Purpose:
    Computes inter-observer agreement (pairwise Dice and mean surface distance) for LIDC-IDRI nodules and LNDb objects.

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
    - report_12.json

Outputs:
    - See docs/file_level_documentation.md.

Dependencies:
    - SimpleITK
    - numpy
    - scipy

Execution:
    PULMONARY_BASE=<project_dir> python scripts/12_interobserver_analysis.py

Portability notes:
    All paths are resolved relative to the project base directory given by the PULMONARY_BASE environment variable. No machine-specific path is required.

Reproducibility notes:
    Environment pinned in requirements.txt and environment/. Seeds and frozen parameters are recorded in the run manifests under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) are not included and must be obtained from their original repositories.
"""

import os
import csv
import json
import logging
import datetime
from itertools import combinations

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as nd
from scipy.optimize import linear_sum_assignment
from collections import Counter

from pulmonary_base import BASE, hash_corpo_manifesto
from caminhos_canonicos import _carregar, sha256, FalhaEntrada, CURRENT_ANNOT

ANALYSIS_DIR    = os.path.join(BASE, "analysis")
LOGS_DIR        = os.path.join(BASE, "logs")
LNDB_MANIFEST   = os.path.join(BASE, "config", "lndb_annotations_manifest.json")
PER_READER_LNDB = os.path.join(BASE, "annotations", "per_reader", "lndb")

IOU_THRESH_LNDB = 0.10
STRUCT_6        = nd.generate_binary_structure(3, 1)

LIDC_PRIMARY_STATUS = "PASS"
LNDB_PRIMARY_STATUS = ("ANALYTIC_UNIT", "ANALYTIC_REPRESENTATIVE")

for _d in [ANALYSIS_DIR, LOGS_DIR]:
    os.makedirs(_d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "interobserver.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def fail(msg):
    log.error(msg)
    raise SystemExit("FAIL_INPUT")


def check_geometry(img_a: sitk.Image, img_b: sitk.Image, rotulo: str) -> None:
    """GEOMETRY_GATE: mesma grelha física (shape, spacing, origin, direction
    com rtol=0/atol=1e-6). Sem isto, Dice/MSD/IoU comparam voxels de sítios
    diferentes. Falha → FAIL_INPUT."""
    ok = (img_a.GetSize() == img_b.GetSize()
          and np.allclose(img_a.GetSpacing(), img_b.GetSpacing(), rtol=0, atol=1e-6)
          and np.allclose(img_a.GetOrigin(), img_b.GetOrigin(), rtol=0, atol=1e-6)
          and np.allclose(img_a.GetDirection(), img_b.GetDirection(), rtol=0, atol=1e-6))
    if not ok:
        fail(f"GEOMETRY_GATE: grelhas incompatíveis em {rotulo}")


def dice_coefficient(arr1: np.ndarray, arr2: np.ndarray) -> float:
    inter = np.logical_and(arr1, arr2).sum()
    denom = arr1.sum() + arr2.sum()
    return float(2 * inter / denom) if denom > 0 else float("nan")


def mean_surface_distance_mm(m1: sitk.Image, m2: sitk.Image) -> float:
    """
    Distância média de superfície simétrica em mm via EDT com crop ao bounding box.
    Crop evita rodar EDT num volume CT inteiro (512×512×200 = 52M voxels).
    """
    arr1 = sitk.GetArrayFromImage(m1).astype(bool)
    arr2 = sitk.GetArrayFromImage(m2).astype(bool)
    sp_xyz = m1.GetSpacing()
    sp     = (sp_xyz[2], sp_xyz[1], sp_xyz[0])

    union = arr1 | arr2
    nz = np.nonzero(union)
    if len(nz[0]) == 0:
        return float("nan")

    pad = 5
    z0 = max(0, int(nz[0].min()) - pad); z1 = min(arr1.shape[0], int(nz[0].max()) + pad + 1)
    y0 = max(0, int(nz[1].min()) - pad); y1 = min(arr1.shape[1], int(nz[1].max()) + pad + 1)
    x0 = max(0, int(nz[2].min()) - pad); x1 = min(arr1.shape[2], int(nz[2].max()) + pad + 1)

    c1 = arr1[z0:z1, y0:y1, x0:x1]
    c2 = arr2[z0:z1, y0:y1, x0:x1]

    surf1 = c1 & ~nd.binary_erosion(c1)
    surf2 = c2 & ~nd.binary_erosion(c2)

    if not surf1.any() or not surf2.any():
        return float("nan")

    dt2 = nd.distance_transform_edt(~surf2, sampling=sp)
    dt1 = nd.distance_transform_edt(~surf1, sampling=sp)
    msd_12 = dt2[surf1].mean()
    msd_21 = dt1[surf2].mean()
    return float((msd_12 + msd_21) / 2)


def load_lidc_sources():
    """Ponteiro do 04 → manifesto (hash) → tabela de objectos + disposição por paciente."""
    p, man, modo = _carregar(CURRENT_ANNOT, "04", "annotations_manifest", "anotacoes do 04")
    m = json.load(open(man))
    cohort_hash = m.get("manifest_sha256") or sha256(man)
    rows_m = m.get("rows") or []
    if not rows_m:
        fail(f"manifesto do 04 sem `rows`: {man}")
    pat = {r["patient_id"]: r for r in rows_m}

    table = p.get("object_table", os.path.join("annotations", "object_tables", "lidc_nodules.csv"))
    table = table if os.path.isabs(table) else os.path.join(BASE, table)
    if not os.path.isfile(table):
        fail(f"tabela de objectos ausente: {table}")
    with open(table, newline="") as f:
        rows = list(csv.DictReader(f))
    for c in ("patient_id", "nodule_num", "reader_key", "mask_relpath"):
        if c not in rows[0]:
            fail(f"tabela sem coluna obrigatória `{c}`: {table}")
    log.info("Fontes LIDC: manifesto %s (%s, %s), tabela %s (%d linhas)",
             man, cohort_hash[:12], modo, os.path.relpath(table, BASE), len(rows))
    return rows, pat, cohort_hash, {"pointer": CURRENT_ANNOT, "manifest": man,
                                    "manifest_sha256": cohort_hash, "table": table,
                                    "table_sha256": sha256(table)}


def load_lndb_sources():
    if not os.path.isfile(LNDB_MANIFEST):
        fail(f"manifesto LNDb ausente: {LNDB_MANIFEST}")
    m = json.load(open(LNDB_MANIFEST))
    declared = m.get("manifest_sha256")
    body = hash_corpo_manifesto(m)
    if not declared or declared != body:
        fail(f"manifesto LNDb: hash do corpo {body[:12]} != declarado {str(declared)[:12]}")
    rows = m.get("rows") or []
    if not rows:
        fail("manifesto LNDb sem `rows`")
    if not os.path.isdir(PER_READER_LNDB):
        fail(f"dir LNDb per_reader ausente: {PER_READER_LNDB}")
    return {r["lndb_id"]: r for r in rows}, declared, {"manifest": LNDB_MANIFEST,
                                                       "manifest_sha256": declared}


def analyze_nodule(patient_id, nodule_num, reader_rows, pat_row, cohort_hash,
                   pairwise_rows, summary_rows):
    status_04   = pat_row.get("status_04", "")
    disposition = pat_row.get("disposition", "")
    in_primary  = (disposition == "INCLUDED") and (status_04 == LIDC_PRIMARY_STATUS)

    loaded = []
    keys = [r["reader_key"] for r in reader_rows]
    base = {"patient_id": patient_id, "nodule_num": nodule_num,
            "in_primary": int(in_primary), "status_04": status_04,
            "disposition": disposition, "cohort_hash": cohort_hash}
    if (status_04 == "HOLD_ANNOTATION_OUTSIDE_RETAINED_BLOCK"
            and all(int(r.get("volume_vox", "1")) == 0 for r in reader_rows)):
        summary_rows.append({**base, "eligible": 0,
                             "exclusion_reason": "ANNOTATION_OUTSIDE_RETAINED_BLOCK",
                             "n_readers": len(set(keys)), "n_annotations": len(keys),
                             "n_pairs": 0, "mean_dice": "", "std_dice": "",
                             "mean_msd_mm": "", "std_msd_mm": ""})
        return
    if len(set(keys)) != len(keys):
        summary_rows.append({**base, "eligible": 0,
                             "exclusion_reason": "MULTI_ANNOTATION_SAME_READER",
                             "n_readers": len(set(keys)), "n_annotations": len(keys),
                             "n_pairs": 0, "mean_dice": "", "std_dice": "",
                             "mean_msd_mm": "", "std_msd_mm": ""})
        return
    for r in sorted(reader_rows, key=lambda x: x["reader_key"]):
        p = r["mask_relpath"]
        p = p if os.path.isabs(p) else os.path.join(BASE, p)
        if not os.path.isfile(p):
            fail(f"máscara ausente: {p}")
        try:
            img = sitk.ReadImage(p, sitk.sitkUInt8)
        except Exception as e:
            fail(f"erro ao ler {p}: {e}")
        raw = sitk.GetArrayFromImage(img)
        if not np.isin(raw, (0, 1)).all():
            fail(f"máscara não binária: {p}")
        arr = raw.astype(bool)
        if not arr.any():
            fail(f"máscara por leitor VAZIA (ambígua — ausência ≠ vazio): {p}")
        loaded.append((r["reader_key"], img, arr))

    n_readers = len(loaded)
    expected_pairs = n_readers * (n_readers - 1) // 2

    dice_vals, msd_vals, n_pairs = [], [], 0
    for (k1, m1, a1), (k2, m2, a2) in combinations(loaded, 2):
        check_geometry(m1, m2, f"{patient_id} nódulo {nodule_num} {k1}/{k2}")
        d   = dice_coefficient(a1, a2)
        msd = mean_surface_distance_mm(m1, m2)
        if np.isnan(d) or d < 0 or d > 1 or np.isnan(msd) or msd < 0:
            fail(f"{patient_id} nódulo {nodule_num} par {k1}/{k2}: Dice={d} MSD={msd}")
        dice_vals.append(d); msd_vals.append(msd); n_pairs += 1
        pairwise_rows.append({**base, "reader1": k1, "reader2": k2,
                              "dice": round(d, 4), "msd_mm": round(msd, 4)})
    if n_pairs != expected_pairs:
        fail(f"{patient_id} nódulo {nodule_num}: pares {n_pairs} != R(R-1)/2 = {expected_pairs}")

    summary_rows.append({
        **base, "eligible": 1, "exclusion_reason": "",
        "n_readers":   n_readers, "n_annotations": n_readers,
        "n_pairs":     n_pairs,
        "mean_dice":   round(float(np.mean(dice_vals)), 4) if dice_vals else "",
        "std_dice":    round(float(np.std(dice_vals)),  4) if dice_vals else "",
        "mean_msd_mm": round(float(np.mean(msd_vals)), 4)  if msd_vals else "",
        "std_msd_mm":  round(float(np.std(msd_vals)),  4)  if msd_vals else "",
    })


def write_csv(path, rows):
    if not rows:
        fail(f"nenhuma linha para salvar: {path}")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    log.info(f"Salvo: {path} ({len(rows)} linhas)")


def _iou_matrix_lndb(lab_a, lab_b, n_a, n_b):
    """Matriz IoU (n_a × n_b) entre componentes de duas máscaras."""
    if n_a == 0 or n_b == 0:
        return np.zeros((n_a, n_b))
    a_flat = lab_a.ravel()
    b_flat = lab_b.ravel()
    sz_a = np.bincount(a_flat, minlength=n_a + 1)[1:]
    sz_b = np.bincount(b_flat, minlength=n_b + 1)[1:]
    mat = np.zeros((n_a, n_b))
    ov = (a_flat > 0) & (b_flat > 0)
    if ov.any():
        for (ai, bi), inter in Counter(
            zip(a_flat[ov].tolist(), b_flat[ov].tolist())
        ).items():
            i, j = ai - 1, bi - 1
            u = sz_a[i] + sz_b[j] - inter
            mat[i, j] = inter / u if u > 0 else 0.0
    return mat


def process_lndb_interobserver(man_rows, cohort_hash):
    """
    Para cada CT LNDb com ≥2 leitores com máscara, concordância de detecção ao
    nível de objecto entre todos os pares de leitores (Hungarian, IoU ≥ τ,
    um-para-um, só arestas admissíveis). Retorna (pairwise_rows, summary_rows, enum).
    """
    pairwise_rows, summary_rows = [], []
    enum = Counter()
    case_dirs = sorted(os.listdir(PER_READER_LNDB))
    log.info(f"LNDb: {len(case_dirs)} CTs com máscaras por leitor …")

    for lndb_id in case_dirs:
        if lndb_id not in man_rows:
            fail(f"{lndb_id}: directório per_reader sem linha no manifesto LNDb")
        mr = man_rows[lndb_id]
        status = mr.get("analytic_status", "")
        in_primary = status in LNDB_PRIMARY_STATUS
        case_dir = os.path.join(PER_READER_LNDB, lndb_id)
        mask_files = sorted(f for f in os.listdir(case_dir) if f.endswith(".nii.gz"))
        if len(mask_files) != int(mr.get("n_per_reader_masks", -1)):
            fail(f"{lndb_id}: {len(mask_files)} máscaras no disco != manifesto {mr.get('n_per_reader_masks')}")
        enum[("cts_with_masks", status)] += 1
        if len(mask_files) < 2:
            continue
        enum[("cts_2plus_readers", status)] += 1

        loaded = []
        for fn in mask_files:
            p = os.path.join(case_dir, fn)
            try:
                img = sitk.ReadImage(p, sitk.sitkUInt8)
            except Exception as e:
                fail(f"{p}: erro ao ler — {e}")
            arr = sitk.GetArrayFromImage(img).astype(np.uint8)
            if not np.isin(arr, (0, 1)).all():
                fail(f"máscara não binária: {p}")
            if loaded:
                check_geometry(loaded[0][3], img, f"{lndb_id} {fn}")
            lab, n = nd.label(arr, structure=STRUCT_6)
            if n == 0:
                fail(f"máscara por leitor VAZIA (ambígua): {p}")
            loaded.append((fn.replace(".nii.gz", ""), lab, n, img))

        base = {"lndb_id": lndb_id, "in_primary": int(in_primary),
                "analytic_status": status, "cohort_hash": cohort_hash}
        case_matched = case_pairs = 0
        for (name_a, lab_a, n_a, _), (name_b, lab_b, n_b, _) in combinations(loaded, 2):
            mat = _iou_matrix_lndb(lab_a, lab_b, n_a, n_b)
            w = np.where(mat >= IOU_THRESH_LNDB, 1000.0 + mat, 0.0)
            rows_h, cols_h = linear_sum_assignment(-w)
            matched_pairs = [(r, c) for r, c in zip(rows_h, cols_h) if mat[r, c] >= IOU_THRESH_LNDB]
            n_matched = len(matched_pairs)
            if len({r for r, _ in matched_pairs}) != n_matched or len({c for _, c in matched_pairs}) != n_matched:
                fail(f"{lndb_id} {name_a}/{name_b}: pareamento não um-para-um")
            n_only_a = n_a - n_matched
            n_only_b = n_b - n_matched
            if n_matched + n_only_a != n_a or n_matched + n_only_b != n_b or n_only_a < 0 or n_only_b < 0:
                fail(f"{lndb_id} {name_a}/{name_b}: identidade matched+only != total")
            concordance = n_matched / max(n_a, n_b)
            if not (0.0 <= concordance <= 1.0):
                fail(f"{lndb_id} {name_a}/{name_b}: concordância fora de [0,1]")
            pairwise_rows.append({**base, "reader_a": name_a, "reader_b": name_b,
                                  "n_nods_a": n_a, "n_nods_b": n_b,
                                  "n_matched": n_matched, "n_only_a": n_only_a,
                                  "n_only_b": n_only_b, "concordance": round(concordance, 4)})
            case_matched += n_matched
            case_pairs += 1
        enum[("reader_pair_ct_combinations", status)] += case_pairs
        summary_rows.append({**base, "n_readers": len(loaded),
                             "mean_nods_per_reader": round(float(np.mean([n for _, _, n, _ in loaded])), 2),
                             "n_pairs": case_pairs, "total_matched": case_matched})

    log.info(f"LNDb interobservador: {len(pairwise_rows)} pares processados")
    return pairwise_rows, summary_rows, enum


def main():
    t0 = datetime.datetime.now().astimezone()
    rows, pat, cohort_hash, prov_lidc = load_lidc_sources()

    nodules = {}
    for r in rows:
        nodules.setdefault((r["patient_id"], r["nodule_num"]), []).append(r)
    for pid in {k[0] for k in nodules}:
        if pid not in pat:
            fail(f"{pid}: na tabela mas ausente do manifesto do 04")
        n_tab = sum(1 for r in rows if r["patient_id"] == pid)
        n_man = int(pat[pid].get("n_per_reader_masks", -1))
        if n_tab != n_man:
            fail(f"{pid}: {n_tab} máscaras na tabela != {n_man} no manifesto")

    pairwise_rows, summary_rows = [], []
    total = len(nodules)
    log.info(f"LIDC-IDRI: processando {total} nódulos …")
    for i, ((pid, nnum), rr) in enumerate(sorted(nodules.items())):
        if i % 200 == 0:
            log.info(f"  {i}/{total} …")
        analyze_nodule(pid, nnum, rr, pat[pid], cohort_hash, pairwise_rows, summary_rows)

    write_csv(os.path.join(ANALYSIS_DIR, "interobserver_pairwise.csv"), pairwise_rows)
    write_csv(os.path.join(ANALYSIS_DIR, "interobserver_summary.csv"), summary_rows)

    man_lndb, lndb_hash, prov_lndb = load_lndb_sources()
    lndb_pairwise, lndb_summary, enum_lndb = process_lndb_interobserver(man_lndb, lndb_hash)
    write_csv(os.path.join(ANALYSIS_DIR, "lndb_interobserver_pairwise.csv"), lndb_pairwise)
    write_csv(os.path.join(ANALYSIS_DIR, "lndb_interobserver_summary.csv"), lndb_summary)

    def lidc_enum(sel):
        s = [r for r in summary_rows if sel(r)]
        e = [r for r in s if r["eligible"]]
        return {"nodules": len(s), "patients": len({r["patient_id"] for r in s}),
                "eligible_nodules": len(e), "eligible_2plus": sum(1 for r in e if r["n_readers"] >= 2),
                "eligible_1": sum(1 for r in e if r["n_readers"] == 1),
                "pairs": sum(r["n_pairs"] for r in e),
                "excluded_by_reason": dict(Counter(r["exclusion_reason"] for r in s if not r["eligible"]))}
    prim = [r for r in pairwise_rows if r["in_primary"]]
    lprim = [r for r in lndb_pairwise if r["in_primary"]]
    report = {
        "script": "12_interobserver_analysis.py", "run_at": t0.isoformat(),
        "provenance": {"lidc": prov_lidc, "lndb": prov_lndb},
        "populations": {"LIDC_PRIMARY": "disposition == INCLUDED and status_04 == PASS",
                        "LNDB_PRIMARY": "analytic_status in " + str(LNDB_PRIMARY_STATUS)},
        "lidc": {"all_labelled": lidc_enum(lambda r: True),
                 "primary": lidc_enum(lambda r: r["in_primary"]),
                 "by_status_04": {s: lidc_enum(lambda r, s=s: r["status_04"] == s)
                                  for s in sorted({r["status_04"] for r in summary_rows})},
                 "primary_mean_dice": round(float(np.mean([r["dice"] for r in prim])), 4) if prim else None,
                 "primary_mean_msd_mm": round(float(np.mean([r["msd_mm"] for r in prim])), 4) if prim else None,
                 "historical_2626_1863_763_7438": "HISTORICAL_SUPERSEDED",
                 "excluded_groups": [f"{r['patient_id']}#{r['nodule_num']}|{r['status_04']}|{r['exclusion_reason']}" for r in summary_rows if not r["eligible"]],
                 "eligibility_rule": "unique(reader_key within nodule); SCRIPT04_DUPLICATE_READER_IMPACT_GATE = CASE A (consensus invariant 6/6)"},
        "lndb": {"enumeration": {f"{k[0]}|{k[1]}": v for k, v in sorted(enum_lndb.items())},
                 "primary_cts_2plus": sum(v for k, v in enum_lndb.items() if k[0] == "cts_2plus_readers" and k[1] in LNDB_PRIMARY_STATUS),
                 "primary_reader_pair_ct_combinations": len(lprim),
                 "primary_mean_concordance": round(float(np.mean([r["concordance"] for r in lprim])), 4) if lprim else None,
                 "all_cts_2plus": sum(v for k, v in enum_lndb.items() if k[0] == "cts_2plus_readers"),
                 "all_reader_pair_ct_combinations": len(lndb_pairwise)},
        "gates": {"combinatorics_R(R-1)/2": "PASS", "metric_domains": "PASS", "geometry": "PASS",
                  "duplicate_pairs": 0, "missing_silently_converted": 0, "source_provenance": "PASS"},
        "connectivity": "SCIPY 6 (generate_binary_structure(3,1))",
    }
    with open(os.path.join(ANALYSIS_DIR, "report_12.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info("Script 12 concluído.")


if __name__ == "__main__":
    try:
        main()
    except FalhaEntrada as e:
        log.error(str(e))
        raise SystemExit("FAIL_INPUT")
