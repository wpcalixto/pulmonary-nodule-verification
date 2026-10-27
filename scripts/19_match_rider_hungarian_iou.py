#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    19_match_rider_hungarian_iou.py

File type:
    Script

Purpose:
    Matches RIDER test-retest components by IoU with the Hungarian algorithm.

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
    - output_hashes.csv

Outputs:
    - manifest.json

Dependencies:
    - SimpleITK
    - numpy
    - pandas
    - scipy

Execution:
    PULMONARY_BASE=<project_dir> python scripts/19_match_rider_hungarian_iou.py

Portability notes:
    All paths are resolved relative to the project base directory given by the PULMONARY_BASE environment variable. No machine-specific path is required.

Reproducibility notes:
    Environment pinned in requirements.txt and environment/. Seeds and frozen parameters are recorded in the run manifests under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) are not included and must be obtained from their original repositories.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage as nd
from scipy.optimize import linear_sum_assignment

from pulmonary_base import BASE
from caminhos_canonicos import raiz_recortes, ct_de, FalhaEntrada

PRED_RIDER_DIR = os.path.join(BASE, "predictions", "rider")
ANALYSIS_DIR   = os.path.join(BASE, "analysis")
LOGS_DIR       = os.path.join(BASE, "logs")

ARCHS = ["2D", "2D_REM", "2D_REL", "3D_F", "3D_FREM", "3D_CF", "3D_CFREM",
         "F100_lowres", "F100_ResEncM", "F100_fullres"]
STRUCT_6 = nd.generate_binary_structure(3, 1)

for _d in (ANALYSIS_DIR, LOGS_DIR):
    os.makedirs(_d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(os.path.join(LOGS_DIR, "match_rider_hungarian_iou.log")),
              logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def fail_input(m):
    log.error(m); raise SystemExit("FAIL_INPUT")


def fail_output(m):
    log.error(m); raise SystemExit("FAIL_OUTPUT")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def same_geometry(a, b):
    return (a.GetSize() == b.GetSize()
            and np.allclose(a.GetSpacing(), b.GetSpacing(), rtol=0, atol=1e-6)
            and np.allclose(a.GetOrigin(), b.GetOrigin(), rtol=0, atol=1e-6)
            and np.allclose(a.GetDirection(), b.GetDirection(), rtol=0, atol=1e-6))


def label_native(path, ref=None):
    """Rótulos nativos com 6-conectividade. Devolve (imagem, rotulado, n)."""
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    if not np.isin(arr, (0, 1)).all():
        fail_input(f"predição não binária: {path}")
    if ref is not None and not same_geometry(img, ref):
        fail_input(f"predição com geometria != CT: {path}")
    lab, n = nd.label(arr > 0, structure=STRUCT_6)
    return img, lab.astype(np.int32), int(n)


def propagate_labels(lab2_native, img2, ref_img, tx):
    """Propaga a IMAGEM DE RÓTULOS Int32 de CT2 para a grelha de CT1 (NN).
    Não há nova rotulagem: os IDs nativos atravessam a transformação."""
    li = sitk.GetImageFromArray(lab2_native)
    li.CopyInformation(img2)
    li = sitk.Cast(li, sitk.sitkInt32)
    out = sitk.Resample(li, ref_img, tx, sitk.sitkNearestNeighbor, 0, sitk.sitkInt32)
    return sitk.GetArrayFromImage(out).astype(np.int32)


def iou_matrix(lab1, n1, lab2r, n2):
    """IoU real na geometria de CT1, por intersecção esparsa."""
    m = np.zeros((n1, n2), float)
    if n1 == 0 or n2 == 0:
        return m
    a = lab1.ravel(); b = lab2r.ravel()
    sa = np.bincount(a, minlength=n1 + 1)[1:]
    sb = np.bincount(b, minlength=n2 + 1)[1:]
    ov = (a > 0) & (b > 0)
    if ov.any():
        pairs = a[ov].astype(np.int64) * (n2 + 1) + b[ov].astype(np.int64)
        cnt = np.bincount(pairs)
        nz = np.nonzero(cnt)[0]
        for k in nz:
            i = int(k // (n2 + 1)) - 1; j = int(k % (n2 + 1)) - 1
            inter = int(cnt[k]); u = sa[i] + sb[j] - inter
            if u > 0:
                m[i, j] = inter / u
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s18-run", required=True, help="pasta do run canónico do Script 18")
    ap.add_argument("--s18-manifest-sha256", required=True)
    ap.add_argument("--inspection", default="", help="JSON com a classificação dos pares FLAG")
    ap.add_argument("--limit-pairs", type=int, default=0)
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()
    t0 = datetime.datetime.now().astimezone()

    man18_p = os.path.join(args.s18_run, "manifest.json")
    if not os.path.isfile(man18_p):
        fail_input(f"run do Script 18 sem manifest.json: {args.s18_run}")
    h18 = sha256(man18_p)
    if h18 != args.s18_manifest_sha256:
        fail_input(f"INPUT_AUTHORITY: SHA do manifesto do 18 {h18[:12]} != {args.s18_manifest_sha256[:12]}")
    man18 = json.load(open(man18_p))
    qc_p = os.path.join(args.s18_run, "rider_registration_qc.csv")
    aqc_p = os.path.join(args.s18_run, "rider_registration_architecture_qc.csv")
    for p in (qc_p, aqc_p, os.path.join(args.s18_run, "output_hashes.csv")):
        if not os.path.isfile(p):
            fail_input(f"produto do Script 18 ausente: {p}")
    qc = pd.read_csv(qc_p); aqc = pd.read_csv(aqc_p)
    oh = pd.read_csv(os.path.join(args.s18_run, "output_hashes.csv"))
    dir_run = os.path.basename(os.path.normpath(args.s18_run))
    if man18["run_id"] != dir_run:
        fail_input(f"run_id do manifesto ({man18['run_id']}) != pasta ({dir_run})")

    th_p = os.path.join(args.s18_run, "transform_hashes.csv")
    if not os.path.isfile(th_p):
        fail_input(f"Script 18 sem transform_hashes.csv (run anterior à persistência "
                   f"da transformação?): {th_p}")
    th = pd.read_csv(th_p)
    if th["patient_id"].duplicated().any():
        fail_input("transform_hashes.csv com patient_id repetido")
    tfm_sha = dict(zip(th["patient_id"], th["sha256"]))
    if set(tfm_sha) != set(qc["patient_id"]):
        fail_input("conjunto de .tfm != conjunto de pares do QC do Script 18")
    tfm_de = {}
    for pid_, sha_esperado in tfm_sha.items():
        tp = os.path.join(args.s18_run, "transforms", f"{pid_}.tfm")
        if not os.path.isfile(tp):
            fail_input(f"TFM_SHA_GATE: .tfm ausente para {pid_}: {tp}")
        h = sha256(tp)
        if h != sha_esperado:
            fail_input(f"TFM_SHA_GATE: SHA do .tfm de {pid_} {h[:12]} != {sha_esperado[:12]}")
        tfm_de[pid_] = tp
    log.info(f"Script 18 canónico: {man18['run_id']} ({h18[:12]}), "
             f"{len(qc)} pares, {len(aqc)} combinações, {len(oh)} saídas, "
             f"{len(tfm_de)} transformações .tfm verificadas por SHA")

    insp = {}
    if args.inspection:
        if not os.path.isfile(args.inspection):
            fail_input(f"ficheiro de inspecção ausente: {args.inspection}")
        insp = {r["patient_id"]: r["classification"] for r in json.load(open(args.inspection))["decisions"]}
    excluded = {}
    for _, r in qc.iterrows():
        st = r["registration_status"]
        if st == "BLOCKED_REGISTRATION":
            excluded[r["patient_id"]] = "BLOCKED_REGISTRATION"
        elif st == "FLAG_FOR_INSPECTION":
            d = insp.get(r["patient_id"])
            if d is None:
                fail_input(f"par {r['patient_id']} FLAG_FOR_INSPECTION sem classificação adjudicada")
            if d == "BLOCKED_REGISTRATION":
                excluded[r["patient_id"]] = "BLOCKED_AFTER_INSPECTION"
            elif d != "QC_ACCEPTED_AFTER_INSPECTION":
                fail_input(f"classificação inválida para {r['patient_id']}: {d}")
    pairs_all = sorted(qc["patient_id"].unique())
    pairs = [p for p in pairs_all if p not in excluded]
    if args.limit_pairs:
        pairs = pairs[: args.limit_pairs]
    log.info(f"Pares: {len(pairs_all)} no 18, {len(excluded)} excluídos {excluded or ''}, "
             f"{len(pairs)} a parear → {len(pairs)*len(ARCHS)} combinações esperadas")

    raiz = raiz_recortes()
    run_id = f"s19_{t0:%Y%m%dT%H%M%S}_{os.getpid()}"
    out_root = os.path.join(args.out_root or os.path.join(ANALYSIS_DIR, "rider_matching"), run_id)
    os.makedirs(out_root, exist_ok=True)

    pair_rows, unm_rows, audit_rows = [], [], []
    n_zero_transformed = 0
    for i, pid in enumerate(pairs):
        r = qc[qc["patient_id"] == pid].iloc[0]
        tx = sitk.ReadTransform(tfm_de[pid])
        ct1 = sitk.ReadImage(ct_de(raiz, "rider", f"{pid}_1"))
        log.info(f"  [{i+1}/{len(pairs)}] {pid}")
        for arch in ARCHS:
            p1 = os.path.join(PRED_RIDER_DIR, arch, f"{pid}_1.nii.gz")
            p2 = os.path.join(PRED_RIDER_DIR, arch, f"{pid}_2.nii.gz")
            for p in (p1, p2):
                if not os.path.isfile(p):
                    fail_input(f"predição nativa ausente: {p}")
            _, lab1, n1 = label_native(p1, ct1)
            img2, lab2, n2 = label_native(p2)
            lab2r = propagate_labels(lab2, img2, ct1, tx)

            reg_p = os.path.join(args.s18_run, arch, f"{pid}_ct2_in_ct1.nii.gz")
            if not os.path.isfile(reg_p):
                reg_p = os.path.join(BASE, aqc[(aqc.patient_id == pid) & (aqc.architecture == arch)]
                                     ["output_path"].iloc[0])
            if not os.path.isfile(reg_p):
                fail_input(f"máscara registada do Script 18 ausente: {reg_p}")
            reg = sitk.GetArrayFromImage(sitk.ReadImage(reg_p))
            if not np.array_equal((lab2r > 0).astype(np.uint8), (reg > 0).astype(np.uint8)):
                fail_output(f"suporte dos rótulos propagados != máscara registada do 18: {pid}/{arch}")

            present = np.bincount(lab2r.ravel(), minlength=n2 + 1)[1:]
            zeros = [j + 1 for j in range(n2) if present[j] == 0]
            n_zero_transformed += len(zeros)

            if n1 == 0 and n2 == 0:
                status = "EMPTY_BOTH"
            elif n1 == 0:
                status = "EMPTY_CT1"
            elif n2 == 0:
                status = "EMPTY_CT2"
            else:
                status = "OK"

            m = iou_matrix(lab1, n1, lab2r, n2)
            pairs_ij = []
            if n1 and n2 and m.size:
                rr, cc = linear_sum_assignment(-m)
                pairs_ij = [(int(a) + 1, int(b) + 1, float(m[a, b])) for a, b in zip(rr, cc) if m[a, b] > 0]
            m1 = {a for a, _, _ in pairs_ij}; m2 = {b for _, b, _ in pairs_ij}
            if len(m1) != len(pairs_ij) or len(m2) != len(pairs_ij):
                fail_output(f"pareamento não um-para-um: {pid}/{arch}")
            for a, b, v in pairs_ij:
                if not v > 0:
                    fail_output(f"par retido com IoU <= 0: {pid}/{arch}")
                pair_rows.append({"patient_id": pid, "architecture": arch,
                                  "component_id_ct1": a, "component_id_ct2_native": b,
                                  "iou": round(v, 6),
                                  "ct2_transformed_voxels": int(present[b - 1])})
            for a in range(1, n1 + 1):
                if a not in m1:
                    unm_rows.append({"patient_id": pid, "architecture": arch, "side": "CT1_only",
                                     "component_id": a, "transformed_voxels": "", "flag": ""})
            for b in range(1, n2 + 1):
                if b not in m2:
                    unm_rows.append({"patient_id": pid, "architecture": arch, "side": "CT2_only",
                                     "component_id": b, "transformed_voxels": int(present[b - 1]),
                                     "flag": "TRANSFORMED_COMPONENT_EMPTY" if present[b - 1] == 0 else ""})
            np_, c1o, c2o = len(pairs_ij), n1 - len(m1), n2 - len(m2)
            if n1 != np_ + c1o or n2 != np_ + c2o:
                fail_output(f"identidade aritmética violada: {pid}/{arch}")
            audit_rows.append({
                "patient_id": pid, "architecture": arch, "status": status,
                "n_ct1_native": n1, "n_ct2_native": n2,
                "n_ct2_registered_islands": int(nd.label(lab2r > 0, structure=STRUCT_6)[1]),
                "n_pairs": np_, "n_ct1_only": c1o, "n_ct2_only": c2o,
                "n_ct2_transformed_empty": len(zeros),
                "identity_ct1_ok": int(n1 == np_ + c1o), "identity_ct2_ok": int(n2 == np_ + c2o),
                "max_iou": round(float(max([v for _, _, v in pairs_ij], default=0.0)), 6),
                "registration_status": r["registration_status"],
                "support_matches_s18": 1})

    da = pd.DataFrame(audit_rows); dp = pd.DataFrame(pair_rows); du = pd.DataFrame(unm_rows)
    esperado = len(pairs) * len(ARCHS)
    if len(da) != esperado:
        fail_output(f"COMPLETENESS: {len(da)} linhas de auditoria != {esperado}")
    esperadas = {(p, a) for p in pairs for a in ARCHS}
    obtidas = set(zip(da.patient_id, da.architecture))
    if len(obtidas) != len(da):
        fail_output("combinação paciente×arquitectura repetida na auditoria")
    if obtidas - esperadas:
        fail_output(f"combinações inesperadas: {sorted(obtidas - esperadas)[:5]}")
    if esperadas - obtidas:
        fail_output(f"combinações em falta: {sorted(esperadas - obtidas)[:5]}")
    if not dp.empty and dp.duplicated(["patient_id", "architecture", "component_id_ct1"]).any():
        fail_output("componente CT1 repetido em pares")
    if not dp.empty and dp.duplicated(["patient_id", "architecture", "component_id_ct2_native"]).any():
        fail_output("componente CT2 repetido em pares")
    if (da["identity_ct1_ok"] == 0).any() or (da["identity_ct2_ok"] == 0).any():
        fail_output("identidade aritmética violada")

    p_pairs = os.path.join(out_root, "rider_iou_pairs.csv")
    p_unm = os.path.join(out_root, "rider_iou_unmatched.csv")
    p_aud = os.path.join(out_root, "rider_iou_audit.csv")
    dp.to_csv(p_pairs, index=False); du.to_csv(p_unm, index=False); da.to_csv(p_aud, index=False)

    prev = os.path.join(ANALYSIS_DIR, "rider_iou_pairs.csv")
    CMP_COLS = ["patient_id", "architecture", "component_id_ct1", "component_id_ct2",
                "in_old", "in_new", "status"]
    cmp_rows = []
    hist_state = ("AVAILABLE" if os.path.isfile(prev)
                  else "NOT_AVAILABLE_NO_HISTORICAL_PRODUCT")
    if os.path.isfile(prev) and not dp.empty:
        old = pd.read_csv(prev)
        kn = set(zip(dp.patient_id, dp.architecture, dp.component_id_ct1, dp.component_id_ct2_native))
        ko = set(zip(old.patient_id, old.architecture, old.get("component_id_ct1", pd.Series(dtype=int)),
                     old.get("component_id_ct2", pd.Series(dtype=int))))
        for k in sorted(kn | ko, key=str):
            cmp_rows.append({"patient_id": k[0], "architecture": k[1], "component_id_ct1": k[2],
                             "component_id_ct2": k[3], "in_old": k in ko, "in_new": k in kn,
                             "status": "kept" if k in kn and k in ko else ("added" if k in kn else "removed")})
    pd.DataFrame(cmp_rows, columns=CMP_COLS).to_csv(
        os.path.join(out_root, "rider_old_vs_iou_matching.csv"), index=False)
    if not cmp_rows and hist_state == "AVAILABLE":
        fail_output("produto histórico existe mas a comparação saiu vazia")

    outs = [p_pairs, p_unm, p_aud]
    pd.DataFrame([{"path": os.path.relpath(p, BASE), "sha256": sha256(p)} for p in outs]).to_csv(
        os.path.join(out_root, "output_hashes.csv"), index=False)
    man = {
        "script": "19_match_rider_hungarian_iou.py", "run_id": run_id,
        "script_sha256": sha256(os.path.abspath(__file__)),
        "run_at": t0.isoformat(), "end_at": datetime.datetime.now().astimezone().isoformat(),
        "source": {"script18_run": man18["run_id"], "script18_manifest_sha256": h18,
                   "s18_run_dir": args.s18_run, "inspection_file": args.inspection},
        "transform_authority": {
            "source": "Script 18 canonical transforms/<patient_id>.tfm",
            "loaded_with": "sitk.ReadTransform",
            "reconstruction_from_csv_parameters": "PROHIBITED — not used anywhere",
            "TFM_SHA_GATE": f"{len(tfm_de)}/{len(qc)} verified against transform_hashes.csv"},
        "identity_rule": {"connectivity": "6 (generate_binary_structure(3,1)), explicit",
                          "CT1": "native labels in CT1 space",
                          "CT2": "native labels in CT2 space, Int32 label image propagated by the "
                                 "Script 18 rigid transform with nearest-neighbor; NEVER relabelled",
                          "support_gate": "propagated label support == Script 18 registered binary mask"},
        "matching": {"iou": "real spatial IoU in CT1 geometry", "assignment": "Hungarian, one-to-one",
                     "retained": "IoU > 0", "historical_pairs": "comparison only, never authority"},
        "HISTORICAL_MATCHING_COMPARISON": {
            "state": hist_state, "rows": len(cmp_rows),
            "interpretation": "zero rows under NOT_AVAILABLE_NO_HISTORICAL_PRODUCT means there is "
                              "no historical RIDER matching product to compare against; it does NOT "
                              "mean zero disagreement nor identical matching"},
        "population": {"pairs_in_s18": len(pairs_all), "excluded": excluded,
                       "pairs_matched": len(pairs), "architectures": len(ARCHS),
                       "audit_rows": len(da)},
        "counts": {"pairs": len(dp), "unmatched": len(du),
                   "ct1_only": int((du.side == "CT1_only").sum()) if not du.empty else 0,
                   "ct2_only": int((du.side == "CT2_only").sum()) if not du.empty else 0,
                   "transformed_component_empty": n_zero_transformed,
                   "status": da["status"].value_counts().to_dict()},
        "gates": {"INPUT_AUTHORITY": "PASS", "CONNECTIVITY_6": "PASS",
                  "NATIVE_CT1_IDENTITY": "PASS", "NATIVE_CT2_IDENTITY": "PASS",
                  "NO_POSTREG_RELABEL": "PASS", "IOU_MATRIX": "PASS",
                  "HUNGARIAN_ONE_TO_ONE": "PASS", "POSITIVE_IOU": "PASS",
                  "CT1_IDENTITY_ARITHMETIC": "PASS", "CT2_IDENTITY_ARITHMETIC": "PASS",
                  "EMPTY_STATE": "PASS", "DUPLICATE_COMPONENT": "PASS",
                  "COMPLETENESS": "PASS", "HASH": "PASS"},
    }
    json.dump(man, open(os.path.join(out_root, "manifest.json"), "w"), indent=2)
    log.info(f"pares {len(dp)} · CT1_only {man['counts']['ct1_only']} · CT2_only {man['counts']['ct2_only']} "
             f"· transformados vazios {n_zero_transformed} · estados {man['counts']['status']}")
    log.info(f"Saída: {out_root}")
    log.info("Script 19 concluído.")


if __name__ == "__main__":
    try:
        main()
    except FalhaEntrada as e:
        log.error(str(e)); raise SystemExit("FAIL_INPUT")
