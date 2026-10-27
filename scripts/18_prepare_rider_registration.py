#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    18_prepare_rider_registration.py

File type:
    Script

Purpose:
    Rigidly registers the second RIDER scan to the first scan per patient and architecture.

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
    - input_manifest.csv
    - rider_registration_qc.csv
    - rider_registration_architecture_qc.csv
    - output_hashes.csv
    - manifest.json

Dependencies:
    - SimpleITK
    - numpy
    - pandas
    - scipy

Execution:
    PULMONARY_BASE=<project_dir> python scripts/18_prepare_rider_registration.py

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

from pulmonary_base import BASE
from caminhos_canonicos import raiz_recortes, ct_de, mascara_pulmonar_de, FalhaEntrada

PRED_RIDER_DIR = os.path.join(BASE, "predictions", "rider")
REG_ROOT       = os.path.join(BASE, "data", "processed", "rider_registered")
ANALYSIS_DIR   = os.path.join(BASE, "analysis")
LOGS_DIR       = os.path.join(BASE, "logs")

ARCHS = ["2D", "2D_REM", "2D_REL", "3D_F", "3D_FREM", "3D_CF", "3D_CFREM",
         "F100_lowres", "F100_ResEncM", "F100_fullres"]
EXPECTED = {"pairs": 31, "scans": 62, "architectures": 10,
            "predictions": 620, "combinations": 310}

DICE_PASS      = 0.90
DICE_BLOCK     = 0.80
DELTA_TOL      = 0.005
DELTA_MAJOR    = 0.05
STRUCT_6       = nd.generate_binary_structure(3, 1)

for _d in (ANALYSIS_DIR, LOGS_DIR):
    os.makedirs(_d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(os.path.join(LOGS_DIR, "prepare_rider_registration.log")),
              logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def fail_input(m):
    log.error(m); raise SystemExit("FAIL_INPUT")


def fail_output(m):
    log.error(m); raise SystemExit("FAIL_OUTPUT")


def fail_registration(m):
    log.error(m); raise SystemExit("FAIL_REGISTRATION")


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


def dice(a, b):
    a = a.astype(bool); b = b.astype(bool)
    d = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / d) if d else float("nan")


def resample_mask(moving, ref, tx):
    return sitk.Resample(moving, ref, tx, sitk.sitkNearestNeighbor, 0, moving.GetPixelID())


def rigid_register(fixed_mask, moving_mask):
    """Euler3D sobre as máscaras pulmonares; devolve (tx, qc)."""
    f = sitk.Cast(fixed_mask, sitk.sitkFloat32)
    m = sitk.Cast(moving_mask, sitk.sitkFloat32)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMeanSquares()
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-6, numberOfIterations=200,
        gradientMagnitudeTolerance=1e-6)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInterpolator(sitk.sitkLinear)
    init = sitk.CenteredTransformInitializer(
        f, m, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    reg.SetInitialTransform(init, inPlace=False)
    tx = reg.Execute(f, m)
    params = [float(p) for p in tx.GetParameters()]
    metric = float(reg.GetMetricValue()); iters = int(reg.GetOptimizerIteration())
    if not all(np.isfinite(params)) or not np.isfinite(metric):
        fail_registration("transformação ou métrica não finita")
    return tx, {"tx_params": params, "tx_fixed_params": [float(p) for p in tx.GetFixedParameters()],
                "final_metric": metric, "n_iterations": iters,
                "stop_condition": reg.GetOptimizerStopConditionDescription(),
                "convergence_flag": bool(iters >= 200)}


def direction_fixture():
    """Teste sintético: um objecto deslocado 10 mm em x tem de voltar ao sítio
    quando reamostrado com a transformação estimada (convenção do SimpleITK)."""
    shape = (40, 60, 60); sp = (1.0, 1.0, 1.0)
    a = np.zeros(shape, np.uint8); a[15:25, 20:40, 20:40] = 1
    b = np.zeros(shape, np.uint8); b[15:25, 20:40, 30:50] = 1
    def img(arr):
        i = sitk.GetImageFromArray(arr); i.SetSpacing(sp); return i
    F, M = img(a), img(b)
    tx, _ = rigid_register(F, M)
    reg = sitk.GetArrayFromImage(resample_mask(M, F, tx))
    d0 = dice(sitk.GetArrayFromImage(M), a); d1 = dice(reg, a)
    return {"dice_before": round(d0, 4), "dice_after": round(d1, 4),
            "PASS": bool(d1 > 0.95 and d1 > d0)}


def status_of(d_after, d_delta):
    if not np.isfinite(d_after) or not np.isfinite(d_delta):
        fail_registration("Dice não finito")
    if d_after < DICE_BLOCK or d_delta < -DELTA_MAJOR:
        return "BLOCKED_REGISTRATION"
    if d_after < DICE_PASS or d_delta < -DELTA_TOL:
        return "FLAG_FOR_INSPECTION"
    return "PASS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-pairs", type=int, default=0, help="ENSAIO: só os N primeiros pares")
    ap.add_argument("--out-root", default=None, help="ENSAIO: raiz alternativa")
    ap.add_argument("--fixture-only", action="store_true")
    args = ap.parse_args()
    t0 = datetime.datetime.now().astimezone()

    fx = direction_fixture()
    log.info(f"DIRECTION_FIXTURE: dice {fx['dice_before']} → {fx['dice_after']} — "
             f"{'PASS' if fx['PASS'] else 'FAIL'}")
    if not fx["PASS"]:
        fail_registration("teste sintético de direcção falhou")
    if args.fixture_only:
        print(json.dumps(fx)); return

    raiz = raiz_recortes()
    run_id = f"s18_{t0:%Y%m%dT%H%M%S}_{os.getpid()}"
    out_root = os.path.join(args.out_root or REG_ROOT, run_id)
    os.makedirs(out_root, exist_ok=True)

    sets = {}
    for a in ARCHS:
        d = os.path.join(PRED_RIDER_DIR, a)
        if not os.path.isdir(d):
            fail_input(f"arquitectura ausente nas predições RIDER: {a}")
        sets[a] = {f[:-7] for f in os.listdir(d)
                   if f.endswith(".nii.gz") and not f.endswith("_cc.nii.gz")}
    base = sets[ARCHS[0]]
    for a in ARCHS[1:]:
        if sets[a] != base:
            fail_input(f"conjunto de exames difere em {a}")
    exams = sorted(base)
    pairs = sorted({e.rsplit("_", 1)[0] for e in exams})
    if len(exams) != EXPECTED["scans"] or len(pairs) != EXPECTED["pairs"]:
        fail_input(f"{len(exams)} exames / {len(pairs)} pares != 62/31")
    for p in pairs:
        if f"{p}_1" not in base or f"{p}_2" not in base:
            fail_input(f"par incompleto: {p}")
    if args.limit_pairs:
        pairs = pairs[: args.limit_pairs]

    rows_in = []
    for p in pairs:
        for k in (1, 2):
            e = f"{p}_{k}"
            ctp = ct_de(raiz, "rider", e)
            lmp = mascara_pulmonar_de(raiz, "rider", e, recortada=True)
            ct = sitk.ReadImage(ctp); lm = sitk.ReadImage(lmp)
            if not same_geometry(ct, lm):
                fail_input(f"máscara pulmonar com geometria != CT em {e}")
            for a in ARCHS:
                pp = os.path.join(PRED_RIDER_DIR, a, f"{e}.nii.gz")
                if not os.path.isfile(pp):
                    fail_input(f"predição ausente: {pp}")
                rows_in.append({"patient_id": p, "acquisition": k, "exam_id": e, "architecture": a,
                                "CT_path": os.path.relpath(ctp, BASE), "CT_SHA256": sha256(ctp),
                                "lung_mask_path": os.path.relpath(lmp, BASE), "lung_mask_SHA256": sha256(lmp),
                                "prediction_path": os.path.relpath(pp, BASE), "prediction_SHA256": sha256(pp)})
    din = pd.DataFrame(rows_in)
    din.to_csv(os.path.join(out_root, "input_manifest.csv"), index=False)
    log.info(f"Manifesto: {len(pairs)} pares, {len(pairs)*2} exames, {len(ARCHS)} arquitecturas, "
             f"{len(din)} predições")

    pat_rows, arch_rows, out_hashes = [], [], []
    for i, p in enumerate(pairs):
        e1, e2 = f"{p}_1", f"{p}_2"
        ct1 = sitk.ReadImage(ct_de(raiz, "rider", e1))
        lm1 = sitk.ReadImage(mascara_pulmonar_de(raiz, "rider", e1, recortada=True))
        lm2 = sitk.ReadImage(mascara_pulmonar_de(raiz, "rider", e2, recortada=True))
        a1 = sitk.GetArrayFromImage(lm1)
        before = dice(sitk.GetArrayFromImage(resample_mask(lm2, lm1, sitk.Transform(3, sitk.sitkIdentity))), a1)
        tx, q = rigid_register(lm1, lm2)
        after = dice(sitk.GetArrayFromImage(resample_mask(lm2, lm1, tx)), a1)
        delta = after - before
        st = status_of(after, delta)
        log.info(f"  [{i+1}/{len(pairs)}] {p}: dice {before:.4f} → {after:.4f} "
                 f"(Δ {delta:+.4f}) {st}{' [conv_flag]' if q['convergence_flag'] else ''}")
        pat_rows.append({"patient_id": p, "dice_before": round(before, 6), "dice_after": round(after, 6),
                         "delta_dice": round(delta, 6), "registration_status": st,
                         "final_metric": q["final_metric"], "n_iterations": q["n_iterations"],
                         "stop_condition": q["stop_condition"],
                         "convergence_flag": int(q["convergence_flag"]),
                         "tx_params": json.dumps([round(x, 6) for x in q["tx_params"]]),
                         "tx_fixed_params": json.dumps([round(x, 6) for x in q["tx_fixed_params"]])})

        for a in ARCHS:
            p1 = os.path.join(PRED_RIDER_DIR, a, f"{e1}.nii.gz")
            p2 = os.path.join(PRED_RIDER_DIR, a, f"{e2}.nii.gz")
            i1 = sitk.ReadImage(p1); i2 = sitk.ReadImage(p2)
            if not same_geometry(i1, ct1):
                fail_input(f"predição CT1 com geometria != CT1: {p1}")
            ar1 = sitk.GetArrayFromImage(i1); ar2 = sitk.GetArrayFromImage(i2)
            for arr, pth in ((ar1, p1), (ar2, p2)):
                if not np.isin(arr, (0, 1)).all():
                    fail_input(f"predição não binária: {pth}")
            try:
                reg = resample_mask(i2, ct1, tx)
            except Exception as ex:
                fail_output(f"reamostragem falhou {a}/{p}: {ex}")
            if not same_geometry(reg, ct1):
                fail_output(f"saída registada com geometria != CT1: {a}/{p}")
            arr = sitk.GetArrayFromImage(reg)
            if not np.isin(arr, (0, 1)).all():
                fail_output(f"saída registada não binária: {a}/{p}")
            od = os.path.join(out_root, a); os.makedirs(od, exist_ok=True)
            op = os.path.join(od, f"{p}_ct2_in_ct1.nii.gz")
            sitk.WriteImage(reg, op)
            n1 = int(nd.label(ar1, structure=STRUCT_6)[1])
            n2n = int(nd.label(ar2, structure=STRUCT_6)[1])
            n2r = int(nd.label(arr, structure=STRUCT_6)[1])
            arch_rows.append({
                "patient_id": p, "architecture": a, "registration_status": st,
                "prediction_CT1_present": 1, "prediction_CT2_present": 1,
                "empty_CT1": int(n1 == 0), "empty_CT2": int(n2n == 0),
                "status_empty": "EMPTY_PREDICTION" if (n1 == 0 or n2n == 0) else "",
                "n_components_CT1": n1, "n_components_CT2_native": n2n,
                "n_components_CT2_registered": n2r, "delta_components": n2r - n2n,
                "COMPONENT_COUNT_CHANGED_BY_RESAMPLING": int(n2r != n2n),
                "voxels_CT2_native": int((ar2 > 0).sum()), "voxels_CT2_registered": int((arr > 0).sum()),
                "output_geometry_gate": "PASS", "binary_gate": "PASS",
                "output_path": os.path.relpath(op, BASE)})
            out_hashes.append({"path": os.path.relpath(op, BASE), "sha256": sha256(op)})

    dp = pd.DataFrame(pat_rows); da = pd.DataFrame(arch_rows)
    dp.to_csv(os.path.join(out_root, "rider_registration_qc.csv"), index=False)
    da.to_csv(os.path.join(out_root, "rider_registration_architecture_qc.csv"), index=False)
    pd.DataFrame(out_hashes).to_csv(os.path.join(out_root, "output_hashes.csv"), index=False)
    if not args.limit_pairs:
        dp.to_csv(os.path.join(ANALYSIS_DIR, "rider_registration_qc.csv"), index=False)

    counts = dp["registration_status"].value_counts().to_dict()
    man = {
        "script": "18_prepare_rider_registration.py", "run_id": run_id,
        "script_sha256": sha256(os.path.abspath(__file__)),
        "run_at": t0.isoformat(), "end_at": datetime.datetime.now().astimezone().isoformat(),
        "population": {"pairs": len(pairs), "scans": len(pairs) * 2,
                       "architectures": len(ARCHS), "predictions": len(din),
                       "combinations": len(da)},
        "registration": {"transform": "Euler3D rigid, one per patient",
                         "estimated_on": "lung masks only (lung_mask_crop.nii.gz)",
                         "interpolator_output": "sitkNearestNeighbor",
                         "resample_reference": "canonical CT1"},
        "qc_metric": {"metric": "LUNG_MASK_DICE", "reference": "lung_mask_CT1",
                      "dice_before": "identity transform", "dice_after": "estimated transform",
                      "gates": {"PASS": f"dice_after >= {DICE_PASS} and delta >= -{DELTA_TOL}",
                                "FLAG_FOR_INSPECTION": f"{DICE_BLOCK} <= dice_after < {DICE_PASS} "
                                                       f"or -{DELTA_MAJOR} <= delta < -{DELTA_TOL}",
                                "BLOCKED_REGISTRATION": f"dice_after < {DICE_BLOCK} or delta < -{DELTA_MAJOR}"}},
        "direction_fixture": fx,
        "status_counts": counts,
        "component_audit": {"connectivity": "6 (generate_binary_structure(3,1))",
                            "changed_by_resampling": int(da["COMPONENT_COUNT_CHANGED_BY_RESAMPLING"].sum()),
                            "note": "auditoria; não bloqueia no Script 18"},
        "empty_predictions": {"CT1": int(da["empty_CT1"].sum()), "CT2": int(da["empty_CT2"].sum())},
        "registered_output_role": "Script 19 spatial correspondence only; Script 20 measures CT2 natively",
        "outputs": {"registered_masks": len(out_hashes)},
    }
    json.dump(man, open(os.path.join(out_root, "manifest.json"), "w"), indent=2)
    log.info(f"Estados: {counts} | componentes alterados pela reamostragem: "
             f"{man['component_audit']['changed_by_resampling']}/{len(da)} | "
             f"vazias CT1 {man['empty_predictions']['CT1']} CT2 {man['empty_predictions']['CT2']}")
    log.info(f"Saída: {out_root}")
    log.info("Script 18 concluído.")


if __name__ == "__main__":
    try:
        main()
    except FalhaEntrada as e:
        log.error(str(e)); raise SystemExit("FAIL_INPUT")
