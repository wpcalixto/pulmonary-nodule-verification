#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    20_extract_rider_morphometry_and_filters.py

File type:
    Script

Purpose:
    Extracts morphometry and filter features for each matched RIDER test-retest pair.

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
    - input_manifest.csv
    - parity_report.json
    - gates.json
    - manifest.json

Dependencies:
    - SimpleITK
    - numpy
    - pandas
    - scipy

Execution:
    PULMONARY_BASE=<project_dir> python scripts/20_extract_rider_morphometry_and_filters.py

Portability notes:
    All paths are resolved relative to the project base directory given by the PULMONARY_BASE environment variable. No machine-specific path is required.

Reproducibility notes:
    Environment pinned in requirements.txt and environment/. Seeds and frozen parameters are recorded in the run manifests under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) are not included and must be obtained from their original repositories.
"""

import argparse
import datetime
import hashlib
import importlib.util
import json
import logging
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.ndimage as nd
import SimpleITK as sitk

BASE = os.environ.get("PULMONARY_BASE") or os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PRED_RIDER_DIR = os.path.join(BASE, "predictions", "rider")
FEATURES_DIR = os.path.join(BASE, "features")
ANALYSIS_DIR = os.path.join(BASE, "analysis")
LOGS_DIR = os.path.join(BASE, "logs")
C_CALIB_DEFAULT = os.path.join(FEATURES_DIR, "c_global_calibration.json")

STRUCT_6 = nd.generate_binary_structure(3, 1)
C_GLOBAL_EXPECTED_ROUND6 = 499.892883

FEATURES_16 = [
    "log_max", "log_mean", "log_p95", "log_scale_mm",
    "blob_max", "blob_mean", "blob_p95", "blob_scale_mm",
    "vessel_max", "vessel_mean", "vessel_p95", "vessel_scale_mm",
    "contrast_hu", "centroid_peak_dist_mm",
    "pleura_min_dist_mm", "pleura_centroid_dist_mm",
]
QC_COLS = ["mu_object_hu", "mu_ring_hu", "ring_voxel_count",
           "ring_truncated_at_image_boundary", "object_voxel_count_in_roi",
           "roi_clipped_at_image_boundary", "component_voxels_total",
           "component_voxels_inside_lung", "component_voxels_outside_lung",
           "component_fraction_inside_lung", "component_intersects_lung_surface",
           "component_entirely_outside_lung", "frac_high_log"]

os.makedirs(LOGS_DIR, exist_ok=True)
log = logging.getLogger("s20")
log.setLevel(logging.INFO)
if not log.handlers:
    _f = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for _h in (logging.StreamHandler(),
               logging.FileHandler(os.path.join(LOGS_DIR, "extract_rider_morphometry_and_filters.log"))):
        _h.setFormatter(_f); log.addHandler(_h)
log.propagate = False


def fail_input(msg):
    log.error(msg); raise SystemExit("FAIL_INPUT")


def fail_output(msg):
    log.error(msg); raise SystemExit("FAIL_OUTPUT")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _carregar_modulo(nome, ficheiro):
    p = os.path.join(SCRIPTS_DIR, ficheiro)
    if not os.path.isfile(p):
        fail_input(f"módulo canónico ausente: {p}")
    sp = importlib.util.spec_from_file_location(nome, p)
    m = importlib.util.module_from_spec(sp)
    sys.modules[nome] = m
    sp.loader.exec_module(m)
    return m, sha256(p)


S13, SHA13 = _carregar_modulo("s13_canon", "13_prepare_filter_objects.py")
S14, SHA14 = _carregar_modulo("s14_canon", "14_compute_filter_maps_gpu.py")
S15, SHA15 = _carregar_modulo("s15_canon", "15_extract_filter_features.py")

from caminhos_canonicos import raiz_recortes, ct_de, mascara_pulmonar_de

if tuple(S14.SCALES_MM) != (0.8, 1.2, 1.8, 2.7, 4.0, 6.0, 8.0):
    fail_input(f"SCALES_MM do Script 14 divergem: {S14.SCALES_MM}")
if tuple(S14.HU_CLIP) != (-1000.0, 400.0):
    fail_input(f"HU_CLIP do Script 14 diverge: {S14.HU_CLIP}")
if (S14.ALPHA, S14.BETA, S14.RING_MM) != (0.5, 0.5, 5.0):
    fail_input("ALPHA/BETA/RING_MM do Script 14 divergem")
if S13.MARGIN_MM != 15.0:
    fail_input(f"MARGIN_MM do Script 13 diverge: {S13.MARGIN_MM}")


def same_geometry(a, b):
    return (a.GetSize() == b.GetSize()
            and np.allclose(a.GetSpacing(), b.GetSpacing(), rtol=0, atol=1e-6)
            and np.allclose(a.GetOrigin(), b.GetOrigin(), rtol=0, atol=1e-6)
            and np.allclose(a.GetDirection(), b.GetDirection(), rtol=0, atol=1e-6))


def ct_nativa(raiz, pid, k):
    """Caminho determinístico do Script 06 — sem glob, sem primeiro resultado."""
    p = ct_de(raiz, "rider", f"{pid}_{k}")
    if not os.path.isfile(p):
        fail_input(f"CT nativa ausente: {p}")
    return p


def mascara_nativa(raiz, pid, k):
    p = mascara_pulmonar_de(raiz, "rider", f"{pid}_{k}", recortada=True)
    if not os.path.isfile(p):
        fail_input(f"máscara pulmonar nativa ausente: {p}")
    return p


def carregar_ct(raiz, pid, k):
    """CT nativa recortada + EDT pleural. Uma vez por aquisição, não por arquitectura."""
    ctp = ct_nativa(raiz, pid, k)
    ct_img = sitk.ReadImage(ctp, sitk.sitkFloat32)
    sp = ct_img.GetSpacing()
    if not all(np.isfinite(sp)) or min(sp) <= 0:
        fail_input(f"espaçamento inválido {sp} em {ctp}")
    sp_zyx = (float(sp[2]), float(sp[1]), float(sp[0]))
    ct_arr = np.clip(sitk.GetArrayFromImage(ct_img), *S14.HU_CLIP).astype(np.float32)
    lmp = mascara_nativa(raiz, pid, k)
    dist, lung, sp_lm = S15.pleura_edt(lmp, ct_img, f"{pid}_{k}")
    if not np.allclose(sp_lm, sp_zyx, rtol=0, atol=1e-6):
        fail_input(f"espaçamento da máscara pulmonar != CT em {pid}_{k}")
    return {"ct_img": ct_img, "ct_arr": ct_arr, "sp_zyx": sp_zyx,
            "dist": dist.astype(np.float32), "lung": lung,
            "ct_path": ctp, "ct_sha": sha256(ctp),
            "lung_path": lmp, "lung_sha": sha256(lmp)}


def carregar_predicao(base_ct, pid, k, arch):
    """Predição nativa rotulada com 6-conectividade, na geometria da CT nativa."""
    predp = os.path.join(PRED_RIDER_DIR, arch, f"{pid}_{k}.nii.gz")
    if not os.path.isfile(predp):
        fail_input(f"predição nativa ausente: {predp}")
    pred_img = sitk.ReadImage(predp)
    if not same_geometry(pred_img, base_ct["ct_img"]):
        fail_input(f"predição com geometria != CT nativa: {predp}")
    pred = sitk.GetArrayFromImage(pred_img)
    if not np.isin(pred, (0, 1)).all():
        fail_input(f"predição não binária: {predp}")
    binary = pred > 0
    labeled, n = nd.label(binary, structure=STRUCT_6)
    return {"ct_arr": base_ct["ct_arr"], "sp_zyx": base_ct["sp_zyx"],
            "labeled": labeled.astype(np.int32), "binary": binary, "n": int(n),
            "dist": base_ct["dist"], "lung": base_ct["lung"],
            "pred_path": predp, "pred_sha": sha256(predp)}


def morfometria(labeled, cid, sp_zyx, rotulo):
    coords = np.argwhere(labeled == cid)
    if coords.size == 0:
        fail_input(f"COMPONENT_IDENTITY: componente {cid} inexistente em {rotulo}")
    n_vox = int(len(coords))
    vol = float(n_vox) * sp_zyx[0] * sp_zyx[1] * sp_zyx[2]
    diam = S13.diam_axial_bbox_mm(coords, sp_zyx)
    if n_vox <= 0 or not np.isfinite(vol) or vol <= 0 or not np.isfinite(diam) or diam <= 0:
        fail_output(f"morfometria não positiva ou não finita em {rotulo}")
    return n_vox, vol, diam, coords


def categoria_tamanho(d, rotulo):
    if not np.isfinite(d) or d <= 0:
        fail_output(f"diâmetro inválido ({d}) em {rotulo}")
    if d < 3.0:
        return "lt3mm"
    if d < 6.0:
        return "3to6mm"
    if d < 15.0:
        return "6to15mm"
    return "ge15mm"


def caracteristicas(acq, cid, c_global, rotulo):
    """As 14 do Script 14 + as 2 pleurais do Script 15, pelas funções canónicas."""
    bbox = S13.object_record(acq["labeled"], cid, acq["sp_zyx"],
                             acq["labeled"].shape, rotulo)
    saida, err = S14.process_roi(acq["ct_arr"], acq["labeled"], acq["binary"],
                                 acq["sp_zyx"], cid, bbox, c_global, "cpu", rotulo)
    if err is not None:
        fail_output(f"ROI irresolúvel em {rotulo}: {err}")
    feats = dict(saida[0])
    comp = (acq["labeled"] == cid)
    feats.update(S15.pleural_features(comp, acq["dist"], acq["lung"], acq["sp_zyx"], rotulo))
    feats["roi_clipped_at_image_boundary"] = bbox["roi_clipped_at_image_boundary"]
    falta = [f for f in FEATURES_16 if f not in feats]
    if falta:
        fail_output(f"características canónicas em falta em {rotulo}: {falta}")
    for f in FEATURES_16:
        if not np.isfinite(float(feats[f])):
            fail_output(f"{f} não finita em {rotulo}")
    return feats


_G = {}


def _init(base, c_global, raiz):
    _G["c_global"] = c_global
    _G["raiz"] = raiz


def _paciente(args_):
    pid, grupo = args_
    raiz, c_global = _G["raiz"], _G["c_global"]
    linhas, entradas = [], []
    base = {k: carregar_ct(raiz, pid, k) for k in (1, 2)}
    for arch in sorted(grupo["architecture"].unique()):
        sub = grupo[grupo["architecture"] == arch]
        acq = {k: carregar_predicao(base[k], pid, k, arch) for k in (1, 2)}
        for k in (1, 2):
            entradas.append({
                "patient_id": pid, "architecture": arch, "acquisition": k,
                "ct_path": os.path.relpath(base[k]["ct_path"], BASE),
                "ct_sha256": base[k]["ct_sha"],
                "lung_mask_path": os.path.relpath(base[k]["lung_path"], BASE),
                "lung_mask_sha256": base[k]["lung_sha"],
                "prediction_path": os.path.relpath(acq[k]["pred_path"], BASE),
                "prediction_sha256": acq[k]["pred_sha"],
                "spacing_zyx": json.dumps(list(base[k]["sp_zyx"]))})
        for _, r in sub.iterrows():
            c1 = int(r["component_id_ct1"]); c2 = int(r["component_id_ct2_native"])
            r1, r2 = f"{pid}/{arch}/CT1/c{c1}", f"{pid}/{arch}/CT2/c{c2}"
            n1, v1, d1, _ = morfometria(acq[1]["labeled"], c1, acq[1]["sp_zyx"], r1)
            n2, v2, d2, _ = morfometria(acq[2]["labeled"], c2, acq[2]["sp_zyx"], r2)
            f1 = caracteristicas(acq[1], c1, c_global, r1)
            f2 = caracteristicas(acq[2], c2, c_global, r2)

            mv = (v1 + v2) / 2.0
            md = (d1 + d2) / 2.0
            if not (mv > 0 and md > 0 and np.isfinite(mv) and np.isfinite(md)):
                fail_output(f"denominador relativo inválido em {pid}/{arch}/{c1}-{c2}")
            linha = {
                "patient_id": pid, "architecture": arch,
                "component_id_ct1": c1, "component_id_ct2_native": c2,
                "iou": float(r["iou"]),
                "ct2_transformed_voxels": int(r["ct2_transformed_voxels"]),
                "voxel_count_ct1": n1, "voxel_count_ct2": n2,
                "spacing_zyx_ct1": json.dumps(list(acq[1]["sp_zyx"])),
                "spacing_zyx_ct2": json.dumps(list(acq[2]["sp_zyx"])),
                "volume_ct1_mm3": v1, "volume_ct2_mm3": v2,
                "diameter_ct1_mm": d1, "diameter_ct2_mm": d2,
                "size_category_ct1": categoria_tamanho(d1, r1),
                "size_category_ct2": categoria_tamanho(d2, r2),
                "delta_volume_mm3": v2 - v1, "abs_delta_volume_mm3": abs(v2 - v1),
                "mean_volume_mm3": mv,
                "rel_delta_volume": (v2 - v1) / mv, "abs_rel_delta_volume": abs(v2 - v1) / mv,
                "pct_rel_delta_volume": 100.0 * (v2 - v1) / mv,
                "pct_abs_rel_delta_volume": 100.0 * abs(v2 - v1) / mv,
                "delta_diameter_mm": d2 - d1, "abs_delta_diameter_mm": abs(d2 - d1),
                "mean_diameter_mm": md,
                "rel_delta_diameter": (d2 - d1) / md, "abs_rel_delta_diameter": abs(d2 - d1) / md,
                "pct_rel_delta_diameter": 100.0 * (d2 - d1) / md,
                "pct_abs_rel_delta_diameter": 100.0 * abs(d2 - d1) / md,
            }
            for f in FEATURES_16:
                a, b = float(f1[f]), float(f2[f])
                linha[f"{f}_ct1"] = a
                linha[f"{f}_ct2"] = b
                linha[f"delta_{f}"] = b - a
                linha[f"abs_delta_{f}"] = abs(b - a)
            for q in QC_COLS:
                if q in f1:
                    linha[f"{q}_ct1"] = f1[q]
                if q in f2:
                    linha[f"{q}_ct2"] = f2[q]
            linhas.append(linha)
    return linhas, entradas


def paridade(c_global, n_amostra, seed=27):
    """Recalcula objectos do BD1 pelo caminho do Script 20 e compara com as
    saídas canónicas do 14/15. Nominalmente igual tem de ser numericamente igual."""
    ref_p = os.path.join(BASE, "figdata", "bd1_filter_features.csv")
    if not os.path.isfile(ref_p):
        fail_input(f"matriz canónica do Script 15 ausente: {ref_p}")
    ref = pd.read_csv(ref_p)
    ref = ref[ref["group"].isin(["TP_pred", "FP_pred"])]
    amostra = ref.sample(n=min(n_amostra, len(ref)), random_state=seed)
    raiz = raiz_recortes()
    filas, difs = [], []
    for _, r in amostra.iterrows():
        pid, arch, cid = r["patient_id"], r["architecture"], int(r["component_id"])
        predp = r["mask_path"] if os.path.isabs(str(r["mask_path"])) \
            else os.path.join(BASE, str(r["mask_path"]))
        ctp = r["ct_path"] if os.path.isabs(str(r["ct_path"])) \
            else os.path.join(BASE, str(r["ct_path"]))
        for p in (predp, ctp):
            if not os.path.isfile(p):
                fail_input(f"objecto BD1 da paridade não resolúvel: {p}")
        ct_img = sitk.ReadImage(ctp, sitk.sitkFloat32)
        sp = ct_img.GetSpacing(); sp_zyx = (float(sp[2]), float(sp[1]), float(sp[0]))
        ct_arr = np.clip(sitk.GetArrayFromImage(ct_img), *S14.HU_CLIP).astype(np.float32)
        arr = sitk.GetArrayFromImage(sitk.ReadImage(predp))
        binary = arr > 0
        labeled, _ = nd.label(binary, structure=STRUCT_6)
        acq = {"ct_arr": ct_arr, "sp_zyx": sp_zyx, "labeled": labeled.astype(np.int32),
               "binary": binary}
        lmp = mascara_pulmonar_de(raiz, "lidc_idri", pid, recortada=True)
        if not os.path.isfile(lmp):
            fail_input(f"máscara pulmonar da paridade ausente: {lmp}")
        acq["dist"], acq["lung"], _ = S15.pleura_edt(lmp, ct_img, pid)
        got = caracteristicas(acq, cid, c_global, f"paridade {pid}/{arch}/c{cid}")
        linha = {"patient_id": pid, "architecture": arch, "component_id": cid}
        for f in FEATURES_16:
            a, b = float(r[f]), float(got[f])
            linha[f] = {"canonico": a, "script20": b, "igual": bool(a == b or abs(a - b) <= 1e-9)}
            if not linha[f]["igual"]:
                difs.append((pid, arch, cid, f, a, b))
        filas.append(linha)
    return {"n": len(filas), "divergencias": len(difs),
            "exemplos_divergentes": difs[:10],
            "features_comparadas": FEATURES_16, "amostras": filas[:3]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s19-run", required=True)
    ap.add_argument("--s19-manifest-sha256", required=True)
    ap.add_argument("--pairs", default="", help="por omissão, rider_iou_pairs.csv do run")
    ap.add_argument("--pairs-sha256", required=True)
    ap.add_argument("--c-calibration", default=C_CALIB_DEFAULT)
    ap.add_argument("--limit-patients", type=int, default=0)
    ap.add_argument("--patients", default="", help="lista explícita, só para ensaio")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--parity-sample", type=int, default=12)
    ap.add_argument("--parity-only", action="store_true")
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()
    t0 = datetime.datetime.now().astimezone()

    if not os.path.isfile(args.c_calibration):
        fail_input(f"calibração de c_global ausente: {args.c_calibration}")
    calib = json.load(open(args.c_calibration))
    c_global = float(calib["c_global"])
    if round(c_global, 6) != C_GLOBAL_EXPECTED_ROUND6:
        fail_input(f"c_global {c_global} != congelado {C_GLOBAL_EXPECTED_ROUND6}")
    if calib.get("excluded") is None or "RIDER" not in calib["excluded"]:
        fail_input("calibração de c_global não declara o RIDER como excluído")
    log.info(f"c_global = {c_global!r} (calibrado no BD1 OOF; RIDER excluído)")

    par = paridade(c_global, args.parity_sample)
    log.info(f"PARIDADE 14/15 → 20: {par['n']} objectos BD1, "
             f"{par['divergencias']} divergências em {len(FEATURES_16)} variáveis")
    if par["divergencias"]:
        for d in par["exemplos_divergentes"]:
            log.error(f"  divergência {d}")
        fail_output("CANONICAL_FILTER/PLEURAL_PARITY_GATE falhou")
    if args.parity_only:
        print(json.dumps({"CANONICAL_PARITY": "PASS", **{k: v for k, v in par.items()
                                                         if k != "amostras"}}, indent=1))
        return

    man_p = os.path.join(args.s19_run, "manifest.json")
    if not os.path.isfile(man_p):
        fail_input(f"run do Script 19 sem manifest.json: {args.s19_run}")
    h = sha256(man_p)
    if h != args.s19_manifest_sha256:
        fail_input(f"INPUT_AUTHORITY: SHA do manifesto do 19 {h[:12]} != {args.s19_manifest_sha256[:12]}")
    man19 = json.load(open(man_p))
    dir_run = os.path.basename(os.path.normpath(args.s19_run))
    if man19["run_id"] != dir_run:
        fail_input(f"run_id do manifesto ({man19['run_id']}) != pasta ({dir_run})")
    pairs_p = args.pairs or os.path.join(args.s19_run, "rider_iou_pairs.csv")
    if os.path.normpath(os.path.dirname(pairs_p)) != os.path.normpath(args.s19_run):
        fail_input(f"pares fora do run canónico: {pairs_p}")
    if not os.path.isfile(pairs_p):
        fail_input(f"rider_iou_pairs.csv ausente: {pairs_p}")
    hp = sha256(pairs_p)
    if hp != args.pairs_sha256:
        fail_input(f"INPUT_AUTHORITY: SHA dos pares {hp[:12]} != {args.pairs_sha256[:12]}")
    oh = pd.read_csv(os.path.join(args.s19_run, "output_hashes.csv"))
    linha = oh[oh["path"].str.endswith("rider_iou_pairs.csv")]
    if len(linha) != 1 or linha["sha256"].iloc[0] != hp:
        fail_input("SHA dos pares não confere com o output_hashes.csv do Script 19")

    pairs = pd.read_csv(pairs_p)
    obrig = {"patient_id", "architecture", "component_id_ct1",
             "component_id_ct2_native", "iou", "ct2_transformed_voxels"}
    falta = obrig - set(pairs.columns)
    if falta:
        fail_input(f"colunas obrigatórias ausentes nos pares: {sorted(falta)}")
    chave = ["patient_id", "architecture", "component_id_ct1", "component_id_ct2_native"]
    if pairs.duplicated(chave).any():
        fail_input("chave de par repetida na entrada")
    if (pairs["iou"] <= 0).any():
        fail_input("par de entrada com IoU <= 0")
    if (pairs["ct2_transformed_voxels"] <= 0).any():
        fail_input("TRANSFORMED_COMPONENT_EMPTY entre os pares correspondidos")
    esperado = len(pairs)
    log.info(f"Script 19 canónico {man19['run_id']}: {esperado} pares correspondidos, "
             f"{pairs.patient_id.nunique()} pacientes, {pairs.architecture.nunique()} arquitecturas")

    pacientes = sorted(pairs["patient_id"].unique())
    if args.patients:
        pedidos = [p.strip() for p in args.patients.split(",") if p.strip()]
        desconhecidos = [p for p in pedidos if p not in pacientes]
        if desconhecidos:
            fail_input(f"pacientes inexistentes nos pares: {desconhecidos}")
        pacientes = sorted(pedidos)
        pairs = pairs[pairs["patient_id"].isin(pacientes)]
        esperado = len(pairs)
    if args.limit_patients:
        pacientes = pacientes[: args.limit_patients]
        pairs = pairs[pairs["patient_id"].isin(pacientes)]
        esperado = len(pairs)

    raiz = raiz_recortes()
    run_id = f"s20_{t0:%Y%m%dT%H%M%S}_{os.getpid()}"
    out_root = os.path.join(args.out_root or os.path.join(FEATURES_DIR, "rider_morphometry"), run_id)
    os.makedirs(out_root, exist_ok=True)

    tarefas = [(p, pairs[pairs["patient_id"] == p]) for p in pacientes]
    linhas, entradas = [], []
    if args.workers > 1:
        with Pool(args.workers, initializer=_init, initargs=(BASE, c_global, raiz),
                  maxtasksperchild=2) as pool:
            for i, (ls, es) in enumerate(pool.imap_unordered(_paciente, tarefas)):
                linhas += ls; entradas += es
                log.info(f"  [{i+1}/{len(tarefas)}] {len(ls)} pares")
    else:
        _init(BASE, c_global, raiz)
        for i, t in enumerate(tarefas):
            ls, es = _paciente(t)
            linhas += ls; entradas += es
            log.info(f"  [{i+1}/{len(tarefas)}] {t[0]}: {len(ls)} pares")

    df = pd.DataFrame(linhas)
    din = pd.DataFrame(entradas)

    g = {}
    g["ROW_COUNT"] = f"{len(df)}/{esperado}"
    if len(df) != esperado:
        fail_output(f"PAIR_KEY_GATE: {len(df)} linhas != {esperado} pares de entrada")
    if df.duplicated(chave).any():
        fail_output("PAIR_KEY_GATE: chave repetida na saída")
    kn = set(map(tuple, df[chave].values)); ke = set(map(tuple, pairs[chave].values))
    if kn != ke:
        fail_output(f"PAIR_KEY_GATE: {len(ke - kn)} em falta, {len(kn - ke)} inesperadas")
    g["PAIR_KEY_GATE"] = "PASS"
    num = ["volume_ct1_mm3", "volume_ct2_mm3", "diameter_ct1_mm", "diameter_ct2_mm"]
    if (df[num] <= 0).any().any() or not np.isfinite(df[num].to_numpy()).all():
        fail_output("volume ou diâmetro não positivo/não finito")
    g["POSITIVE_MORPHOMETRY"] = "PASS"
    cols_feat = [f"{f}_{s}" for f in FEATURES_16 for s in ("ct1", "ct2")]
    if not np.isfinite(df[cols_feat].to_numpy()).all():
        fail_output("característica não finita na saída")
    g["FEATURE_SET_GATE"] = f"{len(FEATURES_16)} analíticas × 2 aquisições"
    if "frac_high_log_ct1" in df.columns:
        g["frac_high_log"] = "QC_ONLY — ANALYTIC_FEATURE = NO"
    cats = set(df["size_category_ct1"]) | set(df["size_category_ct2"])
    if not cats <= {"lt3mm", "3to6mm", "6to15mm", "ge15mm"}:
        fail_output(f"categoria de tamanho fora do congelado: {cats}")
    g["SIZE_CATEGORY_GATE"] = "PASS"
    for a, b, d in (("volume_ct1_mm3", "volume_ct2_mm3", "delta_volume_mm3"),
                    ("diameter_ct1_mm", "diameter_ct2_mm", "delta_diameter_mm")):
        if not np.allclose(df[d], df[b] - df[a], rtol=0, atol=0):
            fail_output(f"diferença {d} não reproduz {b} - {a}")
    g["RELATIVE_DIFFERENCE_GATE"] = "denominador = média das duas aquisições, > 0"
    arch_out = set(df["architecture"])
    g["F100_ResEncM"] = ("NOT_ESTIMABLE_NO_MATCHED_PAIRS"
                         if "F100_ResEncM" not in arch_out else "INESPERADO")
    if "F100_ResEncM" in arch_out:
        fail_output("F100_ResEncM não devia ter pares correspondidos")
    g["CONNECTIVITY_6_GATE"] = "generate_binary_structure(3,1), explícita, CT1 e CT2 nativas"
    g["COMPONENT_IDENTITY_GATE"] = f"{len(df)}/{esperado} IDs do Script 19 resolvidos nativamente"
    g["CANONICAL_FILTER_PARITY_GATE"] = "PASS"
    g["CANONICAL_PLEURAL_PARITY_GATE"] = "PASS"
    g["INPUT_AUTHORITY_GATE"] = "PASS"
    g["NATIVE_GEOMETRY_GATE"] = "PASS"
    g["NO_INTERMEDIATE_ROUNDING"] = "PASS"

    p_csv = os.path.join(out_root, "rider_pair_features.csv")
    df.to_csv(p_csv, index=False)
    din.to_csv(os.path.join(out_root, "input_manifest.csv"), index=False)
    json.dump(par, open(os.path.join(out_root, "parity_report.json"), "w"), indent=1, default=str)
    json.dump(g, open(os.path.join(out_root, "gates.json"), "w"), indent=1)
    pd.DataFrame([{"path": os.path.relpath(p, BASE), "sha256": sha256(p)} for p in
                  (p_csv, os.path.join(out_root, "input_manifest.csv"),
                   os.path.join(out_root, "gates.json"),
                   os.path.join(out_root, "parity_report.json"))]).to_csv(
        os.path.join(out_root, "output_hashes.csv"), index=False)
    man = {
        "script": "20_extract_rider_morphometry_and_filters.py", "run_id": run_id,
        "script_sha256": sha256(os.path.abspath(__file__)),
        "run_at": t0.isoformat(), "end_at": datetime.datetime.now().astimezone().isoformat(),
        "source": {"script19_run": man19["run_id"], "script19_manifest_sha256": h,
                   "pairs_csv": os.path.relpath(pairs_p, BASE), "pairs_sha256": hp},
        "canonical_modules": {"script13": SHA13, "script14": SHA14, "script15": SHA15,
                              "reuse": "import directo — nenhuma reimplementação"},
        "definitions": {
            "connectivity": "6 (generate_binary_structure(3,1)), explícita",
            "measurement_space": "CT1 e CT2 medidas no espaço NATIVO; a máscara "
                                 "registada do Script 18 não é lida",
            "diameter": "MAXIMUM_AXIAL_DIAMETER_MM (bbox por corte axial), "
                        "função canónica do Script 13/11B",
            "size_categories": ["lt3mm", "3to6mm", "6to15mm", "ge15mm"],
            "relative_difference": "(X2 - X1) / ((X1 + X2)/2), simétrica; "
                                   "sem arredondamento intermédio",
            "orientation": "CT1 = primeira aquisição, CT2 = segunda; deltas = CT2 - CT1",
            "c_global": repr(c_global),
            "c_global_source": os.path.relpath(args.c_calibration, BASE),
            "c_global_source_sha256": sha256(args.c_calibration),
            "c_global_rounded_descriptive": round(c_global, 6),
            "c_global_calibration_population": calib.get("population"),
            "c_global_excluded_from_calibration": calib.get("excluded"),
            "rider_participated_in_calibration": False,
            "recalibration_on_rider": "NONE",
            "log_response": {"LOG_RESPONSE_SIGNED": "NO",
                             "LOG_RESPONSE_DEFINITION": "ABS_PHYSICAL_HESSIAN_TRACE",
                             "note": "|trace(H)| com H_ij = sigma_i*sigma_j*d2I/dn_i dn_j; "
                                     "é o cálculo congelado no Script 14 que gerou o BD1"},
            "analytic_features": FEATURES_16,
            "frac_high_log": "EXCLUDED_DEGENERATE_BY_CONSTRUCTION — QC only"},
        "counts": {"pairs": len(df), "patients": int(df.patient_id.nunique()),
                   "architectures": int(df.architecture.nunique()),
                   "expected_pairs": esperado},
        "parity": {"n_objects": par["n"], "divergences": par["divergencias"]},
        "gates": g,
    }
    json.dump(man, open(os.path.join(out_root, "manifest.json"), "w"), indent=2, default=str)
    log.info(f"{len(df)} pares · {df.patient_id.nunique()} pacientes · "
             f"{df.architecture.nunique()} arquitecturas")
    log.info(f"Saída: {out_root}")
    log.info("Script 20 concluído.")


if __name__ == "__main__":
    main()
