#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    22b_compute_lndb_filter_maps.py

File type:
    Script

Purpose:
    Computes the morphological filter features on the LNDb objects with the same scales and parameters as BD1.

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
    - parity_report.json
    - gates.json
    - manifest.json

Dependencies:
    - SimpleITK
    - numpy
    - pandas
    - scipy

Execution:
    PULMONARY_BASE=<project_dir> python scripts/22b_compute_lndb_filter_maps.py

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
FEATURES_DIR = os.path.join(BASE, "features")
LOGS_DIR = os.path.join(BASE, "logs")
CONSENSUS_DIR = os.path.join(BASE, "annotations", "consensus", "lndb")
PRED_DIR = os.path.join(BASE, "predictions", "lndb")
C_CALIB = os.path.join(FEATURES_DIR, "c_global_calibration.json")

C_GLOBAL_ESPERADO_R6 = 499.892883
ESPERADO_TOTAL = 3323
SHA_CANONICOS = {
    "13_prepare_filter_objects.py": "18533b0b1d53c14d46ec08637b3629afdc9846f1dcb2d04630fb8cabcdc3762a",
    "14_compute_filter_maps_gpu.py": "64fe7a82c5cec0146371f7cc7b9c6927932a8816110890f4c293b82cb1e1ea99",
    "15_extract_filter_features.py": "d9adb7af9bf7c0b39c2020118abafd5e85ccf2befb595319e2e6a47267468497"}
STRUCT_6 = nd.generate_binary_structure(3, 1)
CHAVE = ["source", "scan_id", "architecture", "group", "object_id"]
COMBINACOES_VALIDAS = {("pred", "TP_pred"), ("pred", "FP_pred"),
                       ("gt", "TP_gt"), ("gt", "FN_gt")}
FEATURES_16 = [
    "log_max", "log_mean", "log_p95", "log_scale_mm",
    "blob_max", "blob_mean", "blob_p95", "blob_scale_mm",
    "vessel_max", "vessel_mean", "vessel_p95", "vessel_scale_mm",
    "contrast_hu", "centroid_peak_dist_mm",
    "pleura_min_dist_mm", "pleura_centroid_dist_mm",
]
QC_COLS = ["mu_object_hu", "mu_ring_hu", "ring_voxel_count",
           "ring_truncated_at_image_boundary", "object_voxel_count_in_roi",
           "frac_high_log", "component_voxels_total", "component_voxels_inside_lung",
           "component_voxels_outside_lung", "component_fraction_inside_lung",
           "component_intersects_lung_surface", "component_entirely_outside_lung"]

os.makedirs(LOGS_DIR, exist_ok=True)
log = logging.getLogger("s22b")
log.setLevel(logging.INFO)
if not log.handlers:
    _f = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for _h in (logging.StreamHandler(),
               logging.FileHandler(os.path.join(LOGS_DIR, "compute_lndb_filter_maps.log"))):
        _h.setFormatter(_f); log.addHandler(_h)
log.propagate = False


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


def _carregar(nome, ficheiro):
    p = os.path.join(SCRIPTS_DIR, ficheiro)
    if not os.path.isfile(p):
        fail_input(f"módulo canónico ausente: {p}")
    sp = importlib.util.spec_from_file_location(nome, p)
    m = importlib.util.module_from_spec(sp)
    sys.modules[nome] = m
    sp.loader.exec_module(m)
    return m, sha256(p)


S13, SHA13 = _carregar("s13_canon_22b", "13_prepare_filter_objects.py")
S14, SHA14 = _carregar("s14_canon_22b", "14_compute_filter_maps_gpu.py")
S15, SHA15 = _carregar("s15_canon_22b", "15_extract_filter_features.py")
from caminhos_canonicos import raiz_recortes, ct_de, mascara_pulmonar_de

for _f, _sha in (("13_prepare_filter_objects.py", SHA13),
                 ("14_compute_filter_maps_gpu.py", SHA14),
                 ("15_extract_filter_features.py", SHA15)):
    if _sha != SHA_CANONICOS[_f]:
        fail_input(f"MODULE_SHA_GATE: {_f} tem SHA {_sha[:12]} != canónico "
                   f"{SHA_CANONICOS[_f][:12]}")
if tuple(S14.SCALES_MM) != (0.8, 1.2, 1.8, 2.7, 4.0, 6.0, 8.0):
    fail_input(f"SCALES_MM do Script 14 divergem: {S14.SCALES_MM}")
if tuple(S14.HU_CLIP) != (-1000.0, 400.0):
    fail_input(f"HU_CLIP do Script 14 diverge: {S14.HU_CLIP}")
if (S14.ALPHA, S14.BETA, S14.RING_MM) != (0.5, 0.5, 5.0):
    fail_input("ALPHA/BETA/RING_MM do Script 14 divergem")
if S13.MARGIN_MM != 15.0:
    fail_input(f"MARGIN_MM do Script 13 diverge: {S13.MARGIN_MM}")


def caracteristicas(ct_arr, rotulado, binario_fonte, sp_zyx, cid, bbox, dist, lung,
                    c_global, rotulo):
    saida, err = S14.process_roi(ct_arr, rotulado, binario_fonte, sp_zyx, cid, bbox,
                                 c_global, "cpu", rotulo)
    if err is not None:
        fail_output(f"ROI irresolúvel em {rotulo}: {err}")
    feats = dict(saida[0])
    comp = (rotulado == cid)
    feats.update(S15.pleural_features(comp, dist, lung, sp_zyx, rotulo))
    falta = [f for f in FEATURES_16 if f not in feats]
    if falta:
        fail_output(f"características canónicas em falta em {rotulo}: {falta}")
    for f in FEATURES_16:
        if not np.isfinite(float(feats[f])):
            fail_output(f"{f} não finita em {rotulo}")
    return feats


_G = {}


def _init(c_global, raiz):
    _G["c"] = c_global; _G["raiz"] = raiz


def _varrimento(args_):
    sid, grupo = args_
    raiz, c_global = _G["raiz"], _G["c"]
    ctp = ct_de(raiz, "lndb", sid)
    ct_img = sitk.ReadImage(ctp, sitk.sitkFloat32)
    sp = ct_img.GetSpacing(); sp_zyx = (float(sp[2]), float(sp[1]), float(sp[0]))
    ct_arr = np.clip(sitk.GetArrayFromImage(ct_img), *S14.HU_CLIP).astype(np.float32)
    lmp = mascara_pulmonar_de(raiz, "lndb", sid, recortada=True)
    dist, lung, _ = S15.pleura_edt(lmp, ct_img, sid)
    dist = dist.astype(np.float32)

    gtp = os.path.join(CONSENSUS_DIR, sid, "consensus.nii.gz")
    gt_img = sitk.ReadImage(gtp)
    gt_full, _ = nd.label(sitk.GetArrayFromImage(gt_img) > 0, structure=STRUCT_6)
    ci = np.array(gt_img.TransformPhysicalPointToContinuousIndex(ct_img.GetOrigin()))
    if np.abs(ci - np.round(ci)).max() > 1e-3:
        fail_input(f"deslocamento de índice não inteiro em {sid}")
    o = np.round(ci).astype(int)
    Z, Y, X = ct_arr.shape
    gt_lab = gt_full[o[2]:o[2] + Z, o[1]:o[1] + Y, o[0]:o[0] + X].astype(np.int32)
    gt_bin = gt_lab > 0

    preds = {}
    linhas = []
    for _, r in grupo.iterrows():
        rot = f"{r.source}/{sid}/{r.architecture}/{r.group}/{int(r.object_id)}"
        if (r.source, r.group) not in COMBINACOES_VALIDAS:
            fail_output(f"SOURCE_GROUP ilegal em {rot}")
        if r.source == "gt":
            rotulado, binario = gt_lab, gt_bin
        else:
            if r.architecture not in preds:
                pp = os.path.join(PRED_DIR, r.architecture, f"{sid}.nii.gz")
                pb = sitk.GetArrayFromImage(sitk.ReadImage(pp)) > 0
                pl, _ = nd.label(pb, structure=STRUCT_6)
                preds[r.architecture] = (pl.astype(np.int32), pb)
            rotulado, binario = preds[r.architecture]
        bbox = {k: int(r[k]) for k in ("bbox_z0", "bbox_z1", "bbox_y0", "bbox_y1",
                                       "bbox_x0", "bbox_x1")}
        if not (rotulado == int(r.object_id)).any():
            fail_output(f"COMPONENT_IDENTITY: objecto {rot} não existe na fonte")
        f = caracteristicas(ct_arr, rotulado, binario, sp_zyx, int(r.object_id), bbox,
                            dist, lung, c_global, rot)
        linha = {c: r[c] for c in CHAVE}
        linha.update({"match_status": r.match_status, "iou": r.iou, "dice": r.dice,
                      "voxel_count": int(r.voxel_count), "volume_mm3": float(r.volume_mm3),
                      "diam_mm": float(r.diam_mm),
                      "roi_clipped_at_image_boundary": int(r.roi_clipped_at_image_boundary)})
        for c in FEATURES_16 + [q for q in QC_COLS if q in f]:
            linha[c] = f[c]
        linhas.append(linha)
    return linhas


def paridade(c_global, n, seed=27):
    """Recalcula objectos reais do BD1 por este caminho e compara com a matriz
    canónica do Script 15. Nominalmente igual tem de ser numericamente igual."""
    ref_p = os.path.join(BASE, "figdata", "bd1_filter_features.csv")
    if not os.path.isfile(ref_p):
        fail_input(f"matriz canónica do Script 15 ausente: {ref_p}")
    ref = pd.read_csv(ref_p)
    ref = ref[ref["group"].isin(["TP_pred", "FP_pred"])].sample(n=n, random_state=seed)
    raiz = raiz_recortes()
    difs, linhas = [], []
    for _, r in ref.iterrows():
        pid, arch, cid = r["patient_id"], r["architecture"], int(r["component_id"])
        mp = r["mask_path"] if os.path.isabs(str(r["mask_path"])) \
            else os.path.join(BASE, str(r["mask_path"]))
        cp = r["ct_path"] if os.path.isabs(str(r["ct_path"])) \
            else os.path.join(BASE, str(r["ct_path"]))
        for p in (mp, cp):
            if not os.path.isfile(p):
                fail_input(f"objecto BD1 da paridade não resolúvel: {p}")
        ct_img = sitk.ReadImage(cp, sitk.sitkFloat32)
        sp = ct_img.GetSpacing(); sp_zyx = (float(sp[2]), float(sp[1]), float(sp[0]))
        ct_arr = np.clip(sitk.GetArrayFromImage(ct_img), *S14.HU_CLIP).astype(np.float32)
        binario = sitk.GetArrayFromImage(sitk.ReadImage(mp)) > 0
        rotulado, _ = nd.label(binario, structure=STRUCT_6)
        rotulado = rotulado.astype(np.int32)
        bbox = S13.object_record(rotulado, cid, sp_zyx, ct_arr.shape, f"paridade {pid}")
        lmp = mascara_pulmonar_de(raiz, "lidc_idri", pid, recortada=True)
        if not os.path.isfile(lmp):
            fail_input(f"máscara pulmonar da paridade ausente: {lmp}")
        dist, lung, _ = S15.pleura_edt(lmp, ct_img, pid)
        f = caracteristicas(ct_arr, rotulado, binario, sp_zyx, cid, bbox, dist, lung,
                            c_global, f"paridade {pid}/{arch}/c{cid}")
        l = {"patient_id": pid, "architecture": arch, "component_id": cid}
        for c in FEATURES_16:
            a, b = float(r[c]), float(f[c])
            l[c] = {"canonico": a, "script22b": b, "igual": bool(a == b or abs(a - b) <= 1e-9)}
            if not l[c]["igual"]:
                difs.append((pid, arch, cid, c, a, b))
        linhas.append(l)
    nucleo = [c for c in FEATURES_16 if not c.startswith("pleura_")]
    pleu = [c for c in FEATURES_16 if c.startswith("pleura_")]
    def piores(cols):
        d = [(abs(l[c]["script22b"] - l[c]["canonico"]),
              abs(l[c]["script22b"] - l[c]["canonico"]) / max(abs(l[c]["canonico"]), 1e-12))
             for l in linhas for c in cols]
        return (max(x[0] for x in d) if d else 0.0, max(x[1] for x in d) if d else 0.0)
    an, rn = piores(nucleo); ap_, rp = piores(pleu)
    return {"n": len(linhas), "divergencias": len(difs), "exemplos": difs[:10],
            "features_comparadas": FEATURES_16,
            "FILTER_CORE_PARITY_GATE": {
                "variaveis": len(nucleo),
                "divergencias": sum(1 for d in difs if d[3] in nucleo),
                "max_dif_absoluta": an, "max_dif_relativa": rn},
            "PLEURAL_PARITY_GATE": {
                "variaveis": len(pleu),
                "divergencias": sum(1 for d in difs if d[3] in pleu),
                "max_dif_absoluta": ap_, "max_dif_relativa": rp}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s22a-run", required=True)
    ap.add_argument("--s22a-manifest-sha256", required=True)
    ap.add_argument("--c-calibration", default=C_CALIB)
    ap.add_argument("--limit-scans", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--parity-sample", type=int, default=10)
    ap.add_argument("--parity-only", action="store_true")
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()
    t0 = datetime.datetime.now().astimezone()

    congeladas = {"SCALES_MM": (tuple(S14.SCALES_MM), (0.8, 1.2, 1.8, 2.7, 4.0, 6.0, 8.0)),
                  "HU_CLIP": (tuple(S14.HU_CLIP), (-1000.0, 400.0)),
                  "ALPHA": (S14.ALPHA, 0.5), "BETA": (S14.BETA, 0.5),
                  "RING_MM": (S14.RING_MM, 5.0), "MARGIN_MM": (S13.MARGIN_MM, 15.0)}
    for nome, (obtido, esperado) in congeladas.items():
        if obtido != esperado:
            fail_input(f"FROZEN_PARAMETER_GATE: {nome} = {obtido} != {esperado}")
    if set(FEATURES_16) & {"frac_high_log"}:
        fail_input("frac_high_log não pode entrar no conjunto analítico")
    if len(FEATURES_16) != 16:
        fail_input(f"FEATURE_SET_GATE: {len(FEATURES_16)} variáveis != 16")

    if not os.path.isfile(args.c_calibration):
        fail_input(f"calibração de c_global ausente: {args.c_calibration}")
    calib = json.load(open(args.c_calibration))
    c_global = float(calib["c_global"])
    if round(c_global, 6) != C_GLOBAL_ESPERADO_R6:
        fail_input(f"c_global {c_global} != congelado {C_GLOBAL_ESPERADO_R6}")
    if "LNDb" not in (calib.get("excluded") or []):
        fail_input("a calibração de c_global não declara o LNDb como excluído")
    log.info(f"c_global = {c_global!r} (BD1 OOF; LNDb e RIDER excluídos)")

    par = paridade(c_global, args.parity_sample)
    log.info(f"PARIDADE 14/15 → 22B: {par['n']} objectos BD1, {par['divergencias']} "
             f"divergências em {len(FEATURES_16)} variáveis")
    if par["divergencias"]:
        for d in par["exemplos"]:
            log.error(f"  divergência {d}")
        fail_output("CANONICAL_FILTER_PARITY_GATE falhou")
    if args.parity_only:
        print(json.dumps({"CANONICAL_PARITY": "PASS", **par}, indent=1, default=str))
        return

    man_p = os.path.join(args.s22a_run, "manifest.json")
    if not os.path.isfile(man_p):
        fail_input(f"run do Script 22A sem manifest.json: {man_p}")
    h = sha256(man_p)
    if h != args.s22a_manifest_sha256:
        fail_input(f"INPUT_AUTHORITY: SHA do manifesto do 22A {h[:12]} "
                   f"!= {args.s22a_manifest_sha256[:12]}")
    man22a = json.load(open(man_p))
    if man22a["run_id"] != os.path.basename(os.path.normpath(args.s22a_run)):
        fail_input("run_id do manifesto do 22A != pasta")
    rois_p = os.path.join(args.s22a_run, "lndb_object_rois.parquet")
    if not os.path.isfile(rois_p):
        fail_input(f"produto científico do 22A ausente: {rois_p}")
    oh = pd.read_csv(os.path.join(args.s22a_run, "output_hashes.csv"))
    hr = sha256(rois_p)
    linha = oh[oh["path"] == "lndb_object_rois.parquet"]
    if len(linha) != 1 or linha["sha256"].iloc[0] != hr:
        fail_input("SHA do parquet de ROIs não confere com o output_hashes.csv do 22A")
    csv_p = os.path.join(args.s22a_run, "lndb_object_rois.csv")
    if os.path.isfile(csv_p):
        lc = oh[oh["path"] == "lndb_object_rois.csv"]
        if len(lc) == 1 and lc["sha256"].iloc[0] != sha256(csv_p):
            fail_input("SHA da cópia CSV de ROIs não confere com o output_hashes.csv do 22A")
    rois = pd.read_parquet(rois_p)
    if rois.duplicated(CHAVE).any():
        fail_input("chave repetida na tabela de ROIs")
    combos = set(zip(rois.source, rois.group))
    if not combos <= COMBINACOES_VALIDAS:
        fail_input(f"SOURCE_GROUP_GATE: combinações ilegais "
                   f"{sorted(combos - COMBINACOES_VALIDAS)}")
    log.info(f"Script 22A canónico {man22a['run_id']}: {len(rois)} objectos, "
             f"grupos {rois.group.value_counts().to_dict()}")

    scans = sorted(rois["scan_id"].unique())
    if args.limit_scans:
        scans = scans[: args.limit_scans]
        rois = rois[rois["scan_id"].isin(scans)]
    esperado = len(rois)

    raiz = raiz_recortes()
    run_id = f"s22b_{t0:%Y%m%dT%H%M%S}_{os.getpid()}"
    out_root = os.path.join(args.out_root or os.path.join(FEATURES_DIR, "lndb_filter_features"), run_id)
    os.makedirs(out_root, exist_ok=True)

    tarefas = [(s, rois[rois.scan_id == s]) for s in scans]
    linhas = []
    if args.workers > 1:
        with Pool(args.workers, initializer=_init, initargs=(c_global, raiz),
                  maxtasksperchild=4) as pool:
            for i, ls in enumerate(pool.imap_unordered(_varrimento, tarefas)):
                linhas += ls
                if (i + 1) % 25 == 0 or i + 1 == len(tarefas):
                    log.info(f"  [{i+1}/{len(tarefas)}] {len(linhas)} objectos")
    else:
        _init(c_global, raiz)
        for i, t in enumerate(tarefas):
            linhas += _varrimento(t)
            if (i + 1) % 25 == 0 or i + 1 == len(tarefas):
                log.info(f"  [{i+1}/{len(tarefas)}] {t[0]}: {len(linhas)} objectos")

    df = pd.DataFrame(linhas)
    g = {"INPUT_AUTHORITY_GATE": "PASS", "SCRIPT22A_RUN": man22a["run_id"],
         "MODULE_SHA_GATE": "PASS (13/14/15 conferidos contra os SHA canónicos)",
         "FILTER_CORE_PARITY_GATE": par["FILTER_CORE_PARITY_GATE"],
         "PLEURAL_PARITY_GATE": par["PLEURAL_PARITY_GATE"],
         "CANONICAL_FILTER_PARITY_GATE": "PASS",
         "C_GLOBAL": repr(c_global),
         "C_GLOBAL_SOURCE_SHA256": sha256(args.c_calibration),
         "RECALIBRATION_ON_LNDB": "NONE",
         "FEATURE_SET_GATE": f"{len(FEATURES_16)} variáveis analíticas",
         "FROZEN_PARAMETER_GATE": {k: str(v[0]) for k, v in congeladas.items()},
         "frac_high_log": "QC_ONLY — ANALYTIC_FEATURE = NO",
         "RING_SOURCE": "primeiro plano da MESMA fonte (predita ou referência)",
         "SOURCE_GROUP_GATE": "só as quatro combinações legais"}
    if len(df) != esperado:
        fail_output(f"BALANCE_GATE: {len(df)} != {esperado} objectos de entrada")
    g["BALANCE_GATE"] = f"{len(df)}/{esperado}"
    entrada_completa = (not args.limit_scans) and len(pd.read_parquet(rois_p)) == ESPERADO_TOTAL
    if entrada_completa:
        if len(df) != ESPERADO_TOTAL:
            fail_output(f"CANONICAL_COUNT_GATE: {len(df)} != {ESPERADO_TOTAL}")
        g["CANONICAL_COUNT_GATE"] = f"{ESPERADO_TOTAL} objectos"
    else:
        g["CANONICAL_COUNT_GATE"] = f"NAO_APLICAVEL — entrada parcial ({len(df)} objectos)"
    if df.duplicated(CHAVE).any():
        fail_output("KEY_GATE: chave repetida na saída")
    kn = set(map(tuple, df[CHAVE].values)); ke = set(map(tuple, rois[CHAVE].values))
    if kn != ke:
        fail_output(f"KEY_GATE: {len(ke - kn)} em falta, {len(kn - ke)} inesperadas")
    g["KEY_GATE"] = "PASS"
    if not np.isfinite(df[FEATURES_16].to_numpy()).all():
        fail_output("característica não finita na saída")
    g["NONFINITE_GATE"] = "PASS"
    g["GROUP_COUNTS"] = {k: int(v) for k, v in df.group.value_counts().items()}

    prods = {}

    def grava(nome, frame):
        p = os.path.join(out_root, nome); frame.to_csv(p, index=False); prods[nome] = p

    pq = os.path.join(out_root, "lndb_filter_features_raw.parquet")
    df.to_parquet(pq, index=False)
    prods["lndb_filter_features_raw.parquet"] = pq
    grava("lndb_filter_features.csv", df)
    grava("input_manifest.csv", pd.DataFrame([{
        "script22a_run": man22a["run_id"], "script22a_manifest_sha256": h,
        "object_rois_parquet_sha256": hr, "c_calibration": os.path.relpath(args.c_calibration, BASE),
        "c_calibration_sha256": sha256(args.c_calibration)}]))
    json.dump(par, open(os.path.join(out_root, "parity_report.json"), "w"), indent=1, default=str)
    json.dump(g, open(os.path.join(out_root, "gates.json"), "w"), indent=1, default=str)
    prods["parity_report.json"] = os.path.join(out_root, "parity_report.json")
    prods["gates.json"] = os.path.join(out_root, "gates.json")
    pd.DataFrame([{"path": n, "sha256": sha256(p)} for n, p in prods.items()]).to_csv(
        os.path.join(out_root, "output_hashes.csv"), index=False)
    man = {
        "script": "22b_compute_lndb_filter_maps.py", "run_id": run_id,
        "script_sha256": sha256(os.path.abspath(__file__)),
        "run_at": t0.isoformat(), "end_at": datetime.datetime.now().astimezone().isoformat(),
        "source": {"script22a_run": man22a["run_id"], "script22a_manifest_sha256": h,
                   "object_rois_parquet_sha256": hr},
        "canonical_modules": {"script13": SHA13, "script14": SHA14, "script15": SHA15,
                              "reuse": "import directo — nenhuma reimplementação"},
        "definitions": {
            "scales_mm": list(S14.SCALES_MM), "hu_clip": list(S14.HU_CLIP),
            "alpha": S14.ALPHA, "beta": S14.BETA, "ring_mm": S14.RING_MM,
            "roi_margin_mm": S13.MARGIN_MM, "connectivity": 6,
            "c_global": repr(c_global),
            "c_global_source": os.path.relpath(args.c_calibration, BASE),
            "recalibration_on_lndb": "NONE",
            "log_response": {"LOG_RESPONSE_SIGNED": "NO",
                             "LOG_RESPONSE_DEFINITION": "ABS_PHYSICAL_HESSIAN_TRACE"},
            "analytic_features": FEATURES_16,
            "frac_high_log": "EXCLUDED_DEGENERATE_BY_CONSTRUCTION — QC only",
            "no_feature_selection_after_lndb": True},
        "counts": {"scans": len(scans), "objects": len(df), "expected": esperado,
                   "by_group": g["GROUP_COUNTS"]},
        "parity": {"n_objects": par["n"], "divergences": par["divergencias"]},
        "gates": g,
        "promotion": "NOT PROMOTED — produtos ficam no directório do run",
    }
    json.dump(man, open(os.path.join(out_root, "manifest.json"), "w"), indent=2, default=str)
    log.info(f"{len(df)} objectos · grupos {g['GROUP_COUNTS']}")
    log.info(f"Saída: {out_root}")
    log.info("Script 22B concluído.")


if __name__ == "__main__":
    main()
