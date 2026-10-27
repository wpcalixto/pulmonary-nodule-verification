#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    14_compute_filter_maps_gpu.py

File type:
    Script

Purpose:
    Computes multiscale 3D morphological filter responses (LoG, Hessian blobness and vesselness, among others) on the BD1 regions of interest using the GPU.

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
    - {name}.nii.gz
    - roi_failures{suf}.json
    - filter_features_manifest{suf}.json

Dependencies:
    - SimpleITK
    - numpy
    - pandas
    - scipy
    - torch

Execution:
    PULMONARY_BASE=<project_dir> python scripts/14_compute_filter_maps_gpu.py

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
import faulthandler
import functools
import hashlib
import json
import logging
import os

faulthandler.enable()

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage as nd

from pulmonary_base import BASE
from caminhos_canonicos import raiz_recortes, ct_de, FalhaEntrada

FEATURES_DIR = os.path.join(BASE, "features")
FIGDATA_DIR  = os.path.join(BASE, "figdata", "filter_maps", "examples")
LOGS_DIR     = os.path.join(BASE, "logs")

SCALES_MM       = (0.8, 1.2, 1.8, 2.7, 4.0, 6.0, 8.0)
HU_CLIP         = (-1000.0, 400.0)
RING_MM         = 5.0
ALPHA = BETA    = 0.5
N_EXAMPLES      = 5
MAX_CROP_VOXELS = 2_000_000
STRUCT_6        = nd.generate_binary_structure(3, 1)
EPS             = 1e-10

for _d in (FEATURES_DIR, FIGDATA_DIR, LOGS_DIR):
    os.makedirs(_d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(os.path.join(LOGS_DIR, "compute_filter_maps.log")),
              logging.StreamHandler()],
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


TORCH = None
DEVICE = None


def setup_backend(name: str) -> str:
    """`cpu` ou `gpu`; um único backend para toda a execução (sem troca por ROI)."""
    global TORCH, DEVICE
    if name == "cpu":
        return "cpu"
    import torch
    if not torch.cuda.is_available():
        fail_input("backend=gpu pedido mas CUDA indisponível")
    TORCH = torch
    DEVICE = torch.device("cuda")
    return "gpu"


@functools.lru_cache(maxsize=1024)
def _gauss1d(sigma: float, order: int, n: int):
    """Kernel 1D da derivada gaussiana, obtido por resposta ao impulso do SciPy
    (garante numerics idênticos entre CPU e GPU)."""
    radius = max(1, min(int(4.0 * sigma + 0.5), (n - 1) // 2 if n >= 3 else 1))
    size = 2 * radius + 1
    imp = np.zeros(size, dtype=np.float64)
    imp[radius] = 1.0
    return nd.gaussian_filter1d(imp, sigma, order=order, mode="nearest").astype(np.float32)


def _deriv(vol, sp_sigma_vox, orders, backend):
    """Derivada gaussiana separável de ordem `orders` (z,y,x) com σ em voxels."""
    if backend == "cpu":
        out = np.asarray(vol, dtype=np.float64)
        for ax in range(3):
            k = _gauss1d(float(sp_sigma_vox[ax]), int(orders[ax]), vol.shape[ax]).astype(np.float64)
            out = nd.convolve1d(out, k, axis=ax, mode="nearest")
        return out.astype(np.float32)
    x = vol.unsqueeze(0).unsqueeze(0)
    for ax in range(3):
        k = TORCH.from_numpy(_gauss1d(float(sp_sigma_vox[ax]), int(orders[ax]), vol.shape[ax])).to(DEVICE)
        pad = len(k) // 2
        shape = [1, 1, 1, 1, 1]; shape[2 + ax] = -1
        p = [0, 0, 0]; p[ax] = pad
        x = TORCH.nn.functional.pad(x, (p[2], p[2], p[1], p[1], p[0], p[0]), mode="replicate")
        x = TORCH.nn.functional.conv3d(x, k.view(*shape))
    return x.squeeze(0).squeeze(0)


def hessian_physical(crop, sp_zyx, sigma_mm, backend):
    """Hessiana normalizada em unidades físicas: H_ij = σ_i·σ_j·∂²I/∂n_i∂n_j.

    Devolve (Hzz, Hyy, Hxx, Hzy, Hzx, Hyx) — a mesma fonte para LoG (traço) e
    para os autovalores (blobness/vesselness)."""
    s = tuple(sigma_mm / sp for sp in sp_zyx)
    d = lambda o: _deriv(crop, s, o, backend)
    Hzz = d((2, 0, 0)) * (s[0] ** 2)
    Hyy = d((0, 2, 0)) * (s[1] ** 2)
    Hxx = d((0, 0, 2)) * (s[2] ** 2)
    Hzy = d((1, 1, 0)) * (s[0] * s[1])
    Hzx = d((1, 0, 1)) * (s[0] * s[2])
    Hyx = d((0, 1, 1)) * (s[1] * s[2])
    return Hzz, Hyy, Hxx, Hzy, Hzx, Hyx


def frobenius2(H):
    Hzz, Hyy, Hxx, Hzy, Hzx, Hyx = H
    return (Hzz ** 2 + Hyy ** 2 + Hxx ** 2
            + 2.0 * (Hzy ** 2 + Hzx ** 2 + Hyx ** 2))


def eigvals_sym3(H, backend):
    """Autovalores analíticos de matriz simétrica 3×3, vectorizados, ordenados
    por |λ| crescente. Verificado contra `numpy.linalg.eigvalsh`."""
    a, b, c, d, f, e = H
    xp = np if backend == "cpu" else TORCH
    sqrt = np.sqrt if backend == "cpu" else TORCH.sqrt
    acos = np.arccos if backend == "cpu" else TORCH.acos
    cos = np.cos if backend == "cpu" else TORCH.cos
    clip = (lambda v, lo, hi: np.clip(v, lo, hi)) if backend == "cpu" else (lambda v, lo, hi: TORCH.clamp(v, lo, hi))
    q = (a + b + c) / 3.0
    p2 = ((a - q) ** 2 + (b - q) ** 2 + (c - q) ** 2 + 2.0 * (d ** 2 + e ** 2 + f ** 2)) / 6.0
    p = sqrt(clip(p2, 0.0, None) if backend == "cpu" else clip(p2, 0.0, float("inf")))
    p_safe = p + EPS
    aa, bb, cc = (a - q) / p_safe, (b - q) / p_safe, (c - q) / p_safe
    dd, ee, ff = d / p_safe, e / p_safe, f / p_safe
    detB = (aa * (bb * cc - ee * ee) - dd * (dd * cc - ee * ff) + ff * (dd * ee - bb * ff))
    phi = acos(clip(detB / 2.0, -1.0, 1.0)) / 3.0
    e1 = q + 2.0 * p * cos(phi)
    e3 = q + 2.0 * p * cos(phi + 2.0 * np.pi / 3.0)
    e2 = 3.0 * q - e1 - e3
    if backend == "cpu":
        ev = np.stack([e1, e2, e3], axis=-1)
        idx = np.argsort(np.abs(ev), axis=-1)
        return np.take_along_axis(ev, idx, axis=-1)
    ev = TORCH.stack([e1, e2, e3], dim=-1)
    idx = TORCH.argsort(ev.abs(), dim=-1)
    return TORCH.gather(ev, -1, idx)


def vesselness_blobness(ev, S2, c_global, backend):
    """Frangi 1998 (vesselness) e HESSIAN_BLOBNESS_3D (definição do projecto)."""
    l1, l2, l3 = ev[..., 0], ev[..., 1], ev[..., 2]
    absf = np.abs if backend == "cpu" else TORCH.abs
    sqrt = np.sqrt if backend == "cpu" else TORCH.sqrt
    exp = np.exp if backend == "cpu" else TORCH.exp
    where = np.where if backend == "cpu" else TORCH.where
    zeros = (lambda x: np.zeros_like(x)) if backend == "cpu" else (lambda x: TORCH.zeros_like(x))
    RA = absf(l2) / (absf(l3) + EPS)
    RB = absf(l1) / (sqrt(absf(l2 * l3)) + EPS)
    tA = 1.0 - exp(-(RA ** 2) / (2.0 * ALPHA ** 2))
    tB = exp(-(RB ** 2) / (2.0 * BETA ** 2))
    tS = 1.0 - exp(-S2 / (2.0 * c_global ** 2))
    V = tA * tB * tS
    B = tA * (1.0 - tB) * tS
    V = where((l2 >= 0) | (l3 >= 0), zeros(V), V)
    B = where((l1 >= 0) | (l2 >= 0) | (l3 >= 0), zeros(B), B)
    return V, B


_cache = {}


def load_ct(path):
    if path not in _cache:
        img = sitk.ReadImage(path, sitk.sitkFloat32)
        arr = np.clip(sitk.GetArrayFromImage(img), *HU_CLIP).astype(np.float32)
        sp = img.GetSpacing()
        if not all(np.isfinite(sp)) or min(sp) <= 0:
            fail_input(f"espaçamento inválido {sp} em {path}")
        _cache[path] = (arr, (float(sp[2]), float(sp[1]), float(sp[0])), img)
    return _cache[path]


def load_mask(path):
    """Rebinariza e rotula com 6-conectividade — igual aos Scripts 11/13."""
    key = ("m", path)
    if key not in _cache:
        arr = sitk.GetArrayFromImage(sitk.ReadImage(path))
        binary = (arr > 0)
        labeled, n = nd.label(binary, structure=STRUCT_6)
        _cache[key] = (labeled, binary, int(n))
    return _cache[key]


def ring_mask(comp_crop, other_fg_crop, sp_zyx):
    """Anel de RING_MM em distância física, sem o restante primeiro plano da
    mesma fonte. Devolve (ring, truncated)."""
    dist = nd.distance_transform_edt(~comp_crop, sampling=sp_zyx)
    ring = (~comp_crop) & (dist > 0) & (dist <= RING_MM) & (~other_fg_crop)
    return ring, dist


def process_roi(ct_arr, labeled, binary_all, sp_zyx, cid, bbox, c_global, backend):
    z0, z1, y0, y1, x0, x1 = (bbox[k] for k in ("bbox_z0", "bbox_z1", "bbox_y0", "bbox_y1", "bbox_x0", "bbox_x1"))
    crop = ct_arr[z0:z1, y0:y1, x0:x1]
    comp = (labeled[z0:z1, y0:y1, x0:x1] == cid)
    if not comp.any():
        return None, "component_unresolved"
    if crop.size > MAX_CROP_VOXELS:
        return None, f"crop_too_large:{crop.size}"
    other_fg = binary_all[z0:z1, y0:y1, x0:x1] & ~comp

    if backend == "gpu":
        vol = TORCH.from_numpy(np.ascontiguousarray(crop)).to(DEVICE)
        comp_t = TORCH.from_numpy(comp).to(DEVICE)
        neg_inf = float("-inf")
        log_max = TORCH.full_like(vol, neg_inf); log_sc = TORCH.zeros_like(vol)
        bl_max = TORCH.full_like(vol, neg_inf); bl_sc = TORCH.zeros_like(vol)
        ve_max = TORCH.full_like(vol, neg_inf); ve_sc = TORCH.zeros_like(vol)
        for s_mm in SCALES_MM:
            H = hessian_physical(vol, sp_zyx, s_mm, backend)
            lg = (H[0] + H[1] + H[2]).abs()
            S2 = frobenius2(H)
            ev = eigvals_sym3(H, backend)
            V, Bb = vesselness_blobness(ev, S2, c_global, backend)
            for cur, mx, sc in ((lg, log_max, log_sc), (Bb, bl_max, bl_sc), (V, ve_max, ve_sc)):
                m = cur > mx
                mx.copy_(TORCH.where(m, cur, mx)); sc.copy_(TORCH.where(m, TORCH.full_like(sc, s_mm), sc))
        gather = lambda t: t[comp_t].detach().cpu().numpy()
        lgv, lgs = gather(log_max), gather(log_sc)
        blv, bls = gather(bl_max), gather(bl_sc)
        vev, ves = gather(ve_max), gather(ve_sc)
        log_map = log_max.detach().cpu().numpy(); blob_map = bl_max.detach().cpu().numpy(); ves_map = ve_max.detach().cpu().numpy()
        del vol, comp_t, log_max, log_sc, bl_max, bl_sc, ve_max, ve_sc
        TORCH.cuda.empty_cache()
    else:
        neg = np.float32(-np.inf)
        log_max = np.full(crop.shape, neg, np.float32); log_sc = np.zeros(crop.shape, np.float32)
        bl_max = np.full(crop.shape, neg, np.float32); bl_sc = np.zeros(crop.shape, np.float32)
        ve_max = np.full(crop.shape, neg, np.float32); ve_sc = np.zeros(crop.shape, np.float32)
        for s_mm in SCALES_MM:
            H = hessian_physical(crop, sp_zyx, s_mm, backend)
            lg = np.abs(H[0] + H[1] + H[2]); S2 = frobenius2(H)
            ev = eigvals_sym3(H, backend)
            V, Bb = vesselness_blobness(ev, S2, c_global, backend)
            for cur, mx, sc in ((lg, log_max, log_sc), (Bb, bl_max, bl_sc), (V, ve_max, ve_sc)):
                m = cur > mx
                mx[m] = cur[m].astype(np.float32); sc[m] = s_mm
        lgv, lgs = log_max[comp], log_sc[comp]
        blv, bls = bl_max[comp], bl_sc[comp]
        vev, ves = ve_max[comp], ve_sc[comp]
        log_map, blob_map, ves_map = log_max, bl_max, ve_max

    ring, _ = ring_mask(comp, other_fg, sp_zyx)
    if not ring.any():
        return None, "ring_empty"
    mu_obj = float(crop[comp].mean()); mu_ring = float(crop[ring].mean())
    touches = bool(comp[0].any() or comp[-1].any() or comp[:, 0].any() or comp[:, -1].any()
                   or comp[:, :, 0].any() or comp[:, :, -1].any())

    def snap(v):
        """Reconverte a escala lida do mapa float32 à escala congelada exacta."""
        arr = np.asarray(SCALES_MM, dtype=np.float64)
        j = int(np.argmin(np.abs(arr - float(v))))
        if abs(arr[j] - float(v)) > 1e-4:
            fail_output(f"escala dominante {v} fora do conjunto congelado")
        return float(arr[j])

    i_log, i_bl, i_ve = int(np.argmax(lgv)), int(np.argmax(blv)), int(np.argmax(vev))
    feats = {
        "log_max": float(lgv[i_log]), "log_mean": float(lgv.mean()),
        "log_p95": float(np.percentile(lgv, 95)), "log_scale_mm": snap(lgs[i_log]),
        "frac_high_log": float((lgv > np.percentile(lgv, 75)).mean()),
        "blob_max": float(blv[i_bl]), "blob_mean": float(blv.mean()),
        "blob_p95": float(np.percentile(blv, 95)), "blob_scale_mm": snap(bls[i_bl]),
        "vessel_max": float(vev[i_ve]), "vessel_mean": float(vev.mean()),
        "vessel_p95": float(np.percentile(vev, 95)), "vessel_scale_mm": snap(ves[i_ve]),
        "mu_object_hu": round(mu_obj, 3), "mu_ring_hu": round(mu_ring, 3),
        "contrast_hu": round(mu_obj - mu_ring, 3),
        "ring_voxel_count": int(ring.sum()),
        "ring_truncated_at_image_boundary": int(touches),
        "object_voxel_count_in_roi": int(comp.sum()),
    }
    return (feats, (crop, log_map, blob_map, ves_map, (z0, y0, x0))), None


def save_example_maps(maps, ct_img, out_dir):
    crop, log_map, blob_map, ves_map, (z0, y0, x0) = maps
    origin = ct_img.TransformIndexToPhysicalPoint((int(x0), int(y0), int(z0)))
    os.makedirs(out_dir, exist_ok=True)
    for arr, name in ((crop, "ct_crop"), (log_map, "log"), (blob_map, "blobness"), (ves_map, "vesselness")):
        im = sitk.GetImageFromArray(np.asarray(arr, dtype=np.float32))
        im.SetSpacing(ct_img.GetSpacing()); im.SetDirection(ct_img.GetDirection()); im.SetOrigin(origin)
        sitk.WriteImage(im, os.path.join(out_dir, f"{name}.nii.gz"))


_W = {}


def _worker_init(base, c_global, raiz):
    _W["c"] = c_global
    _W["raiz"] = raiz


def _worker_calib(rec):
    """S_max de uma ROI (todas as escalas). Devolve (idx, S_max, sigma_mm)."""
    i, r = rec
    pid = str(r["patient_id"])
    ct_path = os.path.join(BASE, str(r["ct_path"]))
    if not os.path.isfile(ct_path):
        ct_path = ct_de(_W["raiz"], "lidc_idri", pid)
    ct_arr, sp_zyx, _ = load_ct(ct_path)
    z0, z1, y0, y1, x0, x1 = (int(r[k]) for k in ("bbox_z0", "bbox_z1", "bbox_y0", "bbox_y1", "bbox_x0", "bbox_x1"))
    crop = ct_arr[z0:z1, y0:y1, x0:x1]
    best, best_s = 0.0, 0.0
    for s_mm in SCALES_MM:
        v = float(np.sqrt(frobenius2(hessian_physical(crop, sp_zyx, s_mm, "cpu")).max()))
        if not np.isfinite(v):
            return i, None, f"non_finite_S:{s_mm}"
        if v > best:
            best, best_s = v, s_mm
    if len(_cache) > 4:
        _cache.clear()
    return i, best, best_s


def _worker_roi(rec):
    """Processa uma ROI num processo separado. Devolve (idx, feats|None, erro)."""
    i, r = rec
    pid = str(r["patient_id"])
    ct_path = os.path.join(BASE, str(r["ct_path"]))
    if not os.path.isfile(ct_path):
        ct_path = ct_de(_W["raiz"], "lidc_idri", pid)
    mask_path = os.path.join(BASE, str(r["mask_path"]))
    if not os.path.isfile(mask_path):
        return i, None, f"mask_missing:{mask_path}"
    ct_arr, sp_zyx, _ = load_ct(ct_path)
    labeled, binary_all, _ = load_mask(mask_path)
    bbox = {k: int(r[k]) for k in ("bbox_z0", "bbox_z1", "bbox_y0", "bbox_y1", "bbox_x0", "bbox_x1")}
    res, err = process_roi(ct_arr, labeled, binary_all, sp_zyx, int(r["component_id"]),
                           bbox, _W["c"], "cpu")
    if len(_cache) > 6:
        _cache.clear()
    if err:
        return i, None, err
    return i, res[0], None


def load_sources(args):
    rows = []
    for tag, path, sha in (("predicted", args.predicted, args.predicted_sha256),
                           ("reference", args.reference, args.reference_sha256)):
        if not os.path.isfile(path):
            fail_input(f"parquet do Script 13 ausente: {path}")
        h = sha256(path)
        if h != sha:
            fail_input(f"SOURCE_AUTHORITY: SHA de {tag} {h[:12]} != declarado {sha[:12]}")
        d = pd.read_parquet(path)
        rows.append(d)
    pred, ref = rows
    if len(pred) != 43560 or len(ref) != 28870:
        fail_input(f"contagens da fonte: predicted {len(pred)} (esp. 43560), reference {len(ref)} (esp. 28870)")
    return pred, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predicted", required=True); ap.add_argument("--predicted-sha256", required=True)
    ap.add_argument("--reference", required=True); ap.add_argument("--reference-sha256", required=True)
    ap.add_argument("--source-run", default="")
    ap.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    ap.add_argument("--c-global", type=float, default=None, help="obrigatório em produção")
    ap.add_argument("--calibrate-c", action="store_true", help="mede max S no OOF e sai")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only-index", default="", help="ENSAIO: índices separados por vírgula, sobre a ordem canónica")
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--examples-only", action="store_true",
                    help="passagem dedicada: só os N_EXAMPLES primeiros TP_pred da ordem canónica")
    ap.add_argument("--workers", type=int, default=1,
                    help="processos paralelos (só backend=cpu; resultado independente do valor — gate)")
    args = ap.parse_args()
    t0 = datetime.datetime.now().astimezone()
    backend = setup_backend(args.backend)
    log.info(f"backend={backend}")

    pred, ref = load_sources(args)
    all_df = pd.concat([pred, ref], ignore_index=True).sort_values(
        ["patient_id", "subset", "architecture", "group", "component_id"], kind="mergesort").reset_index(drop=True)
    total_expected = len(all_df)
    raiz = raiz_recortes()

    if args.calibrate_c:
        sub = all_df[all_df["subset"] == "oof"].reset_index(drop=True)
        if args.limit:
            sub = sub.iloc[: args.limit]
        if backend != "cpu":
            fail_input("calibração só com backend=cpu (backend canónico)")
        log.info(f"CALIBRAÇÃO de c_global: {len(sub)} ROI OOF, {args.workers} workers "
                 f"(sem usar rótulos TP/FP/FN, sem INTERNAL/LNDb/RIDER)")
        per_roi = np.zeros(len(sub)); sigma_at = np.zeros(len(sub))
        recs = [(i, sub.loc[i]) for i in sub.index]
        with ProcessPoolExecutor(max_workers=max(1, args.workers), initializer=_worker_init,
                                 initargs=(BASE, 0.0, raiz)) as ex:
            for n, (i, v, sg) in enumerate(ex.map(_worker_calib, recs, chunksize=8)):
                if v is None:
                    fail_input(f"calibração: {sg} na ROI {i}")
                per_roi[i] = v; sigma_at[i] = sg
                if n % 2000 == 0:
                    log.info(f"  {n}/{len(sub)} … S_max corrente {per_roi.max():.4f}")
        j = int(np.argmax(per_roi)); s_max = float(per_roi[j]); win = sub.loc[j]
        q = np.percentile(per_roi, [50, 90, 95, 99, 99.9]).tolist()
        out = {"gate": "SCRIPT14_C_CALIBRATION",
               "population": "BD1 OOF (development) only; labels/groups NOT used",
               "excluded": ["INTERNAL", "LNDb", "RIDER"],
               "n_rois": int(len(sub)), "S_definition": "sqrt(l1^2+l2^2+l3^2) = ||H||_F, physical Hessian",
               "S_max_DEV": s_max, "c_global": 0.5 * s_max,
               "S_max_roi": {"patient_id": str(win["patient_id"]), "architecture": str(win["architecture"]),
                             "subset": str(win["subset"]), "group": str(win["group"]),
                             "component_id": int(win["component_id"]), "sigma_mm": float(sigma_at[j])},
               "S_max_per_roi_percentiles": {"p50": q[0], "p90": q[1], "p95": q[2], "p99": q[3],
                                             "p99.9": q[4], "max": s_max},
               "scales_mm": list(SCALES_MM), "hu_clip": list(HU_CLIP), "backend": backend,
               "workers": args.workers, "script_sha256": sha256(os.path.abspath(__file__)),
               "sources": {"predicted": args.predicted, "predicted_sha256": args.predicted_sha256,
                           "reference": args.reference, "reference_sha256": args.reference_sha256,
                           "script13_run": args.source_run},
               "environment": {"python": os.sys.version.split()[0], "numpy": np.__version__,
                               "scipy": __import__("scipy").__version__,
                               "SimpleITK": sitk.Version_VersionString(), "pandas": pd.__version__,
                               "host": os.uname().nodename,
                               "threads_env": {k: os.environ.get(k) for k in
                                               ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}},
               "start": t0.isoformat(), "end": datetime.datetime.now().astimezone().isoformat(), "rc": 0}
        p = os.path.join(FEATURES_DIR, f"c_global_calibration{args.out_suffix}.json")
        json.dump(out, open(p, "w"), indent=2)
        log.info(f"c_global = {out['c_global']:.6f} (S_max_DEV = {s_max:.6f}, "
                 f"ROI {win['patient_id']}/{win['architecture']}/{win['group']}#{int(win['component_id'])} "
                 f"@ sigma {sigma_at[j]} mm) → {p}")
        return

    if args.c_global is None or not np.isfinite(args.c_global) or args.c_global <= 0:
        fail_input("produção exige --c-global positivo e finito (ver modo --calibrate-c)")
    if args.examples_only:
        cand = all_df[all_df["group"] == "TP_pred"].sort_values(
            ["patient_id", "architecture", "component_id"], kind="mergesort")
        sel = cand.groupby("patient_id", sort=True).head(1).head(N_EXAMPLES)
        all_df = sel.reset_index(drop=True)
        if len(all_df) != N_EXAMPLES or all_df["patient_id"].nunique() != N_EXAMPLES:
            fail_output(f"EXAMPLE_SELECTION_RULE_V2: {len(all_df)} ROI / "
                        f"{all_df['patient_id'].nunique()} pacientes distintos")
        log.info(f"EXEMPLOS (regra V2): {len(all_df)} ROI TP_pred de "
                 f"{all_df['patient_id'].nunique()} pacientes distintos")
    elif args.only_index:
        idx = [int(i) for i in args.only_index.split(",")]
        all_df = all_df.iloc[idx].reset_index(drop=True)
    elif args.limit:
        all_df = all_df.iloc[: args.limit].reset_index(drop=True)
    log.info(f"ROIs a processar: {len(all_df)} (total canónico {total_expected})")

    rows, failures = [], []
    example_count, prev_pid = 0, None

    if args.workers > 1:
        if backend != "cpu":
            fail_input("--workers > 1 só é válido com backend=cpu")
        recs = [(i, all_df.loc[i]) for i in all_df.index]
        got = {}
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init,
                                 initargs=(BASE, args.c_global, raiz)) as ex:
            for n, (i, feats, err) in enumerate(ex.map(_worker_roi, recs, chunksize=8)):
                if n % 500 == 0:
                    log.info(f"  {n}/{len(recs)} …")
                if err:
                    r = all_df.loc[i]
                    failures.append({"patient_id": str(r["patient_id"]), "architecture": r["architecture"],
                                     "subset": r["subset"], "group": r["group"],
                                     "component_id": int(r["component_id"]), "reason": err})
                    fail_input(f"ROI falhada ({err}) no índice {i}")
                got[i] = feats
        for i in all_df.index:
            r = all_df.loc[i]
            rows.append({"patient_id": str(r["patient_id"]), "architecture": r["architecture"],
                         "subset": r["subset"], "group": r["group"], "component_id": int(r["component_id"]),
                         "ct_path": r["ct_path"], "mask_path": r["mask_path"], "source": r["source"],
                         "volume_mm3": r["volume_mm3"], "diam_mm": r["diam_mm"],
                         "voxel_count": r["voxel_count"], "backend": backend, **got[i]})
        if not args.limit and not args.only_index:
            log.info("exemplos: gerados em passagem dedicada (ver --only-index)")
        df = pd.DataFrame(rows)
        return _finalize(args, df, pred, ref, total_expected, failures, backend, t0, example_count)

    for i, row in all_df.iterrows():
        if i % 100 == 0:
            log.info(f"  {i}/{len(all_df)} …")
        pid = str(row["patient_id"])
        if pid != prev_pid:
            _cache.clear(); prev_pid = pid
        ct_path = os.path.join(BASE, str(row["ct_path"]))
        if not os.path.isfile(ct_path):
            ct_path = ct_de(raiz, "lidc_idri", pid)
        mask_path = os.path.join(BASE, str(row["mask_path"]))
        if not os.path.isfile(mask_path):
            fail_input(f"máscara ausente: {mask_path}")
        ct_arr, sp_zyx, ct_img = load_ct(ct_path)
        labeled, binary_all, _ = load_mask(mask_path)
        bbox = {k: int(row[k]) for k in ("bbox_z0", "bbox_z1", "bbox_y0", "bbox_y1", "bbox_x0", "bbox_x1")}
        res, err = process_roi(ct_arr, labeled, binary_all, sp_zyx, int(row["component_id"]),
                               bbox, args.c_global, backend)
        if err:
            failures.append({"patient_id": pid, "architecture": row["architecture"], "subset": row["subset"],
                             "group": row["group"], "component_id": int(row["component_id"]), "reason": err})
            fail_input(f"ROI falhada ({err}): {pid}/{row['architecture']}/{row['group']}#{row['component_id']}")
        feats, maps = res
        rows.append({"patient_id": pid, "architecture": row["architecture"], "subset": row["subset"],
                     "group": row["group"], "component_id": int(row["component_id"]),
                     "ct_path": row["ct_path"], "mask_path": row["mask_path"], "source": row["source"],
                     "volume_mm3": row["volume_mm3"], "diam_mm": row["diam_mm"],
                     "voxel_count": row["voxel_count"], "backend": backend, **feats})
        if example_count < N_EXAMPLES and row["group"] == "TP_pred" and (args.examples_only or (not args.limit and not args.only_index)):
            save_example_maps(maps, ct_img, os.path.join(
                FIGDATA_DIR, f"{pid}_{row['architecture']}_{row['subset']}_{row['group']}_c{int(row['component_id'])}"))
            example_count += 1

    df = pd.DataFrame(rows)
    return _finalize(args, df, pred, ref, total_expected, failures, backend, t0, example_count)


def _finalize(args, df, pred, ref, total_expected, failures, backend, t0, example_count):
    key = ["subset", "patient_id", "architecture", "group", "component_id"]
    if df.duplicated(key).any():
        fail_output("linha de saída duplicada")
    num = df.select_dtypes(include=[np.number])
    if not np.isfinite(num.to_numpy()).all():
        fail_output("valor não finito (NaN/Inf) nas características")
    for c in ("log_max", "blob_max", "vessel_max", "log_mean", "blob_mean", "vessel_mean"):
        if (df[c] < 0).any():
            fail_output(f"{c} negativo")
    for c in ("log_scale_mm", "blob_scale_mm", "vessel_scale_mm"):
        if not df[c].isin(SCALES_MM).all():
            fail_output(f"{c} fora do conjunto congelado")
    if (df["mu_object_hu"] < HU_CLIP[0] - 1e-3).any() or (df["mu_object_hu"] > HU_CLIP[1] + 1e-3).any() \
            or (df["mu_ring_hu"] < HU_CLIP[0] - 1e-3).any() or (df["mu_ring_hu"] > HU_CLIP[1] + 1e-3).any():
        fail_output("HU médio fora do clip")
    if (df["log_max"] < df["log_mean"] - 1e-6).any() or (df["blob_max"] < df["blob_mean"] - 1e-6).any() \
            or (df["vessel_max"] < df["vessel_mean"] - 1e-6).any():
        fail_output("max < mean")
    src = pd.concat([pred, ref], ignore_index=True)
    if not args.limit and not args.only_index and not args.examples_only:
        if len(df) != total_expected:
            fail_output(f"linhas {len(df)} != ROI esperadas {total_expected}")
        m = df.merge(src[key + ["volume_mm3", "diam_mm"]], on=key, suffixes=("", "_s13"))
        if len(m) != total_expected:
            fail_output("identidade ROI não casa 1:1 com o Script 13")
        if (m["volume_mm3"] != m["volume_mm3_s13"]).any() or (m["diam_mm"] != m["diam_mm_s13"]).any():
            fail_output("morfometria alterada face ao Script 13")

    suf = args.out_suffix
    out_p = os.path.join(FEATURES_DIR, f"filter_features_raw{suf}.parquet")
    df.to_parquet(out_p, index=False)
    json.dump(failures, open(os.path.join(FEATURES_DIR, f"roi_failures{suf}.json"), "w"), indent=2)
    manifest = {
        "script": "14_compute_filter_maps_gpu.py", "script_sha256": sha256(os.path.abspath(__file__)),
        "run_at": t0.isoformat(), "end_at": datetime.datetime.now().astimezone().isoformat(),
        "backend": backend, "PULMONARY_BASE": BASE,
        "sources": {"predicted": args.predicted, "predicted_sha256": args.predicted_sha256,
                    "reference": args.reference, "reference_sha256": args.reference_sha256,
                    "script13_run": args.source_run},
        "definitions": {"hessian": "physically normalized: H_ij = sigma_i*sigma_j*d2I/dn_i dn_j",
                        "log": "trace(H_sigma), |.|, argmax over scales",
                        "vesselness": "Frangi 1998 canonical (bright tubular)",
                        "blobness": "HESSIAN_BLOBNESS_3D — project-defined, NOT Frangi blobness",
                        "alpha": ALPHA, "beta": BETA, "c_global": args.c_global,
                        "scales_mm": list(SCALES_MM), "hu_clip": list(HU_CLIP),
                        "ring_mm": RING_MM, "ring": "physical EDT ring, excludes other foreground of same source",
                        "connectivity": "6 (generate_binary_structure(3,1))",
                        "dominant_scale": "argmax (never median)"},
        "counts": {"rows": len(df), "expected": total_expected, "failures": len(failures),
                   "examples": example_count},
        "outputs_sha256": {os.path.basename(out_p): sha256(out_p)},
    }
    json.dump(manifest, open(os.path.join(FEATURES_DIR, f"filter_features_manifest{suf}.json"), "w"), indent=2)
    log.info(f"Salvo: {out_p} ({len(df)} linhas); falhas {len(failures)}; exemplos {example_count}")
    log.info("Script 14 concluído.")


if __name__ == "__main__":
    try:
        main()
    except FalhaEntrada as e:
        log.error(str(e))
        raise SystemExit("FAIL_INPUT")
