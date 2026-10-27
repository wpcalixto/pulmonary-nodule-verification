#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    15_extract_filter_features.py

File type:
    Script

Purpose:
    Extracts per-component morphological filter features and the distance to the pleura for BD1, LNDb and RIDER.

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
    - {args.dataset}_filter_features_manifest.json

Dependencies:
    - SimpleITK
    - numpy
    - pandas
    - scipy

Execution:
    PULMONARY_BASE=<project_dir> python scripts/15_extract_filter_features.py

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

FIGDATA_DIR = os.path.join(BASE, "figdata")
LOGS_DIR    = os.path.join(BASE, "logs")
STRUCT_6    = nd.generate_binary_structure(3, 1)
BANCO       = {"bd1": "lidc_idri", "lndb": "lndb", "rider": "rider"}

for _d in (FIGDATA_DIR, LOGS_DIR):
    os.makedirs(_d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(os.path.join(LOGS_DIR, "extract_filter_features.log")),
              logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def fail_input(msg):
    log.error(msg); raise SystemExit("FAIL_INPUT")


def fail_output(msg):
    log.error(msg); raise SystemExit("FAIL_OUTPUT")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_geometry(a, b, rotulo):
    ok = (a.GetSize() == b.GetSize()
          and np.allclose(a.GetSpacing(), b.GetSpacing(), rtol=0, atol=1e-6)
          and np.allclose(a.GetOrigin(), b.GetOrigin(), rtol=0, atol=1e-6)
          and np.allclose(a.GetDirection(), b.GetDirection(), rtol=0, atol=1e-6))
    if not ok:
        fail_input(f"GEOMETRY_GATE: grelhas incompatíveis — {rotulo}")


def pleura_edt(mask_path, ct_img, pid):
    """(distância física à superfície pleural, máscara booleana, spacing zyx)."""
    img = sitk.ReadImage(mask_path)
    check_geometry(ct_img, img, f"CT vs lung_mask {pid}")
    arr = sitk.GetArrayFromImage(img)
    if not np.isin(arr, (0, 1)).all():
        fail_input(f"máscara pulmonar não binária: {mask_path}")
    lung = arr.astype(bool)
    if not lung.any():
        fail_input(f"máscara pulmonar vazia: {mask_path}")
    surface = lung & ~nd.binary_erosion(lung, structure=STRUCT_6, border_value=0)
    if not surface.any():
        fail_output(f"superfície pleural vazia: {mask_path}")
    sp = img.GetSpacing(); sp_zyx = (float(sp[2]), float(sp[1]), float(sp[0]))
    dist = nd.distance_transform_edt(~surface, sampling=sp_zyx)
    if not np.isfinite(dist).all():
        fail_output(f"EDT não finita: {mask_path}")
    return dist, lung, sp_zyx


def pleural_features(comp, dist, lung, sp_zyx, rotulo):
    idx = np.argwhere(comp)
    if idx.size == 0:
        fail_input(f"componente vazio: {rotulo}")
    d_comp = dist[comp]
    pmin = float(d_comp.min())
    centroid = idx.mean(axis=0)
    dd = ((idx - centroid) * np.asarray(sp_zyx)) ** 2
    rep = idx[int(np.argmin(dd.sum(axis=1)))]
    pcen = float(dist[tuple(rep)])
    inside = int(lung[comp].sum()); total = int(len(idx))
    surf_touch = bool((d_comp == 0).any())
    for v, n in ((pmin, "pleura_min_dist_mm"), (pcen, "pleura_centroid_dist_mm")):
        if not np.isfinite(v) or v < 0:
            fail_output(f"{n} inválido ({v}) em {rotulo}")
    return {
        "pleura_min_dist_mm": round(pmin, 4),
        "pleura_centroid_dist_mm": round(pcen, 4),
        "component_voxels_total": total,
        "component_voxels_inside_lung": inside,
        "component_voxels_outside_lung": total - inside,
        "component_fraction_inside_lung": round(inside / total, 6),
        "component_intersects_lung_surface": int(surf_touch),
        "component_entirely_outside_lung": int(inside == 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="filter_features_raw.parquet canónico (Script 14)")
    ap.add_argument("--features-sha256", required=True)
    ap.add_argument("--source-run", default="")
    ap.add_argument("--dataset", default="bd1", choices=("bd1", "lndb", "rider"))
    ap.add_argument("--expected-rows", type=int, default=72430)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default=None, help="ENSAIO: escrever fora de figdata/")
    args = ap.parse_args()
    t0 = datetime.datetime.now().astimezone()
    out_dir = args.out_dir or FIGDATA_DIR
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(args.features):
        fail_input(f"parquet de características ausente: {args.features}")
    h = sha256(args.features)
    if h != args.features_sha256:
        fail_input(f"SOURCE_AUTHORITY: SHA {h[:12]} != declarado {args.features_sha256[:12]}")
    df = pd.read_parquet(args.features)
    if not args.limit and len(df) != args.expected_rows:
        fail_input(f"ROW_COUNT_GATE: {len(df)} linhas != {args.expected_rows}")
    key = ["subset", "patient_id", "architecture", "group", "component_id"]
    for c in key + ["mask_path", "source"]:
        if c not in df.columns:
            fail_input(f"coluna obrigatória ausente na fonte: {c}")
    if df.duplicated(key).any():
        fail_input("KEY_IDENTITY_GATE: chave duplicada na fonte")
    upstream_cols = list(df.columns)
    if args.limit:
        df = df.iloc[: args.limit].copy()
    log.info(f"{args.dataset}: {len(df)} objectos; fonte {os.path.basename(args.features)} ({h[:12]})")

    raiz = raiz_recortes()
    banco = BANCO[args.dataset]
    rows, authority = [], {}
    cache = {}
    for i, r in df.reset_index(drop=True).iterrows():
        if i % 1000 == 0:
            log.info(f"  {i}/{len(df)} …")
        pid = str(r["patient_id"])
        if pid not in cache:
            cache.clear()
            ct_img = sitk.ReadImage(ct_de(raiz, banco, pid))
            mp = mascara_pulmonar_de(raiz, banco, pid, recortada=True)
            dist, lung, sp_zyx = pleura_edt(mp, ct_img, pid)
            cache[pid] = (dist, lung, sp_zyx)
            if pid not in authority:
                im = sitk.ReadImage(mp)
                authority[pid] = {"patient_id": pid, "canonical_lung_mask_path": os.path.relpath(mp, BASE),
                                  "sha256": sha256(mp), "shape": str(im.GetSize()),
                                  "spacing": str(tuple(round(x, 6) for x in im.GetSpacing())),
                                  "origin": str(tuple(round(x, 4) for x in im.GetOrigin())),
                                  "direction": str(tuple(round(x, 6) for x in im.GetDirection()))}
        dist, lung, sp_zyx = cache[pid]
        mpath = os.path.join(BASE, str(r["mask_path"]))
        if not os.path.isfile(mpath):
            fail_input(f"máscara do objecto ausente: {mpath}")
        mk = ("m", mpath)
        if mk not in cache:
            arr = sitk.GetArrayFromImage(sitk.ReadImage(mpath))
            lab, _ = nd.label(arr > 0, structure=STRUCT_6)
            cache[mk] = lab
        comp = cache[mk] == int(r["component_id"])
        rot = f"{pid}/{r['architecture']}/{r['group']}#{int(r['component_id'])}"
        rows.append({**r.to_dict(), **pleural_features(comp, dist, lung, sp_zyx, rot)})

    out = pd.DataFrame(rows)
    missing = [c for c in upstream_cols if c not in out.columns]
    if missing:
        fail_output(f"SCHEMA_PRESERVATION_GATE: colunas upstream perdidas: {missing}")
    for c in upstream_cols:
        if not out[c].equals(df.reset_index(drop=True)[c]):
            fail_output(f"FEATURE_PRESERVATION_GATE: coluna alterada: {c}")
    if out.duplicated(key).any():
        fail_output("chave duplicada na saída")
    num = out.select_dtypes(include=[np.number])
    if not np.isfinite(num.to_numpy()).all():
        fail_output("valor não finito na saída")
    if not args.limit and len(out) != args.expected_rows:
        fail_output(f"linhas {len(out)} != {args.expected_rows}")

    csv_path = os.path.join(out_dir, f"{args.dataset}_filter_features.csv")
    out.to_csv(csv_path, index=False)
    auth_path = os.path.join(out_dir, f"{args.dataset}_lung_mask_authority.csv")
    pd.DataFrame(list(authority.values())).to_csv(auth_path, index=False)
    man = {
        "script": "15_extract_filter_features.py", "script_sha256": sha256(os.path.abspath(__file__)),
        "dataset": args.dataset, "run_at": t0.isoformat(),
        "end_at": datetime.datetime.now().astimezone().isoformat(),
        "source": {"features": args.features, "features_sha256": h, "script14_run": args.source_run},
        "lung_mask_authority": "caminhos_canonicos.mascara_pulmonar_de(..., recortada=True) → lung_mask_crop.nii.gz",
        "pleural_definitions": {
            "surface": "lung & ~binary_erosion(lung, generate_binary_structure(3,1), border_value=0)",
            "distance": "distance_transform_edt(~surface, sampling=spacing_zyx)",
            "pleura_min_dist_mm": "PRIMARY — min over component voxels",
            "pleura_centroid_dist_mm": "SECONDARY — at component voxel nearest to continuous centroid"},
        "counts": {"rows": len(out), "expected": args.expected_rows, "patients": len(authority),
                   "entirely_inside": int((out.component_voxels_outside_lung == 0).sum()),
                   "crossing_surface": int(((out.component_voxels_inside_lung > 0) &
                                            (out.component_voxels_outside_lung > 0)).sum()),
                   "entirely_outside": int(out.component_entirely_outside_lung.sum()),
                   "touching_surface": int(out.component_intersects_lung_surface.sum())},
        "gates": {"SOURCE_AUTHORITY": "PASS", "ROW_COUNT": "PASS", "KEY_IDENTITY": "PASS",
                  "LUNG_MASK_AUTHORITY": "PASS", "GEOMETRY": "PASS",
                  "NUMERIC_DOMAIN": "PASS", "FEATURE_PRESERVATION": "PASS",
                  "SCHEMA_PRESERVATION": "PASS"},
        "outputs_sha256": {os.path.basename(csv_path): sha256(csv_path),
                           os.path.basename(auth_path): sha256(auth_path)},
    }
    json.dump(man, open(os.path.join(out_dir, f"{args.dataset}_filter_features_manifest.json"), "w"), indent=2)
    log.info(f"Salvo: {csv_path} ({len(out)} linhas, {len(out.columns)} colunas)")
    log.info(f"  pleura_min_dist_mm mediana {out.pleura_min_dist_mm.median():.2f} mm; "
             f"objectos a tocar a superfície {man['counts']['touching_surface']}; "
             f"inteiramente fora do pulmão {man['counts']['entirely_outside']}")
    log.info("Script 15 concluído.")


if __name__ == "__main__":
    try:
        main()
    except FalhaEntrada as e:
        log.error(str(e)); raise SystemExit("FAIL_INPUT")
