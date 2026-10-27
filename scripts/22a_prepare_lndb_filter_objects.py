#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    22a_prepare_lndb_filter_objects.py

File type:
    Script

Purpose:
    Prepares LNDb predicted and reference objects for the external morphological-filter analysis.

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
    - failures.json
    - gates.json
    - manifest.json

Dependencies:
    - SimpleITK
    - numpy
    - pandas
    - scipy

Execution:
    PULMONARY_BASE=<project_dir> python scripts/22a_prepare_lndb_filter_objects.py

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

ARQUITECTURAS = ["3D_F", "3D_FREM"]
ESPERADO_POR_GRUPO = {"TP_pred": 594, "TP_gt": 594, "FP_pred": 1029, "FN_gt": 1106}
ESPERADO_TOTAL = sum(ESPERADO_POR_GRUPO.values())
ESPERADO_LINHAS_S22 = 2729
COMBINACOES_VALIDAS = {("pred", "TP_pred"), ("pred", "FP_pred"),
                       ("gt", "TP_gt"), ("gt", "FN_gt")}
GRUPOS = {"pred": ("TP_pred", "FP_pred"), "gt": ("TP_gt", "FN_gt")}
CHAVE = ["source", "scan_id", "architecture", "group", "object_id"]
STRUCT_6 = nd.generate_binary_structure(3, 1)

os.makedirs(LOGS_DIR, exist_ok=True)
log = logging.getLogger("s22a")
log.setLevel(logging.INFO)
if not log.handlers:
    _f = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for _h in (logging.StreamHandler(),
               logging.FileHandler(os.path.join(LOGS_DIR, "prepare_lndb_filter_objects.log"))):
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


S11, SHA11 = _carregar("s11_canon_22a", "11_match_and_score_bd1.py")
S13, SHA13 = _carregar("s13_canon_22a", "13_prepare_filter_objects.py")
from caminhos_canonicos import raiz_recortes, ct_de, mascara_pulmonar_de

if S13.MARGIN_MM != 15.0:
    fail_input(f"MARGIN_MM do Script 13 diverge: {S13.MARGIN_MM}")
if not np.array_equal(S13.STRUCT_6, STRUCT_6):
    fail_input("STRUCT_6 do Script 13 diverge da 6-conectividade explícita")


def mesma_geometria(a, b):
    return (a.GetSize() == b.GetSize()
            and np.allclose(a.GetSpacing(), b.GetSpacing(), rtol=0, atol=1e-6)
            and np.allclose(a.GetOrigin(), b.GetOrigin(), rtol=0, atol=1e-6)
            and np.allclose(a.GetDirection(), b.GetDirection(), rtol=0, atol=1e-6))


def deslocamento_do_recorte(recorte_img, completo_img, rotulo):
    """Índice (z0, y0, x0) do recorte dentro da grelha completa. Exige
    alinhamento a voxel — não há reamostragem em lado nenhum deste script."""
    if not np.allclose(recorte_img.GetSpacing(), completo_img.GetSpacing(), rtol=0, atol=1e-6):
        fail_input(f"espaçamento do recorte != completo em {rotulo}")
    if not np.allclose(recorte_img.GetDirection(), completo_img.GetDirection(), rtol=0, atol=1e-6):
        fail_input(f"direcção do recorte != completo em {rotulo}")
    ci = np.array(completo_img.TransformPhysicalPointToContinuousIndex(recorte_img.GetOrigin()))
    if np.abs(ci - np.round(ci)).max() > 1e-3:
        fail_input(f"deslocamento de índice não inteiro em {rotulo}: {ci}")
    o = np.round(ci).astype(int)
    sz = np.array(recorte_img.GetSize()); rs = np.array(completo_img.GetSize())
    if not ((o >= 0).all() and (o + sz <= rs).all()):
        fail_input(f"recorte fora da grelha completa em {rotulo}")
    return int(o[2]), int(o[1]), int(o[0])


def le_binaria(caminho, rotulo):
    if not os.path.isfile(caminho):
        fail_input(f"{rotulo} ausente: {caminho}")
    img = sitk.ReadImage(caminho)
    arr = sitk.GetArrayFromImage(img)
    if not np.isin(arr, (0, 1)).all():
        fail_input(f"{rotulo} não binária: {caminho}")
    return img, (arr > 0)


def objecto(rotulado, cid, sp_zyx, forma, rotulo, fonte_fora_do_recorte=0):
    """Registo de um objecto, com a ROI do Script 13 (função canónica)."""
    r = S13.object_record(rotulado, cid, sp_zyx, forma, rotulo)
    coords = np.argwhere(rotulado == cid)
    n_vox = int(len(coords))
    vol = float(n_vox) * sp_zyx[0] * sp_zyx[1] * sp_zyx[2]
    diam = S11.diam_mm((rotulado == cid).astype(np.uint8),
                       (sp_zyx[2], sp_zyx[1], sp_zyx[0]))
    if n_vox <= 0 or vol <= 0 or diam <= 0:
        fail_output(f"morfometria não positiva em {rotulo}")
    r.update({"voxel_count": n_vox, "volume_mm3": vol, "diam_mm": diam,
              "spacing_zyx": json.dumps(list(sp_zyx)),
              "source_voxels_outside_crop": fonte_fora_do_recorte})
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s22-run", required=True)
    ap.add_argument("--s22-manifest-sha256", required=True)
    ap.add_argument("--limit-scans", type=int, default=0)
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()
    t0 = datetime.datetime.now().astimezone()

    man_p = os.path.join(args.s22_run, "manifest.json")
    if not os.path.isfile(man_p):
        fail_input(f"run do Script 22 sem manifest.json: {args.s22_run}")
    h22 = sha256(man_p)
    if h22 != args.s22_manifest_sha256:
        fail_input(f"INPUT_AUTHORITY: SHA do manifesto do 22 {h22[:12]} "
                   f"!= {args.s22_manifest_sha256[:12]}")
    man22 = json.load(open(man_p))
    if man22["gates"].get("TP_IOU_THRESHOLD_GATE") is None:
        fail_input("manifesto do Script 22 sem os gates esperados")
    dir_run = os.path.basename(os.path.normpath(args.s22_run))
    if man22["run_id"] != dir_run:
        fail_input(f"run_id do manifesto ({man22['run_id']}) != pasta ({dir_run})")
    obj_p = os.path.join(args.s22_run, "lndb_object_matches.csv")
    if not os.path.isfile(obj_p):
        fail_input(f"tabela de objectos ausente: {obj_p}")
    oh = pd.read_csv(os.path.join(args.s22_run, "output_hashes.csv"))
    linha = oh[oh["path"] == "lndb_object_matches.csv"]
    hobj = sha256(obj_p)
    if len(linha) != 1 or linha["sha256"].iloc[0] != hobj:
        fail_input("SHA da tabela de objectos não confere com o output_hashes.csv do 22")
    obj = pd.read_csv(obj_p)
    if len(obj) != ESPERADO_LINHAS_S22:
        fail_input(f"INPUT_TABLE_COUNT_GATE: {len(obj)} linhas != {ESPERADO_LINHAS_S22} "
                   f"do Script 22 canónico")
    if set(obj["architecture"]) != set(ARQUITECTURAS):
        fail_input(f"arquitecturas inesperadas: {set(obj['architecture'])}")
    if not set(obj["status"]) <= {"TP", "FN", "FP"}:
        fail_input(f"estados inesperados: {set(obj['status'])}")
    log.info(f"Script 22 canónico {man22['run_id']}: {len(obj)} objectos "
             f"({obj.status.value_counts().to_dict()})")

    scans = sorted(obj["scan_id"].unique())
    if args.limit_scans:
        scans = scans[: args.limit_scans]
        obj = obj[obj["scan_id"].isin(scans)]
    n_tp = int((obj.status == "TP").sum())
    esperado = len(obj) + n_tp
    log.info(f"{len(scans)} varrimentos · objectos esperados {esperado} "
             f"= {len(obj)} linhas + {n_tp} TP contados também como referência")

    raiz = raiz_recortes()
    run_id = f"s22a_{t0:%Y%m%dT%H%M%S}_{os.getpid()}"
    out_root = os.path.join(args.out_root or os.path.join(FEATURES_DIR, "lndb_filter_objects"), run_id)
    os.makedirs(out_root, exist_ok=True)

    def confere_com_s22(rec, diam_s22, rot):
        """OBJECT_CONSISTENCY_GATE — o componente recuperado tem de ter o mesmo
        diâmetro que o Script 22 mediu para esse mesmo `object_id`. É isto que
        apanha uma renumeração silenciosa dos rótulos."""
        if not np.isfinite(diam_s22):
            fail_output(f"diâmetro do Script 22 não finito em {rot}")
        if abs(float(rec["diam_mm"]) - float(diam_s22)) > 1e-6:
            fail_output(f"OBJECT_CONSISTENCY: {rot} tem diâmetro {rec['diam_mm']:.6f} mm "
                        f"mas o Script 22 registou {diam_s22:.6f} mm — o `object_id` "
                        f"não aponta para o mesmo componente")

    def valida_combo(src, grp, rot):
        if (src, grp) not in COMBINACOES_VALIDAS:
            fail_output(f"SOURCE_GROUP ilegal ({src}, {grp}) em {rot}")

    linhas, entradas, falhas = [], [], []
    for i, sid in enumerate(scans):
        ctp = ct_de(raiz, "lndb", sid)
        lmp = mascara_pulmonar_de(raiz, "lndb", sid, recortada=True)
        ct_img = sitk.ReadImage(ctp)
        lm_img, lung = le_binaria(lmp, f"máscara pulmonar {sid}")
        if not mesma_geometria(lm_img, ct_img):
            fail_input(f"máscara pulmonar com geometria != CT recortada em {sid}")
        sp = ct_img.GetSpacing()
        if not all(np.isfinite(sp)) or min(sp) <= 0:
            fail_input(f"espaçamento inválido {sp} em {ctp}")
        sp_zyx = (float(sp[2]), float(sp[1]), float(sp[0]))
        forma = sitk.GetArrayFromImage(ct_img).shape

        gtp = os.path.join(CONSENSUS_DIR, sid, "consensus.nii.gz")
        gt_img, gt_bin = le_binaria(gtp, f"referência {sid}")
        gt_lab_full, n_gt = nd.label(gt_bin, structure=STRUCT_6)
        z0, y0, x0 = deslocamento_do_recorte(ct_img, gt_img, f"recorte de {sid}")
        Z, Y, X = forma
        gt_lab = gt_lab_full[z0:z0 + Z, y0:y0 + Y, x0:x0 + X].astype(np.int32)
        dentro = np.zeros(gt_lab_full.shape, bool)
        dentro[z0:z0 + Z, y0:y0 + Y, x0:x0 + X] = True
        gt_fora = int((gt_bin & ~dentro).sum())
        if gt_fora != 0:
            fail_input(f"REFERENCE_CROP_LOSS: {gt_fora} voxels de referência de {sid} "
                       f"fora do recorte pulmonar")
        perdidos = [c for c in range(1, n_gt + 1) if not (gt_lab == c).any()]
        if perdidos:
            fail_input(f"REFERENCE_CROP_LOSS: componentes {perdidos} de {sid} "
                       f"desaparecem no recorte")
        for c in range(1, n_gt + 1):
            if int((gt_lab_full == c).sum()) != int((gt_lab == c).sum()):
                fail_input(f"REFERENCE_CROP_LOSS: componente {c} de {sid} perde voxels "
                           f"no recorte")
        amostra = np.argwhere(gt_lab > 0)
        if len(amostra):
            zz, yy, xx = amostra[len(amostra) // 2]
            p_rec = np.array(ct_img.TransformIndexToPhysicalPoint((int(xx), int(yy), int(zz))))
            p_com = np.array(gt_img.TransformIndexToPhysicalPoint(
                (int(xx) + x0, int(yy) + y0, int(zz) + z0)))
            if np.abs(p_rec - p_com).max() > 1e-4:
                fail_output(f"PHYSICAL_COORDINATE_PRESERVATION falhou em {sid}: "
                            f"{p_rec} != {p_com}")

        entradas.append({"scan_id": sid, "ct_path": os.path.relpath(ctp, BASE),
                         "ct_sha256": sha256(ctp),
                         "lung_mask_path": os.path.relpath(lmp, BASE),
                         "lung_mask_sha256": sha256(lmp),
                         "reference_path": os.path.relpath(gtp, BASE),
                         "reference_sha256": sha256(gtp),
                         "crop_offset_zyx": json.dumps([z0, y0, x0]),
                         "reference_voxels_outside_crop": gt_fora})

        for arch in ARQUITECTURAS:
            sub = obj[(obj.scan_id == sid) & (obj.architecture == arch)]
            predp = os.path.join(PRED_DIR, arch, f"{sid}.nii.gz")
            pr_img, pr_bin = le_binaria(predp, f"predição {arch}/{sid}")
            if not mesma_geometria(pr_img, ct_img):
                fail_input(f"predição com geometria != CT recortada em {arch}/{sid}")
            pr_lab, n_pred = nd.label(pr_bin, structure=STRUCT_6)
            pr_lab = pr_lab.astype(np.int32)
            entradas[-1][f"prediction_{arch}_path"] = os.path.relpath(predp, BASE)
            entradas[-1][f"prediction_{arch}_sha256"] = sha256(predp)

            ids_pred = {int(v) for v in sub.loc[sub.status.isin(["TP", "FP"]),
                                                "pred_component_id"]}
            ids_gt = {int(v) for v in sub.loc[sub.status.isin(["TP", "FN"]),
                                              "gt_component_id"]}
            if ids_pred and max(ids_pred) > n_pred:
                fail_output(f"COMPONENT_IDENTITY: id predito fora do intervalo em {arch}/{sid}")
            if ids_gt and max(ids_gt) > n_gt:
                fail_output(f"COMPONENT_IDENTITY: id de referência fora do intervalo em {arch}/{sid}")
            if len(ids_pred) != int(sub.status.isin(["TP", "FP"]).sum()):
                fail_output(f"ids preditos repetidos em {arch}/{sid}")
            if len(ids_gt) != int(sub.status.isin(["TP", "FN"]).sum()):
                fail_output(f"ids de referência repetidos em {arch}/{sid}")

            for _, r in sub.iterrows():
                if r.status in ("TP", "FP"):
                    cid = int(r.pred_component_id)
                    grupo = "TP_pred" if r.status == "TP" else "FP_pred"
                    rot = f"pred/{sid}/{arch}/{grupo}/{cid}"
                    valida_combo("pred", grupo, rot)
                    rec = objecto(pr_lab, cid, sp_zyx, forma, rot)
                    confere_com_s22(rec, r.diam_mm_pred, rot)
                    linhas.append({"source": "pred", "scan_id": sid, "architecture": arch,
                                   "group": grupo, "object_id": cid,
                                   "match_status": r.status, "iou": float(r.iou),
                                   "dice": float(r.dice) if pd.notna(r.dice) else float("nan"),
                                   "paired_object_id": (int(r.gt_component_id)
                                                        if r.status == "TP" else pd.NA),
                                   **rec})
                if r.status in ("TP", "FN"):
                    cid = int(r.gt_component_id)
                    grupo = "TP_gt" if r.status == "TP" else "FN_gt"
                    rot = f"gt/{sid}/{arch}/{grupo}/{cid}"
                    if not (gt_lab == cid).any():
                        falhas.append({"source": "gt", "scan_id": sid, "architecture": arch,
                                       "group": grupo, "object_id": cid,
                                       "reason": "REFERENCE_COMPONENT_OUTSIDE_LUNG_CROP"})
                        continue
                    valida_combo("gt", grupo, rot)
                    confere_depois = float(r.diam_mm_gt)
                    rec = objecto(gt_lab, cid, sp_zyx, forma, rot,
                                  fonte_fora_do_recorte=int(((gt_lab_full == cid) & ~dentro).sum()))
                    linhas.append({"source": "gt", "scan_id": sid, "architecture": arch,
                                   "group": grupo, "object_id": cid,
                                   "match_status": r.status, "iou": float(r.iou),
                                   "dice": float(r.dice) if pd.notna(r.dice) else float("nan"),
                                   "paired_object_id": (int(r.pred_component_id)
                                                        if r.status == "TP" else pd.NA),
                                   **rec})
                    confere_com_s22(rec, confere_depois, rot)
        if (i + 1) % 25 == 0 or i + 1 == len(scans):
            log.info(f"  [{i+1}/{len(scans)}] {sid}")

    df = pd.DataFrame(linhas)
    df["paired_object_id"] = df["paired_object_id"].astype("Int64")
    din = pd.DataFrame(entradas)

    g = {"INPUT_AUTHORITY_GATE": "PASS",
         "SCRIPT22_RUN": man22["run_id"],
         "CONNECTIVITY_6_GATE": "generate_binary_structure(3,1), explícita",
         "NO_RELABEL_AFTER_CROP": ("a imagem de rótulos da referência é recortada por "
                                   "índice; nunca se rotula de novo"),
         "ROI_RULE": f"caixa do componente + {S13.MARGIN_MM} mm por eixo (função do Script 13)",
         "DIAMETER": "MAXIMUM_AXIAL_DIAMETER_MM (função do Script 11)"}
    if len(df) + len(falhas) != esperado:
        fail_output(f"BALANCE_GATE: {len(df)} sucessos + {len(falhas)} falhas != {esperado}")
    g["BALANCE_GATE"] = f"{len(df)} + {len(falhas)} = {esperado}"
    if df.duplicated(CHAVE).any():
        fail_output("KEY_GATE: chave repetida")
    g["KEY_GATE"] = f"{len(df)} chaves únicas em {CHAVE}"
    for grupo, n in df.group.value_counts().items():
        g.setdefault("GROUP_COUNTS", {})[grupo] = int(n)
    esp_tp = int((obj.status == "TP").sum())
    if g["GROUP_COUNTS"].get("TP_pred", 0) != esp_tp or g["GROUP_COUNTS"].get("TP_gt", 0) != esp_tp:
        fail_output("os TP têm de aparecer uma vez como predito e uma vez como referência")
    g["TP_DUAL_REPRESENTATION"] = f"{esp_tp} preditos e {esp_tp} de referência"
    combos = set(zip(df.source, df.group))
    if not combos <= COMBINACOES_VALIDAS:
        fail_output(f"SOURCE_GROUP_BIJECTION: combinações ilegais {combos - COMBINACOES_VALIDAS}")
    esperado_grupo = {"TP_pred": esp_tp, "TP_gt": esp_tp,
                      "FP_pred": int((obj.status == "FP").sum()),
                      "FN_gt": int((obj.status == "FN").sum())}
    obtido = {k: int(v) for k, v in df.group.value_counts().items()}
    if obtido != esperado_grupo:
        fail_output(f"SOURCE_GROUP_BIJECTION: {obtido} != {esperado_grupo}")
    g["SOURCE_GROUP_BIJECTION_GATE"] = "PASS"
    if not args.limit_scans:
        if len(df) != ESPERADO_TOTAL or obtido != ESPERADO_POR_GRUPO:
            fail_output(f"CANONICAL_COUNT_GATE: {len(df)} / {obtido} != "
                        f"{ESPERADO_TOTAL} / {ESPERADO_POR_GRUPO}")
        g["CANONICAL_COUNT_GATE"] = f"{ESPERADO_TOTAL} linhas, {ESPERADO_POR_GRUPO}"
    g["REFERENCE_CROP_LOSS_GATE"] = "0 voxels e 0 componentes perdidos em todos os varrimentos"
    g["PHYSICAL_COORDINATE_PRESERVATION_GATE"] = "PASS"
    for c in ("voxel_count", "volume_mm3", "diam_mm"):
        if (df[c] <= 0).any() or not np.isfinite(df[c].to_numpy()).all():
            fail_output(f"{c} não positivo ou não finito")
    g["POSITIVE_MORPHOMETRY_GATE"] = "PASS"
    caixas = ["bbox_z0", "bbox_z1", "bbox_y0", "bbox_y1", "bbox_x0", "bbox_x1"]
    if not ((df.bbox_z1 > df.bbox_z0) & (df.bbox_y1 > df.bbox_y0)
            & (df.bbox_x1 > df.bbox_x0)).all():
        fail_output("BBOX_GATE: caixa degenerada")
    g["BBOX_GATE"] = "PASS"
    if S13.MARGIN_MM != 15.0:
        fail_output(f"ROI_MARGIN_GATE: MARGIN_MM = {S13.MARGIN_MM} != 15.0")
    esperado_m = df.apply(lambda r: (
        int(np.ceil(15.0 / json.loads(r.spacing_zyx)[0])),
        int(np.ceil(15.0 / json.loads(r.spacing_zyx)[1])),
        int(np.ceil(15.0 / json.loads(r.spacing_zyx)[2]))), axis=1)
    obtido_m = list(zip(df.margin_vox_z, df.margin_vox_y, df.margin_vox_x))
    mal = [i for i, (a, b) in enumerate(zip(esperado_m, obtido_m)) if tuple(a) != tuple(b)]
    if mal:
        fail_output(f"ROI_MARGIN_GATE: margem != ceil(15/espaçamento) em {len(mal)} objectos")
    g["ROI_MARGIN_GATE"] = "margem = ceil(15 mm / espaçamento) por eixo, em todos os objectos"
    g["INPUT_TABLE_COUNT_GATE"] = f"{len(obj)} linhas do Script 22"
    g["OBJECT_CONSISTENCY_GATE"] = ("diâmetro de cada objecto conferido contra o "
                                    "registado pelo Script 22 para o mesmo object_id")
    g["ROI_CLIPPED"] = int(df.roi_clipped_at_image_boundary.sum())
    g["REFERENCE_OUTSIDE_CROP"] = {"voxels": int(din.reference_voxels_outside_crop.sum()),
                                   "objectos_perdidos": len(falhas)}
    g["FAILURES"] = {"n": len(falhas),
                     "reasons": sorted({f["reason"] for f in falhas}) or []}

    prods = {}

    def grava(nome, frame):
        p = os.path.join(out_root, nome); frame.to_csv(p, index=False); prods[nome] = p

    pq = os.path.join(out_root, "lndb_object_rois.parquet")
    df.to_parquet(pq, index=False)
    prods["lndb_object_rois.parquet"] = pq
    grava("lndb_object_rois.csv", df)
    inv = df.groupby(["source", "architecture", "group"]).agg(
        n_objects=("object_id", "size"), n_scans=("scan_id", "nunique"),
        median_diam_mm=("diam_mm", "median"), median_volume_mm3=("volume_mm3", "median"),
        roi_clipped=("roi_clipped_at_image_boundary", "sum")).reset_index()
    grava("lndb_object_inventory.csv", inv)
    grava("input_manifest.csv", din)
    json.dump(falhas, open(os.path.join(out_root, "failures.json"), "w"), indent=1)
    json.dump(g, open(os.path.join(out_root, "gates.json"), "w"), indent=1, default=str)
    prods["failures.json"] = os.path.join(out_root, "failures.json")
    prods["gates.json"] = os.path.join(out_root, "gates.json")
    pd.DataFrame([{"path": n, "sha256": sha256(p)} for n, p in prods.items()]).to_csv(
        os.path.join(out_root, "output_hashes.csv"), index=False)
    man = {
        "script": "22a_prepare_lndb_filter_objects.py", "run_id": run_id,
        "script_sha256": sha256(os.path.abspath(__file__)),
        "run_at": t0.isoformat(), "end_at": datetime.datetime.now().astimezone().isoformat(),
        "source": {"script22_run": man22["run_id"], "script22_manifest_sha256": h22,
                   "object_matches_sha256": hobj},
        "canonical_modules": {"script11": SHA11, "script13": SHA13},
        "definitions": {
            "grid": "grelha recortada do Script 06 (idêntica à da predição)",
            "reference_labels": "rotulados na grelha completa; imagem de rótulos recortada "
                                "por índice, sem nova rotulagem",
            "prediction_labels": "rotulados na grelha recortada, como no Script 22",
            "connectivity": "6", "roi_margin_mm": S13.MARGIN_MM,
            "sources": {"pred": list(GRUPOS["pred"]), "gt": list(GRUPOS["gt"])},
            "key": CHAVE, "no_rematching": True},
        "counts": {"scans": len(scans), "input_match_rows": len(obj), "objects": len(df),
                   "failures": len(falhas), "expected": esperado,
                   "by_group": g["GROUP_COUNTS"]},
        "gates": g,
        "promotion": "NOT PROMOTED — produtos ficam no directório do run",
    }
    json.dump(man, open(os.path.join(out_root, "manifest.json"), "w"), indent=2, default=str)
    log.info(f"{len(df)} objectos · {len(falhas)} falhas · grupos {g['GROUP_COUNTS']}")
    log.info(f"Saída: {out_root}")
    log.info("Script 22A concluído.")


if __name__ == "__main__":
    main()
