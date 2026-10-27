#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    05_build_lndb_annotations.py

File type:
    Script

Purpose:
    Builds the LNDb annotations from the official masks and challenge CSV files, with manifest-driven case discovery.

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
    - lndb_alternate_label_set.json

Outputs:
    - See docs/file_level_documentation.md.

Dependencies:
    - SimpleITK
    - numpy
    - scipy

Execution:
    PULMONARY_BASE=<project_dir> python scripts/05_build_lndb_annotations.py

Portability notes:
    All paths are resolved relative to the project base directory given by the PULMONARY_BASE environment variable. No machine-specific path is required.

Reproducibility notes:
    Environment pinned in requirements.txt and environment/. Seeds and frozen parameters are recorded in the run manifests under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) are not included and must be obtained from their original repositories.
"""

import os
import sys
import csv
import json
import math
import hashlib
import time
import logging
import argparse
from collections import defaultdict

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as nd

from pulmonary_base import (BASE, check_base, verify_nifti, nifti_info,
                            log_skip, sha256_file, escrever_ponteiro)


SCRIPT05_VERSION            = "2026-07-31.lndb.hardened.v2"
CONSENSUS_SOURCE            = "LNDB_OFFICIAL_GT_MASK_VERBATIM"
TAU_LNDB_APPLIED            = False
PER_READER_MODEL            = "SPHERE_FROM_CENTROID_AND_VOLUME"

REPRESENTATIVE_RULE_VERSION           = "2026-07-31.rule.v2"
NUMBER_OF_FINDINGS_USED_FOR_SELECTION = "NO"
NUMBER_OF_NODULE1_USED_FOR_SELECTION  = "NO"
CONSENSUS_NONEMPTY_USED_FOR_SELECTION = "NO"
ANNOTATION_MERGE_STATUS               = "NOT_PERFORMED"

EMPTY_REFERENCE_LABEL     = "EMPTY_OFFICIAL_REFERENCE"
DISCORDANCE_LABEL         = "OFFICIAL_REFERENCE_VERSUS_READER_ANNOTATION_DISCORDANCE"
FORBIDDEN_LABELS          = ("TRUE_NEGATIVE",)

REQUIRED_MANIFEST_STATUS = "FROZEN_NOMINAL"

GEOM_TOL_MM   = 1e-3
DIR_TOL       = 1e-6
ORTHO_TOL     = 1e-4

AFFINE_TOL_INDEX = 1e-3

MANIFEST_IN  = os.path.join(BASE, "config", "lndb_nominal_manifest.json")
MANIFEST_OUT = os.path.join(BASE, "config", "lndb_annotations_manifest.json")

RAW_LNDB    = os.path.join(BASE, "data", "raw", "lndb")
RAW_DATA    = os.path.join(RAW_LNDB, "data")
RAW_GTMASKS = os.path.join(RAW_LNDB, "gt_masks")

CONSENSUS_DIR  = os.path.join(BASE, "annotations", "consensus", "lndb")
PER_READER_DIR = os.path.join(BASE, "annotations", "per_reader", "lndb")
OBJ_TABLE_DIR  = os.path.join(BASE, "annotations", "object_tables")
ANALYSIS       = os.path.join(BASE, "analysis")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CHARS_ATTRS = ["calcification", "internalStructure", "lobulation",
               "malignancy", "margin", "sphericity", "spiculation", "subtlety"]

NODULE_FIELDS = ["lndb_id", "rad_id", "finding_id", "x_mm", "y_mm", "z_mm",
                 "volume_mm3", "radius_mm", "is_nodule", "texture",
                 "centroid_inside_volume", "analytic_status"] + CHARS_ATTRS

AUDIT_FIELDS = ["lndb_id", "n_readers_documented", "n_readers_with_nodules",
                "geometry_status", "direction_alignment_status",
                "official_mask_value_identity_status", "independent_affine_check_status",
                "per_reader_status", "structural_validity_status",
                "n_vox_consensus", "n_components_consensus", "consensus_empty",
                "adjudication", "n_reader_findings",
                "n_reader_findings_marked_nodule",
                "official_reference_status", "reference_annotation_discordance",
                "n_annotations", "n_nodule_annotations", "n_per_reader_masks",
                "n_centroids_outside_volume", "direction_is_identity",
                "direction_orthonormal", "max_affine_disagreement_index",
                "sha256_consensus", "sha256_gt_source",
                "duplicate_group_id", "analytic_status", "status"]


class FalhaMetodologica(Exception):
    """Condição que invalida o produto. Nunca é absorvida por `continue`."""


def _carregar_manifesto():
    if not os.path.isfile(MANIFEST_IN):
        raise FalhaMetodologica(
            f"manifesto nominal ausente: {MANIFEST_IN}. A descoberta por varredura "
            f"de diretório é proibida; o Script 05 não pode prosseguir.")
    m = json.load(open(MANIFEST_IN))

    declarado = m.get("manifest_sha256")
    corpo = {k: v for k, v in m.items() if k != "manifest_sha256"}
    recalculado = hashlib.sha256(
        json.dumps(corpo, indent=1, sort_keys=True).encode()).hexdigest()
    if declarado != recalculado:
        raise FalhaMetodologica(
            f"SHA-256 do manifesto nominal não confere. declarado={declarado} "
            f"recalculado={recalculado}. O manifesto foi alterado após o congelamento.")

    if m.get("LNDB_MANIFEST_STATUS") != REQUIRED_MANIFEST_STATUS:
        raise FalhaMetodologica(
            f"LNDB_MANIFEST_STATUS = {m.get('LNDB_MANIFEST_STATUS')!r}, "
            f"exigido {REQUIRED_MANIFEST_STATUS!r}.")
    if m.get("LNDB_PROVENANCE_STATUS") != "PASS":
        raise FalhaMetodologica(
            f"LNDB_PROVENANCE_STATUS = {m.get('LNDB_PROVENANCE_STATUS')!r}, exigido 'PASS'.")

    linhas = m.get("rows", [])
    _declarado = m.get("LNDB_NOMINAL_COUNT")
    if _declarado is not None and len(linhas) != int(_declarado):
        raise FalhaMetodologica(
            f"manifesto com {len(linhas)} linhas mas declara LNDB_NOMINAL_COUNT="
            f"{_declarado}.")
    if len({r["lndb_id"] for r in linhas}) != len(linhas):
        raise FalhaMetodologica("identificadores repetidos no manifesto nominal.")

    grupos = m.get("duplicate_groups", {})
    _dup_decl = m.get("LNDB_DUPLICATE_GROUPS")
    if _dup_decl is not None and len(grupos) != int(_dup_decl):
        raise FalhaMetodologica(
            f"{len(grupos)} grupos duplicados mas o manifesto declara {_dup_decl}.")

    log.info(f"manifesto nominal verificado · sha256={declarado[:16]}… · "
             f"{len(linhas)} casos · {len(grupos)} grupos duplicados")
    return m


def _ler_csv(path, obrigatorias):
    if not os.path.isfile(path):
        raise FalhaMetodologica(f"CSV obrigatório ausente: {path}")
    linhas = list(csv.DictReader(open(path, newline="")))
    if not linhas:
        raise FalhaMetodologica(f"CSV vazio: {path}")
    faltam = set(obrigatorias) - set(linhas[0].keys())
    if faltam:
        raise FalhaMetodologica(f"{os.path.basename(path)}: colunas em falta {sorted(faltam)}")
    return linhas


def _carregar_csvs(ids_esperados):
    tn_path = os.path.join(RAW_LNDB, "trainNodules.csv")
    ch_path = os.path.join(RAW_LNDB, "chars_trainNodules.csv")
    ct_path = os.path.join(RAW_LNDB, "trainCTs.csv")

    train = {}
    for r in _ler_csv(tn_path, ["LNDbID", "RadID", "FindingID", "x", "y", "z",
                                "Nodule", "Volume", "Text"]):
        chave = (int(r["LNDbID"]), int(r["RadID"]), int(r["FindingID"]))
        if chave in train:
            raise FalhaMetodologica(f"trainNodules.csv: chave repetida {chave}")
        train[chave] = r

    chars = {}
    for r in _ler_csv(ch_path, ["LNDbID", "RadID", "FindingID"] + CHARS_ATTRS):
        chave = (int(float(r["LNDbID"])), int(float(r["RadID"])), int(float(r["FindingID"])))
        chars[chave] = {a: r.get(a, "") for a in CHARS_ATTRS}

    leitores = {}
    for r in _ler_csv(ct_path, ["LNDbID", "RadN"]):
        leitores[int(r["LNDbID"])] = int(r["RadN"])

    nums = {int(i.split("-")[1]) for i in ids_esperados}
    sem_cobertura = sorted(nums - set(leitores))
    if sem_cobertura:
        raise FalhaMetodologica(
            f"trainCTs.csv não cobre {len(sem_cobertura)} casos do manifesto: {sem_cobertura}")

    sem_anotacao = sorted(nums - {k[0] for k in train})
    log.info(f"trainNodules={len(train)} · chars={len(chars)} · trainCTs={len(leitores)}")
    log.info(f"casos sem QUALQUER linha em trainNodules.csv: {len(sem_anotacao)} {sem_anotacao}")
    return train, chars, leitores, set(sem_anotacao)


def _geom_de(path):
    """Lê SÓ o cabeçalho. Não carrega voxels."""
    r = sitk.ImageFileReader()
    r.SetFileName(path)
    r.ReadImageInformation()
    return {"size": tuple(r.GetSize()), "spacing": tuple(r.GetSpacing()),
            "origin": tuple(r.GetOrigin()), "direction": tuple(r.GetDirection())}


def _geometria_igual(a, b):
    """Compara tamanho, espaçamento, origem E DIREÇÃO. Devolve (ok, motivo)."""
    if a["size"] != b["size"]:
        return False, f"tamanho {a['size']} vs {b['size']}"
    for nome, tol in (("spacing", GEOM_TOL_MM), ("origin", GEOM_TOL_MM),
                      ("direction", DIR_TOL)):
        va, vb = np.asarray(a[nome], float), np.asarray(b[nome], float)
        if va.shape != vb.shape or not np.allclose(va, vb, atol=tol, rtol=0.0):
            d = float(np.max(np.abs(va - vb))) if va.shape == vb.shape else float("nan")
            return False, f"{nome} difere, desvio máximo {d:.3e}"
    return True, "ok"


def _direcao_ortonormal(direction):
    D = np.asarray(direction, float).reshape(3, 3)
    return bool(np.allclose(D.T @ D, np.eye(3), atol=ORTHO_TOL, rtol=0.0))


def _ponto_para_indice(p_mm, geom):
    """
    índice = S⁻¹ · D⁻¹ · (p − origem).   Implementação independente do SimpleITK,
    para que a verificação não repita a rotina que gerou o produto.

    Usa a inversa verdadeira, resolvida por `np.linalg.solve`, e NÃO a transposta.
    Dᵀ só iguala D⁻¹ quando D é exatamente ortonormal; as matrizes gravadas no
    LNDb são ortonormais até ~1e-7, e a substituição por Dᵀ introduzia
    discrepâncias de 1e-6 a 2e-5 índices em 28 exames — ruído numérico, mas
    suficiente para reprovar a conferência cruzada.
    """
    D = np.asarray(geom["direction"], float).reshape(3, 3)
    o = np.asarray(geom["origin"], float)
    s = np.asarray(geom["spacing"], float)
    return np.linalg.solve(D, np.asarray(p_mm, float) - o) / s


def _pintar_esfera(arr, spacing_xyz, idx_xyz, raio_mm):
    """
    arr em ordem numpy (z, y, x). idx_xyz é índice contínuo em ordem (x, y, z).
    A distância física entre índices é ‖D·S·Δi‖; com D ortonormal isso reduz-se
    a ‖S·Δi‖, e é por isso que só o espaçamento entra aqui.
    """
    ix, iy, iz = idx_xyz
    sx, sy, sz = spacing_xyz
    D, H, W = arr.shape

    rx, ry, rz = raio_mm / sx, raio_mm / sy, raio_mm / sz
    x0 = max(0, int(math.floor(ix - rx)) - 1); x1 = min(W, int(math.ceil(ix + rx)) + 2)
    y0 = max(0, int(math.floor(iy - ry)) - 1); y1 = min(H, int(math.ceil(iy + ry)) + 2)
    z0 = max(0, int(math.floor(iz - rz)) - 1); z1 = min(D, int(math.ceil(iz + rz)) + 2)
    if x0 >= x1 or y0 >= y1 or z0 >= z1:
        return

    zz = (np.arange(z0, z1) - iz) * sz
    yy = (np.arange(y0, y1) - iy) * sy
    xx = (np.arange(x0, x1) - ix) * sx
    ZZ, YY, XX = np.meshgrid(zz, yy, xx, indexing="ij")
    arr[z0:z1, y0:y1, x0:x1] |= (XX**2 + YY**2 + ZZ**2 <= raio_mm**2).astype(np.uint8)


def _achados_por_leitor(lndb_num, train):
    """{rad_id: [(x, y, z, raio_mm, finding_id), …]} só com Nodule == 1."""
    por_leitor = defaultdict(list)
    for (lid, rid, fid), r in train.items():
        if lid != lndb_num or int(r["Nodule"]) != 1:
            continue
        vol = float(r["Volume"])
        if not (vol > 0.0 and math.isfinite(vol)):
            raise FalhaMetodologica(
                f"LNDb-{lndb_num:04d} rad={rid} finding={fid}: Volume inválido {vol!r}")
        raio = (3.0 * vol / (4.0 * math.pi)) ** (1.0 / 3.0)
        por_leitor[rid].append((float(r["x"]), float(r["y"]), float(r["z"]), raio, fid))
    return {k: sorted(v, key=lambda t: t[4]) for k, v in por_leitor.items()}


def _rasterizar(geom, achados):
    """Constrói o array da máscara de um leitor. Determinista."""
    W, H, D = geom["size"]
    arr = np.zeros((D, H, W), dtype=np.uint8)
    fora = 0
    for (x, y, z, raio, _fid) in achados:
        idx = _ponto_para_indice((x, y, z), geom)
        if not (0 <= idx[0] < W and 0 <= idx[1] < H and 0 <= idx[2] < D):
            fora += 1
        _pintar_esfera(arr, geom["spacing"], idx, raio)
    return arr, fora


def _sha_array(arr):
    return hashlib.sha256(np.ascontiguousarray(arr)).hexdigest()


def _identidade_consenso(lndb_id, gt_path, geom_ct, sha_gt):
    """
    Identidade de VALOR do consenso gravado contra a máscara oficial.
    Medida em separado para alimentar official_mask_value_identity_pass.
    Devolve (status, motivo, sha256_do_consenso).
    """
    cons = os.path.join(CONSENSUS_DIR, lndb_id, "consensus.nii.gz")
    if not os.path.isfile(cons):
        return "ABSENT", "consensus.nii.gz ausente", ""
    ok, motivo = verify_nifti(cons, ref_geom=nifti_info(gt_path), expect_binary=True)
    if not ok:
        return "FAIL", f"consensus.nii.gz: {motivo}", ""
    for ref, nome in ((_geom_de(gt_path), "máscara oficial"), (geom_ct, "CT")):
        ok, motivo = _geometria_igual(_geom_de(cons), ref)
        if not ok:
            return "FAIL", f"consensus.nii.gz vs {nome}: {motivo}", ""
    sha = _sha_array(sitk.GetArrayFromImage(sitk.ReadImage(cons)))
    if sha != sha_gt:
        return "FAIL", "consensus.nii.gz não é idêntico em valor à máscara oficial", sha
    return "PASS", "ok", sha


def _verificar_per_reader(lndb_id, gt_path, geom_ct, esperado_por_leitor):
    """
    Conjunto exato de ficheiros e conteúdo reproduzido por rasterização
    independente, comparado por SHA-256. Devolve (status, motivo).
    """
    dir_pr = os.path.join(PER_READER_DIR, lndb_id)
    esperados = {f"rad_{r:02d}.nii.gz" for r in esperado_por_leitor}
    observados = set()
    if os.path.isdir(dir_pr):
        observados = {f for f in os.listdir(dir_pr) if f.endswith(".nii.gz")}
    if observados != esperados:
        return "FAIL", (f"sobra {sorted(observados - esperados)} "
                        f"falta {sorted(esperados - observados)}")
    for rid, achados in sorted(esperado_por_leitor.items()):
        p = os.path.join(dir_pr, f"rad_{rid:02d}.nii.gz")
        ok, motivo = verify_nifti(p, ref_geom=nifti_info(gt_path), expect_binary=True)
        if not ok:
            return "FAIL", f"rad_{rid:02d}.nii.gz: {motivo}"
        ok, motivo = _geometria_igual(_geom_de(p), geom_ct)
        if not ok:
            return "FAIL", f"rad_{rid:02d}.nii.gz vs CT: {motivo}"
        ref, _ = _rasterizar(geom_ct, achados)
        if _sha_array(sitk.GetArrayFromImage(sitk.ReadImage(p))) != _sha_array(ref):
            return "FAIL", f"rad_{rid:02d}.nii.gz não reproduz a rasterização"
    return "PASS", "ok"


def _conferir_afim_independente(geom, pontos):
    """
    Cruza a afim independente contra a do SimpleITK. Se as duas discordarem, a
    verificação de rasterização não teria valor — estaria a repetir o gerador.
    Devolve (status, maior_desvio_em_indices).
    """
    img = sitk.Image(1, 1, 1, sitk.sitkUInt8)
    img.SetOrigin(geom["origin"]); img.SetSpacing(geom["spacing"])
    img.SetDirection(geom["direction"])

    cantos = [geom["origin"],
              tuple(np.asarray(geom["origin"], float)
                    + np.asarray(geom["direction"], float).reshape(3, 3)
                    @ (np.asarray(geom["spacing"], float)
                       * (np.asarray(geom["size"], float) - 1.0)))]
    alvos = list(pontos) + cantos

    pior = 0.0
    for p in alvos:
        a = np.asarray(img.TransformPhysicalPointToContinuousIndex(
            tuple(float(v) for v in p)), float)
        b = _ponto_para_indice(p, geom)
        pior = max(pior, float(np.max(np.abs(a - b))))
    return ("PASS" if pior <= AFFINE_TOL_INDEX else "FAIL"), pior


def _verificar_saidas(lndb_id, gt_path, geom_ct, esperado_por_leitor, sha_gt):
    """Composição das duas verificações, para a decisão de skip."""
    st, motivo, _ = _identidade_consenso(lndb_id, gt_path, geom_ct, sha_gt)
    if st != "PASS":
        return False, motivo
    st, motivo = _verificar_per_reader(lndb_id, gt_path, geom_ct, esperado_por_leitor)
    if st != "PASS":
        return False, f"máscaras por leitor: {motivo}"
    return True, "ok"


def processar(lndb_id, linha, train, chars, leitores, sem_anotacao,
              nodule_rows, audit_rows, escrever):
    lndb_num = int(lndb_id.split("-")[1])
    ct_path  = os.path.join(RAW_DATA, f"{lndb_id}.mhd")
    raw_path = os.path.join(RAW_DATA, f"{lndb_id}.raw")
    gt_path  = os.path.join(RAW_GTMASKS, f"LNDb_{lndb_num:04d}.nii.gz")

    for p, rot in ((ct_path, ".mhd"), (raw_path, ".raw"), (gt_path, "gt_mask")):
        if not os.path.isfile(p):
            raise FalhaMetodologica(f"{lndb_id}: {rot} ausente em {p}")

    geom_ct = _geom_de(ct_path)
    geom_gt = _geom_de(gt_path)
    ok, motivo = _geometria_igual(geom_gt, geom_ct)
    if not ok:
        raise FalhaMetodologica(f"{lndb_id}: máscara oficial não alinha com o CT — {motivo}")

    ortonormal = _direcao_ortonormal(geom_ct["direction"])
    if not ortonormal:
        raise FalhaMetodologica(
            f"{lndb_id}: matriz de direção não ortonormal; a rasterização esférica "
            f"em espaço de índice deixa de ser exata.")
    identidade = bool(np.allclose(np.asarray(geom_ct["direction"], float),
                                  np.eye(3).ravel(), atol=DIR_TOL, rtol=0.0))

    gt_arr = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
    valores = np.unique(gt_arr)
    if not np.isin(valores, [0, 1]).all():
        raise FalhaMetodologica(f"{lndb_id}: máscara oficial não é binária, valores={valores[:10]}")
    sha_gt = _sha_array(gt_arr)

    n_vox = int((gt_arr > 0).sum())
    _, n_comp = nd.label(gt_arr > 0)

    esperado = _achados_por_leitor(lndb_num, train)
    anotacoes = [k for k in train if k[0] == lndb_num]
    n_nodulo = sum(1 for k in anotacoes if int(train[k]["Nodule"]) == 1)

    if lndb_num in sem_anotacao and n_vox > 0:
        raise FalhaMetodologica(
            f"{lndb_id}: sem qualquer linha em trainNodules.csv mas a máscara oficial "
            f"tem {n_vox} voxels positivos. Fontes incoerentes.")

    ok_skip, motivo_skip = _verificar_saidas(lndb_id, gt_path, geom_ct, esperado, sha_gt)
    estado = "SKIP_OK_VERIFICADO"
    fora_total = 0

    if not log_skip(log, lndb_id, ok_skip, motivo_skip):
        if not escrever:
            estado = "REPROCESSAMENTO_NECESSARIO"
        else:
            estado = "RECONSTRUIDO"
            d = os.path.join(CONSENSUS_DIR, lndb_id)
            os.makedirs(d, exist_ok=True)
            cons_img = sitk.GetImageFromArray(gt_arr.astype(np.uint8))
            cons_img.CopyInformation(sitk.ReadImage(gt_path))
            tmp = os.path.join(d, "consensus.tmp.nii.gz")
            sitk.WriteImage(sitk.Cast(cons_img, sitk.sitkUInt8), tmp, useCompression=True)
            os.replace(tmp, os.path.join(d, "consensus.nii.gz"))

            dpr = os.path.join(PER_READER_DIR, lndb_id)
            if os.path.isdir(dpr):
                for f in os.listdir(dpr):
                    if f.endswith(".nii.gz"):
                        os.remove(os.path.join(dpr, f))
            if esperado:
                os.makedirs(dpr, exist_ok=True)
            for rid, achados in sorted(esperado.items()):
                arr, fora = _rasterizar(geom_ct, achados)
                fora_total += fora
                img = sitk.GetImageFromArray(arr)
                img.SetOrigin(geom_ct["origin"]); img.SetSpacing(geom_ct["spacing"])
                img.SetDirection(geom_ct["direction"])
                tmp = os.path.join(dpr, f"rad_{rid:02d}.tmp.nii.gz")
                sitk.WriteImage(img, tmp, useCompression=True)
                os.replace(tmp, os.path.join(dpr, f"rad_{rid:02d}.nii.gz"))

            ok2, motivo2 = _verificar_saidas(lndb_id, gt_path, geom_ct, esperado, sha_gt)
            if not ok2:
                raise FalhaMetodologica(f"{lndb_id}: saída inválida após reconstrução — {motivo2}")

    if not fora_total:
        for achados in esperado.values():
            for (x, y, z, _r, _f) in achados:
                idx = _ponto_para_indice((x, y, z), geom_ct)
                W, H, D = geom_ct["size"]
                if not (0 <= idx[0] < W and 0 <= idx[1] < H and 0 <= idx[2] < D):
                    fora_total += 1

    for (lid, rid, fid) in sorted(anotacoes):
        r = train[(lid, rid, fid)]
        vol = float(r["Volume"])
        idx = _ponto_para_indice((float(r["x"]), float(r["y"]), float(r["z"])), geom_ct)
        W, H, D = geom_ct["size"]
        dentro = bool(0 <= idx[0] < W and 0 <= idx[1] < H and 0 <= idx[2] < D)
        nrow = {"lndb_id": lndb_id, "rad_id": rid, "finding_id": fid,
                "x_mm": round(float(r["x"]), 4), "y_mm": round(float(r["y"]), 4),
                "z_mm": round(float(r["z"]), 4), "volume_mm3": round(vol, 3),
                "radius_mm": round((3.0 * vol / (4.0 * math.pi)) ** (1.0 / 3.0), 4)
                             if vol > 0 else 0.0,
                "is_nodule": int(r["Nodule"]), "texture": r.get("Text", ""),
                "centroid_inside_volume": dentro,
                "analytic_status": linha.get("analytic_status", "")}
        nrow.update(chars.get((lid, rid, fid), {a: "" for a in CHARS_ATTRS}))
        nodule_rows.append(nrow)

    st_cons, motivo_cons, sha_cons = _identidade_consenso(lndb_id, gt_path, geom_ct, sha_gt)
    st_pr, motivo_pr = _verificar_per_reader(lndb_id, gt_path, geom_ct, esperado)

    pontos = [(float(train[k]["x"]), float(train[k]["y"]), float(train[k]["z"]))
              for k in anotacoes]
    st_afim, pior_afim = _conferir_afim_independente(geom_ct, pontos)
    if st_afim != "PASS":
        raise FalhaMetodologica(
            f"{lndb_id}: afim independente discorda do SimpleITK em {pior_afim:.3e} "
            f"índices; a verificação da rasterização perderia valor.")

    if n_vox == 0:
        if len(anotacoes) == 0:
            adjudicacao = "NEGATIVE_DOCUMENTED"
        else:
            adjudicacao = "READER_FINDINGS_WITHOUT_OFFICIAL_REFERENCE"
    else:
        adjudicacao = "NOT_APPLICABLE"
    ref_status = EMPTY_REFERENCE_LABEL if n_vox == 0 else ""
    discordancia = (DISCORDANCE_LABEL if (n_vox == 0 and n_nodulo > 0) else "")
    for rotulo in (ref_status, discordancia):
        if rotulo in FORBIDDEN_LABELS:
            raise FalhaMetodologica(f"{lndb_id}: rótulo proibido {rotulo!r}")

    estrutural = "PASS" if (st_cons in ("PASS", "ABSENT") and st_afim == "PASS") else "FAIL"

    audit_rows.append({
        "lndb_id": lndb_id,
        "n_readers_documented": leitores[lndb_num],
        "n_readers_with_nodules": len(esperado),
        "geometry_status": "PASS_GT_CT_FULL_GEOMETRY",
        "direction_alignment_status": "PASS",
        "official_mask_value_identity_status": st_cons,
        "independent_affine_check_status": st_afim,
        "per_reader_status": st_pr,
        "structural_validity_status": estrutural,
        "n_vox_consensus": n_vox,
        "n_components_consensus": int(n_comp),
        "consensus_empty": n_vox == 0,
        "adjudication": adjudicacao,
        "n_reader_findings": len(anotacoes),
        "n_reader_findings_marked_nodule": n_nodulo,
        "official_reference_status": ref_status,
        "reference_annotation_discordance": discordancia,
        "n_annotations": len(anotacoes),
        "n_nodule_annotations": n_nodulo,
        "n_per_reader_masks": len(esperado),
        "n_centroids_outside_volume": fora_total,
        "direction_is_identity": identidade,
        "direction_orthonormal": ortonormal,
        "max_affine_disagreement_index": pior_afim,
        "sha256_consensus": sha_cons,
        "sha256_gt_source": sha_gt,
        "duplicate_group_id": linha.get("duplicate_group_id") or "",
        "analytic_status": "",
        "status": estado,
    })


def _auditoria_intrapar(grupos, audit_por_id, train):
    """
    Mede cada par. NÃO combina anotações — apenas compara.
    """
    linhas = []
    for gid, membros in sorted(grupos.items()):
        if len(membros) != 2:
            raise FalhaMetodologica(f"{gid}: {len(membros)} membros, esperado 2")
        a, b = membros
        ra, rb = audit_por_id[a], audit_por_id[b]

        arr_a = sitk.GetArrayFromImage(sitk.ReadImage(
            os.path.join(RAW_GTMASKS, f"LNDb_{int(a.split('-')[1]):04d}.nii.gz"))) > 0
        arr_b = sitk.GetArrayFromImage(sitk.ReadImage(
            os.path.join(RAW_GTMASKS, f"LNDb_{int(b.split('-')[1]):04d}.nii.gz"))) > 0
        inter = int((arr_a & arr_b).sum())
        soma = int(arr_a.sum()) + int(arr_b.sum())
        dice = (2.0 * inter / soma) if soma else None

        linhas.append({
            "duplicate_group_id": gid, "member_a": a, "member_b": b,
            "image_identity": "IDENTICAL_SOURCE_AND_OUTPUT",
            "gt_mask_identity": "IDENTICAL" if ra["sha256_gt_source"] == rb["sha256_gt_source"]
                                 else "DIFFERENT",
            "gt_dice": None if dice is None else round(dice, 6),
            "gt_intersection_vox": inter,
            "n_vox_a": int(arr_a.sum()), "n_vox_b": int(arr_b.sum()),
            "n_annotations_a": ra["n_annotations"], "n_annotations_b": rb["n_annotations"],
            "n_nodule_annotations_a": ra["n_nodule_annotations"],
            "n_nodule_annotations_b": rb["n_nodule_annotations"],
            "annotations_identical": _anotacoes_iguais(a, b, train),
            "descriptive_only_fields": "n_annotations_*, n_nodule_annotations_*, n_vox_*",
        })
    return linhas


def _anotacoes_iguais(a, b, train):
    """
    Compara o conteúdo anotado dos dois membros. RadID entra na chave porque é o
    índice do leitor dentro do exame; não se assume correspondência de pessoa
    entre os membros do par.
    """
    def chave(x):
        n = int(x.split("-")[1])
        return sorted(
            (rid, fid, round(float(r["x"]), 4), round(float(r["y"]), 4),
             round(float(r["z"]), 4), int(r["Nodule"]), round(float(r["Volume"]), 4))
            for (lid, rid, fid), r in train.items() if lid == n)
    return "IDENTICAL" if chave(a) == chave(b) else "DIFFERENT"


def _selecionar_representantes(pares, audit_por_id):
    """
    Regra 2026-07-31.rule.v2 — CEGA AO CONTEÚDO DA ANOTAÇÃO:
      0. membro com geometria, proveniência e anotação estrutural válidas
      1. maior número documentado de leitores  (RadN, trainCTs.csv)
      2. menor scan_id

    Número de achados, quantidade de Nodule=1 e consenso não vazio NÃO entram.
    Selecionar pelo conteúdo escolheria o representante em função do próprio
    alvo, favorecendo casos positivos. Uma máscara oficial vazia num membro
    escolhido é consequência da regra ser cega ao conteúdo, não motivo para
    substituição.
    """
    escolhidos, excluidos, decisoes = {}, {}, []
    for p in pares:
        a, b = p["member_a"], p["member_b"]

        val = {x: audit_por_id[x]["structural_validity_status"] == "PASS" for x in (a, b)}
        if not any(val.values()):
            raise FalhaMetodologica(
                f"{p['duplicate_group_id']}: nenhum membro passa o critério 0 "
                f"(validade estrutural). Seleção impossível.")
        if val[a] != val[b]:
            ganhador = a if val[a] else b
            criterio = "structural_validity"
        else:
            ra = (audit_por_id[a]["n_readers_documented"], -int(a.split("-")[1]))
            rb = (audit_por_id[b]["n_readers_documented"], -int(b.split("-")[1]))
            if ra == rb:
                raise FalhaMetodologica(
                    f"{p['duplicate_group_id']}: empate total entre {a} e {b}; "
                    f"a regra {REPRESENTATIVE_RULE_VERSION} não desempata.")
            ganhador = a if ra > rb else b
            criterio = ("n_readers_documented" if ra[0] != rb[0] else "lowest_scan_id")

        perdedor = b if ganhador == a else a
        escolhidos[ganhador] = p["duplicate_group_id"]
        excluidos[perdedor] = p["duplicate_group_id"]
        decisoes.append({**p,
                         "representative": ganhador, "excluded": perdedor,
                         "deciding_criterion": criterio,
                         "representative_rule_version": REPRESENTATIVE_RULE_VERSION,
                         "n_readers_documented_a": audit_por_id[a]["n_readers_documented"],
                         "n_readers_documented_b": audit_por_id[b]["n_readers_documented"],
                         "structural_validity_a": audit_por_id[a]["structural_validity_status"],
                         "structural_validity_b": audit_por_id[b]["structural_validity_status"],
                         "excluded_annotations_preserved": True,
                         "excluded_files_preserved": True,
                         "annotation_merge_status": ANNOTATION_MERGE_STATUS})
    return escolhidos, excluidos, decisoes


def _conjunto_alternativo(decisoes, audit_rows):
    """
    Análise de sensibilidade PRÉ-ESPECIFICADA: troca-se o representante nos 12
    pares, gerando um segundo conjunto de 224 exames com as MESMAS imagens e
    anotações diferentes. Quantifica a influência da escolha das anotações sem
    combinar máscaras.

    Pré-especificada quer dizer: fixada aqui, antes de qualquer desempenho de
    modelo ser observado.
    """
    trocados = {d["excluded"]: d["duplicate_group_id"] for d in decisoes}
    fora = {d["representative"] for d in decisoes}
    membros = sorted(r["lndb_id"] for r in audit_rows if r["lndb_id"] not in fora)
    return {
        "analysis_id": "lndb_alternate_label_set",
        "prespecified": True,
        "specified_before_any_model_performance_observed": True,
        "primary_set_n": len(audit_rows) - len(decisoes),
        "alternate_label_set_n": len(membros),
        "all_duplicate_pairs_swapped": len(trocados),
        "annotation_merge_status": ANNOTATION_MERGE_STATUS,
        "masks_combined": False,
        "images_identical_to_primary": True,
        "purpose": "quantificar a influência da escolha das anotações nos 12 pares "
                   "de imagem duplicada, sem combinar máscaras e sem observar "
                   "previamente o desempenho dos modelos.",
        "swapped_in": sorted(trocados), "swapped_out": sorted(fora),
        "members": membros,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-only", action="store_true",
                    help="mede e relata sem escrever qualquer produto")
    args = ap.parse_args()
    escrever = not args.audit_only

    check_base(strict=True)
    snap0 = _snapshot()
    man = _carregar_manifesto()
    linhas = {r["lndb_id"]: r for r in man["rows"]}
    train, chars, leitores, sem_anotacao = _carregar_csvs(linhas.keys())

    if escrever:
        for d in (CONSENSUS_DIR, PER_READER_DIR, OBJ_TABLE_DIR, ANALYSIS):
            os.makedirs(d, exist_ok=True)
    log.info(f"modo = {'AUDITORIA (não escreve)' if args.audit_only else 'CANÓNICO'} · "
             f"{len(linhas)} exames nominais")

    nodule_rows, audit_rows, falhas = [], [], []
    for i, lndb_id in enumerate(sorted(linhas), 1):
        try:
            processar(lndb_id, linhas[lndb_id], train, chars, leitores,
                      sem_anotacao, nodule_rows, audit_rows, escrever)
        except FalhaMetodologica as e:
            log.error(f"  [FALHA] {e}")
            falhas.append(str(e))
        except Exception as e:
            log.error(f"  [ERRO] {lndb_id}: {type(e).__name__}: {e}")
            falhas.append(f"{lndb_id}: {type(e).__name__}: {e}")
        if i % 40 == 0:
            log.info(f"  PROGRESSO {i}/{len(linhas)}")

    if falhas:
        log.error(f"{len(falhas)} falhas. O manifesto de saída NÃO é congelado.")
        for f in falhas[:20]:
            log.error(f"  · {f}")
        mut = _diff_snapshot(snap0, _snapshot())
        log.error(f"  filesystem_mutations = {len(mut)}")
        log.error(f"  undocumented_failures = 0  (as {len(falhas)} estão listadas acima)")
        sys.exit(1)

    if len(audit_rows) != len(linhas):
        log.error(f"auditoria com {len(audit_rows)} linhas, manifesto com {len(linhas)}.")
        sys.exit(1)

    audit_por_id = {r["lndb_id"]: r for r in audit_rows}
    pares = _auditoria_intrapar(man["duplicate_groups"], audit_por_id, train)
    escolhidos, excluidos, decisoes = _selecionar_representantes(pares, audit_por_id)

    for r in audit_rows:
        r["analytic_status"] = ("ANALYTIC_REPRESENTATIVE" if r["lndb_id"] in escolhidos else
                                "EXCLUDED_DUPLICATE_IMAGE_PRESERVED" if r["lndb_id"] in excluidos
                                else "ANALYTIC_UNIT")
    for r in audit_rows:
        if r.get("adjudication") == "READER_FINDINGS_WITHOUT_OFFICIAL_REFERENCE":
            r["analytic_status"] = "EXCLUDED_REFERENCE_READER_DISCORDANCE"
    analiticos = [r for r in audit_rows
                  if r["analytic_status"] not in
                  ("EXCLUDED_DUPLICATE_IMAGE_PRESERVED",
                   "EXCLUDED_REFERENCE_READER_DISCORDANCE")]
    _esperado_analitico = (len(audit_rows)
                           - sum(1 for r in audit_rows if r["analytic_status"] in
                                 ("EXCLUDED_DUPLICATE_IMAGE_PRESERVED",
                                  "EXCLUDED_REFERENCE_READER_DISCORDANCE")))
    if len(analiticos) != _esperado_analitico:
        log.error(f"{len(analiticos)} unidades analíticas, derivado {_esperado_analitico}.")
        sys.exit(1)

    ex = {r["lndb_id"] for r in audit_rows if r["analytic_status"].startswith("EXCLUDED")}
    for r in nodule_rows:
        r["analytic_status"] = ("EXCLUDED_DUPLICATE_IMAGE_PRESERVED" if r["lndb_id"] in ex
                                else "ANALYTIC")

    alternativo = _conjunto_alternativo(decisoes, audit_rows)

    if args.audit_only:
        mutacoes = _diff_snapshot(snap0, _snapshot())
        _relatorio(audit_rows, decisoes, analiticos, nodule_rows, alternativo)
        _bloco_final(audit_rows, decisoes, mutacoes)
        if mutacoes:
            log.error(f"--audit-only mutou o sistema de ficheiros: {mutacoes[:5]}")
            sys.exit(1)
        return

    _escrever_csv(os.path.join(OBJ_TABLE_DIR, "lndb_nodules.csv"), NODULE_FIELDS, nodule_rows)
    _escrever_csv(os.path.join(ANALYSIS, "lndb_annotation_audit.csv"), AUDIT_FIELDS, audit_rows)
    _escrever_csv(os.path.join(ANALYSIS, "lndb_intrapair_audit.csv"),
                  list(decisoes[0].keys()), decisoes)

    saida = {
        "manifest_id": "lndb_annotations_manifest_2026-07-31",
        "script_version": SCRIPT05_VERSION,
        "input_manifest_sha256": man["manifest_sha256"],
        "CONSENSUS_SOURCE": CONSENSUS_SOURCE,
        "TAU_LNDB_APPLIED": TAU_LNDB_APPLIED,
        "PER_READER_MODEL": PER_READER_MODEL,
        "PER_READER_IS_APPROXIMATION": True,
        "ANNOTATION_MERGE_STATUS": ANNOTATION_MERGE_STATUS,
        "REPRESENTATIVE_RULE_VERSION": REPRESENTATIVE_RULE_VERSION,
        "REPRESENTATIVE_RULE": ["0. geometria, proveniência e anotação estrutural válidas",
                                "1. maior número documentado de leitores",
                                "2. menor scan_id"],
        "NUMBER_OF_FINDINGS_USED_FOR_SELECTION": NUMBER_OF_FINDINGS_USED_FOR_SELECTION,
        "NUMBER_OF_NODULE1_USED_FOR_SELECTION": NUMBER_OF_NODULE1_USED_FOR_SELECTION,
        "CONSENSUS_NONEMPTY_USED_FOR_SELECTION": CONSENSUS_NONEMPTY_USED_FOR_SELECTION,
        "EMPTY_REFERENCE_LABEL": EMPTY_REFERENCE_LABEL,
        "ADJUDICATION_RULE": (
            "ausencia de anotacao NAO e negativo. Negativo documentado exige "
            "referencia oficial vazia E zero achados de leitor. Referencia vazia "
            "com achados fica FORA da coorte principal, retida para analise de "
            "sensibilidade. Derivado dos dados, sem lista escrita a mao."),
        "n_negative_documented": sum(1 for r in audit_rows
                                     if r.get("adjudication") == "NEGATIVE_DOCUMENTED"),
        "n_reference_reader_discordance_excluded": sum(
            1 for r in audit_rows
            if r.get("adjudication") == "READER_FINDINGS_WITHOUT_OFFICIAL_REFERENCE"),
        "TRUE_NEGATIVE_LABEL_USED": False,
        "alternate_label_set": alternativo,
        "SCRIPT05_NOMINAL_COUNT": len(audit_rows),
        "SCRIPT05_ANALYTIC_COUNT": len(analiticos),
        "SCRIPT05_EXCLUDED_PRESERVED": len(excluidos),
        "SCRIPT05_UNDOCUMENTED_FAILURES": 0,
        "SCRIPT05_OUTPUT_STATUS": "PASS",
        "SCRIPT05_OUTPUT_MANIFEST_FROZEN": "YES",
        "n_consensus_empty": sum(1 for r in analiticos if r["consensus_empty"]),
        "n_nodule_rows": len(nodule_rows),
        "n_centroids_outside_volume": sum(r["n_centroids_outside_volume"] for r in audit_rows),
        "downstream_discovery_mode": "MANIFEST_ONLY",
        "rad_id_scope": "RadID e indice INTERNO ao exame (1..3); nao identifica a "
                        "mesma pessoa entre exames.",
        "intrapair": decisoes,
        "rows": audit_rows,
    }
    saida["generated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    saida["producer_sha256"] = sha256_file(os.path.abspath(__file__))
    corpo = json.dumps(saida, indent=1, sort_keys=True)
    saida["manifest_sha256"] = hashlib.sha256(corpo.encode()).hexdigest()
    tmp = MANIFEST_OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(saida, f, indent=1, sort_keys=True)
    os.replace(tmp, MANIFEST_OUT)

    _ponteiro = os.path.join(BASE, "runs", "CURRENT_LNDB_ANNOTATIONS.json")
    os.makedirs(os.path.dirname(_ponteiro), exist_ok=True)
    escrever_ponteiro(
        _ponteiro, MANIFEST_OUT,
        script="05", script_version=SCRIPT05_VERSION,
        run_id="s05_%s_%d" % (time.strftime("%Y%m%dT%H%M%S"), os.getpid()),
        published_at=saida["generated"],
        annotations_manifest=os.path.relpath(MANIFEST_OUT, BASE),
        alternate_label_set=os.path.relpath(
            os.path.join(BASE, "config", "lndb_alternate_label_set.json"), BASE),
        nominal_count=saida["SCRIPT05_NOMINAL_COUNT"],
        analytic_count=saida["SCRIPT05_ANALYTIC_COUNT"],
        excluded_preserved=saida["SCRIPT05_EXCLUDED_PRESERVED"],
        n_negative_documented=saida["n_negative_documented"],
        SCRIPT05_OUTPUT_STATUS=saida["SCRIPT05_OUTPUT_STATUS"])
    log.info("  ponteiro: runs/CURRENT_LNDB_ANNOTATIONS.json")

    with open(os.path.join(BASE, "config", "lndb_alternate_label_set.json"), "w") as f:
        json.dump(alternativo, f, indent=1, sort_keys=True)

    log.info(f"manifesto de saída congelado · sha256={saida['manifest_sha256']}")
    _relatorio(audit_rows, decisoes, analiticos, nodule_rows, alternativo)
    _bloco_final(audit_rows, decisoes, _diff_snapshot(snap0, _snapshot()),
                 rotulo_mutacoes="declared_target_writes")


def _snapshot():
    """
    Instantâneo de tudo que o script poderia escrever. `filesystem_mutations`
    é MEDIDO comparando dois instantâneos, não afirmado.
    """
    estado = {}
    for raiz in (CONSENSUS_DIR, PER_READER_DIR, OBJ_TABLE_DIR, ANALYSIS,
                 os.path.dirname(MANIFEST_OUT)):
        if not os.path.isdir(raiz):
            continue
        for d, _sub, ficheiros in os.walk(raiz):
            for f in ficheiros:
                p = os.path.join(d, f)
                try:
                    s = os.stat(p)
                    estado[p] = (s.st_size, s.st_mtime_ns)
                except OSError:
                    estado[p] = None
    return estado


def _diff_snapshot(antes, depois):
    dif = []
    for p in sorted(set(antes) | set(depois)):
        if antes.get(p) != depois.get(p):
            dif.append(p)
    return dif


def _bloco_final(audit_rows, decisoes, mutacoes, rotulo_mutacoes="filesystem_mutations"):
    """Bloco de encerramento. `mutacoes` é sempre medido, nunca presumido."""
    n = len(audit_rows)
    def conta(campo, valor):
        return sum(1 for r in audit_rows if r[campo] == valor)
    linhas = [
        ("nominal_records_checked",             f"{n}"),
        ("negative_documented",                 f"{conta('adjudication','NEGATIVE_DOCUMENTED')}"),
        ("reference_reader_discordance_excluded",
         f"{conta('adjudication','READER_FINDINGS_WITHOUT_OFFICIAL_REFERENCE')}"),
        ("distinct_physical_images",            f"{n - len(decisoes)}"),
        ("duplicate_groups_checked",            f"{len(decisoes)}/{len(decisoes)}"),
        ("direction_alignment_pass",            f"{conta('direction_alignment_status','PASS')}/{n}"),
        ("official_mask_value_identity_pass",   f"{conta('official_mask_value_identity_status','PASS')}/{n}"),
        ("independent_affine_check_pass",       f"{conta('independent_affine_check_status','PASS')}/{n}"),
        ("per_reader_rasterization_pass",       f"{conta('per_reader_status','PASS')}/{n}"),
        ("intrapair_annotation_table_complete", f"{len(decisoes)}/{len(decisoes)}"),
        ("empty_official_reference",            f"{conta('official_reference_status', EMPTY_REFERENCE_LABEL)}"),
        ("reference_reader_discordance",        f"{conta('reference_annotation_discordance', DISCORDANCE_LABEL)}"),
        ("undocumented_failures",               "0"),
        (rotulo_mutacoes,                       f"{len(mutacoes)}"),
    ]
    log.info("─" * 68)
    for k, v in linhas:
        log.info(f"  {k:<38s} = {v}")


def _escrever_csv(path, campos, linhas):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(linhas)
    os.replace(tmp, path)
    log.info(f"{os.path.basename(path)}: {len(linhas)} linhas")


def _relatorio(audit_rows, decisoes, analiticos, nodule_rows, alternativo):
    from collections import Counter
    log.info("─" * 68)
    log.info(f"nominais {len(audit_rows)} · analíticos {len(analiticos)} · "
             f"excluídos preservados {len(audit_rows) - len(analiticos)}")
    log.info(f"estados: {dict(Counter(r['status'] for r in audit_rows))}")
    log.info(f"centróides fora do volume: {sum(r['n_centroids_outside_volume'] for r in audit_rows)}")
    log.info(f"direção não identidade: {sum(1 for r in audit_rows if not r['direction_is_identity'])}")
    log.info(f"linhas de nódulo: {len(nodule_rows)}")
    log.info(f"conjunto alternativo pré-especificado: n={alternativo['alternate_label_set_n']} · "
             f"pares trocados={alternativo['all_duplicate_pairs_swapped']}")
    log.info("")
    log.info("TABELA INTRAPAR  (regra %s — leitores e scan_id apenas)" % REPRESENTATIVE_RULE_VERSION)
    log.info(f"  {'grupo':8s} {'membro':11s} {'leit':>4s} {'ach':>4s} {'nod1':>5s} "
             f"{'vox':>6s} {'dice':>8s}  {'repr':>5s}  motivo")
    for d in decisoes:
        dice = "n/d" if d["gt_dice"] is None else f"{d['gt_dice']:.4f}"
        for lado, m in (("a", d["member_a"]), ("b", d["member_b"])):
            marca = "  <<" if m == d["representative"] else "    "
            motivo = d["deciding_criterion"] if m == d["representative"] else ""
            log.info(f"  {d['duplicate_group_id']:8s} {m:11s} "
                     f"{d['n_readers_documented_' + lado]:>4d} "
                     f"{d['n_annotations_' + lado]:>4d} "
                     f"{d['n_nodule_annotations_' + lado]:>5d} "
                     f"{d['n_vox_' + lado]:>6d} "
                     f"{dice:>8s}{marca}  {motivo}")
    log.info("")
    log.info("Achados, nódulos e voxels acima são DESCRITIVOS: não participam da seleção.")


if __name__ == "__main__":
    try:
        main()
    except FalhaMetodologica as e:
        log.error(f"FALHA METODOLÓGICA: {e}")
        sys.exit(1)
