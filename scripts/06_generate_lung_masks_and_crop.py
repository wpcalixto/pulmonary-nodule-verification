#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    06_generate_lung_masks_and_crop.py

File type:
    Script

Purpose:
    Generates lung masks (LungMask R231) and crops LIDC-IDRI, RIDER and LNDb volumes to the lung bounding box.

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
    - crop_transform.json
    - INVALID.json
    - manifest.json

Outputs:
    - ct.nii.gz
    - lung_mask.nii.gz
    - lung_mask_crop.nii.gz

Dependencies:
    - SimpleITK
    - lungmask
    - numpy
    - torch

Execution:
    PULMONARY_BASE=<project_dir> python scripts/06_generate_lung_masks_and_crop.py

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
import time
import shutil
import hashlib
import logging
import argparse
import platform
from collections import Counter

import numpy as np
import SimpleITK as sitk

from pulmonary_base import BASE, check_base, sha256_file, escrever_ponteiro


SCRIPT06_VERSION       = "2026-07-31.crop.v2"
MARGIN_MM              = 10.0
VOXELWISE_LUNG_FILTER  = False
REFERENCE_GUIDED_CROP  = "NO"
LUNGMASK_MODEL         = "R231"

ACCELERATOR_REQUESTED  = "AUTO"
ACCELERATOR_TORCH_TYPE = "mps"

GEOM_TOL_MM = 1e-3
DIR_TOL     = 1e-6
INVERSE_TOL_MM = 1e-4

def _manifesto_lndb():
    _p = os.path.join(BASE, "runs", "CURRENT_LNDB_ANNOTATIONS.json")
    if os.path.isfile(_p):
        _d = json.load(open(_p))
        _c = _d.get("manifest") or _d.get("annotations_manifest")
        if _c:
            _c = _c if os.path.isabs(_c) else os.path.join(BASE, _c)
            if os.path.isfile(_c):
                return _c
    return os.path.join(BASE, "config", "lndb_annotations_manifest.json")


MANIFESTO_LNDB  = _manifesto_lndb()
MANIFESTO_RIDER = os.path.join(BASE, "config", "rider_pair_selection.json")


NIFTI_DIR = os.path.join(BASE, "data", "interim", "nifti")
CONS_DIR  = os.path.join(BASE, "annotations", "consensus")
PER_READER = os.path.join(BASE, "annotations", "per_reader")
RUNS_ROOT    = os.path.join(BASE, "runs")
CURRENT_JSON = os.path.join(BASE, "runs", "CURRENT.json")

ANNOT_SUBDIR = {"lidc_idri": "lidc", "lndb": "lndb", "rider": None}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


class FalhaMetodologica(Exception):
    """Invalida o produto. Nunca é absorvida por `continue`."""


def _sha_manifesto(m):
    corpo = {k: v for k, v in m.items() if k != "manifest_sha256"}
    return hashlib.sha256(json.dumps(corpo, indent=1, sort_keys=True).encode()).hexdigest()


def _exigir_campo(m, campo, rotulo):
    """
    Le um campo obrigatorio de um manifesto, sem substituto. Um `.get()` com
    omissao aqui gravaria hash vazio no produto — foi assim que `UTILS_SHA256`
    saiu vazio no estagio 04.
    """
    v = m.get(campo)
    if not v or not isinstance(v, str):
        raise FalhaMetodologica(
            f"{rotulo}: campo obrigatorio {campo!r} ausente ou vazio. "
            f"Campos presentes: {sorted(k for k in m if 'sha' in k.lower())}")
    return v


def _carregar_manifesto(path, sha_esperado=None, chave_status=None, valor_status=None):
    if not os.path.isfile(path):
        raise FalhaMetodologica(
            f"manifesto ausente: {path}. A descoberta por varredura é proibida.")
    m = json.load(open(path))
    declarado = m.get("manifest_sha256")
    if declarado is None:
        raise FalhaMetodologica(f"{os.path.basename(path)}: sem manifest_sha256.")
    recalc = _sha_manifesto(m)
    if declarado != recalc:
        raise FalhaMetodologica(
            f"{os.path.basename(path)}: SHA-256 não confere. "
            f"declarado={declarado} recalculado={recalc}. Manifesto alterado após congelamento.")
    if sha_esperado and declarado != sha_esperado:
        raise FalhaMetodologica(
            f"{os.path.basename(path)}: hash {declarado} difere do exigido {sha_esperado}.")
    if chave_status and m.get(chave_status) != valor_status:
        raise FalhaMetodologica(
            f"{os.path.basename(path)}: {chave_status} = {m.get(chave_status)!r}, "
            f"exigido {valor_status!r}.")
    log.info(f"  manifesto {os.path.basename(path)} verificado · sha256={declarado[:16]}…")
    return m


_RELPATH = {}
PONTEIRO_ANOT = os.path.join(BASE, "runs", "CURRENT_LIDC_ANNOTATIONS.json")
CONS_LIDC_PROIBIDO = os.path.join(BASE, "annotations", "consensus", "lidc")
ESQUEMA_PONTEIRO = {"run_id", "script", "published_at", "annotations_manifest",
                    "manifest_sha256", "consensus_dir", "per_reader_dir",
                    "object_table", "report", "gates"}
_ANOT = {}
_FORA_COORTE_COM_MASCARAS = {}


def _resolver_ponteiro_anotacoes():
    """
    Resolve o produto canónico das anotações LIDC ANTES de abrir qualquer
    anotação. Ponteiro ausente, hash divergente ou esquema desconhecido são
    falha — nunca se volta ao caminho antigo.
    """
    if not os.path.isfile(PONTEIRO_ANOT):
        raise FalhaMetodologica(
            f"ponteiro das anotações LIDC ausente: {PONTEIRO_ANOT}. "
            "Sem ele o Script 06 NÃO recorre a annotations/consensus/lidc.")
    p = json.load(open(PONTEIRO_ANOT))
    desconhecidas = set(p) - ESQUEMA_PONTEIRO
    em_falta = ESQUEMA_PONTEIRO - set(p)
    if desconhecidas or em_falta:
        raise FalhaMetodologica(
            f"esquema do ponteiro desconhecido: a mais {sorted(desconhecidas)}, "
            f"em falta {sorted(em_falta)}")
    mp = os.path.join(BASE, p["annotations_manifest"])
    if not os.path.isfile(mp):
        raise FalhaMetodologica(f"manifesto do 04 ausente: {mp}")
    m = json.load(open(mp))
    corpo = hashlib.sha256(json.dumps(
        {k: v for k, v in m.items() if k != "manifest_sha256"},
        indent=1, sort_keys=True).encode()).hexdigest()
    if corpo != m.get("manifest_sha256"):
        raise FalhaMetodologica(
            f"corpo do manifesto do 04 nao confere: {corpo} != {m.get('manifest_sha256')}")
    if m.get("manifest_sha256") != p.get("manifest_sha256"):
        raise FalhaMetodologica("ponteiro e manifesto do 04 declaram hashes diferentes")
    if m.get("SCRIPT04_OUTPUT_STATUS") != "PASS":
        raise FalhaMetodologica(
            f"Script 04 em {m.get('SCRIPT04_OUTPUT_STATUS')}")
    cd = os.path.join(BASE, p["consensus_dir"])
    if not os.path.isdir(cd):
        raise FalhaMetodologica(f"diretório de consenso ausente: {cd}")
    tab = os.path.join(BASE, p["object_table"])
    pr = os.path.join(BASE, p["per_reader_dir"])
    for q, r in ((tab, "tabela de objetos"), (pr, "máscaras por leitor")):
        if not os.path.exists(q):
            raise FalhaMetodologica(f"{r} ausente no produto apontado: {q}")
    _ANOT.update({"pointer": p, "pointer_sha256": sha256_file(PONTEIRO_ANOT),
                  "manifest": m, "manifest_sha256": m["manifest_sha256"],
                  "body_sha256": corpo, "consensus_dir": cd,
                  "object_table": tab, "per_reader": pr})
    log.info("  anotações LIDC via ponteiro | run=%s | consenso=%s"
             % (p["run_id"], p["consensus_dir"]))
    return _ANOT


def _entradas_lidc():
    """
    O conjunto LIDC vem do manifesto de promoção do estágio 04 — 1 002 linhas —
    e nunca de `os.listdir`. As 21 exclusões entram como `SKIPPED_EXCLUDED`: são
    registadas e não processadas.
    """
    a = _resolver_ponteiro_anotacoes()
    linhas = a["manifest"]["rows"]
    obrig = {"patient_id", "has_consensus", "disposition", "status_04"}
    faltam = obrig - set(linhas[0].keys())
    if faltam:
        raise FalhaMetodologica(
            f"manifesto do 04 sem colunas obrigatórias: {sorted(faltam)}")
    if len(linhas) != int(a["manifest"]["n_rows"]):
        raise FalhaMetodologica(
            f"manifesto do 04 com {len(linhas)} linhas != {a['manifest']['n_rows']}")
    ids = [r["patient_id"] for r in linhas]
    if len(ids) != len(set(ids)):
        raise FalhaMetodologica("manifesto do 04 com identificadores repetidos")
    entradas = []
    for r in linhas:
        pid = r["patient_id"]
        entradas.append({
            "dataset": "lidc_idri", "case_id": pid,
            "split": "unassigned",
            "has_consensus": bool(r["has_consensus"]),
            "negative_documented": (r.get("negative_documented") is True
                                    or r.get("status_04") == "NEGATIVE_DOCUMENTED"),
            "n_per_reader_expected": None,
            "analytic_status": r["status_04"],
            "skipped_excluded": not bool(r["has_consensus"]),
            "exclusion_reason": r.get("exclusion_reason", ""),
        })
    obt = {"SOURCE_ROWS": len(entradas),
           "SKIPPED_EXCLUDED": sum(1 for e in entradas if e["skipped_excluded"]),
           "OUTPUT_ELIGIBLE_IDS": sum(1 for e in entradas if not e["skipped_excluded"])}
    esp = {"SOURCE_ROWS": int(a["manifest"]["n_rows"]),
           "SKIPPED_EXCLUDED": int(a["manifest"]["n_rows"])
                               - int(a["manifest"]["n_consensus_published"]),
           "OUTPUT_ELIGIBLE_IDS": int(a["manifest"]["n_consensus_published"])}
    div = {k: (obt[k], v) for k, v in esp.items() if obt[k] != v}
    if div:
        raise FalhaMetodologica(f"contagens LIDC divergem do manifesto: {div}")
    log.info("  LIDC: %d linhas | processar %d | sem referencia %d | coorte por atribuir"
             % (obt["SOURCE_ROWS"], obt["OUTPUT_ELIGIBLE_IDS"], obt["SKIPPED_EXCLUDED"]))
    return entradas, a["manifest"]


def _entradas_lndb():
    m = _carregar_manifesto(MANIFESTO_LNDB, None,
                            "SCRIPT05_OUTPUT_MANIFEST_FROZEN", "YES")
    if m.get("ANNOTATION_MERGE_STATUS") != "NOT_PERFORMED":
        raise FalhaMetodologica("LNDb: ANNOTATION_MERGE_STATUS != NOT_PERFORMED")
    entradas = []
    _FORA = ("EXCLUDED_DUPLICATE_IMAGE_PRESERVED",
             "EXCLUDED_REFERENCE_READER_DISCORDANCE")
    for r in m["rows"]:
        if r["analytic_status"] in _FORA:
            continue
        entradas.append({
            "dataset": "lndb", "case_id": r["lndb_id"], "split": "external",
            "has_consensus": True,
            "n_per_reader_expected": int(r["n_per_reader_masks"]),
            "analytic_status": r["analytic_status"],
        })
    return entradas, m


def _entradas_rider():
    """
    O ficheiro do RIDER segue outra convenção de integridade: não tem auto-hash
    (`manifest_sha256`), e sim `pair_manifest_sha256`, que fixa o CSV de pares.
    A âncora congelada é portanto o CSV, não o JSON. O hash do próprio JSON é
    calculado e registado para proveniência, mas não existe valor congelado
    contra o qual compará-lo — assimetria declarada, não equiparada às outras duas.
    """
    if not os.path.isfile(MANIFESTO_RIDER):
        raise FalhaMetodologica(f"manifesto RIDER ausente: {MANIFESTO_RIDER}")
    m = json.load(open(MANIFESTO_RIDER))
    if m.get("pair_selection_version") != "2026-07-30.acquisition-level.v1":
        raise FalhaMetodologica(
            f"RIDER: pair_selection_version = {m.get('pair_selection_version')!r} inesperada.")
    _csv_pares = os.path.join(BASE, "analysis", "rider_pair_manifest.csv")
    if not os.path.isfile(_csv_pares):
        raise FalhaMetodologica(f"CSV de pares RIDER ausente: {_csv_pares}")
    _sha_csv = sha256_file(_csv_pares)
    if m.get("pair_manifest_sha256") != _sha_csv:
        raise FalhaMetodologica(
            f"RIDER: pair_manifest_sha256 declarado {m.get('pair_manifest_sha256')} "
            f"!= CSV em disco {_sha_csv}")
    observado = sha256_file(MANIFESTO_RIDER)
    m["manifest_sha256"] = observado
    m["RIDER_PAIR_SELECTION_SHA256_OBSERVED"] = observado
    m["RIDER_PAIR_SELECTION_MATCH"] = "PASS"
    m["PAIR_MANIFEST_SHA256_MATCH"] = "PASS"
    m["integrity_anchor"] = "SELF_CONSISTENT_02_MANIFEST_VS_PAIR_CSV"
    log.info(f"  rider_pair_selection.json · JSON={observado[:16]}… "
             f"· CSV de pares={_sha_csv[:16]}… · auto-consistente")
    csv_path = os.path.join(BASE, "analysis", "rider_pair_manifest.csv")
    if not os.path.isfile(csv_path):
        raise FalhaMetodologica(f"manifesto de pares RIDER ausente: {csv_path}")
    sha = sha256_file(csv_path)
    if sha != m["pair_manifest_sha256"]:
        raise FalhaMetodologica(
            f"rider_pair_manifest.csv: sha256 {sha} difere do congelado "
            f"{m['pair_manifest_sha256']}")
    entradas = []
    for r in csv.DictReader(open(csv_path, newline="")):
        if r["pair_status"] != "PASS_PAIR_SELECTION":
            continue
        for sufixo in ("_1", "_2"):
            entradas.append({
                "dataset": "rider", "case_id": r["patient_id"] + sufixo,
                "split": "test_retest", "has_consensus": False,
                "n_per_reader_expected": 0, "analytic_status": "ANALYTIC_UNIT",
            })
    return entradas, m


def _geom(img):
    return {"size": tuple(img.GetSize()), "spacing": tuple(img.GetSpacing()),
            "origin": tuple(img.GetOrigin()), "direction": tuple(img.GetDirection())}


def _geom_igual(a, b):
    if a["size"] != b["size"]:
        return False, f"tamanho {a['size']} vs {b['size']}"
    for nome, tol in (("spacing", GEOM_TOL_MM), ("origin", GEOM_TOL_MM),
                      ("direction", DIR_TOL)):
        va, vb = np.asarray(a[nome], float), np.asarray(b[nome], float)
        if not np.allclose(va, vb, atol=tol, rtol=0.0):
            return False, f"{nome} difere em {float(np.max(np.abs(va - vb))):.3e}"
    return True, "ok"


def _verificar_ct(arr, case_id):
    if not np.isfinite(arr).all():
        n_nan = int(np.isnan(arr).sum()); n_inf = int(np.isinf(arr).sum())
        raise FalhaMetodologica(f"{case_id}: CT com {n_nan} NaN e {n_inf} Inf")


def _verificar_binaria(arr, case_id, rotulo):
    u = np.unique(arr)
    if not np.isin(u, (0, 1)).all():
        raise FalhaMetodologica(f"{case_id}/{rotulo}: não binária, valores={u[:8]}")


_INFERER = None
_LUNGMASK_INFO = None


_ONLY = False


def _lungmask_info():
    global _LUNGMASK_INFO
    if _LUNGMASK_INFO is not None:
        return _LUNGMASK_INFO
    try:
        import importlib.metadata as md
        import lungmask
        import torch
    except ImportError as e:
        raise FalhaMetodologica(f"LungMask/torch indisponíveis: {e}")

    pesos, pesos_sha = "NOT_LOCATED", "NOT_LOCATED"
    cache = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")
    if os.path.isdir(cache):
        cand = sorted(f for f in os.listdir(cache) if "231" in f or "unet" in f.lower())
        if cand:
            pesos = os.path.join(cache, cand[0])
            pesos_sha = sha256_file(pesos)

    global ACCELERATOR_TORCH_TYPE
    mps_ok = bool(getattr(torch.backends, "mps", None)
                  and torch.backends.mps.is_available())
    cuda_ok = bool(torch.cuda.is_available())
    if cuda_ok:
        ACCELERATOR_TORCH_TYPE = "cuda"
    elif mps_ok:
        ACCELERATOR_TORCH_TYPE = "mps"
    else:
        raise FalhaMetodologica(
            "nenhum acelerador disponivel: cuda e mps ambos False. A execucao "
            "nao cai para CPU em silencio.")

    _LUNGMASK_INFO = {
        "ACCELERATOR_REQUESTED": ACCELERATOR_REQUESTED,
        "ACCELERATOR_RESOLVED": ACCELERATOR_TORCH_TYPE.upper(),
        "gpu_name": (torch.cuda.get_device_name(0) if cuda_ok else "APPLE_MPS"),
        "CUDA_AVAILABLE": cuda_ok,
        "MPS_AVAILABLE": mps_ok,
        "ACCELERATOR_EFFECTIVELY_USED": None,
        "lungmask_version": md.version("lungmask"),
        "lungmask_model": LUNGMASK_MODEL,
        "weights_path": pesos,
        "weights_sha256": pesos_sha,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "simpleitk_version": sitk.Version_VersionString(),
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "fallback_api_used": False,
    }
    return _LUNGMASK_INFO


def _lungmask(ct):
    """
    Modelo FIXADO em R231. Sem fallback entre APIs: se LMInferer não existir, a
    execução falha. Um fallback silencioso poderia trocar modelo ou pesos e
    tornar as máscaras incomparáveis entre casos.
    """
    global _INFERER
    if _INFERER is None:
        try:
            from lungmask import LMInferer
        except ImportError as e:
            raise FalhaMetodologica(
                f"LMInferer indisponível ({e}). O fallback para lungmask.mask está "
                f"proibido: trocaria o modelo em silêncio.")
        _lungmask_info()
        _INFERER = LMInferer(modelname=LUNGMASK_MODEL, force_cpu=False, tqdm_disable=True)

        dev = getattr(_INFERER, "device", None)
        tipo = getattr(dev, "type", str(dev))
        if tipo != ACCELERATOR_TORCH_TYPE:
            raise FalhaMetodologica(
                f"troca silenciosa de acelerador: pedido {ACCELERATOR_TORCH_TYPE}, "
                f"LungMask usa {tipo!r}. Execução abortada.")
        if getattr(_INFERER, "force_cpu", None):
            raise FalhaMetodologica("LungMask com force_cpu ativo; acelerador exigido.")
        if getattr(_INFERER, "modelname", None) != LUNGMASK_MODEL:
            raise FalhaMetodologica(
                f"modelo {getattr(_INFERER, 'modelname', None)!r} != {LUNGMASK_MODEL!r}")
        _lungmask_info()["ACCELERATOR_EFFECTIVELY_USED"] = tipo.upper()
        _lungmask_info()["lungmask_batch_size"] = getattr(_INFERER, "batch_size", None)
        log.info(f"  acelerador confirmado no inferidor: {tipo.upper()} · "
                 f"modelo {LUNGMASK_MODEL} · force_cpu=False")

    seg = _INFERER.apply(ct)
    arr = sitk.GetArrayFromImage(seg) if isinstance(seg, sitk.Image) else np.asarray(seg)
    lung = (arr > 0).astype(np.uint8)
    img = sitk.GetImageFromArray(lung)
    img.CopyInformation(ct)
    return img, lung


def _caixa_pulmonar(lung_arr, spacing_xyz, case_id):
    """
    Devolve (bbox, registo_de_margem). O rótulo NÃO participa.
    bbox em ordem de array: (z0, z1, y0, y1, x0, x1).

    A margem pedida é de 10 mm. Converte-se para voxels por `ceil`, logo a margem
    nominal é ≥ 10 mm; junto à borda do volume pode ficar ABAIXO de 10 mm por
    limitação. Ambas as situações ficam registadas por face.
    """
    coords = np.argwhere(lung_arr > 0)
    if coords.size == 0:
        raise FalhaMetodologica(f"{case_id}: máscara pulmonar vazia")

    sp = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    forma = lung_arr.shape
    bbox, registo = [], {"requested_margin_mm": MARGIN_MM}

    for eixo, nome in enumerate(("z", "y", "x")):
        m_vox = int(np.ceil(MARGIN_MM / sp[eixo]))
        lo_pulmao = int(coords[:, eixo].min())
        hi_pulmao = int(coords[:, eixo].max())
        lo = max(lo_pulmao - m_vox, 0)
        hi = min(hi_pulmao + m_vox + 1, forma[eixo])
        aplicado_lo = lo_pulmao - lo
        aplicado_hi = (hi - 1) - hi_pulmao
        bbox += [lo, hi]
        registo[nome] = {
            "requested_margin_vox": m_vox,
            "applied_margin_vox_low": aplicado_lo,
            "applied_margin_vox_high": aplicado_hi,
            "applied_margin_mm_low": round(aplicado_lo * sp[eixo], 4),
            "applied_margin_mm_high": round(aplicado_hi * sp[eixo], 4),
            "clipped_by_volume_boundary_low": aplicado_lo < m_vox,
            "clipped_by_volume_boundary_high": aplicado_hi < m_vox,
            "spacing_mm": round(sp[eixo], 6),
        }
    return tuple(bbox), registo


def _recortar(img, bbox):
    z0, z1, y0, y1, x0, x1 = bbox
    arr = sitk.GetArrayFromImage(img)[z0:z1, y0:y1, x0:x1]
    out = sitk.GetImageFromArray(arr)
    out.SetOrigin(img.TransformIndexToPhysicalPoint((int(x0), int(y0), int(z0))))
    out.SetSpacing(img.GetSpacing())
    out.SetDirection(img.GetDirection())
    return out


def _conferir_inversa(ct, ct_crop, bbox, case_id):
    """A transformação inversa tem de reconstruir a posição física original."""
    z0, _, y0, _, x0, _ = bbox
    for idx in ((0, 0, 0), (1, 2, 3)):
        p_crop = np.asarray(ct_crop.TransformIndexToPhysicalPoint(idx), float)
        orig_idx = (int(idx[0] + x0), int(idx[1] + y0), int(idx[2] + z0))
        p_orig = np.asarray(ct.TransformIndexToPhysicalPoint(orig_idx), float)
        d = float(np.max(np.abs(p_crop - p_orig)))
        if d > INVERSE_TOL_MM:
            raise FalhaMetodologica(
                f"{case_id}: inversa do recorte desloca {d:.3e} mm no índice {idx}")


def _carregar_nomes_rotulos(entradas):
    """
    Os nomes das máscaras por leitor vêm das TABELAS DE OBJETOS congeladas pelos
    Scripts 04 e 05 — nunca de `os.listdir`. Listar o diretório seria descobrir
    anotações por varredura, e aceitaria silenciosamente ficheiros a mais.

    LIDC : coluna `mask_filename` da tabela do produto apontado
    LNDb : rad_{rad_id:02d}.nii.gz                          (lndb_nodules.csv)

    Para o LNDb a contagem derivada é conferida contra `n_per_reader_masks` do
    manifesto. Para o LIDC a fonte é a própria tabela do produto apontado, cujo
    hash está selado no manifesto de promoção.
    """
    tabelas = {
        "lidc_idri": _ANOT["object_table"],
        "lndb": os.path.join(BASE, "annotations", "object_tables", "lndb_nodules.csv"),
    }
    nomes, hashes = {}, {}
    _RELPATH.clear()
    for ds, path in tabelas.items():
        if not os.path.isfile(path):
            raise FalhaMetodologica(f"tabela de objetos ausente: {path}")
        hashes[ds] = sha256_file(path)
        for r in csv.DictReader(open(path, newline="")):
            if ds == "lidc_idri":
                cid = r["patient_id"]
                if r.get("mask_relpath"):
                    rel = r["mask_relpath"]
                    absoluto = (os.path.join(BASE, rel)
                                if rel.startswith("annotations")
                                else os.path.join(BASE, "runs", rel))
                elif r.get("mask_filename"):
                    absoluto = os.path.join(_ANOT["per_reader"], cid,
                                            r["mask_filename"])
                else:
                    raise FalhaMetodologica(
                        "tabela de objetos sem `mask_relpath` nem `mask_filename`")
                f = os.path.basename(absoluto)
                _RELPATH[(cid, f)] = absoluto
            else:
                if int(r["is_nodule"]) != 1:
                    continue
                cid = r["lndb_id"]
                f = f"rad_{int(r['rad_id']):02d}.nii.gz"
            nomes.setdefault(cid, set()).add(f)

    for e in entradas:
        if e["dataset"] == "rider":
            continue
        derivado = len(nomes.get(e["case_id"], set()))
        if e.get("skipped_excluded"):
            if derivado:
                _FORA_COORTE_COM_MASCARAS[e["case_id"]] = derivado
            continue
        if e["n_per_reader_expected"] is None:
            if (e["has_consensus"] and derivado == 0
                    and not e.get("negative_documented")):
                raise FalhaMetodologica(
                    f"{e['case_id']}: tem consenso mas a tabela não deriva máscara alguma.")
            if e.get("negative_documented") and derivado:
                raise FalhaMetodologica(
                    f"{e['case_id']}: negativo documentado mas a tabela deriva "
                    f"{derivado} máscaras — as fontes contradizem-se.")
            continue
        if derivado != e["n_per_reader_expected"]:
            raise FalhaMetodologica(
                f"{e['case_id']}: tabela de objetos deriva {derivado} máscaras por leitor, "
                f"manifesto declara {e['n_per_reader_expected']}.")
    log.info(f"  nomes de rótulos derivados das tabelas congeladas · "
             f"lidc_nodules.csv={hashes['lidc_idri'][:16]}… "
             f"lndb_nodules.csv={hashes['lndb'][:16]}…")
    return nomes, hashes


def _fontes_de_rotulo(e, nomes_por_caso):
    """Caminhos dos rótulos, derivados de manifesto e tabela. Nunca por varredura."""
    chave = ANNOT_SUBDIR[e["dataset"]]
    if chave is None:
        return None, []
    cid = e["case_id"]
    if e["dataset"] == "lidc_idri":
        cons = (os.path.join(_ANOT["consensus_dir"], cid, "consensus.nii.gz")
                if e["has_consensus"] else None)
        d = None
    else:
        cons = (os.path.join(CONS_DIR, chave, cid, "consensus.nii.gz")
                if e["has_consensus"] else None)
        d = os.path.join(PER_READER, chave, cid)
    if e["dataset"] == "lidc_idri":
        per = [_RELPATH[(cid, f)]
               for f in sorted(nomes_por_caso.get(cid, set()))]
    else:
        per = [os.path.join(d, f) for f in sorted(nomes_por_caso.get(cid, set()))]
    return cons, per


def processar(e, dest, plan_only, nomes_rotulos):
    cid, ds = e["case_id"], e["dataset"]
    ct_path = os.path.join(NIFTI_DIR, ds, cid, "ct.nii.gz")
    if not os.path.isfile(ct_path):
        raise FalhaMetodologica(f"{cid}: CT ausente em {ct_path}")

    linha = {"case_id": cid, "dataset": ds, "split": e["split"],
             "analytic_status": e["analytic_status"],
             "sha256_ct_source": sha256_file(ct_path)}

    cons_src, per_src = _fontes_de_rotulo(e, nomes_rotulos)
    if e["has_consensus"] and cons_src and not os.path.isfile(cons_src):
        raise FalhaMetodologica(f"{cid}: manifesto declara consenso mas {cons_src} não existe")
    for p in per_src:
        if not os.path.isfile(p):
            raise FalhaMetodologica(f"{cid}: máscara por leitor declarada e ausente: {p}")

    if plan_only:
        sem_seg = ds == "lidc_idri" and not e["has_consensus"]
        linha.update({
            "status": "PLAN_OK",
            "seg_status": "NO_SEG_PENDING_XML_AUDIT" if sem_seg else "HAS_CONSENSUS",
            "n_labels_declared": len(per_src) + int(bool(cons_src)),
            "truncation_status": ("NOT_AUDITABLE_NO_SEG_PENDING_XML_AUDIT" if sem_seg
                                  else "PENDING_CANONICAL_RUN"),
        })
        return linha

    ct = sitk.ReadImage(ct_path)
    ct_arr = sitk.GetArrayFromImage(ct)
    _verificar_ct(ct_arr, cid)
    g_ct = _geom(ct)

    lung_img, lung_arr = _lungmask(ct)
    _verificar_binaria(lung_arr, cid, "lung_mask")
    ok, motivo = _geom_igual(_geom(lung_img), g_ct)
    if not ok:
        raise FalhaMetodologica(f"{cid}: máscara pulmonar não alinha com a CT — {motivo}")
    n_lung = int(lung_arr.sum())
    if n_lung == 0:
        raise FalhaMetodologica(f"{cid}: máscara pulmonar vazia")

    bbox, margem = _caixa_pulmonar(lung_arr, ct.GetSpacing(), cid)
    bbox_congelada = tuple(bbox)

    rotulos = []
    if cons_src:
        rotulos.append(("consensus.nii.gz", cons_src))
    for p in per_src:
        rotulos.append((os.path.basename(p), p))

    dir_img = os.path.join(dest, ds, "images", cid)
    dir_lbl = os.path.join(dest, ds, "labels", cid)
    dir_lm  = os.path.join(dest, "lung_masks", ds, cid)
    for d in (dir_img, dir_lm):
        os.makedirs(d, exist_ok=True)

    ct_crop = _recortar(ct, bbox_congelada)
    _conferir_inversa(ct, ct_crop, bbox_congelada, cid)
    g_crop = _geom(ct_crop)
    sitk.WriteImage(ct_crop, os.path.join(dir_img, "ct.nii.gz"), useCompression=True)

    lm_crop = _recortar(lung_img, bbox_congelada)
    ok, motivo = _geom_igual(_geom(lm_crop), g_crop)
    if not ok:
        raise FalhaMetodologica(f"{cid}: lung_mask recortada desalinha do CT recortado — {motivo}")
    sitk.WriteImage(lung_img, os.path.join(dir_lm, "lung_mask.nii.gz"), useCompression=True)
    sitk.WriteImage(lm_crop, os.path.join(dir_lm, "lung_mask_crop.nii.gz"), useCompression=True)

    truncados, n_lbl = 0, 0
    if rotulos:
        os.makedirs(dir_lbl, exist_ok=True)
    for nome, src in rotulos:
        m = sitk.ReadImage(src)
        ok, motivo = _geom_igual(_geom(m), g_ct)
        if not ok:
            raise FalhaMetodologica(f"{cid}/{nome}: rótulo não alinha com a CT — {motivo}")
        m_arr = sitk.GetArrayFromImage(m)
        _verificar_binaria(m_arr, cid, nome)
        antes = int((m_arr > 0).sum())
        c = _recortar(m, bbox_congelada)
        c_arr = sitk.GetArrayFromImage(c)
        depois = int((c_arr > 0).sum())
        fora = antes - depois
        truncados += fora
        if fora > 0:
            raise FalhaMetodologica(
                f"{cid}/{nome}: FAIL_TRUNCATION — {fora} voxels fora da caixa pulmonar "
                f"({antes} antes, {depois} depois). A caixa NÃO é ampliada.")
        ok, motivo = _geom_igual(_geom(c), g_crop)
        if not ok:
            raise FalhaMetodologica(f"{cid}/{nome}: recorte desalinha do CT recortado — {motivo}")
        sitk.WriteImage(c, os.path.join(dir_lbl, nome), useCompression=True)
        n_lbl += 1

    if ds == "lidc_idri" and not e["has_consensus"]:
        trunc_status = "NOT_AUDITABLE_NO_SEG_PENDING_XML_AUDIT"
        seg_status = "NO_SEG_PENDING_XML_AUDIT"
    else:
        trunc_status = "PASS"
        seg_status = "HAS_CONSENSUS"

    transform = {
        "case_id": cid, "dataset": ds, "script_version": SCRIPT06_VERSION,
        "original_shape": list(ct_arr.shape),
        "original_origin": list(ct.GetOrigin()),
        "original_spacing": list(ct.GetSpacing()),
        "original_direction": list(ct.GetDirection()),
        "crop_bbox": list(bbox_congelada),
        "cropped_shape": [bbox[1] - bbox[0], bbox[3] - bbox[2], bbox[5] - bbox[4]],
        "cropped_origin": list(ct_crop.GetOrigin()),
        "margin": margem,
        "reference_guided_crop": REFERENCE_GUIDED_CROP,
        "bbox_depends_on": ["ct", "lungmask", "margin_mm"],
        "voxelwise_lung_filter": VOXELWISE_LUNG_FILTER,
        "labels_loaded_after_bbox_frozen": True,
    }
    with open(os.path.join(dir_img, "crop_transform.json"), "w") as f:
        json.dump(transform, f, indent=2, sort_keys=True)

    linha.update({
        "status": "PROCESSED", "seg_status": seg_status,
        "n_lung_vox": n_lung, "n_labels": n_lbl,
        "truncated_voxels": truncados, "truncation_status": trunc_status,
        "original_shape": str(list(ct_arr.shape)),
        "cropped_shape": str(transform["cropped_shape"]),
        "crop_bbox": str(list(bbox_congelada)),
        "reference_guided_crop": REFERENCE_GUIDED_CROP,
        "clipped_any_face": any(margem[a][k] for a in "zyx"
                                for k in ("clipped_by_volume_boundary_low",
                                          "clipped_by_volume_boundary_high")),
        "sha256_ct_crop": sha256_file(os.path.join(dir_img, "ct.nii.gz")),
        "sha256_lung_mask_crop": sha256_file(os.path.join(dir_lm, "lung_mask_crop.nii.gz")),
    })
    return linha


def _exigir_livre(dest):
    if os.path.exists(dest):
        raise FalhaMetodologica("staging preexistente: %s" % dest)


def _quarentenar_incompleta(dest, run_id, motivo, feitos, esperado):
    """
    Move uma execucao incompleta para quarentena e regista INVALID.json.

    A v1 nao tinha isto: ao interromper a execucao das 17:26, 29,5 GB ficaram em
    `staging/` e tiveram de ser movidos a mao. Nenhum ponteiro canonico e escrito
    em caso algum — `CURRENT.json` so existe no fim, apos a auditoria integral.
    """
    if not os.path.isdir(dest):
        return None
    raiz = os.path.join(BASE, "quarantine", "script06_aborted_runs", run_id)
    os.makedirs(os.path.dirname(raiz), exist_ok=True)
    if os.path.exists(raiz):
        raiz = raiz + "_" + time.strftime("%H%M%S")
    os.makedirs(raiz, exist_ok=True)
    n = sum(len(f) for _d, _s, f in os.walk(dest))
    tot = sum(os.path.getsize(os.path.join(d, x))
              for d, _s, f in os.walk(dest) for x in f
              if os.path.isfile(os.path.join(d, x)))
    with open(os.path.join(raiz, "INVALID.json"), "w") as fh:
        json.dump({"record": "INVALID", "run_id": run_id, "script": "06",
                   "script_version": SCRIPT06_VERSION,
                   "status": motivo, "reuse": "FORBIDDEN",
                   "publication": "NEVER_PUBLISHED",
                   "processed": "%d/%d" % (feitos, esperado),
                   "n_files": n, "total_bytes": tot,
                   "CURRENT_JSON_ABSENT_CONFIRMED": not os.path.isfile(CURRENT_JSON),
                   "quarantined_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
                  fh, indent=1, sort_keys=True)
    os.rename(dest, os.path.join(raiz, "payload"))
    log.warning("  execucao incompleta movida para quarentena: %s (%d ficheiros, %.1f GB)"
                % (raiz, n, tot / 1e9))
    return raiz


def _montar_manifesto(run_id, linhas, m_lidc, m_lndb, m_rider, hashes_tabelas,
                      lm_info, dest, ent_lidc, ent_lndb, ent_rider, entradas):
    """
    Constroi o manifesto do Script 06. Extraida de `main()` para poder ser
    exercitada em `--plan-only`: um desencontro entre o que um manifesto a
    montante emite e o que este consome tem de aparecer ANTES das seis horas de
    processamento, nao depois.
    """
    n_no_seg = sum(1 for e in ent_lidc
               if not e.get("skipped_excluded") and not e["has_consensus"])
    ids = [e["case_id"] for e in entradas]
    _n_prod = sum(1 for r in linhas if r.get("status") != "SKIPPED_EXCLUDED")
    _ids_exc = {r["case_id"] for r in linhas if r.get("status") == "SKIPPED_EXCLUDED"}
    _n_out_exc = 0
    for _sub in ("images", "labels"):
        _d = os.path.join(dest, "lidc_idri", _sub)
        if os.path.isdir(_d):
            _n_out_exc += sum(1 for c in os.listdir(_d) if c in _ids_exc)
    _d = os.path.join(dest, "lung_masks", "lidc_idri")
    if os.path.isdir(_d):
        _n_out_exc += sum(1 for c in os.listdir(_d) if c in _ids_exc)
    if _n_out_exc:
        raise FalhaMetodologica(
            f"{_n_out_exc} saídas encontradas para casos excluídos: "
            f"{sorted(_ids_exc)[:5]}")
    esperado_prod = len(linhas) - len(_ids_exc)
    if _n_prod != esperado_prod:
        raise FalhaMetodologica(
            f"produtos {_n_prod} != esperado {esperado_prod} "
            f"({len(linhas)} linhas − {len(_ids_exc)} excluídos)")

    manifesto = {
        "manifest_id": f"script06_{run_id}",
        "run_id": run_id, "script_version": SCRIPT06_VERSION,
        "input_manifest_sha256": {
            "lidc": _exigir_campo(m_lidc, "manifest_sha256", "manifesto 04"),
            "lndb": _exigir_campo(m_lndb, "manifest_sha256", "manifesto 05"),
            "rider": _exigir_campo(m_rider, "manifest_sha256", "manifesto RIDER"),
            "lidc_nodules_csv": hashes_tabelas["lidc_idri"],
            "lndb_nodules_csv": hashes_tabelas["lndb"],
            "lidc_annotations_pointer": _ANOT.get("pointer_sha256", ""),
            "lidc_annotations_manifest": _ANOT.get("manifest_sha256", ""),
        },
        "configuration": {
            "margin_mm": MARGIN_MM, "voxelwise_lung_filter": VOXELWISE_LUNG_FILTER,
            "reference_guided_crop": REFERENCE_GUIDED_CROP,
            "lungmask": lm_info, "binarization_rule": "lungmask_label > 0",
        },
        "SCRIPT06_INPUT_MANIFEST_MATCH": "PASS",
        "SCRIPT06_ANALYTIC_INPUT_COUNT": len(ent_lidc),
        "SCRIPT06_DEVELOPMENT_COUNT": sum(1 for e in ent_lidc if e["split"] == "development"),
        "SCRIPT06_INTERNAL_COUNT": sum(1 for e in ent_lidc if e["split"] == "internal"),
        "SCRIPT06_LNDB_COUNT": len(ent_lndb),
        "SCRIPT06_RIDER_COUNT": len(ent_rider),
        "SCRIPT06_TOTAL_COUNT": len(entradas),
        "SCRIPT06_EXCLUDED_DUPLICATE_READ": 0,
        "SCRIPT06_ORPHAN_CONSENSUS_READ": 0,
        "SCRIPT06_REFERENCE_GUIDED_CROP": REFERENCE_GUIDED_CROP,
        "SCRIPT06_MARGIN_MM": MARGIN_MM,
        "SCRIPT06_MANIFEST_ROWS": len(linhas),
        "SCRIPT06_NEGATIVE_DOCUMENTED": sum(
            1 for e in ent_lidc if e.get("negative_documented")),
        "SCRIPT06_SKIPPED_EXCLUDED": sum(1 for r in linhas
                                         if r.get("status") == "SKIPPED_EXCLUDED"),
        "ANNOTATION_POINTER_SHA256": _ANOT.get("pointer_sha256", ""),
        "PUBLICATION_MANIFEST_SHA256": _ANOT.get("manifest_sha256", ""),
        "PUBLICATION_BODY_SHA256": _ANOT.get("body_sha256", ""),
        "PRODUCER_ROUTE": "XML",
        "LUNGMASK_INFERENCE_REUSED": False,
        "LUNGMASK_REUSE_POLICY": "REMOVIDO — mascara sempre inferida nesta execucao",
        "LEGACY_FALLBACK_USED": False,
        "SCRIPT06_CT_CROPS": sum(1 for r in linhas
                                 if r.get("status") != "SKIPPED_EXCLUDED"),
        "SCRIPT06_LUNG_MASKS": _n_prod,
        "SCRIPT06_CROP_TRANSFORMS": _n_prod,
        "ELIGIBLE_OUTPUTS": _n_prod,
        "OUTPUTS_FOR_EXCLUDED": _n_out_exc,
        "POINTER_04": _ANOT.get("pointer_sha256", ""),
        "SCRIPT06_NO_SEG_PENDING_XML_AUDIT": n_no_seg,
        "SCRIPT06_TRUNCATED_VOXELS": sum(r.get("truncated_voxels", 0) for r in linhas),
        "SCRIPT06_UNDOCUMENTED_FAILURES": 0,
        "SCRIPT06_OUTPUT_STATUS": "PASS",
        "SCRIPT06_OUTPUT_MANIFEST_FROZEN": "YES",
        "LEGACY_LUNG_MASK_CANONICAL_REUSE": "NO",
        "LEGACY_LUNG_CROP_AUDIT_STATUS": "INVALID_FABRICATED",
        "expected_ids": sorted(ids), "observed_ids": sorted(r["case_id"] for r in linhas),
        "missing_ids": sorted(set(ids) - {r["case_id"] for r in linhas}),
        "extra_ids": sorted({r["case_id"] for r in linhas} - set(ids)),
        "duplicate_ids": [],
        "forbidden_ids": "REMOVIDO — exclusoes derivadas dos manifestos e diagnosticos",
        "OUT_OF_COHORT_WITH_READER_MASKS": dict(sorted(_FORA_COORTE_COM_MASCARAS.items())),
        "OUT_OF_COHORT_WITH_READER_MASKS_N": len(_FORA_COORTE_COM_MASCARAS),
        "COHORT_MEMBERSHIP_SOURCE": ("manifesto do Script 04 resolvido por ponteiro "
                                     "e conferido por hash — NUNCA a presenca de "
                                     "ficheiros em annotations/"),
        "rows": linhas,
    }
    return manifesto


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", metavar="IDS", default=None,
                    help="lista de case_id separada por virgula (ensaio)")
    ap.add_argument("--plan-only", action="store_true",
                    help="valida manifestos e entradas; não executa LungMask nem escreve")
    args = ap.parse_args()
    check_base(strict=True)

    run_id = f"s06_{time.strftime('%Y%m%dT%H%M%S')}_{os.getpid()}"
    log.info(f"Script 06 {SCRIPT06_VERSION} · run_id={run_id} · "
             f"modo={'PLAN-ONLY' if args.plan_only else 'CANÓNICO'}")

    log.info("verificando manifestos congelados")
    ent_lidc, m_lidc = _entradas_lidc()
    ent_lndb, m_lndb = _entradas_lndb()
    ent_rider, m_rider = _entradas_rider()
    entradas = ent_lidc + ent_lndb + ent_rider

    if args.only:
        globals()["_ONLY"] = True
        alvo = {x.strip() for x in args.only.split(",") if x.strip()}
        entradas = [e for e in entradas if e["case_id"] in alvo]
        ent_lidc = [e for e in ent_lidc if e["case_id"] in alvo]
        ent_lndb = [e for e in ent_lndb if e["case_id"] in alvo]
        ent_rider = [e for e in ent_rider if e["case_id"] in alvo]
        log.info(f"  --only: {len(entradas)} casos (ENSAIO, nao publica)")

    esperado_por_ds = {
        "lidc_idri": int(m_lidc["n_rows"]),
        "lndb": int(m_lndb["SCRIPT05_ANALYTIC_COUNT"]),
        "rider": sum(1 for e in ent_rider if True),
    }
    if not args.only:
        for ds, esperado in esperado_por_ds.items():
            obtido = sum(1 for e in entradas if e["dataset"] == ds)
            if obtido != esperado:
                raise FalhaMetodologica(
                    f"{ds}: {obtido} entradas, manifesto declara {esperado}")
    ids = [e["case_id"] for e in entradas]
    if len(set(ids)) != len(ids):
        dup = [k for k, v in Counter(ids).items() if v > 1]
        raise FalhaMetodologica(f"identificadores repetidos: {dup}")
    log.info(f"  entradas: LIDC {len(ent_lidc)} · LNDb {len(ent_lndb)} · "
             f"RIDER {len(ent_rider)} · total {len(entradas)}")
    log.info("  contagens conferidas contra os manifestos: %s" % esperado_por_ds)

    nomes_rotulos, hashes_tabelas = _carregar_nomes_rotulos(entradas)

    dest = os.path.join(RUNS_ROOT, run_id)
    if not args.plan_only:
        _exigir_livre(dest)
        os.makedirs(dest, exist_ok=False)
        lm_info = _lungmask_info()
        du = shutil.disk_usage(BASE)
        lm_info["AVAILABLE_DISK_BEFORE_RUN_BYTES"] = du.free
        lm_info["CANONICAL_CASES"] = len(entradas)
        log.info("  ── abertura da execução ──")
        for k, v in (("ACCELERATOR_REQUESTED", lm_info["ACCELERATOR_REQUESTED"]),
                     ("MPS_AVAILABLE", lm_info["MPS_AVAILABLE"]),
                     ("TORCH_VERSION", lm_info["torch_version"]),
                     ("LUNGMASK_VERSION", lm_info["lungmask_version"]),
                     ("LUNGMASK_MODEL", lm_info["lungmask_model"]),
                     ("CANONICAL_CASES", lm_info["CANONICAL_CASES"]),
                     ("AVAILABLE_DISK_BEFORE_RUN_BYTES", du.free)):
            log.info(f"    {k:<34s} = {v}")
    else:
        lm_info = {"status": "NOT_QUERIED_PLAN_ONLY"}

    linhas, falhas = [], []
    for i, e in enumerate(entradas, 1):
        if e.get("skipped_excluded"):
            linhas.append({"dataset": e["dataset"], "case_id": e["case_id"],
                           "split": e["split"], "status": "SKIPPED_EXCLUDED",
                           "seg_status": e["analytic_status"],
                           "analytic_status": e["analytic_status"],
                           "exclusion_reason": e["exclusion_reason"],
                           "crop_bbox": "", "cropped_shape": "", "original_shape": "",
                           "n_labels": 0, "n_lung_vox": 0, "truncated_voxels": 0,
                           "truncation_status": "NOT_APPLICABLE",
                           "clipped_any_face": False, "reference_guided_crop": "NO",
                           "sha256_ct_crop": "", "sha256_ct_source": "",
                           "sha256_lung_mask_crop": ""})
            continue
        try:
            linhas.append(processar(e, dest, args.plan_only, nomes_rotulos))
        except FalhaMetodologica as ex:
            log.error(f"  [FALHA] {ex}")
            falhas.append(str(ex))
        except Exception as ex:
            log.error(f"  [ERRO] {e['case_id']}: {type(ex).__name__}: {ex}")
            falhas.append(f"{e['case_id']}: {type(ex).__name__}: {ex}")
        if i % 50 == 0:
            log.info(f"  PROGRESSO {i}/{len(entradas)}")

    if falhas:
        log.error(f"{len(falhas)} falhas. Nada é publicado; staging preservado em {dest}")
        for f in falhas[:20]:
            log.error(f"  · {f}")
        sys.exit(1)

    n_no_seg = sum(1 for r in linhas if r.get("seg_status") == "NO_SEG_PENDING_XML_AUDIT")

    if args.plan_only:
        log.info("─" * 68)
        log.info("  PLAN-ONLY — nenhum produto escrito, nenhuma publicação")
        for k, v in (("plan_entries_validated", len(linhas)),
                     ("lidc", len(ent_lidc)), ("lndb", len(ent_lndb)),
                     ("rider", len(ent_rider)),
                     ("development", sum(1 for e in ent_lidc if e["split"] == "development")),
                     ("internal", sum(1 for e in ent_lidc if e["split"] == "internal")),
                     ("skipped_excluded", sum(1 for e in ent_lidc
                                              if e.get("skipped_excluded"))),
                     ("lidc_eligible", sum(1 for e in ent_lidc
                                           if not e.get("skipped_excluded"))),
                     ("development_eligible", sum(1 for e in ent_lidc
                                                  if not e.get("skipped_excluded")
                                                  and e["split"] == "development")),
                     ("internal_eligible", sum(1 for e in ent_lidc
                                               if not e.get("skipped_excluded")
                                               and e["split"] == "internal")),
                     ("no_seg_pending_xml_audit", n_no_seg),
                     ("undocumented_failures", 0)):
            log.info(f"  {k:<32s} = {v}")
        _proc = sum(1 for r in linhas if r.get("dataset") == "lidc_idri"
                    and r.get("status") not in ("SKIPPED_EXCLUDED",))
        _pub = int(_ANOT["manifest"]["n_consensus_published"])
        if not _ONLY and _proc != _pub:
            log.error(f"  LIDC processados = {_proc}, publicados pelo 04 = {_pub}")
            sys.exit(1)
        try:
            _dry = _montar_manifesto(run_id, linhas, m_lidc, m_lndb, m_rider,
                                     hashes_tabelas, lm_info, dest, ent_lidc,
                                     ent_lndb, ent_rider, entradas)
        except FalhaMetodologica:
            raise
        except Exception as ex:
            raise FalhaMetodologica(
                f"montagem do manifesto falharia na execucao integral: "
                f"{type(ex).__name__}: {ex}")
        log.info(f"  FINAL_MANIFEST_DRY_BUILD         = PASS "
                 f"({len(_dry)} chaves de topo)")
        log.info("  PLAN_STATUS                      = PASS")
        return

    manifesto = _montar_manifesto(run_id, linhas, m_lidc, m_lndb, m_rider,
                                  hashes_tabelas, lm_info, dest, ent_lidc,
                                  ent_lndb, ent_rider, entradas)
    if manifesto["missing_ids"] or manifesto["extra_ids"]:
        raise FalhaMetodologica(
            f"divergência de identificadores: faltam {manifesto['missing_ids'][:5]} "
            f"sobram {manifesto['extra_ids'][:5]}")

    manifesto["manifest_sha256"] = _sha_manifesto(manifesto)

    destino_final = dest
    if _ONLY:
        log.info("  ENSAIO (--only): ponteiro NAO publicado. Produto em %s" % dest)
        return 0
    with open(os.path.join(destino_final, "manifest.json"), "w") as f:
        json.dump(manifesto, f, indent=1, sort_keys=True)

    _mp = os.path.join(destino_final, "manifest.json")
    _contagens = {k: manifesto.get(k) for k in
                  ("SCRIPT06_ANALYTIC_INPUT_COUNT", "SCRIPT06_DEVELOPMENT_COUNT",
                   "SCRIPT06_INTERNAL_COUNT", "SCRIPT06_LNDB_COUNT",
                   "SCRIPT06_RIDER_COUNT", "SCRIPT06_TOTAL_COUNT",
                   "SCRIPT06_OUTPUT_STATUS") if manifesto.get(k) is not None}
    escrever_ponteiro(
        CURRENT_JSON, _mp,
        run_id=run_id, script="06", script_version=SCRIPT06_VERSION,
        published_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        crops_dir=os.path.relpath(destino_final, BASE),
        **_contagens)
    _alias = os.path.join(BASE, "runs", "CURRENT_LUNG_CROPS.json")
    escrever_ponteiro(
        _alias, _mp,
        run_id=run_id, script="06", script_version=SCRIPT06_VERSION,
        published_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        crops_dir=os.path.relpath(destino_final, BASE),
        supersedes_name="CURRENT.json", **_contagens)

    log.info("─" * 68)
    for k in ("SCRIPT06_INPUT_MANIFEST_MATCH", "SCRIPT06_ANALYTIC_INPUT_COUNT",
              "SCRIPT06_DEVELOPMENT_COUNT", "SCRIPT06_INTERNAL_COUNT",
              "SCRIPT06_LNDB_COUNT", "SCRIPT06_RIDER_COUNT", "SCRIPT06_TOTAL_COUNT",
              "SCRIPT06_EXCLUDED_DUPLICATE_READ", "SCRIPT06_ORPHAN_CONSENSUS_READ",
              "SCRIPT06_REFERENCE_GUIDED_CROP", "SCRIPT06_MARGIN_MM",
              "SCRIPT06_MANIFEST_ROWS", "SCRIPT06_SKIPPED_EXCLUDED",
              "PRODUCER_ROUTE", "LEGACY_FALLBACK_USED",
              "ELIGIBLE_OUTPUTS", "OUTPUTS_FOR_EXCLUDED", "POINTER_04",
              "SCRIPT06_CT_CROPS", "SCRIPT06_LUNG_MASKS", "SCRIPT06_CROP_TRANSFORMS",
              "SCRIPT06_NO_SEG_PENDING_XML_AUDIT", "SCRIPT06_TRUNCATED_VOXELS",
              "SCRIPT06_UNDOCUMENTED_FAILURES", "SCRIPT06_OUTPUT_STATUS",
              "SCRIPT06_OUTPUT_MANIFEST_FROZEN"):
        log.info(f"  {k:<36s} = {manifesto[k]}")
    log.info(f"  {'SCRIPT06_OUTPUT_MANIFEST_SHA256':<36s} = {manifesto['manifest_sha256']}")
    log.info(f"  publicado em {destino_final}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except FalhaMetodologica as e:
        log.error(f"FALHA METODOLOGICA: {e}")
        sys.exit(1)
    except Exception as e:
        log.exception(f"ERRO NAO PREVISTO: {type(e).__name__}: {e}")
        sys.exit(1)
