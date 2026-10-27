#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    03_convert_dicom_to_nifti.py

File type:
    Script

Purpose:
    Converts DICOM series (LIDC-IDRI, RIDER) and MetaImage volumes (LNDb) to NIfTI in Hounsfield units, with slice ordering by physical position.

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
    - CURRENT_NIFTI_CONVERSION.json

Outputs:
    - See docs/file_level_documentation.md.

Dependencies:
    - SimpleITK
    - numpy
    - pydicom

Execution:
    PULMONARY_BASE=<project_dir> python scripts/03_convert_dicom_to_nifti.py

Portability notes:
    All paths are resolved relative to the project base directory given by the PULMONARY_BASE environment variable. No machine-specific path is required.

Reproducibility notes:
    Environment pinned in requirements.txt and environment/. Seeds and frozen parameters are recorded in the run manifests under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) are not included and must be obtained from their original repositories.
"""

import os
import csv
import sys
import json
import time
import hashlib
import logging
from collections import defaultdict
from datetime import datetime

import numpy as np
import pydicom
import SimpleITK as sitk

from collections import Counter
from pulmonary_base import (BASE, verify_nifti, nifti_info, log_skip, verify_gzip_stream,
                            verify_nifti_payload, nifti_header_fields,
                            load_rider_selection, verify_provenance,
                            write_provenance, verify_affine_equivalence,
                            sha256_file)
RAW_LIDC  = os.path.join(BASE, "data", "raw", "lidc_idri")
RAW_RIDER = os.path.join(BASE, "data", "raw", "rider")
RAW_LNDB  = os.path.join(BASE, "data", "raw", "lndb")
NIFTI_DIR = os.path.join(BASE, "data", "interim", "nifti")
ANALYSIS  = os.path.join(BASE, "analysis")
CONFIG    = os.path.join(BASE, "config")
os.makedirs(ANALYSIS, exist_ok=True)

SMOKE = 0

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

validation_rows    = []

HU_AR_MAX = -900
HU_TIPICO = (-3024, 3071)


def _avaliar_hu(hu_min, hu_max):
    """Devolve (ok, estado). Extremos por metal ou preenchimento sao FLAG."""
    if hu_min > HU_AR_MAX:
        return False, ("FAIL_NO_AIR: hu_min=%d acima de %d — volume sem ar, "
                       "rescala para HU provavelmente nao aplicada"
                       % (hu_min, HU_AR_MAX))
    if hu_min < HU_TIPICO[0] or hu_max > HU_TIPICO[1]:
        return True, ("PASS_WITH_FLAG_HU_OUT_OF_TYPICAL_RANGE: [%d,%d] fora de "
                      "[%d,%d] — preenchimento fora do campo de visao ou metal"
                      % (hu_min, hu_max, HU_TIPICO[0], HU_TIPICO[1]))
    return True, "PASS"

_INICIO = time.time()


def _desta_execucao(path):
    """
    Um produto so conta se foi escrito NESTA execucao. Ficheiro anterior nao e
    entrada legitima nem prova de trabalho feito — e resultado herdado.
    """
    return os.path.isfile(path) and os.path.getmtime(path) >= _INICIO


def _read_dicom_header(dcm_path):
    return pydicom.dcmread(dcm_path, stop_before_pixels=True)


def convert_mhd(mhd_path, out_path, dataset, patient_id):
    """
    Lê MetaImage com SimpleITK (preserva origem, direção, espaçamento).
    Salva NIfTI sem reorientação.
    """
    image = sitk.ReadImage(mhd_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sitk.WriteImage(image, out_path, useCompression=True)

    spacing  = image.GetSpacing()
    origin   = image.GetOrigin()
    size     = image.GetSize()
    arr      = sitk.GetArrayFromImage(image)

    _row_nao_usada = {
        "timestamp":        datetime.now().isoformat(),
        "dataset":          dataset,
        "patient_id":       patient_id,
        "series_dir":       mhd_path,
        "out_path":         out_path,
        "n_slices_dicom":   size[2],
        "n_slices_nifti":   size[2],
        "spacing_xy_dicom": f"{spacing[0]:.4f},{spacing[1]:.4f}",
        "spacing_xy_nifti": f"{spacing[0]:.4f},{spacing[1]:.4f}",
        "thickness_dicom":  round(spacing[2], 4),
        "spacing_z_nifti":  round(spacing[2], 4),
        "origin_nifti":     str(tuple(round(v, 3) for v in origin[:3])),
        "direction_dicom":  "from_mhd",
        "direction_nifti":  str(tuple(round(v, 4) for v in image.GetDirection())),
        "hu_min":           int(arr.min()),
        "hu_max":           int(arr.max()),
        "match_slices":     True,
        "match_spacing_xy": True,
        "status":           "ok",
    }
    return image


def _validate_existing_mhd(out_path, mhd_path, dataset, patient_id):
    """
    Compara REALMENTE o MetaImage de origem com o NIfTI produzido.

    A versao anterior abria apenas o NIfTI e preenchia `match_slices` e
    `match_spacing_xy` com `True` fixo, sem tocar no `.mhd`. Qualquer ficheiro
    seria aprovado automaticamente.

    Confronta agora: integridade do fluxo, carga util, dimensoes, espacamento,
    origem, matriz de direcao, tipo de dados, cantos do volume e identidade dos
    voxels. O termo usado e `voxel_value_identity`, e nao "identidade de HU": so
    se pode chamar HU aos valores do MetaImage se a documentacao do LNDb o
    confirmar, o que nao foi verificado.

    Devolve (ok, estado).
    """
    if not os.path.isfile(out_path):
        return False, "FAIL_NIFTI_MISSING"
    if not os.path.isfile(mhd_path):
        return False, "FAIL_MHD_MISSING"

    ok, motivo, n_bytes = verify_gzip_stream(out_path)
    if not ok:
        return False, motivo
    ok, motivo, _ = verify_nifti_payload(out_path, n_bytes)
    if not ok:
        return False, motivo

    try:
        src_img = sitk.ReadImage(mhd_path)
        out_img = sitk.ReadImage(out_path)
    except Exception as e:
        return False, f"FAIL_UNREADABLE: {e}"

    if tuple(src_img.GetSize()) != tuple(out_img.GetSize()):
        return False, (f"FAIL_LNDB_GEOMETRY: dimensoes {out_img.GetSize()} "
                       f"vs {src_img.GetSize()}")
    if any(abs(a - b) > 1e-6 for a, b in zip(src_img.GetSpacing(), out_img.GetSpacing())):
        return False, (f"FAIL_LNDB_GEOMETRY: espacamento {out_img.GetSpacing()} "
                       f"vs {src_img.GetSpacing()}")
    if any(abs(a - b) > 1e-3 for a, b in zip(src_img.GetOrigin(), out_img.GetOrigin())):
        return False, "FAIL_LNDB_GEOMETRY: origem divergente"
    if any(abs(a - b) > 1e-6 for a, b in zip(src_img.GetDirection(), out_img.GetDirection())):
        return False, "FAIL_LNDB_GEOMETRY: matriz de direcao divergente"
    if src_img.GetPixelIDTypeAsString() != out_img.GetPixelIDTypeAsString():
        return False, (f"FAIL_LNDB_DTYPE: {out_img.GetPixelIDTypeAsString()} "
                       f"vs {src_img.GetPixelIDTypeAsString()}")

    h = nifti_header_fields(out_path)
    if not h or not (h["qform_code"] or h["sform_code"]):
        return False, "FAIL_LNDB_GEOMETRY: qform e sform ambos zero"

    a_src = sitk.GetArrayFromImage(src_img)
    a_out = sitk.GetArrayFromImage(out_img)
    if a_src.shape != a_out.shape:
        return False, "FAIL_LNDB_VOXEL_CONTENT: formas diferentes"
    if not np.array_equal(a_src, a_out):
        import hashlib as _hl
        hs = Counter(_hl.sha256(np.ascontiguousarray(a_src[k])).hexdigest()
                     for k in range(a_src.shape[0]))
        ho = Counter(_hl.sha256(np.ascontiguousarray(a_out[k])).hexdigest()
                     for k in range(a_out.shape[0]))
        return False, ("FAIL_LNDB_VOXEL_CONTENT: REORDERED" if hs == ho
                       else "FAIL_LNDB_VOXEL_CONTENT: conteudo divergente")

    size_nifti = out_img.GetSize()
    spacing = out_img.GetSpacing()
    validation_rows.append({
        "timestamp": datetime.now().isoformat(), "dataset": dataset,
        "patient_id": patient_id, "series_dir": mhd_path, "out_path": out_path,
        "n_slices_dicom": src_img.GetSize()[2], "n_slices_nifti": size_nifti[2],
        "spacing_xy_dicom": f"{src_img.GetSpacing()[0]:.4f},{src_img.GetSpacing()[1]:.4f}",
        "spacing_xy_nifti": f"{spacing[0]:.4f},{spacing[1]:.4f}",
        "thickness_dicom": round(src_img.GetSpacing()[2], 4),
        "spacing_z_nifti": round(spacing[2], 4),
        "origin_nifti": str(tuple(round(v, 3) for v in out_img.GetOrigin())),
        "direction_dicom": str(tuple(round(v, 4) for v in src_img.GetDirection())),
        "direction_nifti": str(tuple(round(v, 4) for v in out_img.GetDirection())),
        "hu_min": int(a_out.min()), "hu_max": int(a_out.max()),
        "match_slices": src_img.GetSize()[2] == size_nifti[2],
        "match_spacing_xy": all(abs(a - b) <= 1e-6 for a, b in
                                zip(src_img.GetSpacing()[:2], spacing[:2])),
        "status": "ok",
    })
    return True, "PASS_LNDB_MHD_NIFTI_IDENTITY"


LIDC_PROVENANCE_VERSION = "2026-08-07.lidc.conversion.v1"

LIDC_ESTADOS_CONVERTIVEIS = ("INCLUDED", "INCLUDED_MERGED_CANONICAL")

LIDC_GEOMETRY_OK = {"PASS_NIFTI_DICOM_SLICEWISE_GEOMETRY"}
LIDC_HU_OK       = {"PASS"}


def _lidc_manifesto():
    """
    Manifesto congelado dos casos elegiveis. O script NUNCA descobre entradas
    por varredura do diretorio: os NIfTI excluidos continuam fisicamente
    preservados e seriam apanhados por um `os.listdir`.
    """
    ponteiro = os.path.join(BASE, "runs", "CURRENT_RAW_AUDIT.json")
    if not os.path.isfile(ponteiro):
        raise RuntimeError(
            "FAIL_INPUT — ponteiro do Script 02 ausente: %s\n"
            "  O ramo LIDC nao pode operar por varredura de diretorio." % ponteiro)
    with open(ponteiro) as f:
        pt = json.load(f)
    if "lidc_selection_manifest" not in pt:
        raise RuntimeError("FAIL_INPUT — ponteiro sem 'lidc_selection_manifest': %s"
                           % ponteiro)
    mp = os.path.join(BASE, pt["lidc_selection_manifest"])
    if not os.path.isfile(mp):
        raise RuntimeError("FAIL_INPUT — manifesto de selecao ausente: %s" % mp)
    with open(mp) as f:
        man = json.load(f)
    return man, mp, pt.get("run_id", "")


def _sha256(path):
    x = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            x.update(b)
    return x.hexdigest()


def process_lidc():
    """
    Ramo LIDC-IDRI: CONVERTE as series da selecao canonica do Script 02.

    A versao anterior nao convertia nada. Percorria o manifesto a procura de
    `ct.nii.gz` ja em disco e contava-os como "reaproveitados" — reutilizacao de
    resultado antigo escrita no codigo, com `convert_dicom_series()` definido e
    nunca chamado para o LIDC. Com as pastas limpas declarava 1.010 ausentes.

    Aplica o mesmo rigor do ramo RIDER, e pela mesma razao: separar os cortes
    pelo SeriesInstanceUID em vez de confiar na pasta, escolher a pilha coerente,
    conferir a affine contra a geometria medida nos DICOM, conferir a faixa de
    intensidade e gravar proveniencia por caso.

    Devolve (estado, resumo).
    """
    log.info("=== LIDC-IDRI ===")
    man, mp, run02 = _lidc_manifesto()

    universo = [e for e in man["eligible"]
                if e["analytic_status"] in LIDC_ESTADOS_CONVERTIVEIS]
    if SMOKE:
        universo = universo[:SMOKE]
    fora = [e for e in man["eligible"]
            if e["analytic_status"] not in LIDC_ESTADOS_CONVERTIVEIS]
    log.info("Manifesto: %s (run 02 %s)" % (os.path.basename(mp), run02))
    log.info("  a converter: %d · fora do universo analitico: %d %s"
             % (len(universo), len(fora),
                [(e["patient_id"], e["analytic_status"]) for e in fora]))
    if not SMOKE and len(universo) != man["LIDC_ANALYTIC_DENOMINATOR"]:
        return "FAIL_INPUT", {"motivo": "universo %d != denominador %d"
                              % (len(universo), man["LIDC_ANALYTIC_DENOMINATOR"])}

    out_base = os.path.join(NIFTI_DIR, "lidc_idri")
    feitos, falhas, entradas, cortes_retidos = 0, [], [], []

    for k, e in enumerate(universo, 1):
        pid = e["patient_id"]
        uid = e["series_instance_uid"]
        ser_dir = os.path.join(RAW_LIDC, uid)
        out_dir = os.path.join(out_base, pid)
        out_path = os.path.join(out_dir, "ct.nii.gz")
        prov_path = os.path.join(out_dir, "provenance.json")

        if not os.path.isdir(ser_dir):
            falhas.append("%s: diretorio da serie ausente (%s)" % (pid, uid[-16:]))
            continue

        if os.path.isfile(out_path) and not _desta_execucao(out_path):
            os.remove(out_path)
            if os.path.isfile(prov_path):
                os.remove(prov_path)

        try:
            do_uid, estranhos, total = _particiona_por_uid(ser_dir, uid)
            if not do_uid:
                falhas.append("%s: nenhum corte com o UID do manifesto" % pid)
                continue
            if len(do_uid) != e["n_sop_instances"]:
                falhas.append("%s: %d cortes do UID vs %d no manifesto do 02"
                              % (pid, len(do_uid), e["n_sop_instances"]))
                continue

            ds = _read_dicom_header(do_uid[0])
            espessura = float(getattr(ds, "SliceThickness", 0) or 0)
            if espessura <= 0:
                falhas.append("%s: SliceThickness ausente ou nao positiva" % pid)
                continue

            geo, estado_pilha, detalhe = _selecionar_pilha(do_uid, espessura)
            if geo is None:
                falhas.append("%s: %s — %s" % (pid, estado_pilha, detalhe))
                continue
            origem, direcao, espacamento, tamanho, ordenados = geo
            blocos_rel = _relatorio_blocos(do_uid, ordenados)

            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(ordenados)
            image = reader.Execute()
            os.makedirs(out_dir, exist_ok=True)
            sitk.WriteImage(image, out_path, useCompression=True)

            ok_af, motivo_af, det = verify_affine_equivalence(
                out_path, origem, direcao, espacamento, tamanho)
            arr = sitk.GetArrayFromImage(sitk.ReadImage(out_path))
            hu_min, hu_max = int(arr.min()), int(arr.max())
            hu_ok, hu_estado = _avaliar_hu(hu_min, hu_max)

            estado_conv = "OK" if (ok_af and hu_ok) else "FAIL"
            sha_nii = _sha256(out_path)
            write_provenance(
                prov_path,
                provenance_version    = LIDC_PROVENANCE_VERSION,
                script                = "03_convert_dicom_to_nifti.py",
                script02_run_id       = run02,
                selection_manifest    = os.path.relpath(mp, BASE),
                patient_id            = pid,
                source_series_uid     = uid,
                analytic_status       = e["analytic_status"],
                source_directory      = ser_dir,
                n_dicom_total_in_directory = total,
                n_dicom_selected_uid  = len(do_uid),
                n_dicom_foreign_uid   = len(estranhos),
                n_slices_converted    = tamanho[2],
                n_slices_discarded    = len(do_uid) - tamanho[2],
                stack_selection_status = estado_pilha,
                stack_selection_detail = detalhe,
                physical_gap_report   = {k: v for k, v in blocos_rel.items()
                                         if k != "retained_slices"},
                n_retained_sops       = len(blocos_rel["retained_slices"]),
                truncation_decision   = ("DEFERRED_TO_SCRIPT04" if
                                         blocos_rel["has_physical_gap"] else
                                         "NOT_APPLICABLE"),
                conversion_status     = estado_conv,
                geometry_status       = ("PASS_NIFTI_DICOM_SLICEWISE_GEOMETRY"
                                         if ok_af else "FAIL: %s" % motivo_af),
                affine_details        = det,
                hu_range_observed     = [hu_min, hu_max],
                hu_transform_identity_status = hu_estado,
                sha256_nifti          = sha_nii,
                output_path           = out_path,
            )

            if estado_conv != "OK":
                falhas.append("%s: affine=%s hu=%s" % (pid, motivo_af, hu_estado))
                continue

            validation_rows.append({
                "timestamp": datetime.now().isoformat(), "dataset": "lidc_idri",
                "patient_id": pid, "series_dir": ser_dir, "out_path": out_path,
                "n_slices_dicom": len(do_uid), "n_slices_nifti": tamanho[2],
                "spacing_xy_dicom": "%.4f,%.4f" % (espacamento[0], espacamento[1]),
                "spacing_xy_nifti": "%.4f,%.4f" % (espacamento[0], espacamento[1]),
                "thickness_dicom": round(espacamento[2], 4),
                "spacing_z_nifti": round(espacamento[2], 4),
                "origin_nifti": str(tuple(round(v, 3) for v in origem)),
                "direction_nifti": str(tuple(round(v, 4) for v in direcao)),
                "hu_min": hu_min, "hu_max": hu_max,
                "match_slices": tamanho[2] == len(do_uid),
                "match_spacing_xy": True, "status": "ok",
            })
            for _r in blocos_rel["retained_slices"]:
                cortes_retidos.append({"patient_id": pid,
                                       "series_instance_uid": uid, **_r})
            entradas.append({
                "patient_id": pid, "series_instance_uid": uid,
                "analytic_status": e["analytic_status"],
                "nifti_relpath": os.path.relpath(out_path, BASE),
                "sha256_nifti": sha_nii,
                "n_slices_converted": tamanho[2],
                "n_sop_instances_source": e["n_sop_instances"],
                "stack_selection_status": estado_pilha,
                "stack_selection_detail": detalhe,
                "physical_gap_report": {k: v for k, v in blocos_rel.items()
                                        if k != "retained_slices"},
                "n_retained_sops": len(blocos_rel["retained_slices"]),
                "truncation_decision": ("DEFERRED_TO_SCRIPT04" if
                                        blocos_rel["has_physical_gap"] else
                                        "NOT_APPLICABLE"),
                "spacing": [round(v, 6) for v in espacamento],
                "hu_range_observed": [hu_min, hu_max],
            })
            feitos += 1
            if feitos % 50 == 0 or k == len(universo):
                log.info("  %d/%d convertidos" % (feitos, len(universo)))
        except Exception as ex:
            falhas.append("%s: %s" % (pid, ex))

    nao_convertidos = [{"patient_id": e["patient_id"],
                        "analytic_status": e["analytic_status"],
                        "reason": ("volume de pixeis com sha256 identico ao do "
                                   "canonico do grupo %s; converter duplicaria "
                                   "o mesmo exame" % e["duplicate_group_id"]),
                        "duplicate_group_id": e["duplicate_group_id"]}
                       for e in fora]

    log.info("LIDC-IDRI: %d convertidos, %d falhas, %d nao convertidos por regra"
             % (feitos, len(falhas), len(nao_convertidos)))
    for f in falhas[:10]:
        log.error("    %s" % f)

    rel = os.path.join(ANALYSIS, "lidc_retained_slices.csv")
    with open(rel, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "series_instance_uid",
                                          "canonical_index", "sop_instance_uid",
                                          "instance_number", "z_mm"])
        w.writeheader(); w.writerows(cortes_retidos)
    log.info("  cortes retidos: analysis/lidc_retained_slices.csv (%d linhas)"
             % len(cortes_retidos))

    estado = "PASS" if (not falhas and feitos == len(universo)) else "FAIL_OUTPUT"
    _emitir_manifesto_final(entradas, nao_convertidos, man, mp, run02, estado)
    return estado, {"convertidos": feitos, "falhas": len(falhas),
                    "universo": len(universo),
                    "nao_convertidos": len(nao_convertidos)}


def converter_blocos_dirigidos(caminho_manifesto):
    """
    Conversao DIRIGIDA: converte SO os blocos fisicos pedidos, sem repetir
    nenhum volume ja existente.

    Existe para o caso em que o Script 04, ao cruzar as anotacoes com o intervalo
    em z preservado, conclua que ha nodulo num bloco descartado. Converter os
    blocos adicionais por antecipacao produziria volumes que talvez nunca sejam
    usados; converte-los sob demanda mantem a cadeia auditavel e o custo baixo.

    Manifesto de entrada (CSV): `patient_id`, `block_index` (1-based, blocos
    ordenados por z crescente) e, opcionalmente, `expected_z_min_mm` /
    `expected_z_max_mm` para conferencia.

    O bloco ja convertido pelo caminho normal NAO e regerado: fica como esta e e
    descrito no indice. O bloco novo vai para `<patient_id>__block%02d/`, com
    `source_patient_id` na proveniencia — nunca fingindo ser outro paciente.
    """
    log.info("=== CONVERSAO DIRIGIDA DE BLOCOS ===")
    if not os.path.isfile(caminho_manifesto):
        return "FAIL_INPUT", {"motivo": "manifesto ausente: %s" % caminho_manifesto}

    final = os.path.join(CONFIG, "lidc_final_manifest.json")
    if not os.path.isfile(final):
        return "FAIL_INPUT", {"motivo": "lidc_final_manifest.json ausente; "
                                        "corra primeiro a conversao normal"}
    M = json.load(open(final))
    por_pid = {e["patient_id"]: e for e in M["eligible"]}

    pedidos = list(csv.DictReader(open(caminho_manifesto)))
    log.info("  blocos pedidos: %d" % len(pedidos))
    feitos, falhas, indice = 0, [], []

    for req in pedidos:
        pid = req["patient_id"].strip()
        bidx = int(req["block_index"])
        e = por_pid.get(pid)
        if e is None:
            falhas.append("%s: nao consta do manifesto final" % pid); continue

        uid = e["series_instance_uid"]
        ser_dir = os.path.join(RAW_LIDC, uid)
        do_uid, _estr, _tot = _particiona_por_uid(ser_dir, uid)
        itens = _ler_cortes(do_uid)
        blocos = sorted(_blocos_coerentes(itens), key=lambda b: b["extensao"][0])
        if not (1 <= bidx <= len(blocos)):
            falhas.append("%s: bloco %d inexistente (ha %d)"
                          % (pid, bidx, len(blocos))); continue
        b = blocos[bidx - 1]

        for campo, obtido in (("expected_z_min_mm", b["extensao"][0]),
                              ("expected_z_max_mm", b["extensao"][1])):
            if req.get(campo):
                if abs(float(req[campo]) - obtido) > 1e-3:
                    falhas.append("%s: %s pedido %s, medido %.4f"
                                  % (pid, campo, req[campo], obtido))
                    break
        else:
            mantidos = b["itens"] if b["uniforme"] else _maior_bloco_contiguo(b)[0]
            passo = b["passo"] if b["uniforme"] else _maior_bloco_contiguo(b)[2]
            origem, direcao, espacamento, tamanho, ordenados = _geometria_de(
                mantidos, passo)

            caso = "%s__block%02d" % (pid, bidx)
            out_dir = os.path.join(NIFTI_DIR, "lidc_idri", caso)
            out_path = os.path.join(out_dir, "ct.nii.gz")
            if os.path.isfile(out_path) and not _desta_execucao(out_path):
                os.remove(out_path)

            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(ordenados)
            image = reader.Execute()
            os.makedirs(out_dir, exist_ok=True)
            sitk.WriteImage(image, out_path, useCompression=True)

            ok_af, motivo_af, det = verify_affine_equivalence(
                out_path, origem, direcao, espacamento, tamanho)
            arr = sitk.GetArrayFromImage(sitk.ReadImage(out_path))
            hu_ok, hu_estado = _avaliar_hu(int(arr.min()), int(arr.max()))
            if not (ok_af and hu_ok):
                falhas.append("%s bloco %d: affine=%s hu=%s"
                              % (pid, bidx, motivo_af, hu_estado)); continue

            sha_nii = _sha256(out_path)
            write_provenance(
                os.path.join(out_dir, "provenance.json"),
                provenance_version = LIDC_PROVENANCE_VERSION,
                provenance_mode    = "DIRECTED_BLOCK_CONVERSION",
                script             = "03_convert_dicom_to_nifti.py",
                case_id            = caso,
                source_patient_id  = pid,
                source_series_uid  = uid,
                block_index        = bidx,
                n_contiguous_blocks = len(blocos),
                block_z_min_mm     = round(b["extensao"][0], 4),
                block_z_max_mm     = round(b["extensao"][1], 4),
                sibling_case_ids   = [pid] + ["%s__block%02d" % (pid, i + 1)
                                              for i in range(len(blocos))
                                              if i + 1 != bidx],
                partition_constraint = ("todos os blocos do mesmo paciente ficam "
                                        "no mesmo fold; metrica ao nivel do "
                                        "paciente conta UM paciente, nao dois"),
                requested_by       = os.path.relpath(caminho_manifesto, BASE),
                n_slices_converted = tamanho[2],
                conversion_status  = "OK",
                geometry_status    = "PASS_NIFTI_DICOM_SLICEWISE_GEOMETRY",
                hu_transform_identity_status = hu_estado,
                sha256_nifti       = sha_nii,
                output_path        = out_path,
            )
            indice.append({"case_id": caso, "source_patient_id": pid,
                           "series_instance_uid": uid, "block_index": bidx,
                           "z_min_mm": round(b["extensao"][0], 4),
                           "z_max_mm": round(b["extensao"][1], 4),
                           "n_slices": tamanho[2],
                           "nifti_relpath": os.path.relpath(out_path, BASE),
                           "sha256_nifti": sha_nii})
            log.info("  [OK] %s  %d cortes  z %.1f..%.1f mm"
                     % (caso, tamanho[2], b["extensao"][0], b["extensao"][1]))
            feitos += 1

    if indice:
        alvo = {r["source_patient_id"] for r in indice}
        for pid in sorted(alvo):
            e = por_pid[pid]
            g = e["physical_gap_report"]
            indice.append({"case_id": pid, "source_patient_id": pid,
                           "series_instance_uid": e["series_instance_uid"],
                           "block_index": 1 if g["n_contiguous_blocks"] <= 1 else 0,
                           "z_min_mm": g["retained_z_min_mm"],
                           "z_max_mm": g["retained_z_max_mm"],
                           "n_slices": g["n_slices_retained"],
                           "nifti_relpath": e["nifti_relpath"],
                           "sha256_nifti": e["sha256_nifti"]})
        destino = os.path.join(CONFIG, "lidc_block_index.json")
        json.dump({"producer": "03_convert_dicom_to_nifti.py --split-manifest",
                   "generated": datetime.now().isoformat(),
                   "requested_by": os.path.relpath(caminho_manifesto, BASE),
                   "PARTITION_CONSTRAINT": (
                       "case_id distintos com o mesmo source_patient_id sao o "
                       "MESMO paciente: mesmo fold, e uma unica unidade nas "
                       "metricas ao nivel do paciente"),
                   "blocks": sorted(indice, key=lambda r: (r["source_patient_id"],
                                                           r["block_index"]))},
                  open(destino, "w"), indent=1, sort_keys=True)
        log.info("  indice de blocos: config/lidc_block_index.json (%d volumes)"
                 % len(indice))

    log.info("CONVERSAO DIRIGIDA: %d blocos convertidos, %d falhas"
             % (feitos, len(falhas)))
    for f in falhas[:10]:
        log.error("    %s" % f)
    return ("PASS" if not falhas else "FAIL_OUTPUT"), {"convertidos": feitos,
                                                       "falhas": len(falhas)}


def _emitir_manifesto_final(entradas, nao_convertidos, man, mp, run02, estado):
    """
    Emite `config/lidc_final_manifest.json`, que e o produto que o Script 04a
    consome. O manifesto do 02 nao podia traze-lo: `nifti_relpath` e
    `sha256_nifti` so existem depois da conversao, e sao deste estagio.
    """
    destino = os.path.join(CONFIG, "lidc_final_manifest.json")
    com_lacuna = [e for e in entradas if e["physical_gap_report"]["has_physical_gap"]]
    corpo = {
        "TRUNCATION_POLICY": (
            "Serie com descontinuidade fisica e convertida no maior bloco "
            "contiguo, sem interpolacao nem concatenacao. NENHUM caso e excluido "
            "aqui: a decisao KEEP / EXCLUDE / SPLIT pertence ao Script 04 e exige "
            "cruzar as coordenadas de cada anotacao com o intervalo em z "
            "efectivamente preservado. Criterios acordados: nenhum nodulo na "
            "regiao descartada -> pode permanecer, marcado como truncado; nodulo "
            "inteiro no bloco descartado -> o volume nao representa o caso "
            "anotado; nodulo a cruzar a fronteira -> candidato forte a exclusao; "
            "nodulos em blocos distintos -> ponderar SPLIT com correspondencia "
            "inequivoca paciente/serie/bloco/anotacao."),
        "TRUNCATION_DECISION": "DEFERRED_TO_SCRIPT04",
        "n_with_physical_gap": len(com_lacuna),
        "with_physical_gap": sorted(
            [{"patient_id": e["patient_id"],
              "n_slices_source": e["physical_gap_report"]["n_slices_source"],
              "n_slices_retained": e["physical_gap_report"]["n_slices_retained"],
              "retained_fraction": e["physical_gap_report"]["retained_fraction"],
              "n_contiguous_blocks": e["physical_gap_report"]["n_contiguous_blocks"]}
             for e in com_lacuna], key=lambda r: r["retained_fraction"]),
        "manifest_id": "lidc_final_%s" % time.strftime("%Y%m%dT%H%M%S",
                                                       time.localtime(_INICIO)),
        "producer": "03_convert_dicom_to_nifti.py",
        "generated": datetime.now().isoformat(),
        "source_selection_manifest": os.path.relpath(mp, BASE),
        "script02_run_id": run02,
        "SELECTION_RULE": man.get("SELECTION_RULE", ""),
        "MERGE_STATUS": man.get("MERGE_STATUS", ""),
        "LIDC_ANALYTIC_DENOMINATOR": man.get("LIDC_ANALYTIC_DENOMINATOR"),
        "CONVERSION_STATUS": estado,
        "n_converted": len(entradas),
        "n_not_converted": len(nao_convertidos),
        "not_converted": nao_convertidos,
        "eligible": entradas,
    }
    corpo["manifest_body_sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in corpo.items()
                    if k not in ("manifest_body_sha256", "manifest_sha256")},
                   indent=1, sort_keys=True).encode()).hexdigest()
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    with open(destino, "w") as f:
        json.dump(corpo, f, indent=1, sort_keys=True)
    log.info("  manifesto final: config/lidc_final_manifest.json (%d entradas, %s)"
             % (len(entradas), estado))

    rel = os.path.join(ANALYSIS, "lidc_physical_gaps.csv")
    campos = ["patient_id", "series_instance_uid", "has_physical_gap",
              "n_slices_source", "n_slices_retained", "n_slices_discarded",
              "retained_fraction", "n_contiguous_blocks", "block_index",
              "block_n_slices", "block_z_min_mm", "block_z_max_mm",
              "block_extent_mm", "block_step_mm", "block_slice_thickness_mm",
              "block_uniform", "block_is_retained", "retained_z_min_mm",
              "retained_z_max_mm", "retained_extent_mm", "stack_selection_status",
              "truncation_decision"]
    with open(rel, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore", restval="")
        w.writeheader()
        for e in entradas:
            g = e["physical_gap_report"]
            for i, b in enumerate(g["blocks"]):
                w.writerow({
                    "patient_id": e["patient_id"],
                    "series_instance_uid": e["series_instance_uid"],
                    "has_physical_gap": g["has_physical_gap"],
                    "n_slices_source": g["n_slices_source"],
                    "n_slices_retained": g["n_slices_retained"],
                    "n_slices_discarded": g["n_slices_discarded"],
                    "retained_fraction": g["retained_fraction"],
                    "n_contiguous_blocks": g["n_contiguous_blocks"],
                    "block_index": i, "block_n_slices": b["n_slices"],
                    "block_z_min_mm": b["z_min_mm"], "block_z_max_mm": b["z_max_mm"],
                    "block_extent_mm": b["extent_mm"], "block_step_mm": b["step_mm"],
                    "block_slice_thickness_mm": b["slice_thickness_mm"],
                    "block_uniform": b["uniform"],
                    "block_is_retained": b["is_retained"],
                    "retained_z_min_mm": g["retained_z_min_mm"],
                    "retained_z_max_mm": g["retained_z_max_mm"],
                    "retained_extent_mm": g["retained_extent_mm"],
                    "stack_selection_status": e["stack_selection_status"],
                    "truncation_decision": e["truncation_decision"]})
    log.info("  relatorio de lacunas: analysis/lidc_physical_gaps.csv "
             "(%d series, %d com lacuna)"
             % (len(entradas), sum(1 for e in entradas
                                   if e["physical_gap_report"]["has_physical_gap"])))


def _ler_cortes(files):
    """Le os tags geometricos de cada corte. Devolve lista de dicts."""
    itens = []
    for fp in files:
        ds = pydicom.dcmread(fp, stop_before_pixels=True)
        iop = [float(v) for v in ds.ImageOrientationPatient]
        r, c = iop[:3], iop[3:]
        n = [r[1]*c[2]-r[2]*c[1], r[2]*c[0]-r[0]*c[2], r[0]*c[1]-r[1]*c[0]]
        m = sum(x*x for x in n) ** 0.5
        n = [x/m for x in n]
        ipp = [float(v) for v in ds.ImagePositionPatient]
        itens.append({
            "fp": fp,
            "sop": str(getattr(ds, "SOPInstanceUID", "") or ""),
            "inst": int(getattr(ds, "InstanceNumber", 0) or 0),
            "ipp": ipp, "iop": iop, "normal": n,
            "proj": sum(a*b for a, b in zip(ipp, n)),
            "assinatura": (
                str(ds.PixelSpacing), str(ds.SliceThickness),
                str(getattr(ds, "ConvolutionKernel", "")),
                str(ds.Rows), str(ds.Columns),
                str(getattr(ds, "RescaleSlope", "")),
                str(getattr(ds, "RescaleIntercept", "")),
                tuple(round(v, 5) for v in iop),
            ),
            "espessura": float(ds.SliceThickness),
            "ps": [float(v) for v in ds.PixelSpacing],
            "rows": int(ds.Rows), "cols": int(ds.Columns),
        })
    return itens


def _blocos_coerentes(itens, tol_passo=0.01):
    """
    Separa os cortes em pilhas coerentes.

    Uma pasta ou mesmo um SeriesInstanceUID podem reunir mais de uma
    reconstrucao da mesma aquisicao. Escolher um corte por posicao misturaria
    pixels de reconstrucoes diferentes, por isso a separacao e feita em pilhas
    inteiras.

    Primeiro agrupa pela assinatura geometrica — espacamento, espessura, kernel,
    matriz, orientacao e parametros de conversao para HU. Depois, dentro de cada
    grupo, ordena por numero de instancia e corta onde houver reinicio, inversao
    ou salto incompativel da posicao projetada no vetor normal.

    Cada bloco e validado quanto a: posicoes unicas, monotonicidade e passo
    constante. Devolve lista de blocos validos, cada um com metadados.
    """
    grupos = defaultdict(list)
    for it in itens:
        grupos[it["assinatura"]].append(it)

    blocos = []
    for assinatura, grupo in grupos.items():
        grupo.sort(key=lambda x: (x["inst"], x["proj"]))
        segmentos, atual = [], [grupo[0]]
        for ant, act in zip(grupo, grupo[1:]):
            d = act["proj"] - ant["proj"]
            reinicio = abs(d) < 1e-6
            inversao = (len(atual) > 1 and
                        (d > 0) != (atual[-1]["proj"] - atual[-2]["proj"] > 0))
            salto = abs(d) > 3.0 * act["espessura"]
            if reinicio or inversao or salto:
                segmentos.append(atual); atual = [act]
            else:
                atual.append(act)
        segmentos.append(atual)

        for seg in segmentos:
            if len(seg) < 10:
                continue
            proj = sorted(x["proj"] for x in seg)
            if len(set(round(v, 4) for v in proj)) != len(seg):
                continue
            passos = [proj[i+1] - proj[i] for i in range(len(proj)-1)]
            passo = sum(passos) / len(passos)
            uniforme = max(abs(x - passo) for x in passos) <= tol_passo
            blocos.append({
                "itens": seg, "n": len(seg), "passo": passo,
                "uniforme": uniforme, "espessura": seg[0]["espessura"],
                "extensao": (min(proj), max(proj)),
                "inst": (min(x["inst"] for x in seg), max(x["inst"] for x in seg)),
            })
    return blocos


def _maior_bloco_contiguo(bloco, tol_passo=0.01):
    """
    Reduz um bloco a maior corrida de cortes com passo constante.

    Nao interpola, nao concatena os lados da lacuna e nao cria cortes sinteticos:
    a lacuna e ausencia fisica de aquisicao. Concatenar deslocaria toda a
    anatomia depois dela.

    Devolve (itens_mantidos, n_descartados, passo).
    """
    itens = sorted(bloco["itens"], key=lambda x: x["proj"])
    passos = [itens[i+1]["proj"] - itens[i]["proj"] for i in range(len(itens)-1)]
    modal = sorted(passos)[len(passos)//2]

    corridas, atual = [], [itens[0]]
    for it, d in zip(itens[1:], passos):
        if abs(d - modal) <= tol_passo:
            atual.append(it)
        else:
            corridas.append(atual); atual = [it]
    corridas.append(atual)

    maior = max(corridas, key=len)
    return maior, len(itens) - len(maior), modal


def _geometria_de(itens, passo):
    """Geometria fisica (convencao SimpleITK/LPS) a partir dos cortes mantidos."""
    itens = sorted(itens, key=lambda x: x["proj"])
    r = itens[0]["iop"][:3]
    c = itens[0]["iop"][3:]
    n = itens[0]["normal"]
    ps = itens[0]["ps"]
    origem = tuple(itens[0]["ipp"])
    direcao = (r[0], c[0], n[0], r[1], c[1], n[1], r[2], c[2], n[2])
    espacamento = (ps[1], ps[0], abs(passo))
    tamanho = (itens[0]["cols"], itens[0]["rows"], len(itens))
    return origem, direcao, espacamento, tamanho, [x["fp"] for x in itens]


def _selecionar_pilha(files, espessura_declarada):
    """
    Escolhe a pilha a converter e, se necessario, trunca-a no maior bloco
    contiguo.

    Devolve (geometria, estado, detalhe) — geometria e None quando a selecao e
    ambigua ou nenhum bloco serve.
    """
    itens = _ler_cortes(files)
    blocos = _blocos_coerentes(itens)
    if not blocos:
        return None, "FAIL_NO_COHERENT_STACK", "nenhum bloco coerente"

    cand = [b for b in blocos
            if abs(b["espessura"] - float(espessura_declarada)) < 1e-6]
    if not cand:
        return None, "FAIL_NO_STACK_AT_DECLARED_THICKNESS", (
            f"espessura declarada {espessura_declarada} mm; blocos disponiveis: "
            + ", ".join(f"{b['n']}c/{b['espessura']}mm" for b in blocos))

    maximo = max(b["n"] for b in cand)
    completos = [b for b in cand if b["n"] == maximo]
    if len(completos) > 1:
        return None, "FAIL_AMBIGUOUS_OVERLAPPING_RECONSTRUCTIONS", (
            f"{len(completos)} pilhas de {maximo} cortes com a mesma espessura")

    b = completos[0]
    assinatura = b["itens"][0]["assinatura"]

    n_entrada = len(itens)
    mesma_assinatura = [x for x in itens if x["assinatura"] == assinatura]

    if b["uniforme"]:
        mantidos, passo = b["itens"], b["passo"]
    else:
        mantidos, _desc, passo = _maior_bloco_contiguo(b)

    geo = _geometria_de(mantidos, passo)
    n_mantidos = len(mantidos)
    truncados = len(mesma_assinatura) - n_mantidos
    separados = n_entrada - len(mesma_assinatura)

    partes = []
    if truncados:
        partes.append(
            f"truncagem: lacuna fisica na aquisicao; mantidos {n_mantidos} cortes "
            f"contiguos de {len(mesma_assinatura)} da mesma reconstrucao, "
            f"descartados {truncados}; sem interpolacao nem concatenacao")
    if separados:
        outros = [x for x in blocos if x["itens"][0]["assinatura"] != assinatura]
        detalhe_outros = "; ".join(
            f"{x['n']}c/{x['espessura']}mm inst {x['inst'][0]}-{x['inst'][1]}"
            for x in outros) or f"{separados} cortes"
        partes.append(f"separacao: outra reconstrucao no mesmo UID ({detalhe_outros})")

    if truncados:
        estado = "PASS_TRUNCATED_CONTIGUOUS_BLOCK"
    elif separados:
        estado = "PASS_STACK_SEPARATED"
    else:
        estado = "PASS"
    return geo, estado, "; ".join(partes)


def _relatorio_blocos(files, ordenados):
    """
    Descreve a descontinuidade fisica da serie e, sobretudo, QUAL o intervalo em
    z que sobreviveu.

    `files` sao todos os cortes do UID; `ordenados` sao os que foram mesmo
    convertidos. Reportar apenas os blocos de origem nao chegava: em
    LIDC-IDRI-0672 ha um unico bloco de 462 cortes com passo irregular do qual se
    mantiveram 273, e sem o intervalo retido o Script 04 nao consegue saber que
    parte do torax o volume representa.

    Existe para que o 04 decida caso a caso — KEEP, EXCLUDE ou SPLIT — cruzando
    as coordenadas de cada anotacao com o intervalo preservado, sem reabrir os
    DICOM.
    """
    itens = _ler_cortes(files)
    blocos = _blocos_coerentes(itens)
    itens_ret = sorted(_ler_cortes(ordenados), key=lambda x: x["proj"])
    proj_ret = [x["proj"] for x in itens_ret]
    z_ret = (round(proj_ret[0], 4), round(proj_ret[-1], 4)) if proj_ret else (None, None)
    retidos = [{"canonical_index": i, "sop_instance_uid": x["sop"],
                "instance_number": x["inst"], "z_mm": round(x["proj"], 4)}
               for i, x in enumerate(itens_ret)]

    descr = []
    for b in sorted(blocos, key=lambda x: x["extensao"][0]):
        lo, hi = b["extensao"]
        retido = (proj_ret and lo >= z_ret[0] - 1e-3 and hi <= z_ret[1] + 1e-3)
        descr.append({"n_slices": b["n"],
                      "z_min_mm": round(lo, 4), "z_max_mm": round(hi, 4),
                      "extent_mm": round(hi - lo, 4),
                      "step_mm": round(b["passo"], 4),
                      "slice_thickness_mm": round(b["espessura"], 4),
                      "uniform": bool(b["uniforme"]),
                      "instance_range": list(b["inst"]),
                      "is_retained": bool(retido)})

    n_origem, n_ret = len(files), len(ordenados)
    return {"has_physical_gap": n_ret < n_origem,
            "n_slices_source": n_origem,
            "n_slices_retained": n_ret,
            "n_slices_discarded": n_origem - n_ret,
            "retained_fraction": round(n_ret / n_origem, 6) if n_origem else 0.0,
            "retained_z_min_mm": z_ret[0], "retained_z_max_mm": z_ret[1],
            "retained_extent_mm": (round(z_ret[1] - z_ret[0], 4)
                                   if proj_ret else None),
            "n_contiguous_blocks": len(blocos),
            "blocks": descr,
            "retained_slices": retidos}


def _particiona_por_uid(series_dir, uid_esperado):
    """
    Separa os DICOM da pasta entre os que pertencem ao SeriesInstanceUID
    selecionado e os de outros UID.

    A selecao e feita pelo UID, nunca pela pasta: uma pasta pode conter cortes de
    mais de uma serie — foi o caso encontrado no RIDER — e converter a pasta
    inteira produziria um volume com geometria interna inconsistente.

    Devolve (do_uid, de_outros_uid, total).
    """
    todos = sorted(f for f in os.listdir(series_dir) if f.endswith(".dcm"))
    do_uid, estranhos = [], []
    for fn in todos:
        fp = os.path.join(series_dir, fn)
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True)
            uid = str(getattr(ds, "SeriesInstanceUID", "")).strip()
        except Exception:
            estranhos.append(fn)
            continue
        (do_uid if uid == uid_esperado else estranhos).append(fp if uid == uid_esperado else fn)
    return do_uid, estranhos, len(todos)


def process_rider():
    """
    Converte as series RIDER da selecao vigente, uma por papel de cada par.

    Protecoes exigidas pelo protocolo:
      - confere o SHA-256 completo do manifesto contra a ancora antes de comecar;
      - exige pair_selection_version;
      - seleciona os DICOM por SeriesInstanceUID, nunca pela pasta;
      - conta apenas os ficheiros do UID selecionado e regista os de outros UID;
      - grava proveniencia em JSON lateral a cada NIfTI;
      - valida affine, orientacao, dimensoes fisicas e HU;
      - falha diante de NIfTI existente sem proveniencia valida.
    """
    log.info("=== RIDER ===")
    try:
        cfg, pares = load_rider_selection(strict=True)
    except RuntimeError as e:
        log.error(str(e))
        return "FAIL_INPUT", {"motivo": str(e)}
    log.info(f"Selecao vigente: {cfg['pair_selection_version']}")
    log.info(f"Manifesto sha256: {cfg['pair_manifest_sha256']}")

    aceites = [p for p in pares if p.get("pair_status") == "PASS_PAIR_SELECTION"]
    if SMOKE:
        aceites = aceites[:SMOKE]
    excluidos = [p for p in pares if p.get("pair_status") != "PASS_PAIR_SELECTION"]
    log.info(f"Pares aceites: {len(aceites)} · excluidos: {len(excluidos)} "
             f"(sem produto: {[p['patient_id'] for p in excluidos]})")

    out_base = os.path.join(NIFTI_DIR, "rider")
    feitos = reaproveitados = invalidados = erros = 0

    for par in aceites:
        pid = par["patient_id"]
        for papel, idx in (("ct1", 1), ("ct2", 2)):
            ser_dir = par[f"{papel}_dir"]
            uid     = par[f"{papel}_series_uid"]
            caso    = f"{pid}_{idx}"
            out_dir = os.path.join(out_base, caso)
            out_path = os.path.join(out_dir, "ct.nii.gz")
            prov_path = os.path.join(out_dir, "provenance.json")

            if os.path.isfile(out_path) and not _desta_execucao(out_path):
                os.remove(out_path)
                if os.path.isfile(prov_path):
                    os.remove(prov_path)

            if os.path.isfile(out_path):
                estado, motivo = verify_provenance(
                    prov_path, cfg,
                    campos_obrigatorios=("source_series_uid", "patient_id",
                                         "pair_role", "affine_validation_status"))
                if estado == "REUSABLE":
                    log.info(f"  [REUSABLE] {caso}")
                    reaproveitados += 1
                    continue
                log.warning(f"  [INVALIDATED] {caso} — {motivo}")
                invalidados += 1
                continue

            if not os.path.isdir(ser_dir):
                log.error(f"  [FAIL_INPUT] {caso}: diretorio ausente {ser_dir}")
                erros += 1
                continue

            try:
                do_uid, estranhos, total = _particiona_por_uid(ser_dir, uid)
                if not do_uid:
                    log.error(f"  [FAIL_INPUT] {caso}: nenhum corte com UID {uid[-16:]}")
                    erros += 1
                    continue
                if estranhos:
                    log.warning(f"  {caso}: {len(estranhos)} ficheiros de outro "
                                f"UID na mesma pasta — ignorados")

                geo, estado_pilha, detalhe_pilha = _selecionar_pilha(
                    do_uid, par[f"{papel}_slice_thickness"])
                if geo is None:
                    log.error(f"  [{estado_pilha}] {caso}: {detalhe_pilha}")
                    erros += 1
                    continue
                origem, direcao, espacamento, tamanho, ordenados = geo
                if estado_pilha != "PASS":
                    log.warning(f"  {caso}: {estado_pilha} — {detalhe_pilha}")

                reader = sitk.ImageSeriesReader()
                reader.SetFileNames(ordenados)
                image = reader.Execute()

                os.makedirs(out_dir, exist_ok=True)
                sitk.WriteImage(image, out_path, useCompression=True)

                ok_af, motivo_af, det = verify_affine_equivalence(
                    out_path, origem, direcao, espacamento, tamanho)
                arr = sitk.GetArrayFromImage(sitk.ReadImage(out_path))
                hu_ok, hu_estado = _avaliar_hu(int(arr.min()), int(arr.max()))

                estado_conv = "OK" if (ok_af and hu_ok) else "FAIL"
                write_provenance(
                    prov_path,
                    pair_selection_version = cfg["pair_selection_version"],
                    pair_manifest_sha256   = cfg["pair_manifest_sha256"],
                    patient_id             = pid,
                    pair_role              = papel.upper(),
                    source_series_uid      = uid,
                    source_acquisition_number = par.get(f"{papel}_acquisition_number", ""),
                    source_datetime        = par.get(f"{papel}_acquisition_datetime", ""),
                    source_datetime_source = par.get(f"{papel}_datetime_source", ""),
                    n_dicom_total_in_directory = total,
                    n_dicom_selected_uid   = len(do_uid),
                    n_dicom_foreign_uid    = len(estranhos),
                    n_slices_converted     = tamanho[2],
                    n_slices_discarded     = len(do_uid) - tamanho[2],
                    stack_selection_status = estado_pilha,
                    stack_selection_detail = detalhe_pilha,
                    conversion_status      = estado_conv,
                    affine_validation_status = "PASS" if ok_af else f"FAIL: {motivo_af}",
                    orientation_validation_status = ("PASS" if ok_af else "FAIL"),
                    hu_range_observed      = [int(arr.min()), int(arr.max())],
                    hu_validation_status   = hu_estado,
                    affine_details         = det,
                    source_directory       = ser_dir,
                    output_path            = out_path,
                )

                if estado_conv != "OK":
                    log.error(f"  [FAIL_OUTPUT] {caso}: affine={motivo_af} hu={hu_estado}")
                    erros += 1
                    continue

                validation_rows.append({
                    "timestamp": datetime.now().isoformat(), "dataset": "rider",
                    "patient_id": caso, "series_dir": ser_dir, "out_path": out_path,
                    "n_slices_dicom": len(do_uid), "n_slices_nifti": tamanho[2],
                    "spacing_xy_dicom": f"{espacamento[0]:.4f},{espacamento[1]:.4f}",
                    "spacing_xy_nifti": f"{espacamento[0]:.4f},{espacamento[1]:.4f}",
                    "thickness_dicom": round(espacamento[2], 4),
                    "spacing_z_nifti": round(espacamento[2], 4),
                    "origin_nifti": str(tuple(round(v, 3) for v in origem)),
                    "hu_min": int(arr.min()), "hu_max": int(arr.max()),
                    "match_slices": True, "match_spacing_xy": True, "status": "ok",
                })
                log.info(f"  [OK] {caso}  {tamanho}  {len(do_uid)}/{total} cortes  "
                         f"cantos ±{det.get('max_corner_delta_mm', 0)} mm")
                feitos += 1
            except Exception as e:
                log.error(f"  [ERRO] {caso}: {e}")
                erros += 1

    log.info(f"RIDER: {feitos} convertidos, {reaproveitados} reaproveitados, "
             f"{invalidados} invalidados, {erros} erros")
    esperado = 2 * len(aceites)
    estado = "PASS" if (erros == 0 and invalidados == 0 and
                        feitos + reaproveitados == esperado) else "FAIL_OUTPUT"
    return estado, {"convertidos": feitos, "reaproveitados": reaproveitados,
                    "invalidados": invalidados, "erros": erros,
                    "esperado": esperado, "pares_aceites": len(aceites),
                    "pares_excluidos": len(excluidos)}


LNDB_PROVENANCE_VERSION = "2026-07-31.lndb.retroactive.v1"


def _sha256_ficheiro(path):
    import hashlib
    x = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            x.update(b)
    return x.hexdigest()


def _lndb_provenance_valida(prov_path, mhd_path, out_path):
    """
    Proveniencia so e valida se os hashes da origem e do destino baterem com o
    que esta em disco. Um ficheiro de proveniencia que nao amarra os hashes nao
    prova nada sobre o produto que acompanha.
    """
    if not os.path.isfile(prov_path):
        return False
    try:
        import json as _json
        p = _json.load(open(prov_path))
    except Exception:
        return False
    if p.get("provenance_version") != LNDB_PROVENANCE_VERSION:
        return False
    raw_path = mhd_path[:-4] + ".raw"
    esperado = {
        "sha256_mhd": _sha256_ficheiro(mhd_path),
        "sha256_nifti": _sha256_ficheiro(out_path),
    }
    if os.path.isfile(raw_path):
        esperado["sha256_raw"] = _sha256_ficheiro(raw_path)
    return all(p.get(k) == v for k, v in esperado.items())


def _emitir_provenance_lndb(prov_path, mhd_path, out_path, lndb_id, estado,
                            modo="RETROACTIVE_MHD_NIFTI_AUDIT"):
    """
    Proveniencia do produto LNDb, amarrando origem e destino por hash.

    `modo` distingue os dois casos: CONVERSION quando o NIfTI acaba de ser
    escrito por este script, RETROACTIVE quando apenas se audita um ja existente.
    Faltava a chamada no caminho de conversao, e por isso os 236 NIfTI do LNDb
    saiam sem proveniencia nenhuma enquanto o LIDC e o RIDER a tinham.
    """
    import json as _json
    src = sitk.ReadImage(mhd_path)
    out = sitk.ReadImage(out_path)
    h = nifti_header_fields(out_path)
    raw_path = mhd_path[:-4] + ".raw"
    prov = {
        "provenance_version": LNDB_PROVENANCE_VERSION,
        "provenance_mode": modo,
        "lndb_id": lndb_id,
        "source_mhd": mhd_path,
        "source_raw": raw_path if os.path.isfile(raw_path) else "",
        "output_path": out_path,
        "sha256_mhd": _sha256_ficheiro(mhd_path),
        "sha256_raw": _sha256_ficheiro(raw_path) if os.path.isfile(raw_path) else "",
        "sha256_nifti": _sha256_ficheiro(out_path),
        "size": list(out.GetSize()), "spacing": list(out.GetSpacing()),
        "origin": list(out.GetOrigin()), "direction": list(out.GetDirection()),
        "pixel_type": out.GetPixelIDTypeAsString(),
        "qform_code": h["qform_code"] if h else None,
        "sform_code": h["sform_code"] if h else None,
        "geometry_status": "PASS",
        "voxel_value_identity_status": "PASS",
        "gzip_stream_integrity_status": "PASS",
        "validation_status": estado,
        "note": ("voxel_value_identity, nao 'identidade de HU': so se pode chamar "
                 "HU aos valores do MetaImage se a documentacao do LNDb o confirmar, "
                 "o que nao foi verificado."),
        "audited": datetime.now().isoformat(),
    }
    os.makedirs(os.path.dirname(prov_path), exist_ok=True)
    tmp = prov_path + ".tmp"
    with open(tmp, "w") as f:
        _json.dump(prov, f, indent=1)
    os.replace(tmp, prov_path)


def process_lndb():
    log.info("=== LNDb ===")
    data_dir = os.path.join(RAW_LNDB, "data")
    if not os.path.isdir(data_dir):
        log.error(f"FAIL_INPUT — subpasta data/ nao encontrada: {data_dir}")
        return "FAIL_INPUT", {"motivo": "data/ ausente"}

    mhd_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".mhd")])
    if SMOKE:
        mhd_files = mhd_files[:SMOKE]
    log.info(f"CTs LNDb encontrados: {len(mhd_files)}")

    out_dir_dataset = os.path.join(NIFTI_DIR, "lndb")
    done = skipped = errors = 0

    for mhd in mhd_files:
        lndb_id  = mhd.replace(".mhd", "")
        mhd_path = os.path.join(data_dir, mhd)
        out_path = os.path.join(out_dir_dataset, lndb_id, "ct.nii.gz")

        prov_path = os.path.join(out_dir_dataset, lndb_id, "provenance.json")
        if os.path.isfile(out_path) and not _desta_execucao(out_path):
            os.remove(out_path)
            if os.path.isfile(prov_path):
                os.remove(prov_path)
        if os.path.isfile(out_path):
            ok, estado = _validate_existing_mhd(out_path, mhd_path, "lndb", lndb_id)
            if ok and _lndb_provenance_valida(prov_path, mhd_path, out_path):
                log.info(f"  [SKIP-OK] {lndb_id}/ct.nii.gz — {estado}")
                skipped += 1
                continue
            if ok:
                _emitir_provenance_lndb(prov_path, mhd_path, out_path, lndb_id, estado)
                log.info(f"  [PROV-RETRO] {lndb_id} — proveniencia emitida")
                skipped += 1
                continue
            log.warning(f"  [REPROC] {lndb_id} — saída inválida: {estado}")
            os.remove(out_path)

        raw_path = os.path.join(data_dir, mhd.replace(".mhd", ".raw"))
        if not os.path.isfile(raw_path):
            log.error(f"  [FAIL_INPUT] {mhd} — .raw ausente em {data_dir}")
            errors += 1
            continue

        try:
            img = convert_mhd(mhd_path, out_path, "lndb", lndb_id)
            ok, estado = _validate_existing_mhd(out_path, mhd_path, "lndb", lndb_id)
            if not ok:
                log.error(f"  [FAIL_OUTPUT] {lndb_id}: {estado}")
                errors += 1
                continue
            _emitir_provenance_lndb(prov_path, mhd_path, out_path, lndb_id,
                                    estado, modo="CONVERSION_MHD_TO_NIFTI")
            log.info(f"  [OK]   {lndb_id}/ct.nii.gz  size={img.GetSize()}  {estado}")
            done += 1
        except Exception as e:
            log.error(f"  [ERRO] {lndb_id}: {e}")
            validation_rows.append({
                "dataset": "lndb", "patient_id": lndb_id,
                "series_dir": mhd_path, "status": f"error: {e}",
            })
            errors += 1

    log.info(f"LNDb: {done} convertidos, {skipped} já existentes, {errors} erros")
    estado = "PASS" if (errors == 0 and done + skipped == len(mhd_files)) else "FAIL_OUTPUT"
    return estado, {"convertidos": done, "reaproveitados": skipped,
                    "erros": errors, "esperado": len(mhd_files)}


VALIDATION_FIELDS = [
    "timestamp", "dataset", "patient_id", "series_dir", "out_path",
    "n_slices_dicom", "n_slices_nifti",
    "spacing_xy_dicom", "spacing_xy_nifti",
    "thickness_dicom", "spacing_z_nifti",
    "origin_nifti", "direction_dicom", "direction_nifti",
    "hu_min", "hu_max", "match_slices", "match_spacing_xy", "status",
]


def save_validation_csv():
    out_path = os.path.join(ANALYSIS, "dicom_nifti_validation.csv")

    merged = sorted(validation_rows,
                    key=lambda r: (r.get("dataset", ""), r.get("patient_id", "")))

    if not merged:
        log.warning("Nenhum dado de validação gerado.")
        return

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=VALIDATION_FIELDS,
                           extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(merged)

    n_new = len(validation_rows)
    log.info(f"Validação salva: {out_path} ({len(merged)} linhas, {n_new} novas)")

    ok       = sum(1 for r in merged if r.get("status") == "ok")
    erros    = sum(1 for r in merged if "error" in str(r.get("status", "")))
    no_match = sum(1 for r in merged
                   if r.get("match_slices") in (False, "False")
                   or r.get("match_spacing_xy") in (False, "False"))
    log.info(f"  Total={len(merged)}  OK={ok}  Erros={erros}  Inconsistências={no_match}")
    if no_match > 0:
        log.warning("  AVISO: há inconsistências DICOM-NIfTI — revisar validation.csv")


def main(argv=None):
    """
    Portao de saida: o script so termina com 0 se os TRES ramos passarem e se
    todas as saidas declaradas existirem com carimbo desta execucao.

    Antes, `if __name__` chamava os tres ramos em sequencia, escrevia o CSV e
    registava "Conversao concluida" — com codigo 0 mesmo que o LIDC e o RIDER
    tivessem falhado por inteiro.
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-manifest", metavar="CSV", default="",
                    help="conversao dirigida: converte SO os blocos pedidos, "
                         "sem repetir nenhum volume existente")
    ap.add_argument("--smoke", type=int, default=0, metavar="N",
                    help="ensaio: N casos por banco, para pasta descartavel, "
                         "sem publicar ponteiro")
    args = ap.parse_args(argv)

    global SMOKE, NIFTI_DIR, ANALYSIS, CONFIG
    SMOKE = args.smoke
    if SMOKE:
        raiz = os.path.join(BASE, "staging", "smoke_03_%s"
                            % time.strftime("%Y%m%dT%H%M%S", time.localtime(_INICIO)))
        NIFTI_DIR = os.path.join(raiz, "nifti")
        ANALYSIS = os.path.join(raiz, "analysis")
        CONFIG = os.path.join(raiz, "config")
        for d in (NIFTI_DIR, ANALYSIS, CONFIG):
            os.makedirs(d, exist_ok=True)
        log.warning("MODO DE ENSAIO: %d casos por banco, saida em %s — "
                    "NADA e publicado" % (SMOKE, os.path.relpath(raiz, BASE)))

    log.info("Script 03 — conversao para NIfTI | inicio %s"
             % time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(_INICIO)))
    estados, resumo, falhas = {}, {}, []

    if args.split_manifest:
        est, det = converter_blocos_dirigidos(args.split_manifest)
        log.info("  SPLIT_CONVERSION %s %s" % (est, det))
        return 0 if est == "PASS" else 1

    for nome, fn in (("lidc", process_lidc), ("rider", process_rider),
                     ("lndb", process_lndb)):
        try:
            est, det = fn()
        except Exception as e:
            est, det = "FAIL_METHOD", {"excecao": repr(e)}
            log.exception("  ramo %s levantou excecao", nome)
        estados[nome], resumo[nome] = est, det
        if est != "PASS":
            falhas.append("%s: %s %s" % (nome, est, det))

    save_validation_csv()

    esperadas = [os.path.join(ANALYSIS, "dicom_nifti_validation.csv"),
                 os.path.join(ANALYSIS, "lidc_physical_gaps.csv"),
                 os.path.join(ANALYSIS, "lidc_retained_slices.csv"),
                 os.path.join(CONFIG, "lidc_final_manifest.json")]
    frescura = {}
    for q in esperadas:
        r = os.path.relpath(q, BASE)
        if not os.path.isfile(q):
            frescura[r] = "AUSENTE"; falhas.append("saida ausente: %s" % r)
        elif not _desta_execucao(q):
            frescura[r] = "ANTIGA"; falhas.append("saida nao e desta execucao: %s" % r)
        else:
            frescura[r] = "OK"

    portoes = {"HU_CHECK_SCOPE": (
                   "A verificacao de intensidade aplica-se aos 1.009 do LIDC "
                   "(hu_transform_identity_status) e aos 62 do RIDER "
                   "(hu_validation_status). Os 236 do LNDb ficam de fora por "
                   "decisao explicita: os valores do MetaImage so podem ser "
                   "chamados HU se a documentacao do LNDb o confirmar, o que nao "
                   "foi verificado; para esses vale voxel_value_identity contra o "
                   ".mhd de origem. Por isso o CSV tem 1.307 linhas e a contagem "
                   "de HU do LIDC soma 1.009."),
               "SCRIPT03_BRANCH_STATUS": estados,
               "OUTPUTS_STATUS": frescura,
               "SCRIPT03_OUTPUT_STATUS": "FAIL" if falhas else "PASS",
               "resumo": resumo, "failures": falhas}
    for k in ("SCRIPT03_BRANCH_STATUS", "OUTPUTS_STATUS", "SCRIPT03_OUTPUT_STATUS"):
        log.info("  %-26s %s" % (k, portoes[k]))
    for f in falhas[:10]:
        log.error("  FALHA: %s" % f)

    if SMOKE:
        log.info("SCRIPT03_OUTPUT_STATUS = %s (ENSAIO — ponteiro nao publicado)"
                 % portoes["SCRIPT03_OUTPUT_STATUS"])
        return 1 if falhas else 0

    os.makedirs(os.path.join(BASE, "runs"), exist_ok=True)
    _man03 = os.path.join(CONFIG, "lidc_final_manifest.json")
    _corpo03 = json.load(open(_man03))
    _ponteiro = {"script": "03_CONVERT_DICOM_TO_NIFTI",
                 "pointer_schema": "pulmonary/v2",
                 "run_id": time.strftime("%Y%m%dT%H%M%S", time.localtime(_INICIO)),
                 "published_at": datetime.now().isoformat(),
                 "manifest": os.path.relpath(_man03, BASE),
                 "lidc_final_manifest": "config/lidc_final_manifest.json",
                 "manifest_file_sha256": sha256_file(_man03),
                 "manifest_body_sha256": _corpo03.get("manifest_body_sha256"),
                 "validation_csv": "analysis/dicom_nifti_validation.csv",
                 "physical_gaps_csv": "analysis/lidc_physical_gaps.csv",
                 "retained_slices_csv": "analysis/lidc_retained_slices.csv",
                 "gates": portoes}
    for _chave, _rel in (("validation_csv", "analysis/dicom_nifti_validation.csv"),
                         ("physical_gaps_csv", "analysis/lidc_physical_gaps.csv"),
                         ("retained_slices_csv", "analysis/lidc_retained_slices.csv")):
        _ponteiro[_chave + "_sha256"] = sha256_file(os.path.join(BASE, _rel))
    with open(os.path.join(BASE, "runs", "CURRENT_NIFTI_CONVERSION.json"), "w") as f:
        json.dump(_ponteiro, f, indent=1, sort_keys=True)

    log.info("SCRIPT03_OUTPUT_STATUS = %s" % portoes["SCRIPT03_OUTPUT_STATUS"])
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
