#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    02_audit_raw.py

File type:
    Script

Purpose:
    Audits the raw data: inventory and selection of patients, studies, series, slices and XML annotations, orientation and geometry checks.

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
    - rider_pair_selection.json

Outputs:
    - benchmark.json
    - run_summary.json

Dependencies:
    - numpy
    - pydicom

Execution:
    PULMONARY_BASE=<project_dir> python scripts/02_audit_raw.py

Portability notes:
    All paths are resolved relative to the project base directory given by the PULMONARY_BASE environment variable. No machine-specific path is required.

Reproducibility notes:
    Environment pinned in requirements.txt and environment/. Seeds and frozen parameters are recorded in the run manifests under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) are not included and must be obtained from their original repositories.
"""

import os
import re
import csv
import sys
import json
import time
import hashlib
import collections
import logging
import argparse
import xml.etree.ElementTree as ET
from datetime import datetime
from collections import defaultdict, Counter

import numpy as np
import pydicom

from pulmonary_base import BASE, verify_csv, read_csv_rows, check_base

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("script02")


RAW_LIDC  = os.path.join(BASE, "data/raw/lidc_idri")

RAW_RIDER = os.path.join(BASE, "data/raw/rider")

RAW_LNDB  = os.path.join(BASE, "data/raw/lndb")

ANALYSIS  = os.path.join(BASE, "analysis")

os.makedirs(ANALYSIS, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

inventory_rows  = []

metadata_rows   = []

exclusion_rows  = []

rider_pair_rows = []

lndb_rows       = []

def _dicom_tag(ds, tag, default=""):
    try:
        return str(ds[tag].value)
    except (KeyError, AttributeError):
        return default

def audit_lidc():
    log.info("=== Auditoria LIDC-IDRI ===")
    if not os.path.isdir(RAW_LIDC):
        log.warning(f"Pasta não encontrada: {RAW_LIDC}")
        exclusion_rows.append({"dataset": "LIDC-IDRI", "id": "", "reason": "pasta ausente", "detail": ""})
        return

    series_dirs = sorted([d for d in os.listdir(RAW_LIDC)
                          if os.path.isdir(os.path.join(RAW_LIDC, d))])
    log.info(f"Diretórios de série encontrados: {len(series_dirs)}")

    patient_series = defaultdict(list)

    for ser in series_dirs:
        ser_dir = os.path.join(RAW_LIDC, ser)
        dcm_files = sorted([f for f in os.listdir(ser_dir) if f.endswith(".dcm")])
        if not dcm_files:
            continue
        try:
            ds = pydicom.dcmread(os.path.join(ser_dir, dcm_files[0]),
                                 stop_before_pixels=True)
            modality = _dicom_tag(ds, (0x0008, 0x0060))
            if modality != "CT":
                continue
            pid         = _dicom_tag(ds, (0x0010, 0x0020))
            sid         = _dicom_tag(ds, (0x0020, 0x000D))
            orientation = _dicom_tag(ds, (0x0020, 0x0037))
            spacing_xy  = _dicom_tag(ds, (0x0028, 0x0030))
            thickness   = (_dicom_tag(ds, (0x0050, 0x0088), "") or
                           _dicom_tag(ds, (0x0018, 0x0050)))
            age = _dicom_tag(ds, (0x0010, 0x1010))
            sex = _dicom_tag(ds, (0x0010, 0x0040))
            xml_files   = [f for f in os.listdir(ser_dir) if f.endswith(".xml")]
            try:
                thick_f = float(thickness) if thickness else float("inf")
            except ValueError:
                thick_f = float("inf")
            patient_series[pid].append({
                "series_uid":      ser,
                "study_uid":       sid,
                "n_slices":        len(dcm_files),
                "thick_f":         thick_f,
                "thickness_str":   thickness,
                "orientation":     orientation,
                "spacing_xy":      spacing_xy,
                "age":             age,
                "sex":             sex,
                "xml_annotations": len(xml_files),
            })
            if not thickness:
                exclusion_rows.append({
                    "dataset": "LIDC-IDRI",
                    "id":      f"{pid}/{ser}",
                    "reason":  "slice_thickness ausente",
                    "detail":  "",
                })
        except Exception as e:
            log.warning(f"  Erro em {ser_dir}: {e}")

    for pid, series_list in patient_series.items():
        primary = series_list[0]
        for i, s in enumerate(series_list):
            selected = (i == 0)
            metadata_rows.append({
                "dataset":            "LIDC-IDRI",
                "patient_id":         pid,
                "study_uid":          s["study_uid"],
                "series_uid":         s["series_uid"],
                "n_slices":           s["n_slices"],
                "orientation":        s["orientation"],
                "spacing_xy":         s["spacing_xy"],
                "slice_thickness":    s["thickness_str"],
                "age":                s["age"],
                "sex":                s["sex"],
                "xml_annotations":    s["xml_annotations"],
                "selected_for_nifti": selected,
            })
            if not selected:
                exclusion_rows.append({
                    "dataset": "LIDC-IDRI",
                    "id":      f"{pid}/{s['series_uid']}",
                    "reason":  "segunda_serie_nao_utilizada",
                    "detail":  (f"série primária={primary['series_uid']}; "
                                f"critério: primeira série por ordem alfabética de "
                                f"SeriesInstanceUID (alinhado às anotações do script 04)"),
                })

    n_series_lidc = sum(1 for r in metadata_rows if r["dataset"] == "LIDC-IDRI")
    n_multi = sum(1 for sl in patient_series.values() if len(sl) > 1)
    inventory_rows.append({
        "dataset":          "LIDC-IDRI",
        "n_patients":       len(patient_series),
        "n_series_ct":      n_series_lidc,
        "n_multi_series":   n_multi,
        "n_series_dirs":    len(series_dirs),
    })
    log.info(f"LIDC-IDRI: {len(patient_series)} pacientes, {n_series_lidc} séries CT, "
             f"{n_multi} pacientes com dupla série")

PAIR_SELECTION_VERSION = "2026-07-30.acquisition-level.v1"

RIDER_JANELA_MIN   = 15.0

RIDER_ESPESSURA_MM = 1.25

RIDER_KERNEL       = "LUNG"

RIDER_MATRIZ_ESPERADA = 512

RIDER_ANGULAR_TOL_DEG = 5.0

def _seg(date_str, time_str):
    """(AAAAMMDD, HHMMSS[.frac]) -> segundos absolutos. None se incompleto."""
    if not date_str or not time_str:
        return None
    try:
        t = str(time_str).split(".")[0].zfill(6)
        return (int(date_str) * 86400
                + int(t[:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6]))
    except (ValueError, TypeError):
        return None

def _rider_datetime(ds):
    """
    Resolve o instante da aquisicao segundo a hierarquia ratificada:

      1. AcquisitionDateTime (0008,002A)
      2. AcquisitionDate + AcquisitionTime
      3. data resolvida + SeriesTime
      4. StudyDate + StudyTime  -- apenas identifica a sessao

    StudyTime marca o inicio do estudo e pode ser identico nas duas aquisicoes,
    logo nao comprova sozinho o intervalo. Um par sustentado por StudyTime em
    qualquer dos lados fica fora da analise principal.

    Devolve (segundos, texto_iso, fonte).
    """
    def t(tag):
        try:
            return str(ds[tag].value).strip()
        except (KeyError, AttributeError):
            return ""

    adt = t((0x0008, 0x002A))
    if adt and len(adt) >= 14:
        s = _seg(adt[:8], adt[8:])
        if s is not None:
            return s, f"{adt[:8]}T{adt[8:14]}", "AcquisitionDateTime"

    acq_d, acq_t = t((0x0008, 0x0022)), t((0x0008, 0x0032))
    data = acq_d or t((0x0008, 0x0020))
    if data and acq_t:
        s = _seg(data, acq_t)
        if s is not None:
            fonte = ("AcquisitionDate+AcquisitionTime" if acq_d
                     else "StudyDate+AcquisitionTime")
            return s, f"{data}T{acq_t.split('.')[0]}", fonte

    ser_t = t((0x0008, 0x0031))
    if data and ser_t:
        s = _seg(data, ser_t)
        if s is not None:
            return s, f"{data}T{ser_t.split('.')[0]}", "Date+SeriesTime"

    std_d, std_t = t((0x0008, 0x0020)), t((0x0008, 0x0030))
    if std_d and std_t:
        s = _seg(std_d, std_t)
        if s is not None:
            return s, f"{std_d}T{std_t.split('.')[0]}", "StudyTime"

    return None, "", "INDISPONIVEL"

def _is_axial(iop_str):
    """
    Verdadeiro quando a orientacao corresponde a uma pilha axial.
    A comparacao e numerica: a coleccao grava o mesmo cosseno em formatos
    textuais diferentes, e comparar strings classificaria series axiais como
    nao axiais.
    """
    try:
        v = [float(x) for x in
             str(iop_str).replace("[", "").replace("]", "").split(",")]
        if len(v) != 6:
            return False
        return (abs(v[0]) > 0.98 and abs(v[4]) > 0.98
                and abs(v[2]) < 0.15 and abs(v[5]) < 0.15)
    except (ValueError, TypeError):
        return False

def _primeiro_float(s):
    try:
        return float(str(s).replace("[", "").replace("]", "").split(",")[0])
    except (ValueError, TypeError, IndexError):
        return None

def _par_espacamento(sp_str):
    """PixelSpacing -> (linha_mm, coluna_mm). (None, None) se ilegivel."""
    try:
        v = [float(x) for x in
             str(sp_str).replace("[", "").replace("]", "").split(",")]
        return (v[0], v[1]) if len(v) >= 2 else (None, None)
    except (ValueError, TypeError):
        return (None, None)

def _espacamento_valido(sp_str, rows, cols):
    """
    Validade geometrica do espacamento de UMA serie. Nao compara com a outra
    aquisicao: diferencas intrapar sao legitimas e tratadas no espaco fisico.

    Devolve (ok, motivo).
    """
    a, b = _par_espacamento(sp_str)
    if a is None or b is None:
        return False, "PixelSpacing ilegivel"
    for v in (a, b):
        if not (v > 0) or v != v or v in (float("inf"), float("-inf")):
            return False, f"PixelSpacing nao positivo ou nao finito ({a}, {b})"
    try:
        r, c = int(rows), int(cols)
    except (ValueError, TypeError):
        return False, "Rows/Columns ilegiveis"
    if r != RIDER_MATRIZ_ESPERADA or c != RIDER_MATRIZ_ESPERADA:
        return False, f"matriz {r}x{c}, esperada {RIDER_MATRIZ_ESPERADA}x{RIDER_MATRIZ_ESPERADA}"
    return True, "ok"

def _pilha_consistente(series_dir):
    """
    Confirma que o espacamento e a matriz sao constantes dentro da propria pilha,
    lendo o cabecalho de TODOS os cortes da serie.

    Devolve (ok, motivo, n_cortes_lidos).
    """
    try:
        fs = sorted(f for f in os.listdir(series_dir) if f.endswith(".dcm"))
    except OSError as e:
        return False, f"serie inacessivel: {e}", 0
    if not fs:
        return False, "serie sem ficheiros .dcm", 0

    combos = set()
    for fn in fs:
        try:
            ds = pydicom.dcmread(os.path.join(series_dir, fn), stop_before_pixels=True)
        except Exception as e:
            return False, f"corte ilegivel {fn}: {e}", len(combos)
        combos.add((str(_dicom_tag(ds, (0x0028, 0x0030))),
                    str(_dicom_tag(ds, (0x0028, 0x0010))),
                    str(_dicom_tag(ds, (0x0028, 0x0011)))))
        if len(combos) > 1:
            return False, f"espacamento/matriz variam dentro da pilha: {sorted(combos)}", len(fs)
    return True, "ok", len(fs)

def _normal(iop_str):
    """Vetor normal da pilha, pelo produto vetorial dos cossenos de direcao."""
    try:
        v = [float(x) for x in
             str(iop_str).replace("[", "").replace("]", "").split(",")]
        if len(v) != 6:
            return None
        r, c = v[:3], v[3:]
        n = [r[1]*c[2] - r[2]*c[1], r[2]*c[0] - r[0]*c[2], r[0]*c[1] - r[1]*c[0]]
        m = sum(x*x for x in n) ** 0.5
        return [x/m for x in n] if m > 0 else None
    except (ValueError, TypeError):
        return None

def _angulo_graus(iop_a, iop_b):
    """Angulo entre as normais das duas pilhas, em graus. None se indeterminavel."""
    import math
    na, nb = _normal(iop_a), _normal(iop_b)
    if na is None or nb is None:
        return None
    d = max(-1.0, min(1.0, sum(x*y for x, y in zip(na, nb))))
    return round(math.degrees(math.acos(abs(d))), 3)

def audit_rider():
    """
    Forma os pares teste-reteste do RIDER ao nivel da AQUISICAO.

    O RIDER Lung CT repete a varredura na mesma sessao, mesmo equipamento e
    mesmos parametros, com intervalo inferior a 15 minutos; a reconstrucao
    classica tem 1,25 mm e kernel pulmonar. A regra anterior agrupava por
    StudyInstanceUID e tomava os dois estudos mais antigos, o que em pacientes
    com seguimento longitudinal selecionava exames separados por semanas.

    Condicoes desta coleccao, medidas e nao presumidas:
      - ProtocolName e SeriesDescription vazios em 100% das 561 series;
      - AcquisitionDateTime vazio em 100%;
      - AcquisitionNumber igual a 1 em 100% -- nao discrimina aquisicoes e por
        isso e registado como campo de auditoria, nao como criterio;
      - AcquisitionTime presente em 89%.

    ImageType e campo de auditoria: DERIVED significa apenas que os pixels
    derivam de outra imagem, e a reconstrucao classica de 1,25 mm com kernel
    pulmonar desta coleccao e derivada. A exclusao recai sobre localizadores,
    scouts e pilhas nao axiais ou de espessura incompativel.
    """
    log.info("=== Auditoria RIDER ===")
    if not os.path.isdir(RAW_RIDER):
        log.warning(f"Pasta nao encontrada: {RAW_RIDER}")
        exclusion_rows.append({"dataset": "RIDER", "reason": "pasta ausente", "id": ""})
        return

    series_dirs = sorted([d for d in os.listdir(RAW_RIDER)
                          if os.path.isdir(os.path.join(RAW_RIDER, d))])
    MIN_CORTES, MAX_ESPESSURA_MM = 10, 20.0

    nao_elegiveis = Counter()
    por_paciente = defaultdict(list)
    for ser in series_dirs:
        ser_dir = os.path.join(RAW_RIDER, ser)
        dcm_files = [f for f in os.listdir(ser_dir) if f.endswith(".dcm")]
        if not dcm_files:
            continue
        try:
            ds = pydicom.dcmread(os.path.join(ser_dir, dcm_files[0]),
                                 stop_before_pixels=True)
            if _dicom_tag(ds, (0x0008, 0x0060)) != "CT":
                continue
            image_type = _dicom_tag(ds, (0x0008, 0x0008)).upper()
            iop        = _dicom_tag(ds, (0x0020, 0x0037))
            try:
                espessura = float(_dicom_tag(ds, (0x0018, 0x0050)))
            except ValueError:
                espessura = float("inf")

            motivo = None
            if not _is_axial(iop):
                motivo = f"orientacao nao axial {iop[:40]}"
            elif "LOCALIZER" in image_type or "SCOUT" in image_type:
                motivo = "localizador/scout por ImageType"
            elif len(dcm_files) < MIN_CORTES:
                motivo = f"pilha com {len(dcm_files)} cortes"
            elif espessura > MAX_ESPESSURA_MM:
                motivo = f"espessura {espessura} mm incompativel"
            else:
                geo_ok, geo_motivo = _espacamento_valido(
                    _dicom_tag(ds, (0x0028, 0x0030)),
                    _dicom_tag(ds, (0x0028, 0x0010)),
                    _dicom_tag(ds, (0x0028, 0x0011)))
                if not geo_ok:
                    motivo = f"geometria invalida: {geo_motivo}"
                else:
                    pilha_ok, pilha_motivo, _n = _pilha_consistente(ser_dir)
                    if not pilha_ok:
                        motivo = f"pilha inconsistente: {pilha_motivo}"
            if motivo:
                categoria = ("scout_nao_axial" if "nao axial" in motivo
                             or "localizador" in motivo else "serie_nao_elegivel")
                nao_elegiveis[categoria] += 1
                exclusion_rows.append({
                    "dataset": "RIDER", "id": ser,
                    "reason": categoria,
                    "detail": f"{motivo}; image_type={image_type}",
                })
                continue

            sec, iso, fonte = _rider_datetime(ds)
            reg = {
                "patient_id": _dicom_tag(ds, (0x0010, 0x0020)),
                "study_uid":  _dicom_tag(ds, (0x0020, 0x000D)),
                "series_uid": _dicom_tag(ds, (0x0020, 0x000E)),
                "series_dir": ser_dir,
                "n_slices":   len(dcm_files),
                "acq_number": _dicom_tag(ds, (0x0020, 0x0012)),
                "thickness":  espessura,
                "kernel":     _dicom_tag(ds, (0x0018, 0x1210)),
                "pixel_spacing": _dicom_tag(ds, (0x0028, 0x0030)),
                "rows": _dicom_tag(ds, (0x0028, 0x0010)),
                "cols": _dicom_tag(ds, (0x0028, 0x0011)),
                "iop": iop,
                "image_type": "ORIGINAL" if "ORIGINAL" in image_type else "DERIVED",
                "sec": sec, "iso": iso, "fonte": fonte,
            }
            por_paciente[reg["patient_id"]].append(reg)
            metadata_rows.append({**reg, "dataset": "RIDER"})
        except Exception as e:
            log.warning(f"  Erro em {ser_dir}: {e}")

    n_ok = n_fail = 0
    for pid, series in sorted(por_paciente.items()):
        aquisicoes = defaultdict(list)
        for s in series:
            if s["sec"] is not None:
                aquisicoes[s["sec"]].append(s)
        instantes = sorted(aquisicoes)

        validos, rejeitados = [], []
        for i in range(len(instantes)):
            for j in range(i + 1, len(instantes)):
                minutos = (instantes[j] - instantes[i]) / 60.0
                for a in aquisicoes[instantes[i]]:
                    for b in aquisicoes[instantes[j]]:
                        so_studytime = "StudyTime" == a["fonte"] or "StudyTime" == b["fonte"]
                        angulo = _angulo_graus(a["iop"], b["iop"])

                        falha = None
                        if a["series_uid"] == b["series_uid"]:
                            falha = "mesma serie"
                        elif not (0 < minutos <= RIDER_JANELA_MIN):
                            falha = f"intervalo {minutos:.1f} min fora de (0, 15]"
                        elif a["thickness"] != b["thickness"]:
                            falha = f"espessuras {a['thickness']} != {b['thickness']}"
                        elif a["kernel"] != b["kernel"]:
                            falha = f"kernels {a['kernel']} != {b['kernel']}"
                        elif angulo is None or angulo > RIDER_ANGULAR_TOL_DEG:
                            falha = f"orientacao divergente ({angulo} graus)"
                        elif so_studytime:
                            falha = "intervalo sustentado apenas por StudyTime"
                        if falha:
                            rejeitados.append((minutos, falha, a, b))
                        else:
                            validos.append((
                                abs(a["thickness"] - RIDER_ESPESSURA_MM),
                                0 if a["kernel"] == RIDER_KERNEL else 1,
                                minutos, a, b, angulo))

        if not validos:
            n_fail += 1
            perto = min(rejeitados, key=lambda r: abs(r[0])) if rejeitados else None
            razao = (f"nenhuma dupla de aquisicoes elegivel em ate "
                     f"{RIDER_JANELA_MIN:.0f} min no arquivo disponivel "
                     f"({len(instantes)} aquisicoes distintas)")
            log.warning(f"  {pid}: FAIL_PAIR_SELECTION - {razao}")
            exclusion_rows.append({"dataset": "RIDER", "id": pid,
                                   "reason": "FAIL_PAIR_SELECTION", "detail": razao})
            linha = {
                "patient_id": pid, "ct1_dir": "", "ct2_dir": "",
                "pair_status": "FAIL_PAIR_SELECTION", "failure_reason": razao,
                "n_valid_candidate_pairs": 0,
                "n_candidate_pairs": len(rejeitados),
                "pair_selection_version": PAIR_SELECTION_VERSION,
                "interval_verification_status": "NOT_APPLICABLE",
            }
            if perto:
                m, f, a, b = perto
                linha.update({
                    "nearest_candidate_minutes": round(m, 2),
                    "nearest_candidate_reason":  f,
                    "nearest_candidate_ct1_series_uid": a["series_uid"],
                    "nearest_candidate_ct2_series_uid": b["series_uid"],
                    "nearest_candidate_ct1_datetime":   a["iso"],
                    "nearest_candidate_ct2_datetime":   b["iso"],
                })
            rider_pair_rows.append(linha)
            continue

        validos.sort(key=lambda c: (c[0], c[1], c[2]))
        _, _, minutos, a, b, angulo = validos[0]
        n_ok += 1

        a_sx, a_sy = _par_espacamento(a["pixel_spacing"])
        b_sx, b_sy = _par_espacamento(b["pixel_spacing"])
        a_fx, a_fy = a_sx * int(a["cols"]), a_sy * int(a["rows"])
        b_fx, b_fy = b_sx * int(b["cols"]), b_sy * int(b["rows"])
        fov_inteiro = all(abs(v - round(v)) < 0.01 for v in (a_fx, a_fy, b_fx, b_fy))

        ok1, mot1, n1 = _pilha_consistente(a["series_dir"])
        ok2, mot2, n2 = _pilha_consistente(b["series_dir"])
        if not (ok1 and ok2):
            log.warning(f"  {pid}: pilha inconsistente — "
                        f"CT1 {mot1} · CT2 {mot2}")
        estado_intervalo = ("VERIFIED"
                            if "StudyTime" not in (a["fonte"], b["fonte"])
                            else "UNVERIFIED_STUDYTIME_ONLY")
        rider_pair_rows.append({
            "patient_id": pid,
            "ct1_dir": a["series_dir"], "ct2_dir": b["series_dir"],
            "ct1_date": a["iso"][:8],   "ct2_date": b["iso"][:8],
            "ct1_acquisition_datetime": a["iso"], "ct2_acquisition_datetime": b["iso"],
            "ct1_datetime_source": a["fonte"], "ct2_datetime_source": b["fonte"],
            "interval_verification_status": estado_intervalo,
            "minutes_between": round(minutos, 2),
            "within_15min": 0 < minutos <= RIDER_JANELA_MIN,
            "ct1_study_uid": a["study_uid"],   "ct2_study_uid": b["study_uid"],
            "ct1_series_uid": a["series_uid"], "ct2_series_uid": b["series_uid"],
            "ct1_acquisition_number": a["acq_number"],
            "ct2_acquisition_number": b["acq_number"],
            "ct1_slice_thickness": a["thickness"], "ct2_slice_thickness": b["thickness"],
            "ct1_kernel": a["kernel"], "ct2_kernel": b["kernel"],
            "ct1_image_type": a["image_type"], "ct2_image_type": b["image_type"],
            "ct1_pixel_spacing": a["pixel_spacing"], "ct2_pixel_spacing": b["pixel_spacing"],
            "ct1_matrix": f"{a['rows']}x{a['cols']}",
            "ct2_matrix": f"{b['rows']}x{b['cols']}",
            "ct1_fov_x_mm": round(a_fx, 3), "ct1_fov_y_mm": round(a_fy, 3),
            "ct2_fov_x_mm": round(b_fx, 3), "ct2_fov_y_mm": round(b_fy, 3),
            "fov_difference_x_mm": round(abs(a_fx - b_fx), 3),
            "fov_difference_y_mm": round(abs(a_fy - b_fy), 3),
            "fov_integer_mm": fov_inteiro,
            "ct1_stack_consistent": ok1, "ct2_stack_consistent": ok2,
            "ct1_slices_verified": n1,   "ct2_slices_verified": n2,
            "orientation_angle_deg":       angulo,
            "orientation_tolerance_deg":   RIDER_ANGULAR_TOL_DEG,
            "n_slices_ct1": a["n_slices"], "n_slices_ct2": b["n_slices"],
            "n_valid_candidate_pairs": len(validos),
            "n_candidate_pairs": len(validos) + len(rejeitados),
            "pair_selection_basis": (
                f"espessura {a['thickness']}mm, kernel {a['kernel']}, "
                f"intervalo {minutos:.1f} min, {len(validos)} candidatos validos"),
            "pair_selection_version": PAIR_SELECTION_VERSION,
            "pair_status": "PASS_PAIR_SELECTION",
            "failure_reason": "",
        })
        log.info(f"  {pid}: {a['thickness']}mm/{a['kernel']} {minutos:.1f} min "
                 f"cortes {a['n_slices']}/{b['n_slices']} [{estado_intervalo}]")

    inventory_rows.append({
        "dataset": "RIDER", "n_patients": len(por_paciente),
        "n_pairs": n_ok, "n_fail_pair": n_fail,
        "n_series_dirs": len(series_dirs),
        "n_scouts_excluded":        nao_elegiveis["scout_nao_axial"],
        "n_other_series_excluded":  nao_elegiveis["serie_nao_elegivel"],
        
        "orientation_tolerance_deg": RIDER_ANGULAR_TOL_DEG,
        "pair_selection_version":   PAIR_SELECTION_VERSION,
    })
    log.info(f"RIDER: {len(por_paciente)} pacientes, {n_ok} PASS_PAIR_SELECTION, "
             f"{n_fail} FAIL_PAIR_SELECTION · excluidas "
             f"{nao_elegiveis['scout_nao_axial']} scouts/nao axiais e "
             f"{nao_elegiveis['serie_nao_elegivel']} outras series")

def audit_lndb():
    log.info("=== Auditoria LNDb ===")
    if not os.path.isdir(RAW_LNDB):
        log.warning(f"Pasta não encontrada: {RAW_LNDB}")
        exclusion_rows.append({"dataset": "LNDb", "reason": "pasta ausente", "id": ""})
        return

    data_dir  = os.path.join(RAW_LNDB, "data")
    masks_dir = os.path.join(RAW_LNDB, "gt_masks")

    if not os.path.isdir(data_dir):
        log.warning(f"Subpasta data/ não encontrada: {data_dir}")
        exclusion_rows.append({"dataset": "LNDb", "reason": "data/ ausente", "id": ""})
        return
    mhd_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".mhd")])

    nifti_masks = sorted([f for f in os.listdir(masks_dir)
                           if f.endswith(".nii.gz")]) if os.path.isdir(masks_dir) else []
    masks_set = {f.replace("LNDb_", "").replace(".nii.gz", "").lstrip("0") or "0"
                 for f in nifti_masks}

    train_csv = os.path.join(RAW_LNDB, "trainCTs.csv")
    train_ct_ids = set()
    if os.path.isfile(train_csv):
        with open(train_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                train_ct_ids.add(int(row.get("LNDbID", 0)))
    else:
        log.warning("trainCTs.csv não encontrado — flag in_train_subset indisponível")

    csv_ann = os.path.join(RAW_LNDB, "trainNodules.csv")
    nodule_rows_csv = []
    if os.path.isfile(csv_ann):
        with open(csv_ann) as f:
            reader = csv.DictReader(f)
            nodule_rows_csv = list(reader)
    else:
        log.warning("trainNodules.csv não encontrado — contagens de nódulos indisponíveis")

    cases_with_nodules = set()
    for row in nodule_rows_csv:
        if str(row.get("Nodule", "0")) == "1":
            cases_with_nodules.add(int(row.get("LNDbID", 0)))

    for mhd in mhd_files:
        lid_str = mhd.replace("LNDb-", "").replace(".mhd", "")
        try:
            lid = int(lid_str)
        except ValueError:
            continue
        has_mask = str(lid) in masks_set
        has_nodules = lid in cases_with_nodules
        lndb_rows.append({
            "lndb_id":          f"LNDb-{lid:04d}",
            "mhd_file":         mhd,
            "has_consensus_mask": has_mask,
            "has_nodules":      has_nodules,
            "in_train_subset":  lid in train_ct_ids,
        })
        if not has_mask:
            exclusion_rows.append({
                "dataset": "LNDb",
                "reason":  "máscara de consenso ausente",
                "id":      f"LNDb-{lid:04d}",
            })

    cases_no_nodules = train_ct_ids - cases_with_nodules
    inventory_rows.append({
        "dataset": "LNDb",
        "n_cts": len(mhd_files),
        "n_consensus_masks": len(nifti_masks),
        "n_annotation_rows": len(nodule_rows_csv),
        "n_cases_with_nodules": len(cases_with_nodules),
        "n_cases_no_nodules": len(cases_no_nodules),
        "n_series_dirs": len(mhd_files),
    })
    log.info(f"LNDb: {len(mhd_files)} CTs, {len(nifti_masks)} máscaras de consenso, "
             f"{len(cases_with_nodules)} com nódulos")

EXCLUSION_FIELDNAMES = ["dataset", "id", "reason", "detail"]

def write_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        if not rows:
            log.warning(f"Sem dados e sem fieldnames para {path} — arquivo não criado")
            return
        seen, fieldnames = set(), []
        for row in rows:
            for k in row:
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", restval="")
        w.writeheader()
        if rows:
            w.writerows(rows)
    n = len(rows)
    log.info(f"Salvo: {path} ({n} {'linha' if n == 1 else 'linhas'})")

OUT_INVENTORY = os.path.join(ANALYSIS, "dataset_inventory.csv")

OUT_METADATA  = os.path.join(ANALYSIS, "dicom_metadata_audit.csv")

OUT_EXCLUSION = os.path.join(ANALYSIS, "exclusion_reasons.csv")

OUT_RIDER     = os.path.join(ANALYSIS, "rider_pair_manifest.csv")

OUT_LNDB      = os.path.join(ANALYSIS, "lndb_subset_manifest.csv")

def _count_series_dirs(root):
    if not os.path.isdir(root):
        return 0
    return sum(1 for d in os.listdir(root)
               if os.path.isdir(os.path.join(root, d)))

def verify_outputs():
    """
    Decide se as cinco saídas desta etapa podem ser aceites sem recalcular.

    Verifica VALORES, não presença: contagens contra a fonte em disco, unicidade
    de chaves, existência dos diretórios referenciados e cobertura integral do
    LNDb. Um CSV presente mas desatualizado ou truncado é reprovado aqui — se
    passasse, o script apresentaria a auditoria antiga como se fosse a atual, e
    os scripts 03 e 04, que consomem estes ficheiros, herdariam o erro.

    Devolve (ok, reason).
    """
    ok, reason = verify_csv(
        OUT_INVENTORY,
        required_cols=["dataset", "n_series_dirs"],
        min_rows=3, unique_key=("dataset",),
    )
    if not ok:
        return False, f"dataset_inventory.csv: {reason}"

    inv = {r["dataset"]: r for r in read_csv_rows(OUT_INVENTORY)}
    for ds in ("LIDC-IDRI", "RIDER", "LNDb"):
        if ds not in inv:
            return False, f"dataset_inventory.csv sem linha para {ds}"

    disk_counts = {
        "LIDC-IDRI": _count_series_dirs(RAW_LIDC),
        "RIDER":     _count_series_dirs(RAW_RIDER),
        "LNDb":      len([f for f in os.listdir(os.path.join(RAW_LNDB, "data"))
                          if f.endswith(".mhd")])
                     if os.path.isdir(os.path.join(RAW_LNDB, "data")) else 0,
    }
    for ds, n_disk in disk_counts.items():
        try:
            n_rec = int(inv[ds].get("n_series_dirs", "") or -1)
        except ValueError:
            n_rec = -1
        if n_rec != n_disk:
            return False, (f"{ds}: fonte mudou desde a última auditoria "
                           f"({n_disk} em disco vs {n_rec} registados)")

    ok, reason = verify_csv(
        OUT_METADATA,
        required_cols=["dataset", "series_uid"],
        min_rows=1, unique_key=("dataset", "series_uid"),
    )
    if not ok:
        return False, f"dicom_metadata_audit.csv: {reason}"

    for row in read_csv_rows(OUT_METADATA):
        ds, ser = row.get("dataset", ""), row.get("series_uid", "")
        root = {"LIDC-IDRI": RAW_LIDC, "RIDER": RAW_RIDER}.get(ds)
        if root and ser and not os.path.isdir(os.path.join(root, ser)):
            return False, f"dicom_metadata_audit.csv aponta para série inexistente: {ds}/{ser}"

    ok, reason = verify_csv(OUT_EXCLUSION,
                            required_cols=EXCLUSION_FIELDNAMES, min_rows=0)
    if not ok:
        return False, f"exclusion_reasons.csv: {reason}"

    ok, reason = verify_csv(
        OUT_RIDER,
        required_cols=["patient_id", "ct1_dir", "ct2_dir", "pair_status",
                       "minutes_between", "within_15min", "failure_reason",
                       "ct1_datetime_source", "ct2_datetime_source",
                       "interval_verification_status", "n_valid_candidate_pairs",
                       "pair_selection_version",
                       "ct1_fov_x_mm", "ct1_fov_y_mm", "ct2_fov_x_mm",
                       "ct2_fov_y_mm", "fov_difference_x_mm",
                       "fov_difference_y_mm", "fov_integer_mm",
                       "ct1_stack_consistent", "ct2_stack_consistent"],
        min_rows=1, unique_key=("patient_id",),
    )
    if not ok:
        return False, f"rider_pair_manifest.csv: {reason}"

    motivos_exclusao = {r.get("id", ""): r.get("reason", "")
                        for r in read_csv_rows(OUT_EXCLUSION)}
    linhas = read_csv_rows(OUT_RIDER)
    series_usadas, n_pass, n_fail = [], 0, 0

    for row in linhas:
        pid    = row.get("patient_id", "")
        estado = row.get("pair_status", "")
        d1, d2 = row.get("ct1_dir", ""), row.get("ct2_dir", "")

        if estado == "FAIL_PAIR_SELECTION":
            n_fail += 1
            if d1 or d2:
                return False, (f"{pid}: FAIL_PAIR_SELECTION com diretório preenchido")
            if not row.get("failure_reason", "").strip():
                return False, f"{pid}: FAIL_PAIR_SELECTION sem failure_reason"
            if str(row.get("n_valid_candidate_pairs", "")).strip() != "0":
                return False, (f"{pid}: FAIL_PAIR_SELECTION com "
                               f"n_valid_candidate_pairs="
                               f"{row.get('n_valid_candidate_pairs')!r}")
            if motivos_exclusao.get(pid) != "FAIL_PAIR_SELECTION":
                return False, (f"{pid}: excluído sem motivo correspondente em "
                               f"exclusion_reasons.csv")
            continue

        if estado != "PASS_PAIR_SELECTION":
            return False, f"{pid}: pair_status={estado!r}"
        n_pass += 1

        if d1 == d2:
            return False, f"{pid}: CT1 e CT2 no mesmo diretório"
        for d in (d1, d2):
            if not os.path.isdir(d):
                return False, f"{pid}: diretório ausente: {d}"
        if str(row.get("within_15min", "")).strip().lower() != "true":
            return False, (f"{pid}: aceite com within_15min="
                           f"{row.get('within_15min')!r} "
                           f"({row.get('minutes_between')} min)")
        try:
            minutos = float(row.get("minutes_between", ""))
        except ValueError:
            return False, f"{pid}: minutes_between não numérico"
        if not (0 < minutos <= 15.0):
            return False, f"{pid}: intervalo {minutos} min fora de (0, 15]"

        if row.get("ct1_series_uid") == row.get("ct2_series_uid"):
            return False, f"{pid}: a mesma série em CT1 e CT2"
        if row.get("ct1_acquisition_datetime") == row.get("ct2_acquisition_datetime"):
            return False, (f"{pid}: CT1 e CT2 na mesma aquisição — são "
                           f"reconstruções, não teste–reteste")
        if row.get("ct1_slice_thickness") != row.get("ct2_slice_thickness"):
            return False, f"{pid}: espessuras diferentes entre CT1 e CT2"
        if row.get("ct1_kernel") != row.get("ct2_kernel"):
            return False, f"{pid}: kernels diferentes entre CT1 e CT2"

        if "StudyTime" in (row.get("ct1_datetime_source", ""),
                           row.get("ct2_datetime_source", "")):
            return False, (f"{pid}: intervalo sustentado por StudyTime "
                           f"({row.get('ct1_datetime_source')} / "
                           f"{row.get('ct2_datetime_source')})")
        if row.get("interval_verification_status") != "VERIFIED":
            return False, (f"{pid}: interval_verification_status="
                           f"{row.get('interval_verification_status')!r}")

        series_usadas += [row.get("ct1_series_uid", ""), row.get("ct2_series_uid", "")]

    if len(series_usadas) != len(set(series_usadas)):
        return False, ("rider_pair_manifest.csv: série reutilizada entre pacientes "
                       f"({len(series_usadas)} seleções, "
                       f"{len(set(series_usadas))} únicas)")

    aceites = [r for r in linhas if r.get("pair_status") == "PASS_PAIR_SELECTION"]

    espessuras = {r.get("ct1_slice_thickness") for r in aceites}
    if espessuras != {"1.25"}:
        return False, f"rider_pair_manifest.csv: espessuras selecionadas {espessuras}"

    for r in aceites:
        pid = r.get("patient_id", "")
        if r.get("fov_integer_mm", "").strip().lower() != "true":
            return False, (f"{pid}: FOV nao inteiro em mm "
                           f"({r.get('ct1_fov_x_mm')}, {r.get('ct2_fov_x_mm')})")
        for lado in ("ct1", "ct2"):
            if r.get(f"{lado}_stack_consistent", "").strip().lower() != "true":
                return False, (f"{pid}: espacamento ou matriz variam dentro da "
                               f"pilha de {lado.upper()}")
            if r.get(f"{lado}_matrix") != "512x512":
                return False, (f"{pid}: matriz de {lado.upper()} = "
                               f"{r.get(f'{lado}_matrix')}, esperada 512x512")

    import hashlib
    with open(OUT_RIDER, "rb") as f:
        manifest_sha = hashlib.sha256(f.read()).hexdigest()
    with open(os.path.join(BASE, "config", "rider_pair_selection.json"), "w") as f:
        json.dump({
            "pair_selection_version": PAIR_SELECTION_VERSION,
            "pair_manifest_sha256":   manifest_sha,
            "manifest_path":          OUT_RIDER,
            "n_pass":                 n_pass,
            "n_fail":                 n_fail,
            "n_series_selected":      len(series_usadas),
            "generated":              datetime.now().isoformat(),
        }, f, indent=2)
    log.info(f"RIDER: {n_pass} PASS / {n_fail} FAIL · {len(series_usadas)} séries "
             f"únicas · manifesto sha256 {manifest_sha[:12]} · "
             f"regra {PAIR_SELECTION_VERSION}")

    mhd_ids = {f.replace(".mhd", "") for f in
               os.listdir(os.path.join(RAW_LNDB, "data")) if f.endswith(".mhd")}
    ok, reason = verify_csv(
        OUT_LNDB,
        required_cols=["lndb_id", "mhd_file"],
        min_rows=1, unique_key=("lndb_id",),
        expected_ids=mhd_ids,
        id_col="lndb_id",
    )
    if not ok:
        return False, f"lndb_subset_manifest.csv: {reason}"

    return True, (f"5 CSVs conferidos contra a fonte "
                  f"(LIDC {disk_counts['LIDC-IDRI']}, RIDER {disk_counts['RIDER']}, "
                  f"LNDb {disk_counts['LNDb']} séries)")


PROTOCOL_SHA = "082695723bf7877494a09c33b02593c7a03a244b63bed0b1e7e659b5af0c58e6"

SCHEMA_SHA = "5a4803f804bfb313f540f11631a84a3f397808d69be1df2bd77e6e8dbf794459"

RUN_SERIAL = time.strftime("%Y%m%dT%H%M%S")
_INICIO_EXECUCAO = time.time()

ROUTING_VERSION = "2026-08-03.content_routing.v1"

COLECOES = {
    "lidc_ct": {
        "dataset": "LIDC-IDRI", "root": os.path.join(BASE, "data", "raw", "lidc_idri"),
        "file_types": ["DICOM"], "allowed_modalities": ["CT"],
        "role": "CT_SERIES", "canonical": True},
    "lidc_seg": {
        "dataset": "LIDC-IDRI", "root": os.path.join(BASE, "data", "raw", "lidc_seg"),
        "file_types": ["DICOM"], "allowed_modalities": ["SEG"],
        "role": "SEG_SERIES", "canonical": False,
        "mirror_of": "lidc_annotations",
        "mirror_note": "espelho parcial byte-identico de lidc_annotations/seg"},
    "lidc_annotations": {
        "dataset": "LIDC-IDRI", "root": os.path.join(BASE, "data", "raw", "lidc_annotations"),
        "file_types": ["XML", "DICOM"], "allowed_modalities": ["SEG", "SR"],
        "role": "ANNOTATION_MIXED", "canonical": True,
        "inventory_name": "lidc_annotations"},
    "rider": {
        "dataset": "RIDER", "root": os.path.join(BASE, "data", "raw", "rider"),
        "file_types": ["DICOM"], "allowed_modalities": ["CT"],
        "role": "CT_SERIES", "canonical": True},
    "lndb": {
        "dataset": "LNDb", "root": os.path.join(BASE, "data", "raw", "lndb"),
        "file_types": ["MHD_RAW"], "allowed_modalities": [],
        "role": "MHD_HEADER", "canonical": True},
}

SAIDA_BASE = os.path.join(BASE, "audits", "script02_canonical")

SAIDA = os.path.join(SAIDA_BASE, "run%s" % RUN_SERIAL)

INVENTARIO = os.path.join(SAIDA, "local_physical_files.csv")

FASES = ["inventory", "headers", "pixels", "entities", "report"]

NAO_IMPLEMENTADAS = {"pixels", "report"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

COLS = ["CHECK_ID", "ANALYTICAL_DATASET", "SOURCE_COLLECTION", "ENTITY_ID",
        "ARTIFACT_ROLE", "APPLICABILITY", "OBSERVED_VALUE", "EXPECTED_VALUE",
        "EVIDENCE_SOURCE", "STATUS", "REASON_CODE"]

class Registo:
    """
    Acumulador do registo uniforme. Grava incrementalmente: uma auditoria de
    357 mil ficheiros nao cabe em memoria como lista de dicionarios.
    """

    def __init__(self, caminho):
        novo = not os.path.isfile(caminho)
        self.fh = open(caminho, "a", newline="")
        self.w = csv.DictWriter(self.fh, fieldnames=COLS, extrasaction="ignore", restval="")
        if novo:
            self.w.writeheader()
        self.n = 0
        self.estados = collections.Counter()

    def add(self, check, dataset, colecao, entidade, papel, status,
            observado="", esperado="", evidencia="", razao="", aplicavel=True):
        if not aplicavel:
            status = "PASS"
            assert razao, "NOT_APPLICABLE sem REASON_CODE em %s/%s" % (check, entidade)
        self.w.writerow({
            "CHECK_ID": check, "ANALYTICAL_DATASET": dataset,
            "SOURCE_COLLECTION": colecao, "ENTITY_ID": entidade,
            "ARTIFACT_ROLE": papel,
            "APPLICABILITY": "APPLICABLE" if aplicavel else "NOT_APPLICABLE",
            "OBSERVED_VALUE": observado, "EXPECTED_VALUE": esperado,
            "EVIDENCE_SOURCE": evidencia, "STATUS": status, "REASON_CODE": razao})
        self.n += 1
        self.estados[status] += 1
        if self.n % 5000 == 0:
            self.fh.flush()

    def close(self):
        self.fh.flush()
        self.fh.close()

def _sha(p, bloco=4 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bloco), b""):
            h.update(b)
    return h.hexdigest()

TIPOS = ("DICOM", "XML", "MHD", "RAW", "OTHER", "EMPTY", "UNREADABLE")

def _sniff(p):
    """
    Identifica o tipo PELO CONTEUDO. Nunca pela extensao nem pelo diretorio.

    DICOM      preambulo de 128 bytes seguido de "DICM" (PS3.10), ou, na
               ausencia de preambulo, o primeiro elemento do grupo 0002/0008
               em little-endian — como escrevem alguns arquivos historicos.
    XML        declaracao `<?xml` ou primeiro caracter nao-branco `<`.
    MHD/RAW    cabecalho ITK MetaImage `ObjectType =` ou `NDims =`.

    Devolve (tipo, evidencia). A evidencia entra no registo: uma classificacao
    sem justificacao legivel nao e auditavel.
    """
    try:
        with open(p, "rb") as f:
            h = f.read(256)
    except OSError as e:
        return "UNREADABLE", "errno=%s" % e.errno
    if not h:
        return "EMPTY", "0 bytes"
    if len(h) >= 132 and h[128:132] == b"DICM":
        return "DICOM", "PS3.10_PREAMBLE_DICM"
    s = h.lstrip()
    if s[:5].lower() == b"<?xml":
        return "XML", "XML_DECLARATION"
    if s[:1] == b"<":
        return "XML", "XML_ELEMENT_START"
    u = s[:40].upper()
    if u.startswith(b"OBJECTTYPE") or u.startswith(b"NDIMS"):
        return "MHD", "ITK_METAIMAGE_HEADER"
    if len(h) >= 8 and h[0:2] in (b"\x02\x00", b"\x08\x00"):
        return "DICOM", "DICOM_NO_PREAMBLE_GROUP_%02X%02X" % (h[1], h[0])
    return "OTHER", "NO_KNOWN_SIGNATURE"

def _inventario_por_colecao():
    """Caminhos relativos do inventario congelado, por coleccao. Sem re-hash."""
    inv = collections.defaultdict(list)
    for r in csv.DictReader(open(INVENTARIO)):
        if r["read_status"] == "SYMLINK":
            continue
        inv[r["source_collection"]].append(r["relative_path"])
    return inv

def fase_inventory(reg, colecoes, limite=None):
    """
    Inventario fisico + SHA-256. Constroi o `LOCAL_PHYSICAL_FILES`, base de tudo
    o resto. Sem hash, nao ha `LOCAL_CORPUS_FROZEN`.
    """
    p = os.path.join(SAIDA, "local_physical_files.csv")
    novo = not os.path.isfile(p)
    fh = open(p, "a", newline="")
    cols = ["source_collection", "relative_path", "size_bytes", "mtime",
            "sha256", "read_status"]
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    if novo:
        w.writeheader()
    total_b = total_f = 0
    t0 = time.time()
    for nome, c in colecoes.items():
        raiz = c["root"]
        if not os.path.isdir(raiz):
            reg.add("I00", c["dataset"], nome, nome, "OTHER", "FAIL_INPUT",
                    observado="ausente", esperado=raiz, evidencia=raiz,
                    razao="SOURCE_ROOT_MISSING")
            continue
        n = 0
        for d, _s, fs in os.walk(raiz):
            for f in sorted(fs):
                ab = os.path.join(d, f)
                rel = os.path.relpath(ab, raiz)
                if os.path.islink(ab):
                    w.writerow({"source_collection": nome, "relative_path": rel,
                                "read_status": "SYMLINK"})
                    continue
                try:
                    st = os.stat(ab)
                    h = _sha(ab)
                    estado = "OK"
                    total_b += st.st_size
                except OSError as e:
                    st, h, estado = None, "", "READ_ERROR:%s" % e.errno
                    reg.add("I01", c["dataset"], nome, rel, c["role"], "FAIL_INPUT",
                            observado=estado, esperado="legivel", evidencia=rel,
                            razao="FILE_UNREADABLE")
                w.writerow({"source_collection": nome, "relative_path": rel,
                            "size_bytes": st.st_size if st else "",
                            "mtime": int(st.st_mtime) if st else "",
                            "sha256": h, "read_status": estado})
                n += 1
                total_f += 1
                if limite and n >= limite:
                    break
            if limite and n >= limite:
                break
        log.info("    %-18s %6d ficheiros | %.1f s" % (nome, n, time.time() - t0))
    fh.close()
    log.info("  inventory: %d ficheiros | %.2f GiB | %.0f s"
             % (total_f, total_b / 2**30, time.time() - t0))
    return {"files": total_f, "bytes": total_b, "seconds": round(time.time() - t0, 1)}

CAMPOS_DICOM = ("SOPInstanceUID", "SeriesInstanceUID", "StudyInstanceUID", "PatientID",
                "Modality", "ImageOrientationPatient", "ImagePositionPatient",
                "Rows", "Columns", "PixelSpacing", "SliceThickness",
                "RescaleSlope", "RescaleIntercept", "ImageType", "SeriesDescription",
                "SOPClassUID", "ConvolutionKernel", "AcquisitionNumber", "InstanceNumber",
                "NumberOfFrames")

COLS_REF = ["source_collection", "relative_path", "src_modality", "src_sop_uid",
            "src_series_uid", "src_study_uid", "ref_level", "ref_study_uid",
            "ref_series_uid", "ref_sop_uid", "ref_path"]

def _ler_cabecalho_dicom(p):
    import pydicom
    ds = pydicom.dcmread(p, stop_before_pixels=True, force=False)
    d = {}
    for k in CAMPOS_DICOM:
        v = getattr(ds, k, None)
        d[k] = None if v is None else (list(v) if hasattr(v, "__iter__")
                                       and not isinstance(v, str) else str(v))
    d["TransferSyntaxUID"] = str(getattr(ds.file_meta, "TransferSyntaxUID", ""))
    return d, ds

def _referencias(ds):
    """
    Arestas de referencia DICOM, para o vinculo SEG/SR -> CT.

    SEG  ReferencedSeriesSequence -> SeriesInstanceUID + ReferencedInstanceSequence
    SR   CurrentRequestedProcedureEvidenceSequence -> estudo/serie/SOP

    Extrai-se APOS conhecer a modalidade lida, nunca antes. O SR do LIDC aponta
    para a serie SEG, e esta aponta para a serie CT: a cadeia e SR -> SEG -> CT
    e cada elo e registado separadamente.
    """
    out = []

    def _seq(o, nome):
        v = getattr(o, nome, None)
        return list(v) if v else []

    for rs in _seq(ds, "ReferencedSeriesSequence"):
        suid = str(getattr(rs, "SeriesInstanceUID", "") or "")
        inst = _seq(rs, "ReferencedInstanceSequence") or _seq(rs, "ReferencedImageSequence")
        if not inst:
            out.append({"ref_level": "SERIES", "ref_series_uid": suid,
                        "ref_study_uid": "", "ref_sop_uid": "",
                        "ref_path": "ReferencedSeriesSequence"})
        for it in inst:
            out.append({"ref_level": "SOP", "ref_series_uid": suid,
                        "ref_study_uid": "",
                        "ref_sop_uid": str(getattr(it, "ReferencedSOPInstanceUID", "") or ""),
                        "ref_path": "ReferencedSeriesSequence/ReferencedInstanceSequence"})

    for ev in _seq(ds, "CurrentRequestedProcedureEvidenceSequence"):
        stu = str(getattr(ev, "StudyInstanceUID", "") or "")
        ser = _seq(ev, "ReferencedSeriesSequence")
        if not ser:
            out.append({"ref_level": "STUDY", "ref_series_uid": "",
                        "ref_study_uid": stu, "ref_sop_uid": "",
                        "ref_path": "CurrentRequestedProcedureEvidenceSequence"})
        for se in ser:
            suid = str(getattr(se, "SeriesInstanceUID", "") or "")
            sops = _seq(se, "ReferencedSOPSequence")
            if not sops:
                out.append({"ref_level": "SERIES", "ref_series_uid": suid,
                            "ref_study_uid": stu, "ref_sop_uid": "",
                            "ref_path": "CurrentRequestedProcedureEvidenceSequence/"
                                        "ReferencedSeriesSequence"})
            for sp in sops:
                out.append({"ref_level": "SOP", "ref_series_uid": suid,
                            "ref_study_uid": stu,
                            "ref_sop_uid": str(getattr(sp, "ReferencedSOPInstanceUID", "") or ""),
                            "ref_path": "CurrentRequestedProcedureEvidenceSequence/"
                                        "ReferencedSeriesSequence/ReferencedSOPSequence"})
    return out

def fase_headers(reg, colecoes, limite=None):
    """
    Roteamento por conteudo + D00–D10 de nivel de ficheiro. NAO descodifica
    imagem: `stop_before_pixels`. O `PixelData` fica para a fase `pixels`, que e
    ordens de grandeza mais cara.

    Percorre o INVENTARIO CONGELADO, nao o sistema de ficheiros: assim a
    cobertura da leitura e comparavel, ficheiro a ficheiro, com o que foi
    hasheado. Um ficheiro no inventario que nao apareca aqui e uma falha
    detetavel; se percorresse o disco outra vez, nao seria.
    """
    inv = _inventario_por_colecao()

    p_h = os.path.join(SAIDA, "dicom_headers.csv")
    p_r = os.path.join(SAIDA, "file_routing.csv")
    p_x = os.path.join(SAIDA, "dicom_references.csv")
    fh = open(p_h, "a", newline="")
    fr = open(p_r, "a", newline="")
    fx = open(p_x, "a", newline="")
    cols = ["source_collection", "relative_path"] + list(CAMPOS_DICOM) + \
           ["TransferSyntaxUID", "read_status"]
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore", restval="")
    wr = csv.DictWriter(fr, fieldnames=["source_collection", "relative_path",
                                        "sniffed_type", "sniff_evidence",
                                        "routing_decision", "modality_read"],
                        extrasaction="ignore", restval="")
    wx = csv.DictWriter(fx, fieldnames=COLS_REF, extrasaction="ignore", restval="")
    if not os.path.getsize(p_h):
        w.writeheader()
    if not os.path.getsize(p_r):
        wr.writeheader()
    if not os.path.getsize(p_x):
        wx.writeheader()

    t0 = time.time()
    n_ok = n_err = n_ref = 0
    tipos = collections.defaultdict(collections.Counter)
    modal = collections.defaultdict(collections.Counter)
    inesperadas = collections.Counter()
    nulas = collections.Counter()

    for nome, c in colecoes.items():
        raiz = c["root"]
        alvos = inv.get(c.get("inventory_name", nome), [])
        if not alvos:
            reg.add("D00", c["dataset"], nome, nome, c["role"], "FAIL_INPUT",
                    observado="0 ficheiros no inventario", esperado=">0",
                    evidencia=INVENTARIO, razao="COLLECTION_ABSENT_FROM_INVENTORY")
            continue
        n = 0
        for rel in alvos:
            ab = os.path.join(raiz, rel)
            tipo, ev = _sniff(ab)
            tipos[nome][tipo] += 1
            decisao, mod_lida = "SKIP_NOT_DICOM", ""

            if tipo == "DICOM":
                decisao = "READ_DICOM_HEADER"
                try:
                    hdr, ds = _ler_cabecalho_dicom(ab)
                    hdr.update({"source_collection": nome, "relative_path": rel,
                                "read_status": "OK"})
                    w.writerow(hdr)
                    n_ok += 1
                    mod_lida = (hdr.get("Modality") or "").strip().upper()
                    modal[nome][mod_lida or "(vazia)"] += 1

                    if not mod_lida:
                        nulas[nome] += 1
                        reg.add("D11", c["dataset"], nome, rel, c["role"], "FAIL_INPUT",
                                observado="vazia", esperado="|".join(c["allowed_modalities"]),
                                evidencia="Modality", razao="NULL_MODALITY")
                    elif c["allowed_modalities"] and mod_lida not in c["allowed_modalities"]:
                        inesperadas[nome] += 1
                        reg.add("D12", c["dataset"], nome, rel, c["role"], "WARNING",
                                observado=mod_lida, esperado="|".join(c["allowed_modalities"]),
                                evidencia="Modality", razao="UNEXPECTED_MODALITY")

                    if not hdr.get("SOPInstanceUID"):
                        reg.add("D02", c["dataset"], nome, rel, c["role"], "FAIL_INPUT",
                                observado="ausente", esperado="presente",
                                evidencia="SOPInstanceUID", razao="SOP_UID_MISSING")

                    if mod_lida in ("SEG", "SR"):
                        reg.add("D09", c["dataset"], nome, rel, c["role"], "PASS",
                                aplicavel=False, observado=mod_lida,
                                evidencia="Modality",
                                razao=("SEGMENTATION_HAS_NO_HU_SCALE" if mod_lida == "SEG"
                                       else "SR_HAS_NO_PIXEL_DATA"))
                    elif hdr.get("RescaleSlope") is None or hdr.get("RescaleIntercept") is None:
                        reg.add("D09", c["dataset"], nome, rel, c["role"], "WARNING",
                                observado="ausente", esperado="presente",
                                evidencia="RescaleSlope/Intercept",
                                razao="HU_CONVERSION_FIELDS_MISSING")

                    if mod_lida in ("SEG", "SR"):
                        base = {"source_collection": nome, "relative_path": rel,
                                "src_modality": mod_lida,
                                "src_sop_uid": hdr.get("SOPInstanceUID") or "",
                                "src_series_uid": hdr.get("SeriesInstanceUID") or "",
                                "src_study_uid": hdr.get("StudyInstanceUID") or ""}
                        arestas = _referencias(ds)
                        if not arestas:
                            reg.add("R00", c["dataset"], nome, rel, c["role"], "FAIL_INPUT",
                                    observado="0 arestas", esperado=">=1",
                                    evidencia="Referenced*Sequence",
                                    razao="ANNOTATION_WITHOUT_REFERENCE")
                        for a in arestas:
                            d = dict(base)
                            d.update(a)
                            wx.writerow(d)
                            n_ref += 1
                except Exception as e:
                    w.writerow({"source_collection": nome, "relative_path": rel,
                                "read_status": "ERROR:%s" % type(e).__name__})
                    decisao = "DICOM_PARSE_FAILURE"
                    reg.add("D01", c["dataset"], nome, rel, c["role"], "FAIL_INPUT",
                            observado=type(e).__name__, esperado="legivel",
                            evidencia=rel, razao="DICOM_HEADER_UNREADABLE")
                    n_err += 1
            elif tipo == "XML":
                decisao = ("DEFER_TO_SCRIPT04_XML" if "XML" in c["file_types"]
                           else "UNEXPECTED_XML_IN_COLLECTION")
            elif tipo == "MHD":
                decisao = "DEFER_TO_MHD_READER"
            elif tipo in ("EMPTY", "UNREADABLE"):
                decisao = "FAIL_" + tipo
                reg.add("D01", c["dataset"], nome, rel, c["role"], "FAIL_INPUT",
                        observado=tipo, esperado="legivel", evidencia=ev,
                        razao="FILE_%s" % tipo)

            wr.writerow({"source_collection": nome, "relative_path": rel,
                         "sniffed_type": tipo, "sniff_evidence": ev,
                         "routing_decision": decisao, "modality_read": mod_lida})
            n += 1
            if limite and n >= limite:
                break
        log.info("    %-18s %7d ficheiros | %s | %s | %.0f s"
                 % (nome, n, dict(tipos[nome]), dict(modal[nome]), time.time() - t0))
    fh.close(); fr.close(); fx.close()

    dt = time.time() - t0
    esperados = {k: len(inv.get(c.get("inventory_name", k), []))
                 for k, c in colecoes.items()}
    roteados = {k: sum(v.values()) for k, v in tipos.items()}
    for k in sorted(set(esperados) | set(roteados)):
        e, r = esperados.get(k, 0), roteados.get(k, 0)
        c = colecoes.get(k, {"dataset": "?", "role": "OTHER"})
        reg.add("C01", c["dataset"], k, k, c["role"],
                "PASS" if e == r else "FAIL_OUTPUT",
                observado=r, esperado=e, evidencia=INVENTARIO,
                razao="" if e == r else "ROUTING_COVERAGE_INCOMPLETE")

    log.info("  headers: %d lidos, %d erros, %d arestas | %.0f s | %.0f/s"
             % (n_ok, n_err, n_ref, dt, (n_ok + n_err) / dt if dt else 0))
    return {"headers_ok": n_ok, "parse_failures": n_err, "reference_edges": n_ref,
            "seconds": round(dt, 1),
            "ROUTING_VERSION": ROUTING_VERSION,
            "sniffed_types": {k: dict(v) for k, v in tipos.items()},
            "modalities_read": {k: dict(v) for k, v in modal.items()},
            "NULL_MODALITY": dict(nulas), "UNEXPECTED_MODALITY": dict(inesperadas),
            "inventory_files_by_collection": esperados,
            "routed_files_by_collection": roteados,
            "ROUTING_COVERAGE_COMPLETE": esperados == roteados}

def benchmark(n=300):
    """
    Mede as duas fases caras numa amostra, para dar estimativa MEDIDA — nunca
    extrapolada de outra grandeza. Foi assim que errei a estimativa de
    transferencia: extrapolei taxa agregada para canal unico.
    """
    import pydicom
    res = {}
    for nome, c in COLECOES.items():
        if "DICOM" not in c["file_types"]:
            continue
        alvo = []
        for d, _s, fs in os.walk(c["root"]):
            for f in fs:
                p = os.path.join(d, f)
                if _sniff(p)[0] == "DICOM":
                    alvo.append(p)
                    if len(alvo) >= n:
                        break
            if len(alvo) >= n:
                break
        if not alvo:
            continue
        t0 = time.time()
        b = 0
        for p in alvo:
            b += os.path.getsize(p)
            _sha(p)
        t_hash = time.time() - t0
        t0 = time.time()
        for p in alvo:
            _ler_cabecalho_dicom(p)
        t_hdr = time.time() - t0
        t0 = time.time()
        n_px = 0
        for p in alvo:
            ds = pydicom.dcmread(p)
            if "PixelData" in ds:
                _ = ds.pixel_array
                n_px += 1
        t_px = time.time() - t0
        res[nome] = {"n": len(alvo), "bytes": b, "n_with_pixeldata": n_px,
                     "sha256_s": round(t_hash, 2), "headers_s": round(t_hdr, 2),
                     "pixels_s": round(t_px, 2),
                     "sha256_per_s": round(len(alvo) / t_hash, 1) if t_hash else 0,
                     "headers_per_s": round(len(alvo) / t_hdr, 1) if t_hdr else 0,
                     "pixels_per_s": round(len(alvo) / t_px, 1) if t_px else 0}
        log.info("  %-18s n=%d | sha %.0f/s | headers %.0f/s | pixels %.0f/s"
                 % (nome, len(alvo), res[nome]["sha256_per_s"],
                    res[nome]["headers_per_s"], res[nome]["pixels_per_s"]))
    return res

def main_canonical():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default="")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="limite de ficheiros por coleccao; so para ensaio")
    a = ap.parse_args()
    check_base(strict=True)
    os.makedirs(SAIDA, exist_ok=True)

    log.info("Script 02 canonico — SOMENTE LEITURA, SEM REDE | execucao %s" % RUN_SERIAL)
    log.info("  protocolo %s..." % PROTOCOL_SHA[:16])
    log.info("  esquema   %s..." % SCHEMA_SHA[:16])
    log.info("  roteamento %s" % ROUTING_VERSION)

    if a.benchmark:
        log.info("BENCHMARK — amostra por coleccao com DICOM")
        r = benchmark()
        json.dump(r, open(os.path.join(SAIDA, "benchmark.json"), "w"), indent=1, sort_keys=True)
        return

    fases = FASES if a.phases == "all" else [f for f in a.phases.split(",") if f]
    invalidas = [f for f in fases if f not in FASES]
    if invalidas:
        log.error("  fases desconhecidas: %s" % invalidas)
        sys.exit(1)
    nao_implementadas = [f for f in fases if f in NAO_IMPLEMENTADAS]
    if nao_implementadas and a.phases != "all":
        log.error("  fases ainda NAO IMPLEMENTADAS: %s" % nao_implementadas)
        sys.exit(1)
    if a.phases == "all":
        fases = [f for f in fases if f not in NAO_IMPLEMENTADAS]
        log.warning("  --phases all limitado ao incremento implementado: %s" % fases)
        log.warning("  por implementar: %s" % sorted(NAO_IMPLEMENTADAS))
    fases = [f for f in fases if f != "entities"]

    reg = Registo(os.path.join(SAIDA, "uniform_check_record.csv"))
    resumo = {"RUN_SERIAL": RUN_SERIAL, "phases_run": fases}
    try:
        if "inventory" in fases:
            log.info("FASE inventory")
            resumo["inventory"] = fase_inventory(reg, COLECOES, a.limit)
        if "headers" in fases:
            log.info("FASE headers")
            resumo["headers"] = fase_headers(reg, COLECOES, a.limit)
    finally:
        reg.close()

    h = resumo.get("headers", {})
    portoes = {
        "PROTOCOL_SHA256": PROTOCOL_SHA, "SCHEMA_SHA256": SCHEMA_SHA,
        "ROUTING_VERSION": ROUTING_VERSION,
        "HEADERS_PREVIOUS": "INVALIDATED_ROUTING_ERROR",
        "NETWORK_ACCESS": "NOT_USED",
        "OFFICIAL_MANIFEST_AVAILABLE": "NO",
        "OFFICIAL_CONFRONTATION": "NOT_RUN",
        "SOURCE_PUBLISHED_CHECKSUM_VERIFICATION": "PENDING_EXTERNAL",
        "RIDER_ADJUDICATION_REFERENCE": "ABSENT",
        "SCRIPT02_CANONICAL": "BLOCKED",
        "reason_blocked": "confronto oficial nao executado; execucao offline",
        "DICOM_PARSE_FAILURES": h.get("parse_failures"),
        "NULL_MODALITY": h.get("NULL_MODALITY"),
        "UNEXPECTED_MODALITY": h.get("UNEXPECTED_MODALITY"),
        "ROUTING_COVERAGE_COMPLETE": h.get("ROUTING_COVERAGE_COMPLETE"),
        "PHASES_IMPLEMENTED": ["inventory", "headers", "entities(script proprio)"],
        "PHASES_NOT_IMPLEMENTED": sorted(NAO_IMPLEMENTADAS),
        "COVERAGE_INCOMPLETE_BY_DESIGN": True,
        "LIDC_SELECTION_MANIFEST": (os.path.relpath(MANIFESTO_SELECAO, BASE)
                                    if "lidc" in fases else ""),
        "LIDC_ANALYTIC_DENOMINATOR": (resumo.get("lidc_selection", {})
                                      .get("LIDC_ANALYTIC_DENOMINATOR")),
        "check_record_rows": reg.n,
        "check_record_status": dict(reg.estados),
    }
    resumo["gates"] = portoes
    json.dump(resumo, open(os.path.join(SAIDA, "run_summary.json"), "w"),
              indent=1, sort_keys=True, default=str)
    log.info("-" * 70)
    for k, v in portoes.items():
        if not isinstance(v, dict):
            log.info("  %-42s %s" % (k, v))
    log.info("  %-42s %s" % ("estados do registo", dict(reg.estados)))


MANIFESTO_SELECAO = os.path.join(BASE, "config", "lidc_selection_manifest.json")


def _sha_ficheiro(p, bs=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def _volume_sha256(serie_dir):
    """
    SHA-256 do VOLUME DE PIXEIS, cortes ordenados por posicao fisica.

    E este o criterio de duplicacao, e nao a igualdade de SOPInstanceUID: a
    desidentificacao da TCIA reatribui identificadores, de modo que o mesmo
    exame carregado sob dois PatientID tem UID diferentes e pixeis identicos.
    Medido em LIDC-IDRI-0029 e 0138: listas de SOP distintas, volume sha
    6c04f3fea852811b082854b6 igual nos dois, (273, 512, 512).

    Igualdade de SOP prova identidade quando bate; nao prova diferenca quando
    nao bate. Usa-la sozinha faria o mesmo exame entrar duas vezes na coorte e
    poder cair em folds distintos — vazamento.
    """
    regs = []
    for f in sorted(os.listdir(serie_dir)):
        if not f.lower().endswith(".dcm"):
            continue
        try:
            ds = pydicom.dcmread(os.path.join(serie_dir, f))
            z = float(ds.ImagePositionPatient[2])
            regs.append((z, ds.pixel_array))
        except Exception:
            return None, 0
    if not regs:
        return None, 0
    regs.sort(key=lambda t: t[0])
    arr = np.stack([a for _, a in regs])
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest(), len(regs)


def _geometria_do_grupo(dirs_por_paciente):
    """
    MEDE a geometria dos membros de um grupo duplicado, em vez de a declarar.

    O manifesto afirmava `MERGE_GEOMETRY_VERIFIED` por string fixa, sem que
    nenhuma linha comparasse IOP, PixelSpacing, espessura ou posicoes. Uma
    afirmacao escrita a mao nao e uma verificacao: aqui os quatro campos sao
    lidos dos dois registos e comparados, e o resultado e o que for medido.
    """
    perfis = {}
    for pid, d in sorted(dirs_por_paciente.items()):
        iop = esp = px = None
        zs = []
        try:
            for f in sorted(os.listdir(d)):
                if not f.lower().endswith(".dcm"):
                    continue
                ds = pydicom.dcmread(os.path.join(d, f), stop_before_pixels=True)
                zs.append(round(float(ds.ImagePositionPatient[2]), 4))
                if iop is None:
                    iop = [round(float(x), 6) for x in ds.ImageOrientationPatient]
                    px = [round(float(x), 6) for x in ds.PixelSpacing]
                    esp = round(float(getattr(ds, "SliceThickness", 0) or 0), 4)
        except Exception as e:
            return {"status": "NOT_MEASURABLE", "detail": "%s: %s" % (pid, e)}
        perfis[pid] = {"ImageOrientationPatient": iop, "PixelSpacing": px,
                       "SliceThickness": esp, "n_slices": len(zs),
                       "ImagePositionPatient_z": sorted(zs)}
    campos = ["ImageOrientationPatient", "PixelSpacing", "SliceThickness",
              "n_slices", "ImagePositionPatient_z"]
    ref = perfis[sorted(perfis)[0]]
    igual, divergentes = {}, []
    for c in campos:
        vs = [perfis[p][c] for p in sorted(perfis)]
        igual[c] = all(v == ref[c] for v in vs)
        if not igual[c]:
            divergentes.append(c)
    return {"status": "IDENTICAL" if not divergentes else "DIVERGENT",
            "fields_compared": campos, "fields_equal": igual,
            "divergent_fields": divergentes,
            "n_slices": {p: perfis[p]["n_slices"] for p in sorted(perfis)}}


def _sop_list_sha256(serie_dir):
    """SHA-256 da lista ordenada de SOPInstanceUID. Registado como metadado; NAO
    e o criterio de duplicacao."""
    sops = []
    for f in sorted(os.listdir(serie_dir)):
        if not f.lower().endswith(".dcm"):
            continue
        try:
            ds = pydicom.dcmread(os.path.join(serie_dir, f), stop_before_pixels=True,
                                 specific_tags=["SOPInstanceUID"])
            u = str(getattr(ds, "SOPInstanceUID", "") or "")
            if u:
                sops.append(u)
        except Exception:
            return None, 0
    if not sops:
        return None, 0
    sops.sort()
    return hashlib.sha256("\n".join(sops).encode()).hexdigest(), len(sops)


def _mhd_geometria(mhd_path):
    """Le o cabecalho MetaImage e devolve os campos geometricos normalizados."""
    campos = {}
    with open(mhd_path, "r", errors="replace") as f:
        for linha in f:
            if "=" not in linha:
                continue
            k, v = linha.split("=", 1)
            campos[k.strip()] = v.strip()
    return {k: campos.get(k, "") for k in
            ("NDims", "TransformMatrix", "Offset", "ElementSpacing", "DimSize",
             "ElementType", "AnatomicalOrientation", "BinaryDataByteOrderMSB")}


def emitir_manifesto_nominal_lndb():
    """
    Emite `config/lndb_nominal_manifest.json`: os 236 casos nominais do LNDb e os
    grupos de imagens duplicadas, detetados por MEDICAO.

    Este manifesto existia desde 31/07/2026 sem produtor nenhum no pipeline — era
    lido pelos Scripts 05 e 06 e escrito por nada. Passa a ser produzido aqui,
    onde a deteccao de duplicados ja vive para o LIDC, e pelo mesmo tipo de
    criterio: conteudo igual e o mesmo exame.

    Dois digests por caso, ambos sobre a FONTE, nunca sobre produto convertido:
      source_geometry_sha256  cabecalho MetaImage normalizado (dimensao,
                              espacamento, origem, matriz, tipo de elemento)
      source_semantic_sha256  bytes do `.raw`, isto e, os voxels

    Duplicado = os DOIS digests iguais. Geometria igual com voxels diferentes nao
    e duplicado; voxels iguais com geometria diferente e assinalado como
    divergencia, nunca fundido em silencio.
    """
    data_dir = os.path.join(RAW_LNDB, "data")
    if not os.path.isdir(data_dir):
        raise SystemExit("[FAIL] acervo LNDb ausente: %s" % data_dir)

    mhds = sorted(f for f in os.listdir(data_dir) if f.endswith(".mhd"))
    linhas, por_digest = [], defaultdict(list)
    t0 = time.time()
    for i, f in enumerate(mhds, 1):
        lndb_id = f[:-4]
        mhd = os.path.join(data_dir, f)
        raw = mhd[:-4] + ".raw"
        geom = _mhd_geometria(mhd)
        g_sha = hashlib.sha256(json.dumps(geom, sort_keys=True).encode()).hexdigest()
        s_sha, n_bytes = "", 0
        if os.path.isfile(raw):
            h = hashlib.sha256()
            with open(raw, "rb") as g:
                for b in iter(lambda: g.read(1 << 22), b""):
                    h.update(b); n_bytes += len(b)
            s_sha = h.hexdigest()
        linhas.append({"lndb_id": lndb_id,
                       "nominal_status": ("INCLUDED_NOMINAL" if s_sha
                                          else "EXCLUDED_RAW_MISSING"),
                       "mhd_relpath": os.path.relpath(mhd, BASE),
                       "raw_relpath": os.path.relpath(raw, BASE) if s_sha else "",
                       "dimsize": geom["DimSize"], "spacing": geom["ElementSpacing"],
                       "offset": geom["Offset"], "element_type": geom["ElementType"],
                       "n_slices": int((geom["DimSize"].split() or ["0", "0", "0"])[2]
                                       or 0),
                       "raw_bytes": n_bytes,
                       "sha256_mhd": _sha_ficheiro(mhd),
                       "sha256_raw": s_sha,
                       "source_geometry_sha256": g_sha,
                       "source_semantic_sha256": s_sha,
                       "duplicate_group_id": None, "duplicate_partner": None,
                       "analytic_status": "ANALYTIC_UNIT"})
        if s_sha:
            por_digest[(g_sha, s_sha)].append(lndb_id)
        if i % 50 == 0 or i == len(mhds):
            log.info("    LNDb digests %d/%d (%.0f s)" % (i, len(mhds), time.time() - t0))

    por_voxel = defaultdict(set)
    for (g_sha, s_sha), ids in por_digest.items():
        por_voxel[s_sha].add(g_sha)
    divergentes = sorted(s for s, gs in por_voxel.items() if len(gs) > 1)

    grupos, canonico = {}, {}
    for n, ((g_sha, s_sha), ids) in enumerate(
            sorted(((k, v) for k, v in por_digest.items() if len(v) > 1),
                   key=lambda t: sorted(t[1])), 1):
        gid = "LNDB_DUP_%02d" % n
        ids = sorted(ids)
        grupos[gid] = ids
        for x in ids:
            canonico[x] = (gid, [y for y in ids if y != x])

    por_id = {r["lndb_id"]: r for r in linhas}
    for x, (gid, parceiros) in canonico.items():
        por_id[x]["duplicate_group_id"] = gid
        por_id[x]["duplicate_partner"] = parceiros[0] if len(parceiros) == 1 else parceiros

    nominais = [r for r in linhas if r["nominal_status"] == "INCLUDED_NOMINAL"]
    distinto = len({(r["source_geometry_sha256"], r["source_semantic_sha256"])
                    for r in nominais})
    corpo = {
        "manifest_id": "lndb_nominal_%s" % RUN_SERIAL,
        "producer": "02_audit_raw.py",
        "generated": datetime.now().isoformat(),
        "LNDB_MANIFEST_STATUS": "FROZEN_NOMINAL",
        "LNDB_PROVENANCE_STATUS": "PASS",
        "LNDB_NOMINAL_COUNT": len(nominais),
        "LNDB_DISTINCT_IMAGE_CONTENT": distinto,
        "LNDB_DUPLICATE_GROUPS": len(grupos),
        "LNDB_DUPLICATE_CASES": sum(len(v) for v in grupos.values()),
        "LNDB_ANALYTIC_DENOMINATOR": distinto,
        "LNDB_ANALYTIC_SELECTION": "PENDING_SCRIPT05_INTRAPAIR_AUDIT",
        "DUPLICATE_RULE": ("igualdade simultanea de source_geometry_sha256 "
                           "(cabecalho MetaImage normalizado) e "
                           "source_semantic_sha256 (bytes do .raw)"),
        "GEOMETRY_DIVERGENCE_SAME_VOXELS": divergentes,
        "SOURCE_LEVEL_ONLY": ("este manifesto descreve a FONTE. Os digests do "
                              "NIfTI e o estado de conversao sao do Script 03 e "
                              "vivem na proveniencia de cada caso."),
        "duplicate_groups": grupos,
        "rows": linhas,
    }
    corpo["manifest_sha256"] = hashlib.sha256(
        json.dumps(corpo, indent=1, sort_keys=True).encode()).hexdigest()
    destino = os.path.join(BASE, "config", "lndb_nominal_manifest.json")
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    with open(destino, "w") as f:
        json.dump(corpo, f, indent=1, sort_keys=True)
    log.info("  manifesto nominal LNDb: %d casos · %d grupos duplicados (%d casos) "
             "· conteudo distinto %d"
             % (len(nominais), len(grupos), corpo["LNDB_DUPLICATE_CASES"], distinto))
    for gid, ids in sorted(grupos.items()):
        log.info("    %s: %s" % (gid, ids))
    if divergentes:
        log.warning("    voxels iguais com geometria diferente: %d" % len(divergentes))
    return corpo


def emitir_manifesto_selecao_lidc():
    """
    Emite a selecao canonica de serie por paciente, que o Script 03 converte.

    NAO contem `nifti_path` nem `sha256_nifti`: esses so existem depois da
    conversao e sao do Script 03. O manifesto antigo trazia-os porque era
    proveniencia retroativa, escrita depois do facto.

    Duplicados volumetricos sao detetados pela igualdade da lista de SOP, nao
    por suposicao: duas series com a mesma lista recebem o mesmo
    `duplicate_group_id`, e apenas uma fica `INCLUDED_CANONICAL`.
    """
    base = os.path.join(BASE, "data", "raw", "lidc_idri")
    if not os.path.isdir(base):
        raise SystemExit("[FAIL] acervo LIDC ausente: %s" % base)

    por_paciente = defaultdict(list)
    for uid in sorted(os.listdir(base)):
        d = os.path.join(base, uid)
        if not os.path.isdir(d):
            continue
        pid = None
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(".dcm"):
                try:
                    ds = pydicom.dcmread(os.path.join(d, f), stop_before_pixels=True,
                                         specific_tags=["PatientID", "StudyInstanceUID"])
                    pid = str(getattr(ds, "PatientID", "") or "")
                    stu = str(getattr(ds, "StudyInstanceUID", "") or "")
                except Exception:
                    pid = None
                break
        if not pid:
            continue
        vh, n = _volume_sha256(d)
        sh, _ = _sop_list_sha256(d)
        por_paciente[pid].append({"series_instance_uid": uid, "study_uid": stu,
                                  "volume_sha256": vh, "sop_list_sha256": sh,
                                  "n_sop_instances": n})

    sel = {}
    for pid, lst in por_paciente.items():
        lst.sort(key=lambda r: r["series_instance_uid"])
        sel[pid] = lst[0]

    por_sop = defaultdict(list)
    for pid, r in sel.items():
        if r["volume_sha256"]:
            por_sop[r["volume_sha256"]].append(pid)
    grupos, canonico = [], {}
    for sh, pids in sorted(por_sop.items()):
        if len(pids) < 2:
            continue
        pids = sorted(pids)
        gid = "DUP_%s" % sh[:12]
        def n_sessoes(p):
            xd = os.path.join(BASE, "data", "raw", "lidc_annotations", "xml")
            n = 0
            for r_, _, fs in os.walk(xd):
                for f in fs:
                    if f.endswith(".xml"):
                        try:
                            t = ET.parse(os.path.join(r_, f)).getroot()
                            u = t.find(".//{http://www.nih.gov}SeriesInstanceUid")
                            if u is not None and u.text and \
                               u.text.strip() == sel[p]["series_instance_uid"]:
                                return len(t.findall(
                                    ".//{http://www.nih.gov}readingSession"))
                        except Exception:
                            pass
            return n
        pontos = {p: n_sessoes(p) for p in pids}
        can = sorted(pids, key=lambda p: (-pontos[p], p))[0]
        canonico[can] = gid
        grupos.append({"duplicate_group_id": gid, "volume_sha256": sh,
                       "members": pids, "canonical_patient_id": can,
                       "reading_sessions": pontos,
                       "criterion": "igualdade do sha256 do volume de pixeis "
                                    "(cortes ordenados por posicao fisica); "
                                    "canonico = mais readingSession no XML",
                       "annotation_merge_status": "PENDING_SCRIPT04",
                       "geometry_check": _geometria_do_grupo(
                           {p: os.path.join(base, sel[p]["series_instance_uid"])
                            for p in pids}),
                       "member_series": {p: sel[p]["series_instance_uid"]
                                         for p in pids}})
        for p in pids:
            if p != can:
                canonico[p] = gid

    elig, n_inc, n_can, n_dup = [], 0, 0, 0
    for pid in sorted(sel):
        r = sel[pid]
        gid = canonico.get(pid, "")
        if not gid:
            st = "INCLUDED"; n_inc += 1
        elif any(g["canonical_patient_id"] == pid for g in grupos):
            st = "INCLUDED_MERGED_CANONICAL"; n_can += 1
        else:
            st = "MERGED_INTO_CANONICAL"; n_dup += 1
        if r["volume_sha256"] is None:
            st = "EXCLUDED_UNREADABLE"
        elig.append({"patient_id": pid, "series_instance_uid": r["series_instance_uid"],
                     "study_uid": r["study_uid"], "n_sop_instances": r["n_sop_instances"],
                     "volume_sha256": r["volume_sha256"] or "",
                     "sop_list_sha256": r["sop_list_sha256"] or "",
                     "duplicate_group_id": gid, "analytic_status": st,
                     "n_series_for_patient": len(por_paciente[pid])})

    analiticos = [e for e in elig if e["analytic_status"] in
                  ("INCLUDED", "INCLUDED_MERGED_CANONICAL")]
    man = {"manifest_id": "lidc_selection_%s" % RUN_SERIAL,
           "generated": datetime.now().isoformat(),
           "producer": "02_audit_raw.py",
           "SELECTION_RULE": "primeira serie por ordem alfabetica de SeriesInstanceUID",
           "DUPLICATE_RULE": ("igualdade do sha256 do volume de pixeis, cortes "
                              "ordenados por posicao fisica"),
           "LIDC_PATIENTS": len(sel),
           "LIDC_ANALYTIC_DENOMINATOR": len(analiticos),
           "INCLUDED": n_inc, "INCLUDED_MERGED_CANONICAL": n_can,
           "MERGED_INTO_CANONICAL": n_dup,
           "MERGE_RULE": ("REGRA A APLICAR PELO SCRIPT 04, ainda nao aplicada: "
                          "registos com volume identico sao o mesmo exame; as "
                          "sessoes de leitura de todos os membros passam a contar "
                          "sob o canonico, com chave de leitor (registo, sessao), "
                          "e R conta as sessoes de ambos"),
           "MERGE_STATUS": ("PENDING_SCRIPT04" if grupos else "NOT_APPLICABLE"),
           "MERGE_GEOMETRY_CHECK": {g["duplicate_group_id"]:
                                    g["geometry_check"]["status"] for g in grupos},
           "DENOMINATOR_CONDITIONAL_ON": (
               "LIDC_ANALYTIC_DENOMINATOR ja desconta os membros nao canonicos e "
               "so fica valido quando o Script 04 unir as leituras; enquanto "
               "MERGE_STATUS for PENDING_SCRIPT04, as sessoes dos membros "
               "MERGED_INTO_CANONICAL ainda nao estao representadas"),
           "EXCLUDED_UNREADABLE": sum(1 for e in elig
                                      if e["analytic_status"] == "EXCLUDED_UNREADABLE"),
           "duplicate_groups": grupos,
           "eligible": elig}
    os.makedirs(os.path.dirname(MANIFESTO_SELECAO), exist_ok=True)
    with open(MANIFESTO_SELECAO, "w") as f:
        json.dump(man, f, indent=1, sort_keys=True)
    log.info("  manifesto de selecao: %d pacientes · analiticos %d "
             "(INCLUDED %d + CANONICAL %d) · duplicados %d · ilegiveis %d"
             % (len(sel), len(analiticos), n_inc, n_can, n_dup,
                man["EXCLUDED_UNREADABLE"]))
    for g in grupos:
        log.info("    %s: %s → canonico %s (sessoes %s)"
                 % (g["duplicate_group_id"], g["members"],
                    g["canonical_patient_id"], g["reading_sessions"]))
    return man


FASES_TODAS = ["lidc", "rider", "lndb", "inventory", "headers"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default="all",
                    help="lidc,rider,lndb,inventory,headers ou all (omissao: all)")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="limite de ficheiros por coleccao; so para ensaio")
    a = ap.parse_args()
    check_base(strict=True)
    os.makedirs(SAIDA, exist_ok=True)
    os.makedirs(ANALYSIS, exist_ok=True)

    log.info("Script 02 — auditoria dos dados brutos | execucao %s" % RUN_SERIAL)
    log.info("  roteamento %s" % ROUTING_VERSION)

    if a.benchmark:
        r = benchmark()
        json.dump(r, open(os.path.join(SAIDA, "benchmark.json"), "w"),
                  indent=1, sort_keys=True)
        return 0

    fases = FASES_TODAS if a.phases == "all" else [f for f in a.phases.split(",") if f]
    if not fases:
        log.error("  nenhuma fase selecionada. Use --phases all ou uma lista.")
        return 1
    invalidas = [f for f in fases if f not in FASES_TODAS]
    if invalidas:
        log.error("  fases desconhecidas: %s" % invalidas)
        return 1
    log.info("  fases: %s" % ",".join(fases))

    resumo = {"RUN_SERIAL": RUN_SERIAL, "phases_run": fases,
              "started": datetime.now().isoformat()}
    falhas = []

    parte_a = [f for f in ("lidc", "rider", "lndb") if f in fases]
    if parte_a:
        if "lidc" in fases:
            log.info("FASE lidc")
            audit_lidc()
        if "rider" in fases:
            log.info("FASE rider")
            audit_rider()
        if "lndb" in fases:
            log.info("FASE lndb")
            audit_lndb()
            log.info("  emitindo manifesto nominal LNDb")
            _mln = emitir_manifesto_nominal_lndb()
            resumo["lndb_nominal"] = {k: v for k, v in _mln.items()
                                      if not isinstance(v, (list, dict))}
            for _g, _ids in _mln["duplicate_groups"].items():
                if len(_ids) != 2:
                    falhas.append("%s: %d membros, o 05 audita pares"
                                  % (_g, len(_ids)))
            if _mln["GEOMETRY_DIVERGENCE_SAME_VOXELS"]:
                falhas.append("LNDb: %d digests com voxels iguais e geometria "
                              "diferente" % len(_mln["GEOMETRY_DIVERGENCE_SAME_VOXELS"]))

        write_csv(OUT_INVENTORY, inventory_rows)
        write_csv(OUT_METADATA,  metadata_rows)
        write_csv(OUT_EXCLUSION, exclusion_rows, EXCLUSION_FIELDNAMES)
        write_csv(OUT_RIDER,     rider_pair_rows)
        write_csv(OUT_LNDB,      lndb_rows)
        resumo["parte_a_rows"] = {
            "dataset_inventory": len(inventory_rows),
            "dicom_metadata_audit": len(metadata_rows),
            "exclusion_reasons": len(exclusion_rows),
            "rider_pair_manifest": len(rider_pair_rows),
            "lndb_subset_manifest": len(lndb_rows)}
        log.info("  Parte A gravada: %s" % resumo["parte_a_rows"])

        ok_a, motivo_a = verify_outputs()
        resumo["parte_a_verificacao"] = motivo_a
        if not ok_a:
            log.error("  FAIL_OUTPUT — Parte A nao passou na verificacao: %s" % motivo_a)
            falhas.append("PARTE_A: %s" % motivo_a)
        else:
            log.info("  Parte A verificada: %s" % motivo_a)

        if "lidc" in fases:
            log.info("  emitindo manifesto de selecao LIDC")
            _man = emitir_manifesto_selecao_lidc()
            resumo["lidc_selection"] = {k: v for k, v in _man.items()
                                        if not isinstance(v, list)}
            for _g in _man.get("duplicate_groups", []):
                _st = _g["geometry_check"]["status"]
                log.info("    %s geometria: %s %s"
                         % (_g["duplicate_group_id"], _st,
                            _g["geometry_check"].get("divergent_fields") or ""))
                if _st != "IDENTICAL":
                    falhas.append("%s geometria %s: %s"
                                  % (_g["duplicate_group_id"], _st,
                                     _g["geometry_check"].get("divergent_fields")
                                     or _g["geometry_check"].get("detail")))
                if _g["annotation_merge_status"] == "PERFORMED":
                    falhas.append("%s: o Script 02 nao pode declarar fusao "
                                  "PERFORMED — quem funde e o Script 04"
                                  % _g["duplicate_group_id"])

    reg = Registo(os.path.join(SAIDA, "uniform_check_record.csv"))
    if "inventory" in fases:
        log.info("FASE inventory")
        resumo["inventory"] = fase_inventory(reg, COLECOES, a.limit)
    if "headers" in fases:
        log.info("FASE headers")
        resumo["headers"] = fase_headers(reg, COLECOES, a.limit)

    h = resumo.get("headers", {}) or {}
    portoes = {
        "HEADERS_OK": h.get("headers_ok"),
        "REFERENCE_EDGES": h.get("reference_edges"),
        "DICOM_PARSE_FAILURES": h.get("parse_failures"),
        "NULL_MODALITY": h.get("NULL_MODALITY"),
        "UNEXPECTED_MODALITY": h.get("UNEXPECTED_MODALITY"),
        "ROUTING_COVERAGE_COMPLETE": h.get("ROUTING_COVERAGE_COMPLETE"),
        "PHASES_RUN": fases,
        "PHASES_NOT_IMPLEMENTED": ["pixels", "report"],
        "ENTITIES_PHASE": "OUT_OF_SCOPE — nenhum estagio 00-27 consome as suas saidas",
        "LIDC_SELECTION_MANIFEST": (os.path.relpath(MANIFESTO_SELECAO, BASE)
                                    if "lidc" in fases else ""),
        "LIDC_ANALYTIC_DENOMINATOR": (resumo.get("lidc_selection", {})
                                      .get("LIDC_ANALYTIC_DENOMINATOR")),
        "check_record_rows": reg.n,
        "check_record_status": dict(reg.estados),
    }

    if "headers" in fases:
        for k in ("DICOM_PARSE_FAILURES", "NULL_MODALITY", "UNEXPECTED_MODALITY"):
            v = portoes.get(k)
            if isinstance(v, dict):
                if v:
                    falhas.append("%s=%s" % (k, v))
            elif v not in (0, None):
                falhas.append("%s=%s" % (k, v))
        if a.limit is None and portoes.get("ROUTING_COVERAGE_COMPLETE") is False:
            falhas.append("ROUTING_COVERAGE_COMPLETE=False: inventario %s != roteado %s"
                          % (h.get("inventory_files_by_collection"),
                             h.get("routed_files_by_collection")))
        if not portoes.get("HEADERS_OK"):
            falhas.append("HEADERS_OK vazio ou zero")
    if reg.n == 0 and any(f in fases for f in ("inventory", "headers")):
        falhas.append("check_record_rows=0 — nenhuma verificacao registada")

    esperadas = []
    if parte_a:
        esperadas += [OUT_INVENTORY, OUT_METADATA, OUT_EXCLUSION, OUT_RIDER, OUT_LNDB]
    if "lidc" in fases:
        esperadas.append(MANIFESTO_SELECAO)
    if "lndb" in fases:
        esperadas.append(os.path.join(BASE, "config", "lndb_nominal_manifest.json"))
    if "inventory" in fases:
        esperadas.append(os.path.join(SAIDA, "local_physical_files.csv"))
    if "headers" in fases:
        esperadas += [os.path.join(SAIDA, "dicom_headers.csv"),
                      os.path.join(SAIDA, "file_routing.csv"),
                      os.path.join(SAIDA, "dicom_references.csv")]
    _t0 = _INICIO_EXECUCAO
    frescura = {}
    for q in esperadas:
        r = os.path.relpath(q, BASE)
        if not os.path.isfile(q):
            falhas.append("saida ausente: %s" % r); frescura[r] = "AUSENTE"; continue
        mt = os.path.getmtime(q)
        if mt < _t0:
            falhas.append("saida NAO e desta execucao: %s (%s)"
                          % (r, time.strftime("%Y-%m-%d %H:%M", time.localtime(mt))))
            frescura[r] = "ANTIGA"
        else:
            frescura[r] = "OK"
    portoes["OUTPUTS_EXPECTED"] = len(esperadas)
    portoes["OUTPUTS_FRESH"] = sum(1 for v in frescura.values() if v == "OK")
    portoes["OUTPUTS_STATUS"] = frescura
    portoes["SCRIPT02_OUTPUT_STATUS"] = "FAIL" if falhas else "PASS"
    resumo["gates"] = portoes
    resumo["failures"] = falhas
    json.dump(resumo, open(os.path.join(SAIDA, "run_summary.json"), "w"),
              indent=1, sort_keys=True, default=str)

    log.info("-" * 70)
    for k, v in portoes.items():
        if not isinstance(v, dict):
            log.info("  %-40s %s" % (k, v))
    if falhas:
        log.error("  SCRIPT02_OUTPUT_STATUS = FAIL: %s" % "; ".join(falhas))
        return 1
    pm = os.path.join(SAIDA, "run_summary.json")
    ptr = {"run_id": RUN_SERIAL, "script": "02_AUDIT_RAW",
           "manifest": pm, "manifest_sha256": _sha_ficheiro(pm),
           "audit_dir": os.path.relpath(SAIDA, BASE),
           "dicom_headers": os.path.relpath(
               os.path.join(SAIDA, "dicom_headers.csv"), BASE),
           "lidc_selection_manifest": os.path.relpath(MANIFESTO_SELECAO, BASE),
           "published_at": datetime.now().isoformat()}
    pp = os.path.join(BASE, "runs", "CURRENT_RAW_AUDIT.json")
    os.makedirs(os.path.dirname(pp), exist_ok=True)
    tmp = pp + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ptr, f, indent=1, sort_keys=True)
    os.replace(tmp, pp)
    log.info("  ponteiro: runs/CURRENT_RAW_AUDIT.json -> %s" % RUN_SERIAL)
    log.info("  SCRIPT02_OUTPUT_STATUS = PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
