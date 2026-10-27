#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    04_build_lidc_annotations.py

File type:
    Script

Purpose:
    Builds the canonical LIDC-IDRI nodule annotations from the XML route (pylidc 0.2.1) on the canonical image grid.

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
    - report_04.json

Outputs:
    - See docs/file_level_documentation.md.

Dependencies:
    - SimpleITK
    - numpy
    - pylidc

Execution:
    PULMONARY_BASE=<project_dir> python scripts/04_build_lidc_annotations.py

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
import glob
import json
import math
import time
import shutil
import inspect
import hashlib
import logging
import argparse
import collections
import xml.etree.ElementTree as ET

import numpy as np
import pylidc as pl
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lidc_grid import carregar_grade_ct, mask_to_sitk, FalhaGrade

BASE = os.environ["PULMONARY_BASE"]
PONTEIRO_02 = os.path.join(BASE, "runs", "CURRENT_RAW_AUDIT.json")
PONTEIRO_03 = os.path.join(BASE, "runs", "CURRENT_NIFTI_CONVERSION.json")
HEADERS = None
RAW_XML = os.path.join(BASE, "data", "raw", "lidc_annotations", "xml")
RAW_LIDC = os.path.join(BASE, "data", "raw", "lidc_idri")
ANNOT   = os.path.join(BASE, "annotations")
ANALYSIS = os.path.join(BASE, "analysis")
CONFIG  = os.path.join(BASE, "config")
PONTEIRO_SAIDA = os.path.join(BASE, "runs", "CURRENT_LIDC_ANNOTATIONS.json")
MANIFESTO_SAIDA = os.path.join(CONFIG, "lidc_annotations_manifest.json")

NS = {"n": "http://www.nih.gov"}
TAU = 0.60
TOL_SLICE_MM = 0.001
ESTADOS = ("INCLUDED", "INCLUDED_MERGED_CANONICAL")
MAX_ANOTACOES_POR_CLUSTER = 4

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def sha_ficheiro(p, bs=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def impressao_arvore(raiz):
    """Nome+tamanho+mtime. Deteta escrita sem ler centenas de GB."""
    h = hashlib.sha256()
    for d, _, fs in os.walk(raiz):
        for f in sorted(fs):
            p = os.path.join(d, f)
            try:
                st = os.stat(p)
            except OSError:
                continue
            h.update(("%s|%d|%d\n" % (os.path.relpath(p, raiz), st.st_size,
                                      int(st.st_mtime))).encode())
    return h.hexdigest()


_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _ipp(bruto):
    """
    O Script 02 gravou `ImagePositionPatient` como repr de lista Python
    (`"['-191.2', '-152.9', '-48.0']"`), não no formato DICOM com `\\`.
    Aceita os dois e exige exatamente três componentes — um valor a menos é
    geometria inválida, não motivo para adivinhar.
    """
    if not bruto:
        return None
    v = [float(x) for x in _NUM.findall(bruto)]
    return v if len(v) == 3 else None


def carregar_geometria_ct(headers_path):
    """
    {SeriesInstanceUID → [(sop, ipp) ordenado por projeção no normal]}.

    A fonte é o `dicom_headers.csv` da execução vigente do Script 02, resolvido
    por ponteiro. Nenhum DICOM bruto é reaberto: a árvore fica intocada por
    construção.
    """
    por_serie = collections.defaultdict(list)
    with open(headers_path) as f:
        for r in csv.DictReader(f):
            if r["Modality"] != "CT" or r["read_status"] != "OK":
                continue
            v = _ipp(r["ImagePositionPatient"])
            if v is None:
                continue
            try:
                inst = int(float(r.get("InstanceNumber") or -1))
            except ValueError:
                inst = -1
            por_serie[r["SeriesInstanceUID"]].append((r["SOPInstanceUID"], v, inst))
    return por_serie


def indices_canonicos(ipps, grade, pid):
    """
    Projeta cada IPP no normal da grade e converte em índice de fatia.

    A tolerância de 0,001 mm é do contrato: o desvio entre a posição projetada e
    o centro da fatia tem de ser desprezável. Um desvio maior não é arredondado
    em silêncio — levanta `FalhaGrade`.
    """
    n = grade["normal"]
    o = np.asarray(grade["origin"], float)
    dz = float(grade["spacing"][2])
    nz = grade["shape_zyx"][0]
    fora = []
    idxs = []
    for reg_ in ipps:
        sop, ipp = reg_[0], reg_[1]
        proj = float(np.dot(np.asarray(ipp, float) - o, n))
        f = proj / dz
        i = int(round(f))
        desvio = abs(f - i) * dz
        if desvio > TOL_SLICE_MM:
            fora.append((sop, round(desvio, 6)))
        if not (0 <= i < nz):
            raise FalhaGrade("%s: indice %d fora de %d fatias" % (pid, i, nz))
        idxs.append(i)
    if fora:
        raise FalhaGrade("%s: %d fatias fora da tolerancia de %.3f mm (pior %.6f mm)"
                         % (pid, len(fora), TOL_SLICE_MM,
                            max(d for _, d in fora)))
    return idxs


def indexar_xml():
    idx = {}
    dup = collections.Counter()
    for x in glob.glob(os.path.join(RAW_XML, "**", "*.xml"), recursive=True):
        try:
            r = ET.parse(x).getroot()
        except ET.ParseError:
            continue
        u = r.find(".//n:SeriesInstanceUid", NS)
        if u is not None and u.text:
            uid = u.text.strip()
            dup[uid] += 1
            idx.setdefault(uid, x)
    return idx, {u: c for u, c in dup.items() if c > 1}


NS_XLSX = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
PLANILHA = os.path.join(BASE, "data", "reference",
                        "lidc-idri-nodule-counts-6-23-2015.xlsx")
PLANILHA_CACHE = {}


def ler_planilha_nodulos(p):
    """XLSX sem openpyxl: e um zip com XML. Criterio: UNIAO entre leitores."""
    import zipfile
    z = zipfile.ZipFile(p)
    ss = [t.text or "" for t in
          ET.fromstring(z.read("xl/sharedStrings.xml")).iter(NS_XLSX + "t")]
    out = {}
    for r in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(NS_XLSX + "row"):
        v = []
        for c in r:
            e = c.find(NS_XLSX + "v")
            x = e.text if e is not None else ""
            if c.get("t") == "s" and x:
                x = ss[int(x)]
            v.append(x)
        if v and str(v[0]).startswith("LIDC-IDRI"):
            out[v[0]] = {"total": v[1], "ge3mm": v[2]}
    return out


def adjudicar_sem_anotacao(pid, caminho_xml):
    """
    Ausencia de anotacao indexada NAO e negativo. Para entrar como negativo e
    preciso EVIDENCIA POSITIVA de ausencia do alvo, vinda de duas fontes que
    concordem: o XML do LIDC e a planilha oficial de contagens de 2015.

      NEGATIVE_DOCUMENTED        4 sessoes, nenhum `unblindedReadNodule`, so
                                 `nonNodule`; planilha total=0 e >=3mm=0.
      SUBTHRESHOLD_NODULES_ONLY  ha nodulos <3 mm marcados; planilha >=3mm=0 com
                                 total>0. Fora da coorte principal — transforma-los
                                 em fundo seria censurar rotulo.
      INCONCLUSIVE               fontes discordam, faltam sessoes, sem linha.

    Devolve (classe, evidencia).
    """
    ev = {}
    try:
        root = ET.parse(caminho_xml).getroot()
    except Exception as e:
        return "INCONCLUSIVE", {"detail": "XML ilegivel: %s" % e}
    ses = root.findall(".//n:readingSession", NS)
    com = sem = 0
    for s_ in ses:
        for n_ in s_.findall("n:unblindedReadNodule", NS):
            if n_.find("n:characteristics", NS) is not None:
                com += 1
            else:
                sem += 1
    nn = sum(len(s_.findall("n:nonNodule", NS)) for s_ in ses)
    ev = {"reading_sessions": len(ses), "xml_with_characteristics": com,
          "xml_without_characteristics": sem, "xml_non_nodules": nn}

    linha = PLANILHA_CACHE.get(pid)
    if linha is None:
        return "INCONCLUSIVE", dict(ev, detail="sem linha na planilha de 2015")
    try:
        tot, ge3 = int(float(linha["total"])), int(float(linha["ge3mm"]))
    except (TypeError, ValueError):
        return "INCONCLUSIVE", dict(ev, detail="contagens ilegiveis")
    ev.update(sheet_total=tot, sheet_ge3mm=ge3)

    if len(ses) != 4:
        return "INCONCLUSIVE", dict(ev, detail="%d sessoes, esperado 4" % len(ses))
    if com > 0 or ge3 > 0:
        return "INCONCLUSIVE", dict(ev, detail="nodulo >=3mm declarado (xml=%d, "
                                               "planilha=%d) sem indexacao" % (com, ge3))
    if sem == 0 and tot == 0:
        return "NEGATIVE_DOCUMENTED", dict(
            ev, detail="quatro leitores, nenhuma marca de nodulo, planilha total=0")
    if sem > 0 and tot > 0:
        return "SUBTHRESHOLD_NODULES_ONLY", dict(
            ev, detail="%d marcas <3mm; planilha total=%d, >=3mm=0" % (sem, tot))
    return "INCONCLUSIVE", dict(ev, detail="xml_sem_char=%d vs planilha total=%d"
                                           % (sem, tot))


def leitores_do_xml(caminho):
    """
    Sequência ordenada (indice_sessao, servicingRadiologistID, noduleID) dos
    nódulos com `characteristics` — o mesmo critério do `pylidc`.
    """
    root = ET.parse(caminho).getroot()
    seq = []
    for si, s in enumerate(root.findall(".//n:readingSession", NS)):
        rid = s.find("n:servicingRadiologistID", NS)
        rid = rid.text.strip() if rid is not None and rid.text else ""
        for nod in s.findall("n:unblindedReadNodule", NS):
            if nod.find("n:characteristics", NS) is None:
                continue
            nid = nod.find("n:noduleID", NS)
            seq.append((si, rid, nid.text.strip() if nid is not None and nid.text else ""))
    return seq


def _carregar_registo(uid, xml_idx):
    """
    Carrega as anotações de UMA série e prova a identidade do leitor pela ordem.

    Devolve (anns, seq, erro). `erro` é None quando correu bem.
    """
    scan = pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == uid).first()
    if scan is None:
        return None, None, None, "FAIL_SCAN_NOT_FOUND"
    cx = xml_idx.get(uid)
    if cx is None:
        return None, None, None, "FAIL_XML_NOT_FOUND"
    anns = (pl.query(pl.Annotation).filter(pl.Annotation.scan_id == scan.id)
            .order_by(pl.Annotation.id).all())
    seq = leitores_do_xml(cx)
    if [str(x._nodule_id) for x in anns] != [t[2] for t in seq]:
        return None, None, None, ("FAIL_READER_ASSIGNMENT",
                                  "sequencia noduleID da SQLite != ordem do XML "
                                  "(db=%d xml=%d)" % (len(anns), len(seq)))
    return scan, anns, seq, None


def _rasterizar(anns, geo_uid, grade, pid, retido=None):
    """
    Rasteriza as anotações de um registo na grade canónica.

    `retido` é o intervalo (z_min, z_max) do bloco físico que o Script 03 de
    facto converteu. Quando a série foi truncada, a grade tem menos fatias do
    que a série de origem, e o `pylidc` rasteriza sobre a série INTEIRA: sem
    restringir, a associação de fatias falha e o caso morre com um rótulo
    técnico em vez de ser diagnosticado.

    Devolve (mascaras, erro, nota).
    """
    ipps = geo_uid
    if not ipps:
        return None, "FAIL_GEOMETRY_NOT_FOUND", ""
    o = np.asarray(grade["origin"], float)
    n = grade["normal"]
    ipps = sorted(ipps, key=lambda t: float(np.dot(np.asarray(t[1], float) - o, n)))
    sop_ret = (retido or {}).get("sop_index") or {}
    def _preferido(a_, b_):
        if not sop_ret:
            return a_
        return a_ if (a_[0] in sop_ret or b_[0] not in sop_ret) else b_
    por_z = {}
    ordem = []
    for t in ipps:
        z = round(float(np.dot(np.asarray(t[1], float) - o, n)), 4)
        if z in por_z:
            por_z[z] = _preferido(por_z[z], t)
        else:
            por_z[z] = t
            ordem.append(z)
    unicos = [por_z[z] for z in ordem]
    ipps = unicos
    n_origem = len(ipps)

    if retido is None:
        try:
            k_para_canonico = indices_canonicos(ipps, grade, pid)
        except FalhaGrade as e:
            return None, "FAIL_SLICE_ASSOCIATION", str(e)
        pares = list(enumerate(k_para_canonico))
    else:
        idx = retido["sop_index"]
        pares = [(k, idx[t[0]]) for k, t in enumerate(ipps) if t[0] in idx]
        if len(pares) != grade["shape_zyx"][0]:
            return None, "FAIL_RETAINED_SLICE_COUNT", (
                "%d cortes do manifesto do 03 encontrados na serie, grade com %d"
                % (len(pares), grade["shape_zyx"][0]))
        if sorted(c for _, c in pares) != list(range(grade["shape_zyx"][0])):
            return None, "FAIL_RETAINED_INDEX_GAP", (
                "indices canonicos do 03 nao formam 0..%d"
                % (grade["shape_zyx"][0] - 1))

    nz, ny, nx = grade["shape_zyx"]
    mascaras = {}
    for x in anns:
        try:
            m = x.boolean_mask(pad=10000, include_contour_points=False)
            m = np.asarray(m, dtype=bool).transpose(2, 0, 1)
        except Exception as e:
            return None, "FAIL_RASTERIZATION", "annotation %d (%s): %s: %s" % (
                x.id, x._nodule_id, type(e).__name__, e)
        if m.shape[0] != n_origem:
            return None, "FAIL_SLICE_COUNT", "mascara com %d fatias, serie com %d" % (
                m.shape[0], n_origem)
        if (m.shape[1], m.shape[2]) != (ny, nx):
            return None, "FAIL_INPLANE_SHAPE", "mascara %s, grade %s" % (
                (m.shape[1], m.shape[2]), (ny, nx))
        can = np.zeros((nz, ny, nx), dtype=bool)
        for k, kc in pares:
            if m[k].any():
                can[kc] |= m[k]
        mascaras[x.id] = can
    return mascaras, None, ""


def _unir_clusters_entre_registos(clusters_a, clusters_b, masc, desloc):
    """
    Casa os agrupamentos dos dois registos fundidos pela SOBREPOSIÇÃO das suas
    máscaras, e não por suposição de ordem.

    `Scan.cluster_annotations()` opera dentro de um `Scan`; com dois registos do
    MESMO exame físico é preciso decidir quais agrupamentos representam o mesmo
    nódulo. Como os volumes têm sha256 de pixels idêntico e a geometria foi
    medida igual, o mesmo nódulo ocupa exatamente as mesmas coordenadas: a
    sobreposição é medida, não inferida.

    Devolve a lista de agrupamentos fundidos, cada um como lista de chaves
    (registo, annotation_id).
    """
    def uniao(ids, pref):
        u = None
        for i in ids:
            m = masc[(pref, i)]
            u = m.copy() if u is None else (u | m)
        return u

    ua = [uniao([x.id for x in c], "A") for c in clusters_a]
    ub = [uniao([x.id for x in c], "B") for c in clusters_b]

    pares = []
    for i, A in enumerate(ua):
        for j, B in enumerate(ub):
            inter = int(np.logical_and(A, B).sum())
            if inter:
                un = int(np.logical_or(A, B).sum())
                pares.append((inter / un, i, j))
    pares.sort(reverse=True)

    usados_a, usados_b, fundidos = set(), set(), []
    casamentos = []
    for iou, i, j in pares:
        if i in usados_a or j in usados_b:
            continue
        usados_a.add(i); usados_b.add(j)
        fundidos.append([("A", x.id) for x in clusters_a[i]] +
                        [("B", x.id) for x in clusters_b[j]])
        casamentos.append({"cluster_a": i + 1, "cluster_b": j + 1,
                           "iou": round(iou, 6)})
    for i, c in enumerate(clusters_a):
        if i not in usados_a:
            fundidos.append([("A", x.id) for x in c])
    for j, c in enumerate(clusters_b):
        if j not in usados_b:
            fundidos.append([("B", x.id) for x in c])
    return fundidos, casamentos


def _diagnostico_truncagem(anns, gap):
    """
    Relata o que geometricamente aconteceu numa série truncada — sem decidir o
    que fazer com o caso.

    Usa a posição física em z de cada contorno (`image_z_position`), que é a
    coordenada da série DE ORIGEM. A primeira versão media isto sobre as
    máscaras rasterizadas, o que não funciona: quando a série é truncada a
    rasterização falha antes, e o caso morria com `FAIL_SLICE_ASSOCIATION` sem
    nunca produzir a evidência que justifica a decisão.

    Diagnóstico e ação são camadas separadas: dizer `EXCLUDE_ANNOTATION_OUTSIDE`
    misturaria o facto com a consequência e daria a exclusão por definitiva
    quando o bloco descartado pode ser recuperado pela conversão dirigida.

    Devolve (classe_da_serie, linhas_por_anotacao).
    """
    if not gap or not gap.get("has_physical_gap"):
        return "NOT_APPLICABLE", []
    rlo, rhi = gap["retained_z_min_mm"], gap["retained_z_max_mm"]
    descartados = [b for b in gap["blocks"] if not b["is_retained"]]

    linhas, classes = [], []
    for x in anns:
        zs = [c.image_z_position for c in x.contours]
        if not zs:
            linhas.append({"annotation_id": x.id, "nodule_id": str(x._nodule_id),
                           "ann_z_min_mm": "", "ann_z_max_mm": "",
                           "class": "NO_CONTOURS",
                           "justification": "anotacao sem contornos"})
            continue
        zlo, zhi = float(min(zs)), float(max(zs))
        dentro = (zlo >= rlo) and (zhi <= rhi)
        sobrepoe = (zlo <= rhi) and (zhi >= rlo)
        em_descartado = [b for b in descartados
                         if not (zhi < b["z_min_mm"] or zlo > b["z_max_mm"])]
        if dentro:
            cl = "ANNOTATION_INSIDE_RETAINED_BLOCK"
            just = "anotacao [%.1f, %.1f] dentro do preservado [%.1f, %.1f] mm" % (
                zlo, zhi, rlo, rhi)
        elif not sobrepoe:
            cl = "ANNOTATION_OUTSIDE_RETAINED_BLOCK"
            just = ("anotacao [%.1f, %.1f] fora do preservado [%.1f, %.1f] mm; "
                    "cai em bloco(s) descartado(s) %s" % (
                        zlo, zhi, rlo, rhi,
                        ";".join("%.1f..%.1f" % (b["z_min_mm"], b["z_max_mm"])
                                 for b in em_descartado) or "nenhum"))
        else:
            cl = "ANNOTATION_CROSSES_BOUNDARY"
            just = ("anotacao [%.1f, %.1f] atravessa a fronteira do preservado "
                    "[%.1f, %.1f] mm" % (zlo, zhi, rlo, rhi))
        classes.append(cl)
        linhas.append({"annotation_id": x.id, "nodule_id": str(x._nodule_id),
                       "ann_z_min_mm": round(zlo, 3), "ann_z_max_mm": round(zhi, 3),
                       "class": cl, "justification": just})

    if "ANNOTATION_CROSSES_BOUNDARY" in classes:
        serie = "ANNOTATION_CROSSES_BOUNDARY"
    elif "ANNOTATION_OUTSIDE_RETAINED_BLOCK" in classes:
        serie = ("ANNOTATION_OUTSIDE_RETAINED_BLOCK"
                 if "ANNOTATION_INSIDE_RETAINED_BLOCK" not in classes
                 else "ANNOTATION_SPLIT_ACROSS_BLOCKS")
    elif classes:
        serie = "ANNOTATION_INSIDE_RETAINED_BLOCK"
    else:
        serie = "NO_ANNOTATION"
    return serie, linhas


def processar(reg, xml_idx, geo, dest, gap, fusoes, reti_sops=None):
    pid = reg["patient_id"]
    uid = reg["series_instance_uid"]
    a = {"patient_id": pid, "series_instance_uid": uid,
         "status": None, "consensus_written": False,
         "n_annotations": 0, "n_readers": 0, "n_clusters": 0,
         "n_reading_sessions": 0, "radiologist_id_distinct": 0,
         "n_vox_consensus": 0, "n_per_reader_masks": 0,
         "merged_records": "", "merge_status": "NOT_APPLICABLE",
         "truncation_class": "NOT_APPLICABLE", "negative_documented": False,
         "adjudication_evidence": None, "notes": ""}

    scan, anns, seq, err = _carregar_registo(uid, xml_idx)
    if err:
        a["status"], a["notes"] = err if isinstance(err, tuple) else (err, "")
        return a, None, None, []
    if not anns:
        classe, ev = adjudicar_sem_anotacao(pid, xml_idx[uid])
        a["status"] = classe
        a["adjudication_evidence"] = ev
        a["truncation_class"], linhas = _diagnostico_truncagem([], gap)
        if classe != "NEGATIVE_DOCUMENTED":
            return a, None, None, linhas
        try:
            grade = carregar_grade_ct(pid)
        except FalhaGrade as e:
            a["status"] = "FAIL_GRID"; a["notes"] = str(e)
            return a, None, None, linhas
        nz, ny, nx = grade["shape_zyx"]
        a["negative_documented"] = True
        a["n_vox_consensus"] = 0
        a["notes"] = ev.get("detail", "")
        return (a, {}, ("WRITE", {}, [], grade,
                        np.zeros((nz, ny, nx), dtype=bool), [], {}, uid, []),
                linhas)

    try:
        grade = carregar_grade_ct(pid)
    except FalhaGrade as e:
        a["status"] = "FAIL_GRID"
        a["notes"] = str(e)
        return a, None, None, []

    a["truncation_class"], linhas_trunc = _diagnostico_truncagem(anns, gap)
    retido = None
    if gap and gap.get("has_physical_gap"):
        if not reti_sops:
            a["status"] = "FAIL_RETAINED_SLICES_MISSING"
            a["notes"] = "serie truncada sem lista de cortes retidos do Script 03"
            return a, None, None, linhas_trunc
        retido = {"sop_index": reti_sops}

    masc_a, err, nota = _rasterizar(anns, geo.get(uid), grade, pid, retido)
    if err:
        a["status"] = err
        a["notes"] = nota
        return a, None, None, linhas_trunc

    leitor = {("A", x.id): ("A", seq[i][0], seq[i][1]) for i, x in enumerate(anns)}
    masc = {("A", k): v for k, v in masc_a.items()}
    anots = {("A", x.id): x for x in anns}
    clusters_a = scan.cluster_annotations()
    sessoes = {("A", t[0]) for t in seq}
    rad_ids = {t[1] for t in seq}
    n_anns = len(anns)

    membros = fusoes.get(pid, [])
    casamentos = []
    if membros:
        clusters_b_todos = []
        for m_pid, m_uid in membros:
            scan_b, anns_b, seq_b, err_b = _carregar_registo(m_uid, xml_idx)
            if err_b or not anns_b:
                a["status"] = "FAIL_MERGE_MEMBER"
                _d = err_b[1] if isinstance(err_b, tuple) else err_b
                a["notes"] = "%s (%s): %s" % (m_pid, m_uid[-12:],
                                              _d or "sem anotacoes elegiveis")
                return a, None, None, []
            masc_b, err_b, nota_b = _rasterizar(anns_b, geo.get(m_uid), grade,
                                                m_pid, retido)
            if err_b:
                a["status"] = "FAIL_MERGE_RASTERIZATION"
                a["notes"] = "%s: %s %s" % (m_pid, err_b, nota_b)
                return a, None, None, []
            for i, x in enumerate(anns_b):
                leitor[("B", x.id)] = ("B", seq_b[i][0], seq_b[i][1])
                masc[("B", x.id)] = masc_b[x.id]
                anots[("B", x.id)] = x
            sessoes |= {("B", t[0]) for t in seq_b}
            rad_ids |= {t[1] for t in seq_b}
            n_anns += len(anns_b)
            clusters_b_todos = scan_b.cluster_annotations()
        clusters, casamentos = _unir_clusters_entre_registos(
            clusters_a, clusters_b_todos, masc, None)
        a["merged_records"] = ",".join(p for p, _ in membros)
        a["merge_status"] = "PERFORMED"
    else:
        clusters = [[("A", x.id) for x in c] for c in clusters_a]

    a["n_annotations"] = n_anns
    a["n_reading_sessions"] = len(sessoes)
    a["radiologist_id_distinct"] = len(rad_ids)
    a["n_clusters"] = len(clusters)
    a["n_readers"] = len({leitor[k][:2] for k in leitor})

    ambiguo = [i for i, c in enumerate(clusters, 1)
               if len(c) > MAX_ANOTACOES_POR_CLUSTER * (1 + len(membros))]
    if ambiguo:
        a["status"] = "FAIL_CLUSTER_AMBIGUOUS"
        a["notes"] = "clusters com delineamentos a mais: %s" % ",".join(
            "%d(%d)" % (i, len(clusters[i - 1])) for i in ambiguo)
        return a, masc, ("CONSENSUS_NOT_WRITTEN", leitor, clusters, grade,
                         None, None, anots, uid, casamentos), linhas_trunc

    nz, ny, nx = grade["shape_zyx"]
    consenso = np.zeros((nz, ny, nx), dtype=bool)
    detalhe = []
    for i, c in enumerate(clusters, 1):
        por_leitor = collections.defaultdict(list)
        for k in c:
            por_leitor[leitor[k][:2]].append(k)
        soma = np.zeros((nz, ny, nx), dtype=np.int16)
        for rid, ks in por_leitor.items():
            u = np.zeros((nz, ny, nx), dtype=bool)
            for k in ks:
                u |= masc[k]
            soma += u.astype(np.int16)
        R = len(por_leitor)
        limiar = int(math.ceil(TAU * R))
        nod = soma >= limiar
        consenso |= nod
        detalhe.append({"cluster": i, "R_readers": R, "n_annotations": len(c),
                        "threshold": limiar, "n_vox": int(nod.sum())})

    if a["truncation_class"] in ("ANNOTATION_OUTSIDE_RETAINED_BLOCK",
                                 "ANNOTATION_CROSSES_BOUNDARY",
                                 "ANNOTATION_SPLIT_ACROSS_BLOCKS"):
        a["status"] = "HOLD_" + a["truncation_class"]
        a["notes"] = ("consenso nao escrito: %d voxels no volume convertido, "
                      "anotacao fora ou a atravessar o bloco preservado"
                      % int(consenso.sum()))
        return a, masc, ("CONSENSUS_NOT_WRITTEN", leitor, clusters, grade,
                         None, None, anots, uid, casamentos), linhas_trunc

    a["status"] = "PASS"
    a["n_vox_consensus"] = int(consenso.sum())
    return a, masc, ("WRITE", leitor, clusters, grade, consenso, detalhe,
                     anots, uid, casamentos), linhas_trunc


NOVE = ("subtlety", "internalStructure", "calcification", "sphericity",
        "margin", "lobulation", "spiculation", "texture", "malignancy")


def gravar(dest, pid, uid, grade, mascaras, leitor, clusters, consenso, anotacoes):
    """
    Escreve as máscaras por leitor e o consenso, e devolve as linhas da tabela de
    objetos.

    A chave de leitor é `(registo, reading_session_index)` — o registo distingue
    o canónico do membro fundido, e sem ele as sessões dos dois colidiriam.
    NÃO é identidade global de radiologista: o `servicingRadiologistID` do XML é
    o mesmo literal ('anon') em 626 das 873 séries com mais de uma sessão, e por
    isso foi rejeitado. O Script 12 não pode tratar sessões de pacientes
    diferentes como o mesmo observador.
    """
    pr = os.path.join(dest, "per_reader", "lidc", pid)
    os.makedirs(pr, exist_ok=True)
    id_cluster = {}
    for i, c in enumerate(clusters, 1):
        for k in c:
            id_cluster[k] = i
    linhas = []
    for chave, m in sorted(mascaras.items(), key=lambda t: (t[0][0], t[0][1])):
        registo, aid = chave
        nome = "nodule%02d_rec%s_session%02d_ann%d.nii.gz" % (
            id_cluster.get(chave, 0), registo, leitor[chave][1], aid)
        p = os.path.join(pr, nome)
        if os.path.exists(p):
            raise SystemExit("[FAIL] recusa sobrescrever %s" % p)
        sitk.WriteImage(mask_to_sitk(m.astype(np.uint8), grade), p, True)
        vox = np.argwhere(m)
        vol = int(vox.shape[0])
        cz = cy = cx = diam = 0.0
        if vol:
            cz, cy, cx = vox.mean(axis=0)
            diam = 2.0 * (3.0 * vol / (4.0 * math.pi)) ** (1.0 / 3.0)
        a = anotacoes[chave]
        r = {"patient_id": pid, "ct_series_uid": uid,
             "source_record": registo,
             "nodule_num": id_cluster.get(chave, 0),
             "reading_session_index": leitor[chave][1],
             "reader_key": "%s:%d" % (registo, leitor[chave][1]),
             "servicing_radiologist_id_raw": leitor[chave][2],
             "annotation_id": aid, "nodule_id": str(a._nodule_id),
             "mask_filename": nome,
             "mask_relpath": os.path.join("annotations", "per_reader", "lidc",
                                          pid, nome),
             "centroid_z": round(float(cz), 2), "centroid_y": round(float(cy), 2),
             "centroid_x": round(float(cx), 2),
             "volume_vox": vol, "diam_mm_approx": round(diam, 2)}
        for c in NOVE:
            v = getattr(a, c, None)
            r[c] = "" if v is None else v
        linhas.append(r)
    if consenso is None:
        return linhas
    cd = os.path.join(dest, "consensus", "lidc", pid)
    os.makedirs(cd, exist_ok=True)
    p = os.path.join(cd, "consensus.nii.gz")
    if os.path.exists(p):
        raise SystemExit("[FAIL] recusa sobrescrever %s" % p)
    sitk.WriteImage(mask_to_sitk(consenso.astype(np.uint8), grade), p, True)
    return linhas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="processar apenas as N primeiras series (ensaio)")
    ap.add_argument("--only", default="",
                    help="ensaio dirigido: lista de patient_id separada por virgula")
    args = ap.parse_args()
    ENSAIO = bool(args.only or args.limit)

    t0 = time.time()
    run_id = "s04_%s" % time.strftime("%Y%m%dT%H%M%S")
    global ANNOT, ANALYSIS, CONFIG
    if ENSAIO:
        _raiz = os.path.join(BASE, "staging_ensaio", run_id)
        ANNOT = os.path.join(_raiz, "annotations")
        ANALYSIS = os.path.join(_raiz, "analysis")
        CONFIG = os.path.join(_raiz, "config")
        for _d in (ANNOT, ANALYSIS, CONFIG):
            os.makedirs(_d, exist_ok=True)
        log.warning("ENSAIO: saida em %s — nada publicado"
                    % os.path.relpath(_raiz, BASE))
    dest = ANNOT
    for q in (os.path.join(ANNOT, "annotations", "consensus", "lidc"),
              os.path.join(ANNOT, "consensus", "lidc"),
              os.path.join(ANNOT, "per_reader", "lidc")):
        if os.path.isdir(q):
            shutil.rmtree(q)
    for d in (ANNOT, ANALYSIS, CONFIG, os.path.dirname(PONTEIRO_SAIDA)):
        os.makedirs(d, exist_ok=True)
    falhas_portao = []

    sig = inspect.signature(pl.Annotation.boolean_mask)
    par = sig.parameters.get("include_contour_points")
    if pl.__version__ != "0.2.1":
        raise SystemExit("[FAIL] PYLIDC_VERSION %s != 0.2.1" % pl.__version__)
    if par is None or par.default is not False:
        raise SystemExit("[FAIL] include_contour_points ausente ou != False")

    sqlite = os.path.join(os.path.dirname(pl.__file__), "pylidc.sqlite")
    sq_antes = sha_ficheiro(sqlite)

    for nome, q in (("Script 02", PONTEIRO_02), ("Script 03", PONTEIRO_03)):
        if not os.path.isfile(q):
            raise SystemExit("[FAIL] ponteiro do %s ausente: %s" % (nome, q))
    _p2 = json.load(open(PONTEIRO_02))
    _p3 = json.load(open(PONTEIRO_03))

    headers = os.path.join(BASE, _p2["dicom_headers"])
    if not os.path.isfile(headers):
        raise SystemExit("[FAIL] dicom_headers.csv ausente: %s" % headers)
    sel_path = os.path.join(BASE, _p2["lidc_selection_manifest"])
    man_sel = json.load(open(sel_path))
    fin_path = os.path.join(BASE, _p3["lidc_final_manifest"])
    if not os.path.isfile(fin_path):
        raise SystemExit("[FAIL] lidc_final_manifest.json ausente: %s" % fin_path)
    man_fin = json.load(open(fin_path))
    if man_fin.get("CONVERSION_STATUS") != "PASS":
        raise SystemExit("[FAIL] conversao do Script 03 em %s"
                         % man_fin.get("CONVERSION_STATUS"))

    alvo = sorted(man_fin["eligible"], key=lambda r: r["patient_id"])
    esperado = int(man_sel["LIDC_ANALYTIC_DENOMINATOR"])
    if len(alvo) != esperado:
        raise SystemExit("[FAIL] alvo %d != denominador %d" % (len(alvo), esperado))
    gaps = {e["patient_id"]: e.get("physical_gap_report") for e in alvo}

    retidos = collections.defaultdict(dict)
    reti_csv = os.path.join(BASE, _p3.get("retained_slices_csv", ""))
    if not os.path.isfile(reti_csv):
        raise SystemExit("[FAIL] lista de cortes retidos ausente: %s" % reti_csv)
    with open(reti_csv) as f:
        for r in csv.DictReader(f):
            retidos[r["patient_id"]][r["sop_instance_uid"]] = int(r["canonical_index"])
    log.info("  cortes retidos declarados pelo 03: %d series"  % len(retidos))

    FUSOES = {}
    for g in man_sel.get("duplicate_groups", []):
        can = g["canonical_patient_id"]
        FUSOES[can] = [(q, u) for q, u in g["member_series"].items() if q != can]

    if args.only:
        pedidos = [x.strip() for x in args.only.split(",") if x.strip()]
        alvo = [r for r in alvo if r["patient_id"] in pedidos]
        em_falta = set(pedidos) - {r["patient_id"] for r in alvo}
        if em_falta:
            raise SystemExit("[FAIL] pedidos fora do universo: %s" % sorted(em_falta))
    elif args.limit:
        alvo = alvo[:args.limit]


    pids_alvo = {r["patient_id"] for r in alvo}
    FUSOES_ESPERADAS = {k for k in FUSOES if k in pids_alvo}
    gaps = {k: v for k, v in gaps.items() if k in pids_alvo}

    log.info("Script 04 — rota XML | run_id=%s" % run_id)
    log.info("  pylidc %s · include_contour_points=False · monkeypatches 0"
             % pl.__version__)
    log.info("  entrada: 02=%s · 03=%s" % (_p2.get("run_id"), _p3.get("run_id")))
    log.info("  series alvo: %d (denominador congelado %d)" % (len(alvo), esperado))
    log.info("  fusoes a executar: %s"
             % {k: [q for q, _ in v] for k, v in FUSOES.items()})
    log.info("  series com lacuna fisica: %d"
             % sum(1 for g in gaps.values() if g and g.get("has_physical_gap")))
    log.info("  destino: staging/%s" % run_id)

    arv_antes = impressao_arvore(RAW_LIDC)
    xml_idx, xml_dup = indexar_xml()
    log.info("  XML indexados: %d (uids repetidos: %d)" % (len(xml_idx), len(xml_dup)))
    geo = carregar_geometria_ct(headers)
    global PLANILHA_CACHE
    if not os.path.isfile(PLANILHA):
        raise SystemExit("[FAIL] planilha ausente: %s" % PLANILHA)
    PLANILHA_CACHE = ler_planilha_nodulos(PLANILHA)
    log.info("  planilha 2015: %d pacientes (2.a fonte da adjudicacao)"
             % len(PLANILHA_CACHE))
    log.info("  geometria para %d series CT" % len(geo))

    auditoria, detalhes, objetos, decisoes = [], [], [], []
    casamentos_fusao = []
    contagem = collections.Counter()
    for n, reg in enumerate(alvo, 1):
        try:
            a, masc, plano, linhas_trunc = processar(
                reg, xml_idx, geo, dest, gaps.get(reg["patient_id"]), FUSOES,
                retidos.get(reg["patient_id"]))
        except Exception as e:
            a = {"patient_id": reg["patient_id"],
                 "series_instance_uid": reg["series_instance_uid"],
                 "status": "FAIL_UNEXPECTED", "consensus_written": False,
                 "n_annotations": 0, "n_readers": 0, "n_clusters": 0,
                 "n_reading_sessions": 0, "radiologist_id_distinct": 0,
                 "n_vox_consensus": 0, "n_per_reader_masks": 0,
                 "merged_records": "", "merge_status": "NOT_APPLICABLE",
                 "truncation_class": "NOT_APPLICABLE",
                 "notes": "%s: %s" % (type(e).__name__, e)}
            masc, plano, linhas_trunc = None, None, []
        if plano is not None and (masc or a.get("negative_documented")):
            grade, anots, uid_serie = plano[3], plano[6], plano[7]
            consenso = plano[4] if plano[0] == "WRITE" else None
            objetos += gravar(dest, a["patient_id"], uid_serie, grade, masc,
                              plano[1], plano[2], consenso, anots)
            if plano[0] == "WRITE":
                a["consensus_written"] = True
                for d in plano[5]:
                    d["patient_id"] = a["patient_id"]
                    detalhes.append(d)
            for c in plano[8]:
                c["patient_id"] = a["patient_id"]
                casamentos_fusao.append(c)
            a["n_per_reader_masks"] = len(masc)

        g = gaps.get(a["patient_id"])
        if g and g.get("has_physical_gap"):
            retido = [b for b in g["blocks"] if b["is_retained"]]
            desc = [b for b in g["blocks"] if not b["is_retained"]]
            for lt in (linhas_trunc or [{"annotation_id": "", "nodule_id": "",
                                         "ann_z_min_mm": "", "ann_z_max_mm": "",
                                         "class": "NO_ANNOTATION",
                                         "justification": "serie sem anotacao elegivel"}]):
                acao = {"ANNOTATION_INSIDE_RETAINED_BLOCK": "KEEP_TRUNCATED",
                        "ANNOTATION_OUTSIDE_RETAINED_BLOCK": "SPLIT_CANDIDATE_OR_EXCLUDE_CASE",
                        "ANNOTATION_CROSSES_BOUNDARY": "EXCLUDE_CASE",
                        "NO_CONTOURS": "REVIEW",
                        "EMPTY_MASK": "REVIEW",
                        "NO_ANNOTATION": "REVIEW"}.get(lt["class"], "REVIEW")
                decisoes.append({
                    "case_id": a["patient_id"],
                    "series_instance_uid": a["series_instance_uid"],
                    "n_contiguous_blocks": g["n_contiguous_blocks"],
                    "retained_z_min_mm": g["retained_z_min_mm"],
                    "retained_z_max_mm": g["retained_z_max_mm"],
                    "retained_n_slices": g["n_slices_retained"],
                    "retained_block_index": (1 + g["blocks"].index(retido[0])
                                             if retido else ""),
                    "discarded_blocks": ";".join(
                        "idx%d:%.1f..%.1f:%dc" % (1 + g["blocks"].index(b),
                                                  b["z_min_mm"], b["z_max_mm"],
                                                  b["n_slices"]) for b in desc),
                    "annotation_id": lt["annotation_id"],
                    "nodule_id": lt.get("nodule_id", ""),
                    "ann_z_min_mm": lt["ann_z_min_mm"],
                    "ann_z_max_mm": lt["ann_z_max_mm"],
                    "diagnostic_class": lt["class"],
                    "possible_action": acao,
                    "justification": lt["justification"],
                    "series_class": a["truncation_class"],
                    "series_status": a["status"]})
        auditoria.append(a)
        contagem[a["status"]] += 1
        if n % 25 == 0 or n == len(alvo):
            dt = time.time() - t0
            log.info("  %4d/%d  %.0f s  (%.2f s/serie)  estados: %s"
                     % (n, len(alvo), dt, dt / n, dict(contagem)))

    arv_depois = impressao_arvore(RAW_LIDC)
    sq_depois = sha_ficheiro(sqlite)

    def escrever(nome, linhas, cols=None, raiz=None):
        if not linhas:
            return 0
        q = os.path.join(raiz or ANALYSIS, nome)
        os.makedirs(os.path.dirname(q), exist_ok=True)
        with open(q, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols or list(linhas[0].keys()),
                               extrasaction="ignore", restval="")
            w.writeheader(); w.writerows(linhas)
        return len(linhas)

    escrever("audit_04.csv", auditoria)
    escrever("object_tables/lidc_nodules.csv", objetos, raiz=ANNOT)
    escrever("consensus_thresholds.csv", detalhes,
             ["patient_id", "cluster", "R_readers", "n_annotations",
              "threshold", "n_vox"])
    n_dec = escrever("truncation_decision_manifest.csv", decisoes)
    escrever("merge_cluster_matching.csv", casamentos_fusao)

    fusoes_feitas = {a["patient_id"] for a in auditoria if a["merge_status"] == "PERFORMED"}
    if FUSOES_ESPERADAS - fusoes_feitas:
        falhas_portao.append("fusoes declaradas e nao executadas: %s"
                             % sorted(FUSOES_ESPERADAS - fusoes_feitas))
    if sq_antes != sq_depois:
        falhas_portao.append("SQLITE alterada")
    if arv_antes != arv_depois:
        falhas_portao.append("arvore bruta alterada")
    n_lacuna = sum(1 for g in gaps.values() if g and g.get("has_physical_gap"))
    if n_lacuna and not n_dec:
        falhas_portao.append("%d series com lacuna e nenhuma linha de decisao" % n_lacuna)

    dec = collections.Counter(d["diagnostic_class"] for d in decisoes)
    gates = {
        "PYLIDC_VERSION": pl.__version__,
        "INCLUDE_CONTOUR_POINTS": "False",
        "PYLIDC_MONKEYPATCHES": 0,
        "BOOLEAN_MASK_SIGNATURE": str(sig),
        "SCRIPT02_RUN_ID": _p2.get("run_id"),
        "SCRIPT03_RUN_ID": _p3.get("run_id"),
        "SELECTED_SERIES_EXPECTED": esperado,
        "SELECTED_SERIES_SCANNED": len(auditoria),
        "SQLITE_SCANS_TOTAL": pl.query(pl.Scan).count(),
        "SQLITE_READ_ONLY": sq_antes == sq_depois,
        "SQLITE_SHA256": sq_antes,
        "RAW_TREE_WRITES": 0 if arv_antes == arv_depois else 1,
        "SLICE_ASSOCIATION_TOL_MM": TOL_SLICE_MM,
        "CONSENSUS_RULE": "ceil(%.2f x R), R = readers with valid mask" % TAU,
        "READER_KEY": "(source_record, reading_session_index)",
        "READER_KEY_REJECTED": ("servicingRadiologistID — anonimizado e identico "
                                "em todas as sessoes na maioria das series"),
        "TRIAL_MODE": ENSAIO,
        "MERGE_DECLARED": sorted(FUSOES_ESPERADAS),
        "MERGE_PERFORMED": sorted(fusoes_feitas),
        "MERGE_CLUSTER_MATCH_CRITERION": ("sobreposicao medida das mascaras entre "
                                          "os registos; volumes com sha256 de "
                                          "pixeis identico e geometria medida igual"),
        "STATUS_COUNTS": dict(contagem),
        "CONSENSUS_WRITTEN": sum(1 for a in auditoria if a["consensus_written"]),
        "TRUNCATED_SERIES": n_lacuna,
        "TRUNCATION_DIAGNOSTIC_COUNTS": dict(dec),
        "TRUNCATION_DECISION_ROWS": n_dec,
        "TRUNCATION_ACTION_LAYER": ("o 04 relata o facto geometrico; a accao "
                                    "(KEEP_TRUNCATED / SPLIT_CANDIDATE / "
                                    "EXCLUDE_CASE) e decidida sobre este "
                                    "manifesto, nao aqui"),
        "XML_DUPLICATE_SERIES_UIDS": len(xml_dup),
        "PER_READER_MASKS": len(objetos),
        "SELECTION_MANIFEST_SHA256": sha_ficheiro(sel_path),
        "FINAL_MANIFEST_SHA256": sha_ficheiro(fin_path),
        "SCRIPT04_OUTPUT_STATUS": "FAIL" if falhas_portao else "PASS",
        "FAILURES": falhas_portao,
    }
    json.dump({"RUN_ID": run_id, "generated": time.strftime("%Y-%m-%dT%H:%M:%S BRT"),
               "SCRIPT": "04_XML_ROUTE", "gates": gates},
              open(os.path.join(ANALYSIS, "report_04.json"), "w"),
              indent=1, sort_keys=True, default=str)

    rows = []
    for a in auditoria:
        pub = bool(a["consensus_written"])
        rows.append({
            "patient_id": a["patient_id"],
            "series_instance_uid": a["series_instance_uid"],
            "status_04": a["status"],
            "has_consensus": pub,
            "consensus_published": pub,
            "disposition": "INCLUDED" if pub else "EXCLUDED",
            "cohort": "UNASSIGNED",
            "consensus_relpath": (os.path.join("annotations", "consensus", "lidc",
                                               a["patient_id"], "consensus.nii.gz")
                                  if pub else ""),
            "n_per_reader_masks": a["n_per_reader_masks"],
            "merge_status": a["merge_status"],
            "truncation_class": a["truncation_class"],
            "negative_documented": bool(a.get("negative_documented")),
            "adjudication_evidence": a.get("adjudication_evidence"),
            "exclusion_reason": "" if pub else a["status"],
            "detail": (a.get("notes", "") or "")[:140]})
    corpo = {
        "manifest_id": "lidc_annotations_%s" % run_id,
        "producer": "04_build_lidc_annotations.py",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S BRT"),
        "SCRIPT04_OUTPUT_MANIFEST_FROZEN": "YES",
        "SCRIPT04_OUTPUT_STATUS": gates["SCRIPT04_OUTPUT_STATUS"],
        "SCRIPT02_RUN_ID": _p2.get("run_id"),
        "SCRIPT03_RUN_ID": _p3.get("run_id"),
        "CONSENSUS_RULE": gates["CONSENSUS_RULE"],
        "READER_KEY": gates["READER_KEY"],
        "n_rows": len(rows),
        "n_consensus_published": sum(1 for r in rows if r["has_consensus"]),
        "rows": rows,
    }
    corpo["manifest_sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in corpo.items() if k != "manifest_sha256"},
                   indent=1, sort_keys=True).encode()).hexdigest()
    with open(os.path.join(CONFIG, os.path.basename(MANIFESTO_SAIDA)), "w") as f:
        json.dump(corpo, f, indent=1, sort_keys=True)

    if ENSAIO:
        log.warning("  ENSAIO: ponteiro NAO publicado")
        return 1 if falhas_portao else 0

    tmp = PONTEIRO_SAIDA + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"script": "04_BUILD_LIDC_ANNOTATIONS", "run_id": run_id,
                   "published_at": time.strftime("%Y-%m-%dT%H:%M:%S BRT"),
                   "annotations_manifest": os.path.relpath(MANIFESTO_SAIDA, BASE),
                   "manifest_sha256": corpo["manifest_sha256"],
                   "consensus_dir": "annotations/consensus/lidc",
                   "per_reader_dir": "annotations/per_reader/lidc",
                   "object_table": "annotations/object_tables/lidc_nodules.csv",
                   "report": "analysis/report_04.json",
                   "gates": gates}, f, indent=1, sort_keys=True, default=str)
    os.replace(tmp, PONTEIRO_SAIDA)
    log.info("  publicado: %d casos · manifesto %s… · ponteiro CURRENT_LIDC_ANNOTATIONS.json"
             % (len(rows), corpo["manifest_sha256"][:16]))

    log.info("-" * 76)
    for k in ("SCRIPT02_RUN_ID", "SCRIPT03_RUN_ID", "SELECTED_SERIES_SCANNED",
              "SQLITE_READ_ONLY", "RAW_TREE_WRITES", "CONSENSUS_RULE",
              "READER_KEY", "MERGE_DECLARED", "MERGE_PERFORMED",
              "CONSENSUS_WRITTEN", "PER_READER_MASKS", "TRUNCATED_SERIES",
              "TRUNCATION_DIAGNOSTIC_COUNTS", "TRUNCATION_DECISION_ROWS",
              "SCRIPT04_OUTPUT_STATUS"):
        log.info("  %-32s %s" % (k, gates[k]))
    log.info("  %-32s %s" % ("STATUS_COUNTS", dict(contagem)))
    for f in falhas_portao:
        log.error("  FALHA: %s" % f)
    log.info("  annotations/ | %s | %.0f s" % (run_id, time.time() - t0))
    return 1 if falhas_portao else 0


if __name__ == "__main__":
    sys.exit(main())
