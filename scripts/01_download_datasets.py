#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    01_download_datasets.py

File type:
    Script

Purpose:
    Downloads LIDC-IDRI and RIDER Lung CT (TCIA/NBIA API, series by series) and LNDb (Zenodo, file by file) with automatic resumption.

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
    - See docs/file_level_documentation.md.

Dependencies:
    - requests
    - tcia_utils

Execution:
    PULMONARY_BASE=<project_dir> python scripts/01_download_datasets.py

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
import hashlib
import logging
import time
from datetime import datetime
from pathlib import Path

from pulmonary_base import BASE
DATA_RAW      = os.path.join(BASE, "data/raw")
LOGS_DIR      = os.path.join(BASE, "logs")
LIDC_ANNOT    = os.path.join(DATA_RAW, "lidc_annotations")

for d in [os.path.join(DATA_RAW, "lidc_idri"),
          os.path.join(DATA_RAW, "rider"),
          os.path.join(DATA_RAW, "lndb"),
          os.path.join(LIDC_ANNOT, "seg"),
          os.path.join(LIDC_ANNOT, "sr"),
          LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "download.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

TCIA_LIDC        = "LIDC-IDRI"
TCIA_RIDER       = "RIDER Lung CT"
TCIA_BASE_URL    = "https://nbia.cancerimagingarchive.net/nbia-api/services/v4"
LNDB_ZENODO_URL  = "https://zenodo.org/api/records/8348419"

manifest_rows  = []
checksum_rows  = []

PROGRESS_FILE  = os.path.join(LOGS_DIR, "download_progress.json")

MANIFEST_FIELDS = [
    "type", "dataset", "file", "series_uid", "patient_id",
    "study_uid", "modality", "series_date", "image_count",
    "manufacturer", "n_files", "size_bytes", "sha256", "checksum_scope",
    "acquired", "source", "version", "notes",
    "verification_status", "expected_images", "observed_files",
]
CHECKSUM_FIELDS = ["file", "sha256", "size_bytes", "checksum_scope"]

SCOPE_SERIES = "dir_listing(name,size)+sha256(first_dcm)"
SCOPE_FILE   = "full_file_content"


def load_progress():
    if os.path.isfile(PROGRESS_FILE):
        return json.load(open(PROGRESS_FILE))
    return {"lidc_idri": [], "rider": [], "lndb": [], "lidc_seg": [], "lidc_sr": []}


def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def sha256_file(path, chunk=1 << 20):
    """SHA-256 do conteúdo completo de um arquivo."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def sha256_dir_fingerprint(directory, ext=".dcm"):
    """
    Impressão digital rápida de um diretório: SHA-256 da lista ordenada de
    (nome, tamanho_bytes) de cada arquivo com a extensão dada.
    Não lê o conteúdo dos arquivos — é instantâneo mesmo para >100 DICOMs.
    """
    h = hashlib.sha256()
    entries = sorted(
        (fn, os.path.getsize(os.path.join(directory, fn)))
        for fn in os.listdir(directory)
        if fn.endswith(ext)
    )
    for fn, sz in entries:
        h.update(f"{fn}:{sz}\n".encode())
    return h.hexdigest()


def record_tcia_series(series_dir, series_meta, dataset, collection_url):
    """
    Registra metadados completos de uma série TCIA baixada.
    series_meta: dict retornado por nbia.getSeries (uma entrada da lista).
    """
    if not os.path.isdir(series_dir):
        return

    dcm_files = sorted([f for f in os.listdir(series_dir) if f.endswith(".dcm")])
    if not dcm_files:
        return

    n_files    = len(dcm_files)
    total_size = sum(os.path.getsize(os.path.join(series_dir, f)) for f in dcm_files)

    dir_fp      = sha256_dir_fingerprint(series_dir)
    first_sha256 = sha256_file(os.path.join(series_dir, dcm_files[0]))
    combined    = hashlib.sha256(f"{dir_fp}|{first_sha256}".encode()).hexdigest()

    uid         = series_meta.get("SeriesInstanceUID", "")
    version     = (series_meta.get("MaxSubmissionTimestamp", "") or
                   series_meta.get("SeriesDate", ""))

    row = {
        "type":         "tcia_series",
        "dataset":      dataset,
        "file":         series_dir,
        "series_uid":   uid,
        "patient_id":   series_meta.get("PatientID", ""),
        "study_uid":    series_meta.get("StudyInstanceUID", ""),
        "modality":     series_meta.get("Modality", ""),
        "series_date":  series_meta.get("SeriesDate", ""),
        "image_count":  series_meta.get("ImageCount", ""),
        "manufacturer": series_meta.get("Manufacturer", ""),
        "n_files":      n_files,
        "size_bytes":   total_size,
        "sha256":         combined,
        "checksum_scope": SCOPE_SERIES,
        "acquired":       datetime.now().isoformat(),
        "source":         f"TCIA:{collection_url}",
        "version":        version,
        "notes":          f"sha256_first_dcm={first_sha256}",
    }
    _esp = series_meta.get("ImageCount", "")
    try:
        _esp_i = int(str(_esp).strip())
    except (TypeError, ValueError):
        _esp_i = None
    if n_files == 0:
        _vs = "FAIL_EMPTY"
    elif not combined:
        _vs = "FAIL_NO_CHECKSUM"
    elif _esp_i is None:
        _vs = "PASS_NO_EXPECTED_COUNT"
    elif n_files == _esp_i:
        _vs = "PASS"
    else:
        _vs = "FAIL_FILE_COUNT_MISMATCH"
    row["verification_status"] = _vs
    row["expected_images"] = _esp
    row["observed_files"] = n_files
    manifest_rows.append(row)
    checksum_rows.append({
        "file":           series_dir,
        "sha256":         combined,
        "size_bytes":     total_size,
        "checksum_scope": SCOPE_SERIES,
    })


def record_file(path, source, version="", dataset="", notes=""):
    """Registra metadados de um arquivo individual (usado para LNDb)."""
    if not os.path.isfile(path):
        return
    size = os.path.getsize(path)
    chk  = sha256_file(path)
    _vs_f = "PASS" if (size > 0 and chk) else ("FAIL_EMPTY" if size == 0
                                               else "FAIL_NO_CHECKSUM")
    manifest_rows.append({
        "verification_status": _vs_f,
        "expected_images": "", "observed_files": 1,
        "type":        "file",
        "dataset":     dataset,
        "file":        path,
        "series_uid":  "",
        "patient_id":  "",
        "study_uid":   "",
        "modality":    "",
        "series_date": "",
        "image_count": "",
        "manufacturer":"",
        "n_files":     1,
        "size_bytes":  size,
        "sha256":      chk,
        "checksum_scope": SCOPE_FILE,
        "acquired":    datetime.now().isoformat(),
        "source":      source,
        "version":     version,
        "notes":       notes,
    })
    checksum_rows.append({"file": path, "sha256": chk, "size_bytes": size,
                          "checksum_scope": SCOPE_FILE})


def series_is_complete(series_dir, series_meta=None, min_files=1):
    """
    Completude de uma série descarregada.

    Quando o TCIA informa `ImageCount`, esse é o critério: a série só está
    completa se tiver pelo menos essa quantidade de ficheiros. O antigo limiar de
    um único `.dcm` aceitava como completa uma série com 1 de 200 cortes, o que
    faria o download ser dado por terminado sobre um volume truncado.

    `min_files` só é usado quando não há `ImageCount` na metadata.
    """
    if not os.path.isdir(series_dir):
        return False
    dcm = [f for f in os.listdir(series_dir) if f.endswith(".dcm")]
    if not dcm:
        return False

    esperado = None
    if series_meta:
        ic = str(series_meta.get("ImageCount", "") or "").strip()
        if ic.isdigit():
            esperado = int(ic)

    if esperado is not None:
        if len(dcm) < esperado:
            log.warning(f"  Série incompleta: {os.path.basename(series_dir)} "
                        f"— {len(dcm)} de {esperado} ficheiros")
            return False
        return True

    return len(dcm) >= min_files


def clean_tmp_files(directory):
    for root, _, files in os.walk(directory):
        for fn in files:
            if fn.endswith(".tmp"):
                path = os.path.join(root, fn)
                os.remove(path)
                log.warning(f"  Removido arquivo parcial: {path}")


def download_lidc_idri(progress):
    log.info("=== LIDC-IDRI ===")
    out_dir = os.path.join(DATA_RAW, "lidc_idri")
    clean_tmp_files(out_dir)

    try:
        from tcia_utils import nbia
    except ImportError:
        log.error("tcia_utils não instalado. Execute: pip install tcia_utils")
        return

    series_cache = os.path.join(LOGS_DIR, "lidc_series_list.json")
    if os.path.isfile(series_cache):
        with open(series_cache) as f:
            series_list = json.load(f)
        log.info(f"Lista em cache: {len(series_list)} séries")
    else:
        log.info("Obtendo lista de séries LIDC-IDRI …")
        series_list = nbia.getSeries(collection=TCIA_LIDC, modality="CT")
        with open(series_cache, "w") as f:
            json.dump(series_list, f, default=str)
        log.info(f"Séries encontradas: {len(series_list)}")

    done      = set(progress.get("lidc_idri", []))
    total     = len(series_list)
    remaining = [s for s in series_list
                 if s.get("SeriesInstanceUID", "") not in done]
    log.info(f"Já baixadas: {len(done)}/{total}  Restantes: {len(remaining)}")

    registered = 0
    for series in series_list:
        uid        = series.get("SeriesInstanceUID", "")
        series_dir = os.path.join(out_dir, uid)
        if series_is_complete(series_dir, series):
            record_tcia_series(series_dir, series, "lidc_idri",
                               f"TCIA collection={TCIA_LIDC}")
            registered += 1
    log.info(f"LIDC-IDRI: {registered}/{total} séries registradas no manifesto")

    for i, series in enumerate(remaining):
        uid        = series.get("SeriesInstanceUID", f"unknown_{i}")
        series_dir = os.path.join(out_dir, uid)

        if series_is_complete(series_dir, series):
            done.add(uid)
            progress["lidc_idri"] = list(done)
            save_progress(progress)
            continue

        log.info(f"  [{i+1}/{len(remaining)}] Baixando: {uid} …")
        try:
            nbia.downloadSeries([series], path=out_dir, format="dcm")
            done.add(uid)
            progress["lidc_idri"] = list(done)
            save_progress(progress)
        except Exception as e:
            log.error(f"  Erro em {uid}: {e} — será retomado na próxima execução")

    log.info(f"LIDC-IDRI: {len(done)}/{total} séries concluídas")


def _download_lidc_annot(progress, modality, out_subdir, progress_key,
                         series_cache_name):
    """
    Baixa séries de anotação LIDC-IDRI (SEG ou SR) via NBIA API.

    SEG  → data/raw/lidc_annotations/seg/{SeriesInstanceUID}/  (máscaras por leitor)
    SR   → data/raw/lidc_annotations/sr/{SeriesInstanceUID}/   (atributos radiológicos)

    Passa todas as séries pendentes de uma vez para nbia.downloadSeries() de modo
    que ela use max_workers=10 internamente (~10× mais rápido que sequencial).

    Script 04 usará:
      SEG → reconstruir masks per_reader e consenso (τ=0.60)
      SR  → extrair malignancy, texture, margin, subtlety, calcification, etc.
    """
    log.info(f"=== LIDC-IDRI {modality} ===")
    out_dir = os.path.join(LIDC_ANNOT, out_subdir)

    try:
        from tcia_utils import nbia
    except ImportError:
        log.error("tcia_utils não instalado.")
        return

    cache_path = os.path.join(LOGS_DIR, series_cache_name)
    if os.path.isfile(cache_path):
        with open(cache_path) as f:
            series_list = json.load(f)
        log.info(f"Lista em cache: {len(series_list)} séries {modality}")
    else:
        log.info(f"Obtendo lista de séries {modality} LIDC-IDRI …")
        series_list = nbia.getSeries(collection=TCIA_LIDC, modality=modality)
        with open(cache_path, "w") as f:
            json.dump(series_list, f, default=str)
        log.info(f"Séries encontradas: {len(series_list)}")

    done  = set(progress.get(progress_key, []))
    total = len(series_list)

    for series in series_list:
        uid        = series.get("SeriesInstanceUID", "")
        series_dir = os.path.join(out_dir, uid)
        if series_is_complete(series_dir, series):
            record_tcia_series(series_dir, series, f"lidc_{modality.lower()}",
                               f"TCIA collection={TCIA_LIDC}")
            done.add(uid)

    progress[progress_key] = list(done)
    save_progress(progress)

    remaining = [s for s in series_list
                 if s.get("SeriesInstanceUID", "") not in done]
    log.info(f"Já presentes: {len(done)}/{total}  A baixar: {len(remaining)}")

    if not remaining:
        log.info(f"LIDC-IDRI {modality}: todos já presentes.")
        return

    log.info(f"Iniciando download paralelo de {len(remaining)} séries {modality} …")
    try:
        nbia.downloadSeries(remaining, path=out_dir, format="dcm")
    except Exception as e:
        log.error(f"Erro no download em lote {modality}: {e}")

    for series in remaining:
        uid        = series.get("SeriesInstanceUID", "")
        series_dir = os.path.join(out_dir, uid)
        if series_is_complete(series_dir, series):
            record_tcia_series(series_dir, series, f"lidc_{modality.lower()}",
                               f"TCIA collection={TCIA_LIDC}")
            done.add(uid)

    progress[progress_key] = list(done)
    save_progress(progress)

    failed = total - len(done)
    log.info(f"LIDC-IDRI {modality}: {len(done)}/{total} concluídas"
             + (f"  ({failed} falhas — serão retomadas na próxima execução)" if failed else ""))


def download_lidc_seg(progress):
    _download_lidc_annot(
        progress,
        modality="SEG",
        out_subdir="seg",
        progress_key="lidc_seg",
        series_cache_name="lidc_seg_series_list.json",
    )


def download_lidc_sr(progress):
    _download_lidc_annot(
        progress,
        modality="SR",
        out_subdir="sr",
        progress_key="lidc_sr",
        series_cache_name="lidc_sr_series_list.json",
    )


def download_rider(progress):
    log.info("=== RIDER Lung CT ===")
    out_dir = os.path.join(DATA_RAW, "rider")
    clean_tmp_files(out_dir)

    try:
        from tcia_utils import nbia
    except ImportError:
        log.error("tcia_utils não instalado.")
        return

    series_cache = os.path.join(LOGS_DIR, "rider_series_list.json")
    if os.path.isfile(series_cache):
        with open(series_cache) as f:
            series_list = json.load(f)
        log.info(f"Lista em cache: {len(series_list)} séries")
    else:
        log.info("Obtendo lista de séries RIDER …")
        series_list = nbia.getSeries(collection=TCIA_RIDER, modality="CT")
        with open(series_cache, "w") as f:
            json.dump(series_list, f, default=str)
        log.info(f"Séries encontradas: {len(series_list)}")

    done      = set(progress.get("rider", []))
    total     = len(series_list)
    remaining = [s for s in series_list
                 if s.get("SeriesInstanceUID", "") not in done]
    log.info(f"Já baixadas: {len(done)}/{total}  Restantes: {len(remaining)}")

    registered = 0
    for series in series_list:
        uid        = series.get("SeriesInstanceUID", "")
        series_dir = os.path.join(out_dir, uid)
        if series_is_complete(series_dir, series):
            record_tcia_series(series_dir, series, "rider",
                               f"TCIA collection={TCIA_RIDER}")
            registered += 1
    log.info(f"RIDER: {registered}/{total} séries registradas no manifesto")

    for i, series in enumerate(remaining):
        uid        = series.get("SeriesInstanceUID", f"unknown_{i}")
        series_dir = os.path.join(out_dir, uid)

        if series_is_complete(series_dir, series):
            done.add(uid)
            progress["rider"] = list(done)
            save_progress(progress)
            continue

        log.info(f"  [{i+1}/{len(remaining)}] Baixando: {uid} …")
        try:
            nbia.downloadSeries([series], path=out_dir, format="dcm")
            done.add(uid)
            progress["rider"] = list(done)
            save_progress(progress)
        except Exception as e:
            log.error(f"  Erro em {uid}: {e} — será retomado na próxima execução")

    log.info(f"RIDER: {len(done)}/{total} séries concluídas")


def _lndb_load_zenodo_meta():
    """Retorna (version, doi) consultando o cache local ou o Zenodo."""
    import requests
    meta_cache = os.path.join(LOGS_DIR, "lndb_zenodo_meta.json")
    if os.path.isfile(meta_cache):
        with open(meta_cache) as f:
            meta = json.load(f)
    else:
        log.info("Consultando Zenodo …")
        resp = requests.get(LNDB_ZENODO_URL, timeout=60)
        resp.raise_for_status()
        meta = resp.json()
        with open(meta_cache, "w") as f:
            json.dump(meta, f)
    version = meta.get("metadata", {}).get("version", "1.0")
    doi     = meta.get("doi", LNDB_ZENODO_URL)
    return version, doi, meta


def audit_lndb_extracted(progress):
    """
    Registra no manifesto os arquivos MHD/RAW/NIfTI já extraídos em data/raw/lndb/.
    Sempre re-registra a cada execução (o manifesto é reconstruído a cada run).
    Usa a flag 'lndb_extracted' no progresso apenas para controle de extração,
    não para pular o registro — igual ao comportamento de LIDC/RIDER.
    """
    out_dir   = os.path.join(DATA_RAW, "lndb")
    data_dir  = os.path.join(out_dir, "data")
    masks_dir = os.path.join(out_dir, "gt_masks")

    mhds = sorted(Path(data_dir).glob("*.mhd"))  if os.path.isdir(data_dir)  else []
    nifs = sorted(Path(masks_dir).glob("*.nii.gz")) if os.path.isdir(masks_dir) else []

    if not mhds:
        return

    try:
        import requests
        version, doi, _ = _lndb_load_zenodo_meta()
    except Exception:
        version, doi = "1.0", LNDB_ZENODO_URL

    log.info(f"LNDb: registrando {len(mhds)} MHDs + RAWs + {len(nifs)} NIfTIs no manifesto …")
    source = f"Zenodo:{doi}"

    for mhd in mhds:
        record_file(str(mhd),  source=source, version=version, dataset="lndb",
                    notes="extracted_via_rclone")
        raw = mhd.with_suffix(".raw")
        if raw.exists():
            record_file(str(raw), source=source, version=version, dataset="lndb",
                        notes="extracted_via_rclone")

    for nif in nifs:
        record_file(str(nif), source=source, version=version, dataset="lndb",
                    notes="gt_mask_extracted_via_rclone")

    for aux in ["zenodo_meta.json",
                "New version of the DMP Common Standard Ontology.pdf"]:
        p = os.path.join(out_dir, aux)
        if os.path.isfile(p):
            record_file(p, source=source, version=version, dataset="lndb")

    if "lndb_extracted" not in progress.get("lndb", []):
        done = list(progress.get("lndb", []))
        done.append("lndb_extracted")
        progress["lndb"] = done
        save_progress(progress)

    log.info(f"LNDb: {len(mhds)*2 + len(nifs)} arquivos registrados no manifesto")


def download_lndb(progress):
    """
    Arquivos de imagem (RARs) são pulados se os MHDs já foram extraídos via rclone.
    Arquivos de metadados (CSVs, ZIPs de anotação) são sempre verificados e baixados
    se ausentes — independentemente de os RARs estarem presentes ou não.
    Arquivos do conjunto de teste e auxiliares não utilizados são ignorados.
    """
    log.info("=== LNDb (Zenodo) ===")
    out_dir = os.path.join(DATA_RAW, "lndb")
    clean_tmp_files(out_dir)

    IMAGE_ARCHIVES = {
        "data0.rar", "data1.rar", "data2.rar",
        "data3.rar", "data4.rar", "data5.rar",
    }
    SKIP_FILES = {
        "testdata0.rar", "testdata1.rar",
        "masks.rar",
        "scripts.zip",
        "submission.zip",
        "README.pdf",
    }

    data_dir = os.path.join(out_dir, "data")
    has_mhds = os.path.isdir(data_dir) and any(Path(data_dir).glob("*.mhd"))
    has_rars = any(Path(out_dir).glob("*.rar"))

    if has_mhds and not has_rars:
        audit_lndb_extracted(progress)

    try:
        import requests
        import zipfile
    except ImportError:
        log.error("requests não instalado. Execute: pip install requests")
        return

    try:
        version, doi, meta = _lndb_load_zenodo_meta()
    except Exception as e:
        log.error(f"Não foi possível obter metadados do Zenodo: {e}")
        return

    files_info = meta.get("files", [])
    source     = f"Zenodo:{doi}"
    log.info(f"LNDb versão: {version} | DOI: {doi} | arquivos: {len(files_info)}")

    done = set(progress.get("lndb", []))

    for fi in files_info:
        fname         = fi["key"]
        expected_size = fi.get("size", 0)
        url           = fi["links"]["self"]
        dest          = os.path.join(out_dir, fname)
        tmp           = dest + ".tmp"

        if fname in SKIP_FILES:
            log.info(f"  Ignorado (não usado no projeto): {fname}")
            continue

        if fname in IMAGE_ARCHIVES and (has_mhds or fname in done):
            log.info(f"  Imagens já extraídas, pulando RAR: {fname}")
            continue

        if os.path.isfile(dest):
            if fname not in done:
                log.info(f"  Presente (não registrado): {fname} — registrando")
                record_file(dest, source=source, version=version, dataset="lndb",
                            notes=f"zenodo_size={expected_size}")
                done.add(fname)
                progress["lndb"] = list(done)
                save_progress(progress)
            else:
                log.info(f"  Já existe: {fname}")
                record_file(dest, source=source, version=version, dataset="lndb",
                            notes=f"zenodo_size={expected_size}")
            continue

        headers    = {}
        downloaded = 0
        if os.path.isfile(tmp):
            downloaded = os.path.getsize(tmp)
            headers["Range"] = f"bytes={downloaded}-"
            log.info(f"  Retomando {fname} a partir de {downloaded/1e6:.1f} MB …")
        else:
            log.info(f"  Baixando {fname} …")

        try:
            r = requests.get(url, stream=True, timeout=600, headers=headers)
            if r.status_code == 416:
                os.rename(tmp, dest)
            else:
                r.raise_for_status()
                mode = "ab" if downloaded > 0 else "wb"
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                os.rename(tmp, dest)

            record_file(dest, source=source, version=version, dataset="lndb",
                        notes=f"zenodo_size={expected_size}")
            done.add(fname)
            progress["lndb"] = list(done)
            save_progress(progress)

            if fname.endswith(".zip"):
                log.info(f"  Extraindo {fname} …")
                with zipfile.ZipFile(dest, "r") as zf:
                    zf.extractall(out_dir)
                    log.info(f"    → {zf.namelist()}")
            elif fname.endswith(".rar"):
                log.info(f"  Extraindo {fname} …")
                import subprocess
                result = subprocess.run(
                    ["unrar", "x", "-o+", dest, out_dir],
                    capture_output=True, text=True
                )
                if result.returncode != 0:
                    log.warning(f"  unrar falhou: {result.stderr[:200]}")

        except Exception as e:
            log.error(f"  Erro em {fname}: {e} — será retomado na próxima execução")

    _relevantes = {f["key"] for f in files_info
                   if f["key"] not in SKIP_FILES and f["key"] not in IMAGE_ARCHIVES}
    n_done  = len([f for f in done if f in _relevantes])
    n_total = len(_relevantes)
    log.info(f"LNDb metadados: {n_done}/{n_total} arquivos processados")


def save_manifests():
    if not manifest_rows:
        log.warning("Nenhum dado para manifesto — nada novo foi baixado nesta execução.")
        return

    mf = os.path.join(LOGS_DIR, "download_manifest.csv")
    with open(mf, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(manifest_rows)
    log.info(f"Manifesto: {mf} ({len(manifest_rows)} entradas)")

    cf = os.path.join(LOGS_DIR, "download_checksums.csv")
    with open(cf, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CHECKSUM_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(checksum_rows)
    log.info(f"Checksums: {cf} ({len(checksum_rows)} entradas)")

    import collections as _c
    _est = _c.Counter(r.get("verification_status", "SEM_ESTADO") for r in manifest_rows)
    _falhas = {k: v for k, v in _est.items() if k.startswith("FAIL") or k == "SEM_ESTADO"}
    log.info("Estados de verificacao por linha:")
    for _k, _v in sorted(_est.items()):
        log.info(f"  {_k:<28s} {_v}")
    _por_ds = _c.defaultdict(_c.Counter)
    for r in manifest_rows:
        _por_ds[r.get("dataset", "?")][r.get("verification_status", "SEM_ESTADO")] += 1
    for _ds in sorted(_por_ds):
        _ok = _por_ds[_ds].get("PASS", 0) + _por_ds[_ds].get("PASS_NO_EXPECTED_COUNT", 0)
        log.info(f"  {_ds:<14s} {_ok}/{sum(_por_ds[_ds].values())} verificados")
    if _falhas:
        log.error(f"SCRIPT01_OUTPUT_STATUS = FAIL — linhas nao verificadas: {dict(_falhas)}")
        _amostra = [r["file"] for r in manifest_rows
                    if r.get("verification_status", "SEM_ESTADO") in _falhas][:5]
        log.error(f"  exemplos: {_amostra}")
        sys.exit(1)
    log.info("SCRIPT01_OUTPUT_STATUS = PASS — todas as linhas verificadas")


REFERENCIAS = [
    {
        "nome": "lidc-idri-nodule-counts-6-23-2015.xlsx",
        "url": ("https://wiki.cancerimagingarchive.net/download/attachments/"
                "1966254/lidc-idri%20nodule%20counts%20%286-23-2015%29.xlsx"),
        "sha256": "9860039cbd9af7a679c09fc239ca1553d351a969fa9a0acf50128e1210a2b5a5",
        "usado_por": "07_draw_bd1_partition.py",
        "papel": ("contagem oficial de nodulos por paciente: Total Number of "
                  "Nodules* e Number of Nodules >=3mm**, criterio UNIAO entre "
                  "leitores (>=1 dos 4). Estratifica a particao BD1."),
    },
]


def download_referencias(progress):
    """
    Descarrega os recursos de referencia publicados com o LIDC-IDRI.

    Nao sao produto do pipeline nem resultado nosso: sao dados de origem, como o
    DICOM. Entram no manifesto com hash proprio para que o Script 07 nao dependa
    de um ficheiro posto no caminho a mao.

    Ficheiro ja presente com o hash esperado nao e redescarregado; com hash
    diferente e falha, nao substituicao silenciosa.
    """
    out_dir = os.path.join(BASE, "data", "reference")
    os.makedirs(out_dir, exist_ok=True)
    log.info("=== Referencias LIDC-IDRI ===")
    for r in REFERENCIAS:
        dest = os.path.join(out_dir, r["nome"])
        if os.path.isfile(dest):
            obs = sha256_file(dest)
            if obs == r["sha256"]:
                log.info(f"  {r['nome']}: presente e conferido ({obs[:16]}…)")
                record_file(dest, r["url"], dataset="reference",
                            notes=f"usado_por={r['usado_por']}")
                continue
            log.error(f"  {r['nome']}: hash divergente. esperado {r['sha256']} "
                      f"obtido {obs}. Nao substituo em silencio.")
            sys.exit(1)
        log.info(f"  {r['nome']}: ausente — descarregando de {r['url']}")
        try:
            import requests
            resp = requests.get(r["url"], timeout=120)
            resp.raise_for_status()
            tmp = dest + ".part"
            with open(tmp, "wb") as f:
                f.write(resp.content)
            obs = sha256_file(tmp)
            if obs != r["sha256"]:
                os.remove(tmp)
                log.error(f"  {r['nome']}: descarregado com hash {obs}, "
                          f"esperado {r['sha256']}")
                sys.exit(1)
            os.replace(tmp, dest)
            log.info(f"  {r['nome']}: descarregado e conferido")
        except SystemExit:
            raise
        except Exception as e:
            log.error(f"  {r['nome']}: falha na descarga — {type(e).__name__}: {e}")
            sys.exit(1)
        record_file(dest, r["url"], dataset="reference",
                    notes=f"usado_por={r['usado_por']}")


if __name__ == "__main__":
    progress = load_progress()
    download_lidc_idri(progress)
    download_lidc_seg(progress)
    download_lidc_sr(progress)
    download_rider(progress)
    download_lndb(progress)
    download_referencias(progress)
    save_manifests()
    log.info("Download concluído.")
