#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    10_infer_and_export_predictions.py

File type:
    Script

Purpose:
    Runs nnU-Net v2 inference and exports binary masks, probability maps and connected components for BD1, RIDER and LNDb.

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
    - SimpleITK
    - numpy
    - scipy

Execution:
    PULMONARY_BASE=<project_dir> python scripts/10_infer_and_export_predictions.py

Portability notes:
    All paths are resolved relative to the project base directory given by the PULMONARY_BASE environment variable. No machine-specific path is required.

Reproducibility notes:
    Environment pinned in requirements.txt and environment/. Seeds and frozen parameters are recorded in the run manifests under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) are not included and must be obtained from their original repositories.
"""

import os
import csv
import json
import time
import shutil
import logging
import subprocess
from datetime import datetime


from pulmonary_base import BASE
NNUNET_RAW     = os.path.join(BASE, "nnUNet_raw")
NNUNET_PRE     = os.path.join(BASE, "nnUNet_preprocessed")
NNUNET_RESULTS = os.path.join(BASE, "models")
PRED_DIR       = os.path.join(BASE, "predictions")
ANALYSIS_DIR   = os.path.join(BASE, "analysis")
CONFIG_DIR     = os.path.join(BASE, "config")
LOGS_DIR       = os.path.join(BASE, "logs")

DATASET_ID   = 1
DATASET_NAME = f"Dataset{DATASET_ID:03d}_PulmonaryNodules"
IMAGES_TR    = os.path.join(NNUNET_RAW, DATASET_NAME, "imagesTr")
IMAGES_TS    = os.path.join(NNUNET_RAW, DATASET_NAME, "imagesTs")

ARCHITECTURES = [
    {"name": "2D",       "config": "2d",
     "trainer": None,
     "planner": None,                  "plans_id": "nnUNetPlans"},
    {"name": "2D_REM",   "config": "2d",
     "trainer": None,
     "planner": "ResEncUNetMPlanner",  "plans_id": "nnUNetResEncUNetMPlans"},
    {"name": "2D_REL",   "config": "2d",
     "trainer": "nnUNetTrainerCheckEvery10",
     "planner": "ResEncUNetLPlanner",  "plans_id": "nnUNetResEncUNetLPlans"},
    {"name": "3D_F",     "config": "3d_fullres",
     "trainer": None,
     "planner": None,                  "plans_id": "nnUNetPlans"},
    {"name": "3D_FREM",  "config": "3d_fullres",
     "trainer": None,
     "planner": "ResEncUNetMPlanner",  "plans_id": "nnUNetResEncUNetMPlans"},
    {"name": "F100_lowres",  "config": "3d_lowres",
     "trainer": "nnUNetTrainer_Foreground100",
     "planner": None,                  "plans_id": "nnUNetPlans"},
    {"name": "F100_ResEncM", "config": "3d_lowres",
     "trainer": "nnUNetTrainer_Foreground100",
     "planner": "ResEncUNetMPlanner",  "plans_id": "nnUNetResEncUNetMPlans"},
    {"name": "3D_CF",    "config": "3d_cascade_fullres",
     "trainer": "nnUNetTrainer_Foreground100",
     "planner": None,                  "plans_id": "nnUNetPlans",
     "prev_stage_name": "F100_lowres"},
    {"name": "3D_CFREM", "config": "3d_cascade_fullres",
     "trainer": "nnUNetTrainer_Foreground100",
     "planner": "ResEncUNetMPlanner",  "plans_id": "nnUNetResEncUNetMPlans",
     "prev_stage_name": "F100_ResEncM"},
    {"name": "F100_fullres", "config": "3d_fullres",
     "trainer": "nnUNetTrainer_Foreground100",
     "planner": None,                  "plans_id": "nnUNetPlans"},
]

LNDB_ARCH_NAMES = ["3D_F", "3D_FREM"]

_arch_filter = os.environ.get("PULMONARY_ARCH_FILTER", "")
if _arch_filter:
    _allowed = {n.strip() for n in _arch_filter.split(",")}
    ARCHITECTURES = [a for a in ARCHITECTURES if a["name"] in _allowed]

for _d in [ANALYSIS_DIR, LOGS_DIR, PRED_DIR]:
    os.makedirs(_d, exist_ok=True)

os.environ["nnUNet_raw"]          = NNUNET_RAW
os.environ["nnUNet_preprocessed"] = NNUNET_PRE
os.environ["nnUNet_results"]      = NNUNET_RESULTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "inference.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)
manifest_rows = []

_DONE_MARKER = "_predict_done.json"


def load_degenerate_architectures():
    """Lê training_summary.csv e retorna set de nomes de arquitetura degenerados."""
    summary = os.path.join(ANALYSIS_DIR, "training_summary.csv")
    degenerate = set()
    if not os.path.isfile(summary):
        log.warning("training_summary.csv não encontrado — nenhuma arquitetura excluída.")
        return degenerate
    with open(summary) as f:
        for row in csv.DictReader(f):
            if row.get("status") == "degenerate":
                degenerate.add(row["architecture"])
    if degenerate:
        log.warning(f"Arquiteturas degeneradas excluídas da inferência: {sorted(degenerate)}")
    return degenerate


def load_folds():
    """Carrega config/bd1_folds.json → {fold_idx: {'train': [...], 'val': [...]}}."""
    path = os.path.join(CONFIG_DIR, "bd1_folds.json")
    if not os.path.isfile(path):
        log.error(f"Folds não encontrados: {path}. Execute script 07 primeiro.")
        return {}
    with open(path) as f:
        data = json.load(f)
    return {i: data["folds"][f"fold_{i}"] for i in range(5)}


def _make_flat_dir(img_base, flat_dir, patient_ids=None):
    """
    Cria diretório flat com {pid}_0000.nii.gz (symlink/cópia) para nnUNetv2_predict.
    img_base: pasta com subpastas {pid}/ct.nii.gz (estrutura do script 06).
    """
    os.makedirs(flat_dir, exist_ok=True)
    if patient_ids is None:
        patient_ids = sorted([
            p for p in os.listdir(img_base)
            if os.path.isdir(os.path.join(img_base, p))
        ])
    missing = 0
    for pid in patient_ids:
        src = os.path.join(img_base, pid, "ct.nii.gz")
        dst = os.path.join(flat_dir, f"{pid}_0000.nii.gz")
        if not os.path.isfile(src):
            missing += 1
            continue
        if not os.path.exists(dst):
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copy2(src, dst)
    if missing:
        log.warning(f"  {missing} CTs ausentes em {img_base}")
    return patient_ids


def _compute_cc(pred_dir):
    """
    Para cada {case}.nii.gz em pred_dir, salva {case}_cc.nii.gz com
    componentes conectados rotulados (scipy.ndimage.label).
    """
    import numpy as np
    import SimpleITK as sitk
    from scipy import ndimage as nd

    n = 0
    for fn in sorted(os.listdir(pred_dir)):
        if not fn.endswith(".nii.gz") or fn.endswith("_cc.nii.gz"):
            continue
        cc_path = os.path.join(pred_dir, fn.replace(".nii.gz", "_cc.nii.gz"))
        if os.path.isfile(cc_path):
            continue
        img = sitk.ReadImage(os.path.join(pred_dir, fn))
        arr = sitk.GetArrayFromImage(img).astype(np.uint8)
        labeled, _ = nd.label(arr)
        cc_img = sitk.GetImageFromArray(labeled.astype(np.int32))
        cc_img.CopyInformation(img)
        sitk.WriteImage(cc_img, cc_path, useCompression=True)
        n += 1
    if n:
        log.info(f"    CC: {n} arquivos → {os.path.relpath(pred_dir, BASE)}")


def run_predict(arch, input_folder, output_folder, folds=None, prev_stage_dir=None):
    """
    Executa nnUNetv2_predict. Pula se _predict_done.json existir.
    prev_stage_dir: pasta com predições do estágio lowres (obrigatório para cascade).
    Registra em manifest_rows. Computa componentes conectados ao final.
    """
    t_start = datetime.now().isoformat()
    done_path = os.path.join(output_folder, _DONE_MARKER)

    if os.path.isfile(done_path):
        log.info(f"  [SKIP] {arch['name']} → {os.path.relpath(output_folder, BASE)}")
        manifest_rows.append({
            "architecture": arch["name"],
            "config":       arch["config"],
            "plans_id":     arch["plans_id"],
            "folds":        str(folds),
            "input":        input_folder,
            "output":       output_folder,
            "status":       "skipped",
            "elapsed_s":    0,
            "timestamp":    t_start,
        })
        return "skipped"

    if arch.get("prev_stage_name") and prev_stage_dir is None:
        log.error(f"  CASCADE sem prev_stage_dir: {arch['name']} — pulando.")
        return "error_no_prev_stage"

    if prev_stage_dir and not os.path.isdir(prev_stage_dir):
        log.error(f"  prev_stage_dir ausente: {prev_stage_dir} — {arch['name']} pulado.")
        return "error_prev_stage_missing"

    os.makedirs(output_folder, exist_ok=True)

    cmd = [
        "nnUNetv2_predict",
        "-i", input_folder,
        "-o", output_folder,
        "-d", str(DATASET_ID),
        "-c", arch["config"],
        "--save_probabilities",
    ]
    if arch.get("trainer"):
        cmd += ["-tr", arch["trainer"]]
    if arch.get("planner"):
        cmd += ["-p", arch["plans_id"]]
    if folds is not None:
        cmd += ["-f"] + [str(f) for f in folds]
    if prev_stage_dir:
        cmd += ["-prev_stage_predictions", prev_stage_dir]

    log.info(f"  Predizendo: {arch['name']}  folds={folds}  "
             f"→ {os.path.relpath(output_folder, BASE)}")

    t0     = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0

    status = "ok" if result.returncode == 0 else f"error_{result.returncode}"
    if result.returncode != 0:
        log.error(f"  FALHA: {arch['name']} (código {result.returncode})")

    manifest_rows.append({
        "architecture": arch["name"],
        "config":       arch["config"],
        "plans_id":     arch["plans_id"],
        "folds":        str(folds),
        "input":        input_folder,
        "output":       output_folder,
        "status":       status,
        "elapsed_s":    round(elapsed),
        "timestamp":    t_start,
    })

    if result.returncode == 0:
        with open(done_path, "w") as f:
            json.dump({"timestamp": datetime.now().isoformat(),
                       "arch": arch["name"], "elapsed_s": round(elapsed)}, f)
        _compute_cc(output_folder)

    return status


def setup_bd1_raw_dirs():
    """
    Cria imagesTr (development) e imagesTs (internal) a partir de bd1_split.csv
    se esses diretórios ainda não existirem.
    """
    split_csv = os.path.join(CONFIG_DIR, "bd1_split.csv")
    if not os.path.isfile(split_csv):
        log.warning(f"bd1_split.csv não encontrado: {split_csv} — BD1 será pulado.")
        return

    dev_ids, int_ids = [], []
    with open(split_csv) as f:
        for row in csv.DictReader(f):
            pid, split = row["patient_id"], row["split"]
            if split == "development":
                dev_ids.append(pid)
            elif split == "internal":
                int_ids.append(pid)

    lidc_images = os.path.join(BASE, "data", "processed", "lidc_idri", "images")
    if not os.path.isdir(lidc_images):
        log.warning(f"LIDC-IDRI processado ausente: {lidc_images} — BD1 será pulado.")
        return

    os.makedirs(os.path.dirname(IMAGES_TR), exist_ok=True)

    if not os.path.isdir(IMAGES_TR) or not os.listdir(IMAGES_TR):
        log.info(f"  Criando imagesTr ({len(dev_ids)} casos)…")
        _make_flat_dir(lidc_images, IMAGES_TR, dev_ids)

    if not os.path.isdir(IMAGES_TS) or not os.listdir(IMAGES_TS):
        log.info(f"  Criando imagesTs ({len(int_ids)} casos)…")
        _make_flat_dir(lidc_images, IMAGES_TS, int_ids)


def infer_bd1_oof(active_archs, folds):
    """
    Cada caso do conjunto de desenvolvimento é predito pelo fold que não o treinou.
    Para cada fold n: cria subdir de input com apenas os casos de validação de n,
    executa nnUNetv2_predict -f n, salva em predictions/bd1_oof/{arch}/fold_{n}/.
    """
    log.info("=== BD1 out-of-fold (desenvolvimento) ===")
    if not os.path.isdir(IMAGES_TR):
        log.error(f"imagesTr não encontrado: {IMAGES_TR}.")
        return
    if not folds:
        log.error("Folds não carregados — OOF cancelado.")
        return

    for arch in active_archs:
        log.info(f"  Arquitetura: {arch['name']}")
        for fold_idx, fold_data in folds.items():
            val_cases = fold_data["val"]

            fold_input = os.path.join(
                PRED_DIR, "bd1_oof", arch["name"], f"_input_fold_{fold_idx}"
            )
            os.makedirs(fold_input, exist_ok=True)
            for pid in val_cases:
                src = os.path.join(IMAGES_TR, f"{pid}_0000.nii.gz")
                dst = os.path.join(fold_input, f"{pid}_0000.nii.gz")
                if os.path.isfile(src) and not os.path.exists(dst):
                    try:
                        os.symlink(src, dst)
                    except OSError:
                        shutil.copy2(src, dst)

            fold_out = os.path.join(
                PRED_DIR, "bd1_oof", arch["name"], f"fold_{fold_idx}"
            )
            prev_stage_dir = None
            if arch.get("prev_stage_name"):
                prev_stage_dir = os.path.join(
                    PRED_DIR, "bd1_oof", arch["prev_stage_name"], f"fold_{fold_idx}"
                )
            run_predict(arch, fold_input, fold_out, folds=[fold_idx],
                        prev_stage_dir=prev_stage_dir)


def infer_bd1_internal(active_archs):
    """Prediz o subconjunto interno com ensemble de todos os 5 folds."""
    log.info("=== BD1 ensemble (avaliação interna) ===")
    if not os.path.isdir(IMAGES_TS):
        log.error(f"imagesTs não encontrado: {IMAGES_TS}. Execute script 08 primeiro.")
        return
    for arch in active_archs:
        out_dir = os.path.join(PRED_DIR, "bd1_internal", arch["name"])
        prev_stage_dir = None
        if arch.get("prev_stage_name"):
            prev_stage_dir = os.path.join(PRED_DIR, "bd1_internal", arch["prev_stage_name"])
        run_predict(arch, IMAGES_TS, out_dir, folds=list(range(5)),
                    prev_stage_dir=prev_stage_dir)


def infer_rider(active_archs):
    """Todas as arquiteturas não degeneradas; CT1 e CT2 juntos (sufixos _1/_2 nos nomes)."""
    log.info("=== RIDER — todas as arquiteturas ===")
    rider_images = os.path.join(BASE, "data", "processed", "rider", "images")
    if not os.path.isdir(rider_images):
        log.warning(f"RIDER processado ausente: {rider_images}")
        return

    flat_dir = os.path.join(PRED_DIR, "rider", "_input")
    all_pids = _make_flat_dir(rider_images, flat_dir)
    log.info(f"  {len(all_pids)} casos RIDER (CT1 + CT2)")

    for arch in active_archs:
        out_dir = os.path.join(PRED_DIR, "rider", arch["name"])
        prev_stage_dir = None
        if arch.get("prev_stage_name"):
            prev_stage_dir = os.path.join(PRED_DIR, "rider", arch["prev_stage_name"])
        run_predict(arch, flat_dir, out_dir, folds=list(range(5)),
                    prev_stage_dir=prev_stage_dir)


def infer_lndb():
    """3D_F e 3D_FREM pré-especificados (independentemente de degeneração)."""
    log.info("=== LNDb — 3D_F e 3D_FREM (pré-especificados) ===")
    lndb_images = os.path.join(BASE, "data", "processed", "lndb", "images")
    if not os.path.isdir(lndb_images):
        log.warning(f"LNDb processado ausente: {lndb_images}")
        return

    flat_dir = os.path.join(PRED_DIR, "lndb", "_input")
    _make_flat_dir(lndb_images, flat_dir)

    lndb_archs = [a for a in ARCHITECTURES if a["name"] in LNDB_ARCH_NAMES]
    for arch in lndb_archs:
        out_dir = os.path.join(PRED_DIR, "lndb", arch["name"])
        run_predict(arch, flat_dir, out_dir, folds=list(range(5)))


def save_manifest():
    if not manifest_rows:
        log.info("Nenhuma inferência nova registrada.")
        return
    out = os.path.join(ANALYSIS_DIR, "inference_manifest.csv")
    fieldnames = ["architecture", "config", "plans_id", "folds",
                  "input", "output", "status", "elapsed_s", "timestamp"]
    mode = "a" if os.path.isfile(out) else "w"
    with open(out, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if mode == "w":
            w.writeheader()
        w.writerows(manifest_rows)

    ok   = sum(1 for r in manifest_rows if r["status"] == "ok")
    skip = sum(1 for r in manifest_rows if r["status"] == "skipped")
    err  = sum(1 for r in manifest_rows if r["status"].startswith("error"))
    log.info(f"inference_manifest.csv: {len(manifest_rows)} runs "
             f"(ok={ok} skip={skip} erro={err})")


def main():
    degenerate   = load_degenerate_architectures()
    active_archs = [a for a in ARCHITECTURES if a["name"] not in degenerate]
    folds        = load_folds()

    setup_bd1_raw_dirs()
    infer_bd1_oof(active_archs, folds)
    infer_bd1_internal(active_archs)
    infer_rider(active_archs)
    infer_lndb()
    save_manifest()
    log.info("Script 10 concluído.")


if __name__ == "__main__":
    main()
