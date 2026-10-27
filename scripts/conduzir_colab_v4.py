#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    conduzir_colab_v4.py

File type:
    Script

Purpose:
    Fold conductor for Google Colab: runs the folds of one architecture in order, advancing only after verifying the complete artefact list.

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
    - {PRE}/{DS}/splits_final.json

Outputs:
    - See docs/file_level_documentation.md.

Dependencies:
    - torch

Execution:
    PULMONARY_BASE=<project_dir> python scripts/conduzir_colab_v4.py

Portability notes:
    All paths are resolved relative to the project base directory given by the PULMONARY_BASE environment variable. No machine-specific path is required.

Reproducibility notes:
    Environment pinned in requirements.txt and environment/. Seeds and frozen parameters are recorded in the run manifests under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) are not included and must be obtained from their original repositories.
"""

import json, glob, os, re, shutil, subprocess, sys, time
import torch

ARQ   = os.environ.get("PULMONAR_ARQ", "").strip()
if not ARQ:
    sys.exit("PULMONAR_ARQ nao definido — ex.: PULMONAR_ARQ=3D_F")
FOLDS = [int(x) for x in os.environ.get("PULMONAR_FOLDS", "0,1,2,3,4").split(",")]

L, PY = "/content/pulmonar", sys.executable
ORD, PRE, DS = f"{L}/ordens", f"{L}/nnUNet_preprocessed", "Dataset001_PulmonaryNodules"
VELHO, NOVO, SUSP = f"{L}/resultados", f"{L}/resultados_limpo", f"{L}/suspeitos"
RAW = f"{L}/nnUNet_raw"
MAU = ("AUSENTE", "NAO CARREGA", "SUSPEITA", "SEM DICE", "CASOS A MENOS", "ILEGIVEL")
os.makedirs(RAW, exist_ok=True); os.makedirs(SUSP, exist_ok=True)

def log(*a): print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)

ordem0 = f"{ORD}/ordem_{ARQ}_fold{FOLDS[0]}.json"
if not os.path.exists(ordem0):
    sys.exit("ordem em falta: %s" % ordem0)
arq = json.load(open(ordem0))["architecture"]
SUB = "%s__%s__%s" % (arq["trainer"], arq["plans_id"], arq["config"])
splits = json.load(open(f"{PRE}/{DS}/splits_final.json"))
log("arquitectura %s · pasta %s · folds %s" % (ARQ, SUB, FOLDS))

def pasta(raiz, f): return f"{raiz}/{DS}/{SUB}/fold_{f}"

def carrega(p):
    try:
        c = torch.load(p, map_location="cpu", weights_only=False)
        return True, c.get("current_epoch"), round(float(c.get("_best_ema") or 0), 4)
    except Exception as e:
        return False, str(e)[:60], None

def dice_do_summary(p, N):
    if not os.path.exists(p): return "AUSENTE", ""
    try:
        d = json.load(open(p))
        fm = d.get("foreground_mean") or {}
        if "Dice" not in fm: return "SEM DICE", str(list(fm.keys()))[:50]
        n = len(d.get("metric_per_case") or [])
        if n != N: return "CASOS A MENOS", "%d de %d" % (n, N)
        v = float(fm["Dice"])
        return "ok", "Dice %.4f · %d casos%s" % (v, n, "  <<< DEGENERADO" if v < 1e-6 else "")
    except Exception as e:
        return "ILEGIVEL", str(e)[:50]

def inspeccionar(d, f):
    N, rel, ok = len(splits[f]["val"]), [], set()
    if not os.path.isdir(d): return [("(pasta)", "AUSENTE", d)], ok

    for nome, obrig in (("checkpoint_final.pth", True), ("checkpoint_best.pth", True),
                        ("checkpoint_latest.pth", False)):
        p = f"{d}/{nome}"
        if not os.path.exists(p):
            rel.append((nome, "AUSENTE" if obrig else "ausente", "")); continue
        bom, ep, ema = carrega(p)
        rel.append((nome, "ok" if bom else "NAO CARREGA",
                    "epoca %s · EMA %s" % (ep, ema) if bom else str(ep)))
        if bom: ok.add(nome)

    logs = sorted(glob.glob(f"{d}/training_log_*.txt"))
    for Lg in logs:
        eps = [int(m.group(1)) for m in
               re.finditer(r": Epoch (\d+)\s*$", open(Lg, errors="ignore").read(), re.M)]
        rel.append((os.path.basename(Lg), "ok",
                    "epocas %s-%s" % (min(eps) if eps else "?", max(eps) if eps else "?")))
        ok.add(os.path.basename(Lg))
    if not logs: rel.append(("training_log_*.txt", "AUSENTE", ""))

    for nome in ("progress.png", "debug.json", "estado_fold.json"):
        if os.path.exists(f"{d}/{nome}"): rel.append((nome, "ok", "")); ok.add(nome)
        else: rel.append((nome, "AUSENTE", ""))

    preds = glob.glob(f"{d}/validation/*.nii.gz")
    cf = f"{d}/checkpoint_final.pth"
    t0 = os.path.getmtime(cf) if os.path.exists(cf) else 0
    if not preds:
        rel.append(("validation/", "AUSENTE", "esperadas %d" % N))
    else:
        velhas = [p for p in preds if os.path.getmtime(p) < t0]
        est = "ok" if (len(preds) == N and not velhas) else "SUSPEITA"
        rel.append(("validation/", est, "%d de %d · %d anteriores ao checkpoint"
                    % (len(preds), N, len(velhas))))
        if est == "ok": ok.add("validation")

    est, det = dice_do_summary(f"{d}/validation/summary.json", N)
    rel.append(("summary.json", est, det))
    if est == "ok": ok.add("summary.json")
    return rel, ok

def mostrar(rel, titulo):
    log("---", titulo)
    for nome, est, det in rel:
        m = "  " if est in ("ok", "ausente") else ">>"
        print("   %s %-32s %-14s %s" % (m, nome, est, det), flush=True)

def completo(d, f):
    rel, _ = inspeccionar(d, f)
    return (not [n for n, e, _ in rel if e in MAU]), [n for n, e, _ in rel if e in MAU], rel

def migrar(f):
    dv, dn = pasta(VELHO, f), pasta(NOVO, f)
    if not os.path.isdir(dv):
        log("fold %d — nao ha pasta antiga, nada a migrar" % f); return
    rel, ok = inspeccionar(dv, f)
    mostrar(rel, "fold %d — o que existe na pasta antiga" % f)
    os.makedirs(dn, exist_ok=True)
    for nome in sorted(ok):
        if nome in ("validation", "summary.json"): continue
        s = f"{dv}/{nome}"
        if os.path.isfile(s) and not os.path.exists(f"{dn}/{nome}"):
            shutil.copy2(s, f"{dn}/{nome}"); log("   migrado:", nome)
    s = f"{dv}/checkpoint_latest.pth"
    if os.path.exists(s) and not os.path.exists(f"{dv}/checkpoint_final.pth") \
       and not os.path.exists(f"{dn}/checkpoint_latest.pth"):
        shutil.copy2(s, f"{dn}/checkpoint_latest.pth")
        log("   migrado: checkpoint_latest.pth (para retomar)")
    if "validation" in ok and "summary.json" in ok:
        if not os.path.isdir(f"{dn}/validation"):
            shutil.copytree(f"{dv}/validation", f"{dn}/validation")
            log("   migrada: validation/ (passou na verificacao)")
    elif os.path.isdir(f"{dv}/validation"):
        alvo = f"{SUSP}/fold_{f}_validation_{time.strftime('%H%M%S')}"
        shutil.copytree(f"{dv}/validation", alvo)
        log("   validacao NAO passou — copia guardada em", alvo)
    for j in ("plans.json", "dataset.json", "dataset_fingerprint.json"):
        s = f"{VELHO}/{DS}/{SUB}/{j}"
        if os.path.exists(s):
            os.makedirs(f"{NOVO}/{DS}/{SUB}", exist_ok=True)
            if not os.path.exists(f"{NOVO}/{DS}/{SUB}/{j}"):
                shutil.copy2(s, f"{NOVO}/{DS}/{SUB}/{j}")

def arrumar(f):
    d = pasta(NOVO, f)
    p = f"{d}/checkpoint_latest.pth"
    if os.path.exists(f"{d}/checkpoint_final.pth") and os.path.exists(p):
        os.remove(p); log("   arrumado: checkpoint_latest.pth removido (fold terminado)")

def regenerar_validacao(f):
    """run_training(only_run_validation=True) em subprocesso: o ambiente tem de
    estar posto ANTES de importar nnunetv2. Nome completo do dataset em vez do
    id `001` — a resolucao do id varre o nnUNet_raw, que aqui esta vazio."""
    codigo = ("import os\n"
              f"os.environ['nnUNet_preprocessed']={PRE!r}\n"
              f"os.environ['nnUNet_results']={NOVO!r}\n"
              f"os.environ['nnUNet_raw']={RAW!r}\n"
              "from nnunetv2.run.run_training import run_training\n"
              f"run_training({DS!r}, {arq['config']!r}, {f}, {arq['trainer']!r}, "
              f"{arq['plans_id']!r}, only_run_validation=True)\n")
    log("fold %d — a regenerar validacao (so predicao, sem treinar)" % f)
    r = subprocess.run([PY, "-u", "-c", codigo])
    log("   rc =", r.returncode)
    return r.returncode == 0

def treinar(f):
    d = pasta(NOVO, f)
    cont = ["--continuar"] if os.path.exists(f"{d}/checkpoint_latest.pth") else []
    log("fold %d — a treinar%s" % (f, " (retoma)" if cont else " (do zero)"))
    r = subprocess.run([PY, "-u", f"{L}/Scripts/novos/09_train_nnunet.py",
                        "--ordem", f"{ORD}/ordem_{ARQ}_fold{f}.json",
                        "--preprocessed", PRE, "--resultados", NOVO, "--python", PY] + cont,
                       env=dict(os.environ, PULMONARY_BASE=f"{L}/Pulmonary_Project",
                                PYTHONUNBUFFERED="1"))
    log("   rc =", r.returncode)
    return r.returncode == 0

for f in FOLDS:
    log("=========== %s FOLD %d ===========" % (ARQ, f))
    if not os.path.exists(f"{ORD}/ordem_{ARQ}_fold{f}.json"):
        log("*** ordem em falta para o fold %d — PARO ***" % f); sys.exit(1)
    migrar(f)
    d = pasta(NOVO, f)
    bom, falta, rel = completo(d, f)
    if bom:
        log("fold %d ja completo — nada a fazer" % f)
    else:
        log("fold %d em falta: %s" % (f, ", ".join(falta)))
        if set(falta) <= {"validation/", "summary.json"} and os.path.exists(f"{d}/checkpoint_final.pth"):
            regenerar_validacao(f)
        else:
            treinar(f)
            if not os.path.isdir(f"{d}/validation"):
                regenerar_validacao(f)
    arrumar(f)
    bom, falta, rel = completo(d, f)
    mostrar(rel, "fold %d — verificacao final" % f)
    if not bom:
        log("*** FOLD %d NAO PASSOU. Em falta: %s" % (f, ", ".join(falta)))
        log("*** PARO AQUI. Nao avanco sobre verificacao falhada.")
        sys.exit(1)
    log("=== %s FOLD %d COMPLETO ===" % (ARQ, f))

log("=== %s: TODOS OS FOLDS COMPLETOS ===" % ARQ)
