#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    17a_apply_filter_verifier_internal.py

File type:
    Script

Purpose:
    Applies the frozen verifier to the internal-validation set without re-tuning.

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
    - SHA256SUMS.txt

Outputs:
    - internal_verifier_metrics.csv
    - gates.json
    - manifest.json

Dependencies:
    - numpy
    - pandas

Execution:
    PULMONARY_BASE=<project_dir> python scripts/17a_apply_filter_verifier_internal.py

Portability notes:
    All paths are resolved relative to the project base directory given by the PULMONARY_BASE environment variable. No machine-specific path is required.

Reproducibility notes:
    Environment pinned in requirements.txt and environment/. Seeds and frozen parameters are recorded in the run manifests under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) are not included and must be obtained from their original repositories.
"""

import argparse
import ast
import datetime
import hashlib
import json
import logging
import os
import shutil
import tempfile

import numpy as np
import pandas as pd

BASE = os.environ.get("PULMONARY_BASE") or os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
ANALYSIS = os.path.join(BASE, "analysis")
CONFIG = os.path.join(BASE, "config")
LOGS = os.path.join(BASE, "logs")

INTERNAL_N_PATIENTS = 158
N_GT_POR_ARQ = 412
LIMIAR_CONGELADO = 0.3417980765530837
MATCHING_IOU_PRIMARIO = 0.10
ARQUITECTURAS = ["2D", "2D_REL", "2D_REM", "3D_CF", "3D_CFREM", "3D_F", "3D_FREM",
                 "F100_ResEncM", "F100_fullres", "F100_lowres"]

os.makedirs(LOGS, exist_ok=True)
log = logging.getLogger("s17a")
log.setLevel(logging.INFO)
if not log.handlers:
    _f = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for _h in (logging.StreamHandler(),
               logging.FileHandler(os.path.join(LOGS, "apply_verifier_internal.log"))):
        _h.setFormatter(_f); log.addHandler(_h)
log.propagate = False


def falha(tipo, msg):
    log.error(f"{tipo}: {msg}")
    raise SystemExit(f"{tipo}: {msg}")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def auto_verificacao_de_ambito():
    """O que é proibido tem de ser proibido no código, não só na intenção.

    Procura por AST — não por texto — chamadas que violariam o âmbito congelado.
    Uma docstring que mencione «bootstrap» não é um bootstrap; uma chamada a
    `.fit()` é um ajuste, e reprova.
    """
    proibidas = {
        "fit", "fit_transform", "fit_predict", "partial_fit",
        "CalibratedClassifierCV", "IsotonicRegression", "calibration_curve",
        "roc_curve", "precision_recall_curve", "linspace", "arange", "logspace",
        "argmax", "argmin",
        "GridSearchCV", "RandomizedSearchCV", "cross_val_score",
        "cross_val_predict", "train_test_split", "KFold", "StratifiedKFold",
        "GroupKFold", "SelectKBest", "RFE",
        "resample", "bootstrap", "shuffle", "permutation", "default_rng",
        "RandomState", "wilcoxon", "mannwhitneyu", "ttest_ind", "ttest_rel",
        "roc_auc_score", "average_precision_score"}
    arvore = ast.parse(open(os.path.abspath(__file__), encoding="utf-8").read())
    achadas = []
    for n in ast.walk(arvore):
        if isinstance(n, ast.Call):
            f, partes = n.func, []
            while isinstance(f, ast.Attribute):
                partes.append(f.attr); f = f.value
            if isinstance(f, ast.Name):
                partes.append(f.id)
            if partes and partes[0] in proibidas:
                achadas.append(".".join(reversed(partes)))
    def parece_caminho(s):
        if not s or any(c.isspace() for c in s):
            return False
        return ("/" in s or s.endswith((".csv", ".json", ".parquet", ".pkl",
                                        ".nii.gz")))
    caminhos = [n.value for n in ast.walk(arvore)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
                and parece_caminho(n.value)]
    externos = [s for s in caminhos if "lndb" in s.lower()]
    return {"FORBIDDEN_CALLS_IN_CODE": achadas,
            "EXTERNAL_BANK_PATHS_IN_CODE": externos,
            "n_path_literals_inspected": len(caminhos),
            "SCOPE_SELF_CHECK": "PASS" if not achadas and not externos
                                else "FAIL_METHOD"}


def confirma_entradas_internas(args):
    """O âmbito também tem de valer no que foi realmente passado: um caminho de
    entrada que aponte para o banco externo reprova, por muito correcto que o
    código esteja."""
    entradas = {k: v for k, v in vars(args).items()
                if isinstance(v, str) and v and k != "out_root"}
    externos = {k: v for k, v in entradas.items() if "lndb" in v.lower()}
    if externos:
        falha("FAIL_METHOD", f"entradas apontam para o banco externo: {externos}")
    return {"NO_EXTERNAL_BANK_IN_RESOLVED_INPUTS": "PASS",
            "n_inputs_checked": len(entradas)}


def carrega_pacote(pkl, sha_esperado, man17_caminho, cfg_caminho):
    """Três autoridades independentes têm de concordar: o binário, o manifesto do
    Script 17 e a configuração publicada. Discordância é falha, não aviso."""
    import pickle
    if not os.path.isfile(pkl):
        falha("FAIL_INPUT", f"pacote do verificador ausente: {pkl}")
    sha_real = sha256(pkl)
    if sha_esperado and sha_real != sha_esperado:
        falha("FAIL_PROVENANCE",
              f"SHA do pacote {sha_real[:12]} != declarado {sha_esperado[:12]}")
    if not os.path.isfile(man17_caminho):
        falha("FAIL_INPUT", f"manifesto do Script 17 ausente: {man17_caminho}")
    man17 = json.load(open(man17_caminho))
    sha_man = (man17.get("outputs_sha256") or {}).get("filter_verifier_A.pkl")
    if not sha_man:
        falha("FAIL_SCHEMA",
              "`outputs_sha256.filter_verifier_A.pkl` ausente do manifesto do 17")
    if sha_man != sha_real:
        falha("FAIL_PROVENANCE",
              f"SHA do binário {sha_real[:12]} != manifesto do 17 {sha_man[:12]}")

    payload = pickle.load(open(pkl, "rb"))
    for k in ("model", "features", "threshold", "model_label"):
        if k not in payload:
            falha("FAIL_SCHEMA", f"payload sem a chave `{k}`")
    if payload["model_label"] != "A":
        falha("FAIL_PROVENANCE", f"modelo {payload['model_label']!r} != A (canónico)")
    if not hasattr(payload["model"], "predict_proba"):
        falha("FAIL_SCHEMA", "payload['model'] não expõe predict_proba")

    if not os.path.isfile(cfg_caminho):
        falha("FAIL_INPUT", f"configuração do limiar ausente: {cfg_caminho}")
    cfg = json.load(open(cfg_caminho))
    ma = cfg.get("model_A") or {}
    if "threshold" not in ma or "features" not in ma:
        falha("FAIL_SCHEMA", "`model_A.threshold`/`model_A.features` ausentes da config")

    r_pkl, r_cfg = repr(float(payload["threshold"])), repr(float(ma["threshold"]))
    r_frz = repr(LIMIAR_CONGELADO)
    thr17 = ((man17.get("results") or {}).get("A") or {}).get("threshold")
    if thr17 is None:
        falha("FAIL_SCHEMA", "`results.A.threshold` ausente do manifesto do 17")
    if not (r_pkl == r_cfg == r_frz == repr(float(thr17))):
        falha("FAIL_PROVENANCE",
              f"limiar divergente: pkl {r_pkl} · config {r_cfg} · "
              f"manifesto17 {repr(float(thr17))} · congelado {r_frz}")

    if list(payload["features"]) != list(ma["features"]):
        falha("FAIL_PROVENANCE", "ordem das variáveis difere entre pacote e config")
    n_esperado = payload["model"].named_steps["clf"].n_features_in_
    if len(payload["features"]) != n_esperado:
        falha("FAIL_SCHEMA",
              f"{len(payload['features'])} variáveis declaradas != "
              f"{n_esperado} esperadas pelo estimador")
    tem_nomes = getattr(payload["model"], "feature_names_in_", None)

    gates = {
        "PACKAGE_SHA_GATE": "PASS",
        "package_sha256": sha_real,
        "THRESHOLD_PARITY_GATE": "PASS",
        "threshold": r_pkl,
        "threshold_authorities_agreeing": ["pkl", "config", "script17_manifest",
                                           "frozen_literal"],
        "FEATURE_ORDER_GATE": "PASS",
        "n_features": len(payload["features"]),
        "feature_order_enforced_by": "POSITIONAL_FROM_PAYLOAD_FEATURES",
        "pipeline_exposes_feature_names_in": "NO" if tem_nomes is None else "YES",
        "feature_order_note": (
            "o Pipeline foi ajustado com ndarray e não guarda feature_names_in_; "
            "a ordem é imposta posicionalmente a partir de payload['features'] e "
            "conferida contra config/filter_verifier_threshold.json"),
        "sklearn_version_declared": payload.get("sklearn_version"),
        "sklearn_version_running": __import__("sklearn").__version__,
    }
    return payload, float(payload["threshold"]), man17, gates


def coorte_interna(split_csv):
    if not os.path.isfile(split_csv):
        falha("FAIL_INPUT", f"split ausente: {split_csv}")
    sp = pd.read_csv(split_csv)
    for c in ("patient_id", "split"):
        if c not in sp.columns:
            falha("FAIL_SCHEMA", f"coluna `{c}` ausente de {split_csv}")
    ids = set(sp.loc[sp.split.str.strip().str.lower() == "internal", "patient_id"])
    if len(ids) != INTERNAL_N_PATIENTS:
        falha("FAIL_VALUE", f"{len(ids)} internos != {INTERNAL_N_PATIENTS}")
    return ids


def liga_objectos(feat_csv, obj_csv, ids_internos):
    """OBJECT_LINKAGE_GATE.

    Cada objecto predito interno das características tem de ligar **um para um**
    a uma linha do Script 11B pela chave `(patient_id, architecture,
    component_id)`. Objecto que não ligue é falha explícita — nunca é descartado
    em silêncio, que é como um denominador encolhe sem ninguém notar.
    """
    if not os.path.isfile(feat_csv):
        falha("FAIL_INPUT", f"características ausentes: {feat_csv}")
    if not os.path.isfile(obj_csv):
        falha("FAIL_INPUT", f"emparelhamentos do Script 11B ausentes: {obj_csv}")

    F = pd.read_csv(feat_csv)
    obrig = {"patient_id", "architecture", "subset", "group", "component_id", "source"}
    if not obrig.issubset(F.columns):
        falha("FAIL_SCHEMA", f"colunas ausentes das características: "
                             f"{sorted(obrig - set(F.columns))}")
    F = F[F.subset == "internal"].copy()
    if len(F) == 0:
        falha("FAIL_EMPTY_SUPPORT", "nenhuma linha interna nas características")
    if set(F.patient_id) != ids_internos:
        falha("FAIL_VALUE",
              f"pacientes das características != split: "
              f"+{len(set(F.patient_id) - ids_internos)} "
              f"-{len(ids_internos - set(F.patient_id))}")

    pred = F[F.source == "pred"].copy()
    chave = ["patient_id", "architecture", "component_id"]
    if pred.duplicated(chave).any():
        falha("FAIL_VALUE",
              f"{int(pred.duplicated(chave).sum())} objectos preditos com chave repetida")

    O = pd.read_csv(obj_csv)
    obrig_o = {"subset", "patient_id", "model", "pred_component_id", "gt_component_id",
               "status"}
    if not obrig_o.issubset(O.columns):
        falha("FAIL_SCHEMA", f"colunas ausentes do Script 11B: "
                             f"{sorted(obrig_o - set(O.columns))}")
    O = O[(O.subset == "internal") & (O.status.isin(["TP", "FP"]))].copy()
    O = O.rename(columns={"model": "architecture",
                          "pred_component_id": "component_id"})
    for d in (pred, O):
        d["component_id"] = pd.to_numeric(d["component_id"], errors="coerce")
    if pred["component_id"].isna().any() or O["component_id"].isna().any():
        falha("FAIL_SCHEMA", "component_id não numérico em pelo menos um dos lados")
    if O.duplicated(chave).any():
        falha("FAIL_VALUE", "objectos preditos repetidos no Script 11B")

    j = pred.merge(O[chave + ["status"]], on=chave, how="left", indicator=True)
    orfaos = j[j._merge == "left_only"]
    if len(orfaos):
        falha("FAIL_PROVENANCE",
              f"{len(orfaos)} objectos preditos sem linha correspondente no "
              f"Script 11B: {orfaos[chave].head(3).to_dict('records')}")
    sem_caracteristica = O.merge(pred[chave], on=chave, how="left", indicator=True)
    faltam = sem_caracteristica[sem_caracteristica._merge == "left_only"]
    if len(faltam):
        falha("FAIL_INPUT",
              f"{len(faltam)} objectos do Script 11B sem características "
              f"extraídas: {faltam[chave].head(3).to_dict('records')}")

    esperado = j.status.map({"TP": "TP_pred", "FP": "FP_pred"})
    discordantes = int((j.group != esperado).sum())
    if discordantes:
        falha("FAIL_VALUE",
              f"{discordantes} objectos com `group` discordante do `status` do 11B")

    fn_por_arq = (F[F.group == "FN_gt"].groupby("architecture").size().to_dict())
    tp_gt_por_arq = (F[F.group == "TP_gt"].groupby("architecture").size().to_dict())
    gates = {
        "OBJECT_LINKAGE_GATE": "PASS",
        "n_internal_rows": int(len(F)),
        "n_predicted_objects": int(len(pred)),
        "n_reference_objects": int(len(F[F.source == "gt"])),
        "linkage_key": "(patient_id, architecture, component_id)",
        "orphans_in_features": 0, "missing_features_for_script11b_objects": 0,
        "label_disagreements_with_script11b": 0,
        "COHORT_GATE": "PASS",
        "n_internal_patients": int(F.patient_id.nunique()),
    }
    return pred, fn_por_arq, tp_gt_por_arq, gates


def aplica(pred, payload, thr, fn_por_arq):
    """Pontuar e decidir. A ordem das colunas vem do pacote, posicionalmente."""
    ordem = list(payload["features"])
    faltam = [c for c in ordem if c not in pred.columns]
    if faltam:
        falha("FAIL_SCHEMA", f"variáveis do verificador ausentes das características: {faltam}")
    X = pred[ordem].to_numpy(float)
    if not np.isfinite(X).all():
        n = int((~np.isfinite(X)).any(axis=1).sum())
        falha("FAIL_INPUT",
              f"{n} objectos com variável ausente ou não finita — a PM §27 exige "
              f"relatar todos os casos sem variável, nunca imputar")
    prob = payload["model"].predict_proba(X)[:, 1]
    if not np.isfinite(prob).all():
        falha("FAIL_OUTPUT", "probabilidade não finita do verificador")
    pred = pred.copy()
    pred["verifier_probability"] = prob
    pred["decision"] = np.where(prob >= thr, "ACCEPT", "REJECT")
    pred["status_original"] = pred.group.map({"TP_pred": "TP", "FP_pred": "FP"})
    pred["status_filtered"] = np.where(
        pred.decision == "ACCEPT", pred.status_original,
        np.where(pred.status_original == "TP", "FN", "REMOVED"))

    linhas = []
    for arch in ARQUITECTURAS:
        s = pred[pred.architecture == arch]
        fn0 = int(fn_por_arq.get(arch, 0))
        tp0 = int((s.group == "TP_pred").sum())
        fp0 = int((s.group == "FP_pred").sum())
        tpf = int(((s.group == "TP_pred") & (s.decision == "ACCEPT")).sum())
        fpf = int(((s.group == "FP_pred") & (s.decision == "ACCEPT")).sum())
        fnf = fn0 + (tp0 - tpf)
        if tp0 + fn0 != N_GT_POR_ARQ or tpf + fnf != N_GT_POR_ARQ:
            falha("FAIL_OUTPUT",
                  f"ARITHMETIC_GATE em {arch}: {tp0}+{fn0} e {tpf}+{fnf} != "
                  f"{N_GT_POR_ARQ}; o número de objectos de referência não pode "
                  f"mudar com a filtragem")
        if tpf > tp0 or fpf > fp0:
            falha("FAIL_OUTPUT", f"o verificador não pode aumentar TP nem FP em {arch}")
        for etiqueta, tp, fp, fn in (("original", tp0, fp0, fn0),
                                     ("filtered", tpf, fpf, fnf)):
            sens = tp / (tp + fn)
            prec = tp / (tp + fp) if (tp + fp) else float("nan")
            f1 = (2 * prec * sens / (prec + sens)) \
                if (np.isfinite(prec) and (prec + sens) > 0) else float("nan")
            linhas.append({
                "architecture": arch, "subset": "internal", "detector": etiqueta,
                "matching_iou": MATCHING_IOU_PRIMARIO,
                "tp": tp, "fp": fp, "fn": fn,
                "n_gt": N_GT_POR_ARQ, "n_pred": tp + fp,
                "n_exams": INTERNAL_N_PATIENTS,
                "sensitivity": sens, "precision": prec, "f1": f1,
                "fp_per_exam": fp / INTERNAL_N_PATIENTS,
                "verifier_threshold": thr if etiqueta == "filtered" else "",
                "threshold_source": ("SCRIPT17_DEVELOPMENT_OOF"
                                     if etiqueta == "filtered" else ""),
                "inference": "POINT_ESTIMATE_ONLY"})
    return pred, pd.DataFrame(linhas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.join(BASE, "figdata",
                                                       "bd1_filter_features.csv"))
    ap.add_argument("--verifier", default=os.path.join(BASE, "models",
                                                       "filter_verifier_A.pkl"))
    ap.add_argument("--verifier-sha256", default="")
    ap.add_argument("--script17-manifest",
                    default=os.path.join(ANALYSIS, "filter_verifier_manifest.json"))
    ap.add_argument("--threshold-config",
                    default=os.path.join(CONFIG, "filter_verifier_threshold.json"))
    ap.add_argument("--split", default=os.path.join(CONFIG, "bd1_split.csv"))
    ap.add_argument("--script11-matches", required=True,
                    help="bd1_object_matches.csv do run canónico do Script 11B")
    ap.add_argument("--out-root",
                    default=os.path.join(ANALYSIS, "internal_verifier_application"))
    args = ap.parse_args()
    t0 = datetime.datetime.now().astimezone()

    log.info("Script 17A — verificador congelado aplicado ao BD1 interno")
    ambito = auto_verificacao_de_ambito()
    if ambito["SCOPE_SELF_CHECK"] != "PASS":
        falha("FAIL_METHOD", f"auto-verificação de âmbito reprovou: {ambito}")
    ambito.update(confirma_entradas_internas(args))
    log.info(f"  âmbito: sem chamadas proibidas · "
             f"{ambito['n_path_literals_inspected']} literais de caminho "
             f"inspeccionados · {ambito['n_inputs_checked']} entradas resolvidas, "
             f"nenhuma do banco externo")

    payload, thr, man17, g_pkg = carrega_pacote(
        args.verifier, args.verifier_sha256, args.script17_manifest,
        args.threshold_config)
    log.info(f"  pacote: {g_pkg['n_features']} variáveis · limiar {g_pkg['threshold']} "
             f"· SHA {g_pkg['package_sha256'][:12]}")

    ids = coorte_interna(args.split)
    pred, fn_por_arq, tp_gt_por_arq, g_lig = liga_objectos(
        args.features, args.script11_matches, ids)
    log.info(f"  ligação: {g_lig['n_predicted_objects']} objectos preditos, "
             f"{g_lig['n_internal_patients']} pacientes, 0 órfãos")

    pred, met = aplica(pred, payload, thr, fn_por_arq)
    for _, x in met[met.detector == "filtered"].iterrows():
        o = met[(met.architecture == x.architecture) & (met.detector == "original")].iloc[0]
        log.info(f"  {x.architecture:13} sens {o.sensitivity:.3f}→{x.sensitivity:.3f} · "
                 f"FP/exame {o.fp_per_exam:.2f}→{x.fp_per_exam:.2f}")

    gates = {**g_pkg, **g_lig,
             "ARITHMETIC_GATE": "PASS",
             "arithmetic_identities": "TP+FN=N_GT e TP+FP=N_pred, antes e depois",
             "N_GT_INVARIANT_UNDER_FILTERING": "PASS",
             "SCOPE_SELF_CHECK": ambito["SCOPE_SELF_CHECK"],
             "NO_FIT": "PASS", "NO_CALIBRATION": "PASS",
             "NO_FEATURE_SELECTION": "PASS", "NO_THRESHOLD_SEARCH": "PASS",
             "NO_THRESHOLD_SWEEP": "PASS", "NO_LNDB_ACCESS": "PASS",
             "MISSING_VARIABLES_REPORTED": "PASS",
             "n_objects_with_missing_variables": 0,
             "INFERENCE": "POINT_ESTIMATES_ONLY",
             "CONFIDENCE_INTERVALS": "NOT_RUN",
             "HYPOTHESIS_TESTS": "NOT_RUN",
             "BOOTSTRAP": "NOT_RUN",
             "MULTIPLICITY_CORRECTION": "NOT_APPLICABLE",
             "INFERENCE_NOT_PRESPECIFIED": (
                 "No inferential analysis was performed because confidence "
                 "intervals, hypothesis tests, and resampling procedures were not "
                 "prespecified for the Script 17A internal application."),
             "SCRIPT17A_OVERALL_GATE": "PASS"}

    run_id = f"s17a_{t0:%Y%m%dT%H%M%S}_{os.getpid()}"
    destino = os.path.join(args.out_root, run_id)
    staging = tempfile.mkdtemp(prefix="s17a_staging_")
    try:
        out = os.path.join(staging, run_id)
        os.makedirs(out)
        colunas = ["patient_id", "architecture", "subset", "component_id", "group",
                   "status_original", "verifier_probability", "decision",
                   "status_filtered", "diam_mm", "volume_mm3"]
        pred[[c for c in colunas if c in pred.columns]].sort_values(
            ["architecture", "patient_id", "component_id"]).to_csv(
            os.path.join(out, "internal_verifier_object_decisions.csv"), index=False)
        met.to_csv(os.path.join(out, "internal_verifier_metrics.csv"), index=False)
        json.dump(gates, open(os.path.join(out, "gates.json"), "w"), indent=1)

        man = {"script": "17a_apply_filter_verifier_internal.py", "run_id": run_id,
               "script_sha256": sha256(os.path.abspath(__file__)),
               "run_at": t0.isoformat(),
               "end_at": datetime.datetime.now().astimezone().isoformat(),
               "objective": ("PM §27 — aplicar o verificador congelado ao BD1 "
                             "interno sem ajuste adicional"),
               "sources": {
                   "features": args.features,
                   "features_sha256": sha256(args.features),
                   "verifier": args.verifier,
                   "verifier_sha256": g_pkg["package_sha256"],
                   "script17_manifest_sha256": sha256(args.script17_manifest),
                   "threshold_config_sha256": sha256(args.threshold_config),
                   "split_sha256": sha256(args.split),
                   "script11b_matches": args.script11_matches,
                   "script11b_matches_sha256": sha256(args.script11_matches)},
               "definitions": {
                   "population": f"BD1 interno, {INTERNAL_N_PATIENTS} pacientes",
                   "matching": ("Script 11B, Húngaro consciente do limiar; sem novo "
                                f"emparelhamento; IoU primário {MATCHING_IOU_PRIMARIO}"),
                   "verifier": {"model": "A", "threshold": repr(thr),
                                "threshold_rule": "YOUDEN_OOF",
                                "THRESHOLD_SOURCE": "SCRIPT17_DEVELOPMENT_OOF",
                                "THRESHOLD_TUNING_ON_INTERNAL": "NO",
                                "RECALIBRATION_ON_INTERNAL": "NONE",
                                "features": list(payload["features"]),
                                "feature_order_enforced_by":
                                    "POSITIONAL_FROM_PAYLOAD_FEATURES"},
                   "filtering_rule": (
                       "objecto predito com probabilidade >= limiar é ACCEPT; "
                       "TP rejeitado passa a FN (o objecto de referência volta a "
                       "não emparelhado); FP rejeitado é REMOVED; FN original "
                       "não muda"),
                   "primary_metrics": ["sensitivity", "precision", "f1",
                                       "fp_per_exam"],
                   "audit_counts": ["tp", "fp", "fn", "n_gt", "n_pred", "n_exams"],
                   "no_additional_endpoints": "YES",
                   "null_denominator": ("precisão e F1 ficam NaN quando N_pred=0; "
                                        "uma configuração que não prediz nada não "
                                        "tem precisão zero, tem precisão indefinida"),
                   "inference": "POINT_ESTIMATES_ONLY"},
               "counts": {"predicted_objects": int(len(pred)),
                          "accepted": int((pred.decision == "ACCEPT").sum()),
                          "rejected": int((pred.decision == "REJECT").sum()),
                          "metric_rows": int(len(met)),
                          "architectures": len(ARQUITECTURAS)},
               "gates": gates,
               "scope_self_check": ambito,
               "promotion": "NOT PROMOTED — produtos ficam no directório do run"}
        json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=2,
                  default=str)

        saidas = {f: sha256(os.path.join(out, f)) for f in sorted(os.listdir(out))}
        pd.DataFrame([{"path": k, "sha256": v} for k, v in saidas.items()]).to_csv(
            os.path.join(out, "output_hashes.csv"), index=False)
        with open(os.path.join(out, "SHA256SUMS.txt"), "w") as f:
            for k, v in saidas.items():
                f.write(f"{v}  {k}\n")

        os.makedirs(args.out_root, exist_ok=True)
        if os.path.exists(destino):
            falha("FAIL_OUTPUT", f"o directório do run já existe: {destino}")
        shutil.move(out, destino)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    log.info(f"SCRIPT17A_OVERALL_GATE = PASS · saída: {destino}")
    log.info("Script 17A concluído.")


if __name__ == "__main__":
    main()
