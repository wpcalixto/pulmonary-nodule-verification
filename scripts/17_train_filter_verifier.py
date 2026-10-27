#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    17_train_filter_verifier.py

File type:
    Script

Purpose:
    Trains the morphological post-detection verifier on out-of-fold predictions and freezes its decision threshold.

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
    - filter_verifier_threshold.json
    - filter_verifier_manifest.json

Dependencies:
    - numpy
    - pandas
    - sklearn

Execution:
    PULMONARY_BASE=<project_dir> python scripts/17_train_filter_verifier.py

Portability notes:
    All paths are resolved relative to the project base directory given by the PULMONARY_BASE environment variable. No machine-specific path is required.

Reproducibility notes:
    Environment pinned in requirements.txt and environment/. Seeds and frozen parameters are recorded in the run manifests under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) are not included and must be obtained from their original repositories.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pulmonary_base import BASE

MODELS_DIR   = os.path.join(BASE, "models")
CONFIG_DIR   = os.path.join(BASE, "config")
ANALYSIS_DIR = os.path.join(BASE, "analysis")
LOGS_DIR     = os.path.join(BASE, "logs")

SEED        = 27
N_FOLDS_CV  = 5
N_BOOTSTRAP = 2000
ELIGIBILITY_AUROC = 0.60

MODEL_A_FEATURES = [
    "log_max", "log_mean", "log_p95", "log_scale_mm",
    "blob_max", "blob_mean", "blob_p95", "blob_scale_mm",
    "vessel_max", "vessel_mean", "vessel_p95", "vessel_scale_mm",
    "contrast_hu", "centroid_peak_dist_mm",
    "pleura_min_dist_mm", "pleura_centroid_dist_mm",
]
SIZE_FEATURE = "size_log1p_diam_mm"
FORBIDDEN = {"frac_high_log", "volume_mm3"}

LOGIT = dict(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000,
             class_weight=None, random_state=SEED)

EXPECTED = {"total_rows": 72430, "patients": 718, "TP_pred": 14673,
            "FP_pred": 23314, "rows": 37987}

for _d in (MODELS_DIR, CONFIG_DIR, ANALYSIS_DIR, LOGS_DIR):
    os.makedirs(_d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(os.path.join(LOGS_DIR, "train_filter_verifier.log")),
              logging.StreamHandler()],
)
log = logging.getLogger(__name__)


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


def build_model():
    return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(**LOGIT))])


def youden_oof(y, proba):
    """Limiar de Youden nas probabilidades OOF; empate → menor limiar finito."""
    fpr, tpr, thr = roc_curve(y, proba)
    J = tpr - fpr
    jmax = J.max()
    cand = thr[(J >= jmax - 1e-12) & np.isfinite(thr)]
    if len(cand) == 0:
        fail_output("nenhum limiar finito no máximo de J")
    t = float(np.min(cand))
    if not np.isfinite(t):
        fail_output("limiar não finito")
    return t, float(jmax)


def metrics_at(y, proba, t):
    pred = (proba >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * prec * sens / (prec + sens) if prec and sens and (prec + sens) else np.nan
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "sensitivity": sens,
            "specificity": spec, "precision": prec, "f1": f1}


def cv_oof(df, feats, groups, y, folds):
    """Probabilidades OOF e métricas por fold, com os MESMOS folds para A e B."""
    X = df[feats].to_numpy(float)
    oof = np.full(len(y), np.nan)
    rows = []
    for k, (tr, va) in enumerate(folds):
        if len(set(groups[tr]) & set(groups[va])):
            fail_output(f"fold {k}: paciente em treino e validação")
        if y[tr].sum() == 0 or (1 - y[tr]).sum() == 0 or y[va].sum() == 0 or (1 - y[va]).sum() == 0:
            fail_output(f"fold {k}: classe ausente")
        m = build_model()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m.fit(X[tr], y[tr])
            if any("converg" in str(x.message).lower() for x in w):
                fail_output(f"fold {k}: LogisticRegression não convergiu")
        p = m.predict_proba(X[va])[:, 1]
        oof[va] = p
        rows.append({"fold": k, "n_train_patients": len(set(groups[tr])),
                     "n_valid_patients": len(set(groups[va])),
                     "TP_train": int(y[tr].sum()), "FP_train": int((1 - y[tr]).sum()),
                     "TP_valid": int(y[va].sum()), "FP_valid": int((1 - y[va]).sum()),
                     "auroc_fold": float(roc_auc_score(y[va], p)),
                     "average_precision_fold": float(average_precision_score(y[va], p))})
    if np.isnan(oof).any():
        fail_output("probabilidade OOF em falta")
    return oof, rows


def boot_ci(y, pa, pb, groups, n_boot=N_BOOTSTRAP, seed=SEED):
    """IC por paciente para AUROC/AP de A e B e para as diferenças B−A (pareado)."""
    idx_by = {}
    for i, g in enumerate(groups):
        idx_by.setdefault(g, []).append(i)
    pids = np.array(list(idx_by)); rng = np.random.default_rng(seed)
    acc = {k: [] for k in ("auroc_A", "ap_A", "auroc_B", "ap_B", "d_auroc", "d_ap")}
    for _ in range(n_boot):
        sel = np.concatenate([idx_by[p] for p in rng.choice(pids, len(pids), replace=True)])
        yy = y[sel]
        if yy.sum() < 2 or (1 - yy).sum() < 2:
            continue
        a1 = roc_auc_score(yy, pa[sel]); a2 = roc_auc_score(yy, pb[sel])
        p1 = average_precision_score(yy, pa[sel]); p2 = average_precision_score(yy, pb[sel])
        acc["auroc_A"].append(a1); acc["auroc_B"].append(a2)
        acc["ap_A"].append(p1); acc["ap_B"].append(p2)
        acc["d_auroc"].append(a2 - a1); acc["d_ap"].append(p2 - p1)
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return {k: {"ci_lo": q(v)[0], "ci_hi": q(v)[1]} for k, v in acc.items() if v}, len(acc["auroc_A"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--features-sha256", required=True)
    ap.add_argument("--script16-manifest", required=True)
    ap.add_argument("--source-run", default="")
    ap.add_argument("--out-dir", default=None, help="ENSAIO: escrever fora de models/config/analysis")
    ap.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    args = ap.parse_args()
    t0 = datetime.datetime.now().astimezone()
    mdir = args.out_dir or MODELS_DIR
    cdir = args.out_dir or CONFIG_DIR
    adir = args.out_dir or ANALYSIS_DIR
    for d in (mdir, cdir, adir):
        os.makedirs(d, exist_ok=True)

    if not os.path.isfile(args.features):
        fail_input(f"entrada ausente: {args.features}")
    h = sha256(args.features)
    if h != args.features_sha256:
        fail_input(f"SOURCE_AUTHORITY: SHA {h[:12]} != declarado {args.features_sha256[:12]}")
    full = pd.read_csv(args.features)
    if len(full) != EXPECTED["total_rows"]:
        fail_input(f"entrada com {len(full)} linhas != {EXPECTED['total_rows']}")

    if not os.path.isfile(args.script16_manifest):
        fail_input(f"manifesto do Script 16 ausente: {args.script16_manifest}")
    m16 = json.load(open(args.script16_manifest))
    s16_dir = os.path.dirname(os.path.abspath(args.script16_manifest))
    cand = [os.path.join(s16_dir, "..", "figdata", "filter_bd1_size_strata.csv"),
            os.path.join(s16_dir, "filter_bd1_size_strata.csv"),
            os.path.join(BASE, "figdata", "filter_bd1_size_strata.csv")]
    s16 = next((p for p in cand if os.path.isfile(p)), None)
    if s16 is None:
        fail_input("filter_bd1_size_strata.csv do Script 16 não encontrado")
    z = pd.read_csv(s16)
    za = z[(z["comparison"] == "TP_pred_vs_FP_pred") & (z["status"] == "OK")]
    best = float(za["auroc"].max())
    if not (best > ELIGIBILITY_AUROC):
        fail_input(f"elegibilidade não satisfeita: AUROC máx {best:.3f} <= {ELIGIBILITY_AUROC}")
    log.info(f"ELIGIBILITY: Script 16 canónico, AUROC máx estratificado em A = {best:.3f} "
             f"(> {ELIGIBILITY_AUROC}); fonte {os.path.basename(s16)}")

    df = full[(full["subset"] == "oof") & (full["group"].isin(["TP_pred", "FP_pred"]))].copy()
    n_tp = int((df["group"] == "TP_pred").sum()); n_fp = int((df["group"] == "FP_pred").sum())
    n_pat = int(df["patient_id"].nunique())
    log.info(f"OOF/predição: {len(df)} objectos, {n_pat} pacientes (TP_pred={n_tp}, FP_pred={n_fp})")
    if (len(df), n_pat, n_tp, n_fp) != (EXPECTED["rows"], EXPECTED["patients"],
                                        EXPECTED["TP_pred"], EXPECTED["FP_pred"]):
        fail_input(f"população {len(df)}/{n_pat}/{n_tp}/{n_fp} != congelada")
    if (full["subset"] == "internal").sum() and (df["subset"] == "internal").any():
        fail_input("INTERNAL entrou na matriz de treino")

    falta = [f for f in MODEL_A_FEATURES if f not in df.columns]
    if falta:
        fail_input(f"características ausentes: {falta}")
    usados = set(MODEL_A_FEATURES) & FORBIDDEN
    if usados:
        fail_input(f"característica proibida na lista: {usados}")
    if df["diam_mm"].isna().any() or (df["diam_mm"] <= 0).any():
        fail_input("diam_mm inválido")
    df[SIZE_FEATURE] = np.log1p(df["diam_mm"].astype(float))
    feats_b = MODEL_A_FEATURES + [SIZE_FEATURE]
    for f in feats_b:
        v = df[f].to_numpy(float)
        if not np.isfinite(v).all():
            fail_input(f"NaN/Inf em {f} (sem imputação autorizada)")

    y = (df["group"] == "TP_pred").astype(int).to_numpy()
    groups = df["patient_id"].to_numpy()
    folds = list(GroupKFold(n_splits=N_FOLDS_CV).split(df, y, groups))

    res, oof = {}, {}
    for label, feats in (("A", MODEL_A_FEATURES), ("B", feats_b)):
        log.info(f"=== MODEL_{label}: {len(feats)} características")
        o, fold_rows = cv_oof(df, feats, groups, y, folds)
        oof[label] = o
        auroc = float(roc_auc_score(y, o)); apv = float(average_precision_score(y, o))
        thr, jmax = youden_oof(y, o)
        met = metrics_at(y, o, thr)
        fp_per_exam = met["fp"] / n_pat
        log.info(f"  OOF AUROC={auroc:.4f} AP={apv:.4f} | limiar YOUDEN_OOF={thr:.4f} "
                 f"(J={jmax:.4f}) sens={met['sensitivity']:.3f} prec={met['precision']:.3f} "
                 f"FP/exame={fp_per_exam:.2f}")
        res[label] = {"features": feats, "auroc_oof": auroc, "average_precision_oof": apv,
                      "threshold": thr, "threshold_rule": "YOUDEN_OOF",
                      "tie_break": "LOWEST_THRESHOLD_AT_MAX_J", "youden_J": jmax,
                      "metrics_at_threshold": {**met, "fp_per_exam": fp_per_exam,
                                               "label": "DEVELOPMENT_THRESHOLD_TUNING_METRIC"},
                      "folds": fold_rows}

    ci, used = boot_ci(y, oof["A"], oof["B"], groups, n_boot=args.n_boot)
    log.info(f"Bootstrap por paciente ({used}/{args.n_boot}): "
             f"AUROC A {ci['auroc_A']['ci_lo']:.4f}–{ci['auroc_A']['ci_hi']:.4f}; "
             f"ΔAUROC B−A {ci['d_auroc']['ci_lo']:+.4f}–{ci['d_auroc']['ci_hi']:+.4f}")

    paths = {}
    for label, feats in (("A", MODEL_A_FEATURES), ("B", feats_b)):
        m = build_model()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m.fit(df[feats].to_numpy(float), y)
            if any("converg" in str(x.message).lower() for x in w):
                fail_output(f"MODEL_{label} final não convergiu")
        fx = df[feats].to_numpy(float)[:20]
        pre = m.predict_proba(fx)[:, 1]
        name = "filter_verifier_A.pkl" if label == "A" else "filter_verifier_B_plus_size.pkl"
        p = os.path.join(mdir, name)
        with open(p, "wb") as f:
            pickle.dump({"model": m, "features": feats, "model_label": label,
                         "threshold": res[label]["threshold"],
                         "threshold_rule": "YOUDEN_OOF", "tie_break": "LOWEST_THRESHOLD_AT_MAX_J",
                         "canonical": label == "A", "sklearn_version": sklearn.__version__,
                         "python_version": sys.version.split()[0], "seed": SEED,
                         "logistic_params": LOGIT}, f)
        with open(p, "rb") as f:
            back = pickle.load(f)
        post = back["model"].predict_proba(fx)[:, 1]
        if not np.allclose(pre, post, rtol=0, atol=0):
            fail_output(f"MODEL_{label}: predição difere após serialização")
        res[label]["model_path"] = os.path.relpath(p, BASE)
        res[label]["model_sha256"] = sha256(p)
        res[label]["coefficients"] = dict(zip(feats, m.named_steps["clf"].coef_[0].round(6).tolist()))
        res[label]["intercept"] = float(m.named_steps["clf"].intercept_[0])
        paths[label] = p
    alias = os.path.join(mdir, "filter_verifier.pkl")
    with open(paths["A"], "rb") as s, open(alias, "wb") as d:
        d.write(s.read())

    thr_json = {
        "trained": True, "canonical_model": "A",
        "canonical_reason": "MODEL_A é o verificador; a escolha NÃO depende do AUROC observado. "
                            "MODEL_B é comparador pré-especificado que quantifica o ganho do tamanho.",
        "model_A": {k: res["A"][k] for k in ("features", "threshold", "threshold_rule", "tie_break",
                                             "auroc_oof", "average_precision_oof", "model_path",
                                             "model_sha256")},
        "model_B": {k: res["B"][k] for k in ("features", "threshold", "threshold_rule", "tie_break",
                                             "auroc_oof", "average_precision_oof", "model_path",
                                             "model_sha256")},
        "seed": SEED, "n_dev_patients": n_pat, "n_TP_pred": n_tp, "n_FP_pred": n_fp,
        "threshold_population": "OOF only; INTERNAL/LNDb/RIDER never used",
    }
    json.dump(thr_json, open(os.path.join(cdir, "filter_verifier_threshold.json"), "w"), indent=2)

    cv_rows = []
    for label in ("A", "B"):
        for r in res[label]["folds"]:
            cv_rows.append({"model": label, **r})
        cv_rows.append({"model": label, "fold": "pooled_oof",
                        "auroc_fold": res[label]["auroc_oof"],
                        "average_precision_fold": res[label]["average_precision_oof"],
                        "threshold": res[label]["threshold"],
                        **{k: v for k, v in res[label]["metrics_at_threshold"].items() if k != "label"}})
    cv_path = os.path.join(adir, "filter_verifier_cv.csv")
    pd.DataFrame(cv_rows).to_csv(cv_path, index=False)

    man = {
        "script": "17_train_filter_verifier.py", "script_sha256": sha256(os.path.abspath(__file__)),
        "run_at": t0.isoformat(), "end_at": datetime.datetime.now().astimezone().isoformat(),
        "source": {"features": args.features, "features_sha256": h,
                   "script15_run": args.source_run, "script16_manifest": args.script16_manifest,
                   "eligibility_source": "SCRIPT16_CANONICAL", "eligibility_best_auroc": best},
        "population": {"subset": "oof", "groups": ["TP_pred", "FP_pred"], "rows": len(df),
                       "patients": n_pat, "TP_pred": n_tp, "FP_pred": n_fp,
                       "internal_rows_in_training": 0},
        "models": {"canonical": "A", "A": "16 morphological/anatomical features",
                   "B": "A + log1p(diam_mm)", "volume_mm3": "not used"},
        "excluded_features": {"frac_high_log": "EXCLUDED_DEGENERATE_BY_CONSTRUCTION",
                              "volume_mm3": "redundant with diameter; not used"},
        "logistic": {**{k: str(v) for k, v in LOGIT.items()}, "sklearn_version": sklearn.__version__,
                     "python_version": sys.version.split()[0]},
        "cv": {"scheme": "GroupKFold by patient_id", "n_splits": N_FOLDS_CV,
               "same_folds_for_A_and_B": True, "scaler": "fit inside each fold (Pipeline)"},
        "threshold": {"rule": "YOUDEN_OOF", "tie_break": "LOWEST_THRESHOLD_AT_MAX_J",
                      "computed_on": "out-of-fold probabilities only",
                      "metrics_label": "DEVELOPMENT_THRESHOLD_TUNING_METRIC"},
        "bootstrap": {"unit": "patient_id", "n": args.n_boot, "seed": SEED,
                      "replicates_used": used, "paired_B_minus_A": True, "ci": ci},
        "results": {k: {kk: res[k][kk] for kk in ("auroc_oof", "average_precision_oof", "threshold",
                                                  "youden_J", "metrics_at_threshold", "coefficients",
                                                  "intercept", "model_path", "model_sha256")}
                    for k in ("A", "B")},
        "outputs_sha256": {os.path.basename(p): sha256(p) for p in
                           [paths["A"], paths["B"], alias, cv_path,
                            os.path.join(cdir, "filter_verifier_threshold.json")]},
    }
    json.dump(man, open(os.path.join(adir, "filter_verifier_manifest.json"), "w"), indent=2)
    log.info(f"Modelos: {paths['A']}, {paths['B']} (alias canónico {alias})")
    log.info("Script 17 concluído.")


if __name__ == "__main__":
    main()
