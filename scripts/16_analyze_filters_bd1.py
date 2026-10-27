#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    16_analyze_filters_bd1.py

File type:
    Script

Purpose:
    Statistical analysis of the filter features in BD1 (TP versus FP predictions; TP versus FN references).

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
    - filter_bd1_analysis_manifest.json

Dependencies:
    - numpy
    - pandas
    - scipy
    - sklearn
    - statsmodels

Execution:
    PULMONARY_BASE=<project_dir> python scripts/16_analyze_filters_bd1.py

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
import warnings

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, average_precision_score

from pulmonary_base import BASE

FIGDATA_DIR  = os.path.join(BASE, "figdata")
ANALYSIS_DIR = os.path.join(BASE, "analysis")
LOGS_DIR     = os.path.join(BASE, "logs")

N_BOOTSTRAP = 2000
SEED        = 27
ALPHA       = 0.05

SIZE_EDGES = [0.0, 3.0, 6.0, 15.0, float("inf")]
SIZE_CATS  = ["lt3mm", "3to6mm", "6to15mm", "ge15mm"]

MORPHOLOGICAL = [
    "log_max", "log_mean", "log_p95", "log_scale_mm",
    "blob_max", "blob_mean", "blob_p95", "blob_scale_mm",
    "vessel_max", "vessel_mean", "vessel_p95", "vessel_scale_mm",
    "contrast_hu", "centroid_peak_dist_mm",
]
ANATOMICAL_PRIMARY   = ["pleura_min_dist_mm"]
ANATOMICAL_SECONDARY = ["pleura_centroid_dist_mm"]
FEATURES = MORPHOLOGICAL + ANATOMICAL_PRIMARY + ANATOMICAL_SECONDARY

EXCLUDED = {
    "frac_high_log": ("EXCLUDED_DEGENERATE_BY_CONSTRUCTION",
                      "threshold defined from each object's own 75th percentile, "
                      "forcing the fraction to ~0.25 by construction"),
    "diam_mm": ("SIZE_COVARIATE", "used as stratum and continuous covariate, not as feature"),
    "volume_mm3": ("SIZE_COVARIATE", "size, not a filter response"),
    "mu_object_hu": ("QC_METADATA", "intermediate of contrast_hu"),
    "mu_ring_hu": ("QC_METADATA", "intermediate of contrast_hu"),
    "voxel_count": ("QC_METADATA", "size proxy"),
    "ring_voxel_count": ("QC_METADATA", "ring geometry"),
    "ring_truncated_at_image_boundary": ("QC_METADATA", "boundary flag"),
    "object_voxel_count_in_roi": ("QC_METADATA", "size proxy"),
    "component_voxels_total": ("QC_METADATA", "size proxy"),
    "component_voxels_inside_lung": ("QC_METADATA", "location flag"),
    "component_voxels_outside_lung": ("QC_METADATA", "location flag"),
    "component_fraction_inside_lung": ("QC_METADATA", "location flag"),
    "component_intersects_lung_surface": ("QC_METADATA", "location flag"),
    "component_entirely_outside_lung": ("QC_METADATA", "location flag"),
}

COMPARISONS = [("A", "TP_pred_vs_FP_pred", "TP_pred", "FP_pred", "pred"),
               ("B", "TP_gt_vs_FN_gt", "TP_gt", "FN_gt", "gt")]

EXPECTED = {
    "patients": 718, "rows": 62737,
    "A": {"total": (14673, 23314), "lt3mm": (859, 10488), "3to6mm": (7039, 8219),
          "6to15mm": (5237, 3772), "ge15mm": (1538, 835)},
    "B": {"total": (14673, 10077), "lt3mm": (168, 2862), "3to6mm": (6142, 4118),
          "6to15mm": (6502, 2578), "ge15mm": (1861, 519)},
}

for _d in (FIGDATA_DIR, ANALYSIS_DIR, LOGS_DIR):
    os.makedirs(_d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(os.path.join(LOGS_DIR, "analyze_filters_bd1.log")),
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


def holm(p):
    p = np.asarray(p, float); n = len(p)
    idx = np.argsort(p); adj = np.empty(n)
    for rank, i in enumerate(idx):
        adj[i] = min(p[i] * (n - rank), 1.0)
    for k in range(1, n):
        adj[idx[k]] = max(adj[idx[k]], adj[idx[k - 1]])
    return adj


def bh(p):
    p = np.asarray(p, float); n = len(p)
    idx = np.argsort(p)
    a = np.minimum(np.minimum.accumulate((p[idx] * n / np.arange(1, n + 1))[::-1])[::-1], 1.0)
    out = np.empty(n); out[idx] = a
    return out


def point_stats(pos, neg):
    """AUROC/AUPRC no sentido ORIGINAL da característica; TP = classe positiva."""
    if len(pos) < 2 or len(neg) < 2:
        return None
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    s = np.concatenate([pos, neg])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u, p = mannwhitneyu(pos, neg, alternative="two-sided")
    return {"auroc": float(roc_auc_score(y, s)), "average_precision": float(average_precision_score(y, s)),
            "u_stat": float(u), "p_raw": float(p),
            "median_pos": float(np.median(pos)), "q25_pos": float(np.percentile(pos, 25)),
            "q75_pos": float(np.percentile(pos, 75)),
            "median_neg": float(np.median(neg)), "q25_neg": float(np.percentile(neg, 25)),
            "q75_neg": float(np.percentile(neg, 75)),
            "direction": "raw_feature",
            "direction_of_difference": "higher_in_TP" if np.median(pos) > np.median(neg)
                                       else ("lower_in_TP" if np.median(pos) < np.median(neg) else "equal"),
            "n_pos": int(len(pos)), "n_neg": int(len(neg))}


def bootstrap_patient(sub, feat, gpos, gneg, n_boot=N_BOOTSTRAP, seed=SEED):
    """IC 95 % por reamostragem de PACIENTES: cada réplica sorteia patient_id uma
    vez e traz todas as linhas desse paciente, nas duas classes."""
    pids = sub["patient_id"].unique()
    if len(pids) < 2:
        return {}
    by = {p: g for p, g in sub.groupby("patient_id", sort=False)}
    rng = np.random.default_rng(seed)
    dif, au, ap = [], [], []
    for _ in range(n_boot):
        pick = rng.choice(pids, len(pids), replace=True)
        parts = [by[p] for p in pick]
        r = pd.concat(parts, ignore_index=True)
        pv = r.loc[r["group"] == gpos, feat].to_numpy()
        nv = r.loc[r["group"] == gneg, feat].to_numpy()
        if len(pv) < 2 or len(nv) < 2:
            continue
        dif.append(np.median(pv) - np.median(nv))
        y = np.concatenate([np.ones(len(pv)), np.zeros(len(nv))])
        s = np.concatenate([pv, nv])
        au.append(roc_auc_score(y, s)); ap.append(average_precision_score(y, s))
    if len(dif) < n_boot * 0.5:
        return {"bootstrap_replicates_used": len(dif)}
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    d = q(dif); a = q(au); p = q(ap)
    return {"median_diff_ci_lo": d[0], "median_diff_ci_hi": d[1],
            "auroc_ci_lo": a[0], "auroc_ci_hi": a[1],
            "average_precision_ci_lo": p[0], "average_precision_ci_hi": p[1],
            "bootstrap_replicates_used": len(dif)}


def size_adjusted(sub, feat, gpos, gneg, seed=SEED, n_boot=N_BOOTSTRAP):
    """MODEL_SIZE vs MODEL_SIZE_FILTER; IC do coeficiente por bootstrap de pacientes."""
    import statsmodels.api as sm
    d = sub[["patient_id", "group", "diam_mm", feat]].copy()
    d["y"] = (d["group"] == gpos).astype(int)
    d["size"] = np.log1p(d["diam_mm"].astype(float))
    X0 = sm.add_constant(d[["size"]], has_constant="add")
    X1 = sm.add_constant(d[["size", feat]], has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            m0 = sm.Logit(d["y"], X0).fit(disp=0, maxiter=200)
            m1 = sm.Logit(d["y"], X1).fit(disp=0, maxiter=200)
        except Exception as e:
            return {"status": "NOT_ESTIMABLE", "reason": f"logit: {type(e).__name__}"}
    if not (m0.mle_retvals.get("converged") and m1.mle_retvals.get("converged")):
        return {"status": "NOT_ESTIMABLE", "reason": "logit did not converge"}
    y = d["y"].to_numpy()
    a0 = float(roc_auc_score(y, m0.predict(X0))); a1 = float(roc_auc_score(y, m1.predict(X1)))
    p0 = float(average_precision_score(y, m0.predict(X0))); p1 = float(average_precision_score(y, m1.predict(X1)))
    pids = d["patient_id"].unique(); by = {p: g for p, g in d.groupby("patient_id", sort=False)}
    rng = np.random.default_rng(seed); coefs = []
    for _ in range(n_boot):
        r = pd.concat([by[p] for p in rng.choice(pids, len(pids), replace=True)], ignore_index=True)
        if r["y"].nunique() < 2:
            continue
        Xr = sm.add_constant(r[["size", feat]], has_constant="add")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                mr = sm.Logit(r["y"], Xr).fit(disp=0, maxiter=200)
                if mr.mle_retvals.get("converged"):
                    coefs.append(float(mr.params[feat]))
            except Exception:
                pass
    ci = (float(np.percentile(coefs, 2.5)), float(np.percentile(coefs, 97.5))) if len(coefs) >= 100 else (np.nan, np.nan)
    return {"status": "OK", "coef_feature": float(m1.params[feat]),
            "coef_feature_se": float(m1.bse[feat]), "p_raw": float(m1.pvalues[feat]),
            "coef_ci_lo": ci[0], "coef_ci_hi": ci[1], "bootstrap_replicates_used": len(coefs),
            "coef_size": float(m1.params["size"]),
            "auroc_size_only": a0, "auroc_size_filter": a1, "delta_auroc": a1 - a0,
            "ap_size_only": p0, "ap_size_filter": p1, "delta_ap": p1 - p0,
            "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--features-sha256", required=True)
    ap.add_argument("--source-run", default="")
    ap.add_argument("--out-dir", default=None, help="ENSAIO: escrever fora de figdata/analysis")
    ap.add_argument("--features-subset", default="", help="ENSAIO: subconjunto de características")
    ap.add_argument("--skip-adjusted", action="store_true", help="ENSAIO")
    ap.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    args = ap.parse_args()
    t0 = datetime.datetime.now().astimezone()
    figd = args.out_dir or FIGDATA_DIR
    anad = args.out_dir or ANALYSIS_DIR
    os.makedirs(figd, exist_ok=True); os.makedirs(anad, exist_ok=True)

    if not os.path.isfile(args.features):
        fail_input(f"entrada ausente: {args.features}")
    h = sha256(args.features)
    if h != args.features_sha256:
        fail_input(f"SOURCE_AUTHORITY: SHA {h[:12]} != declarado {args.features_sha256[:12]}")
    full = pd.read_csv(args.features)
    if len(full) != 72430:
        fail_input(f"entrada com {len(full)} linhas != 72430")
    key = ["subset", "patient_id", "architecture", "group", "component_id"]
    falta_key = [c for c in key + ["diam_mm"] if c not in full.columns]
    if falta_key:
        fail_input(f"colunas obrigatórias ausentes na entrada: {falta_key}")
    if full.duplicated(key).any():
        fail_input("chave duplicada na entrada")
    bad = set(full["group"].unique()) - {"TP_pred", "FP_pred", "TP_gt", "FN_gt"}
    if bad:
        fail_input(f"grupo inesperado: {bad}")

    df = full[full["subset"] == "oof"].copy()
    n_int = len(full) - len(df)
    log.info(f"OOF {len(df)} linhas / {df['patient_id'].nunique()} pacientes; INTERNAL {n_int} descartado")
    if len(df) != EXPECTED["rows"] or df["patient_id"].nunique() != EXPECTED["patients"]:
        fail_input(f"OOF {len(df)}/{df['patient_id'].nunique()} != {EXPECTED['rows']}/{EXPECTED['patients']}")
    if n_int != 9693:
        fail_input(f"INTERNAL {n_int} != 9693")

    if df["diam_mm"].isna().any() or (df["diam_mm"] <= 0).any() or not np.isfinite(df["diam_mm"]).all():
        fail_input("diam_mm inválido")
    df["size_cat"] = pd.cut(df["diam_mm"], SIZE_EDGES, right=False, labels=SIZE_CATS).astype(str)
    if (df["size_cat"] == "nan").any():
        fail_input("size_cat indefinido")

    feats = [f for f in FEATURES if f in df.columns]
    falta = [f for f in FEATURES if f not in df.columns]
    if falta:
        fail_input(f"características ausentes na entrada: {falta}")
    if args.features_subset:
        feats = [f for f in args.features_subset.split(",") if f in feats]
    for f in feats:
        if not np.isfinite(df[f]).all():
            fail_input(f"característica não finita: {f}")
    log.info(f"Características analíticas ({len(feats)}): {feats}")
    log.info("Excluídas: " + str({k: v[0] for k, v in EXCLUDED.items()}))

    for tag, _, gpos, gneg, _src in COMPARISONS:
        e = EXPECTED[tag]
        got = (int((df["group"] == gpos).sum()), int((df["group"] == gneg).sum()))
        if got != e["total"]:
            fail_input(f"{tag}: contagens {got} != congeladas {e['total']}")
        for cat in SIZE_CATS:
            s = df[df["size_cat"] == cat]
            g = (int((s["group"] == gpos).sum()), int((s["group"] == gneg).sum()))
            if g != e[cat]:
                fail_input(f"{tag}/{cat}: {g} != congeladas {e[cat]}")
    log.info("COUNT_GATE: contagens OOF idênticas às congeladas")

    pool, auc_rows, size_rows, adj_rows = [], [], [], []
    for tag, cname, gpos, gneg, src in COMPARISONS:
        sub = df[df["group"].isin([gpos, gneg])]
        log.info(f"=== {cname} (fonte {src}): n_pos={int((sub['group']==gpos).sum())} "
                 f"n_neg={int((sub['group']==gneg).sum())}")
        for f in feats:
            st = point_stats(sub.loc[sub["group"] == gpos, f].to_numpy(),
                             sub.loc[sub["group"] == gneg, f].to_numpy())
            base = {"family": f"POOL_{tag}", "comparison": cname, "source": src, "feature": f,
                    "stratum": "pooled", "analysis_role": "SECONDARY_DESCRIPTIVE"}
            if st is None:
                pool.append({**base, "status": "NOT_ESTIMABLE", "reason": "n < 2"}); continue
            ci = bootstrap_patient(sub, f, gpos, gneg, n_boot=args.n_boot)
            pool.append({**base, "status": "OK", **st, **ci})
            auc_rows.append({"family": f"POOL_{tag}", "comparison": cname, "feature": f,
                             "architecture": "all", "stratum": "pooled", **st, **ci})
        for cat in SIZE_CATS:
            s = sub[sub["size_cat"] == cat]
            for f in feats:
                st = point_stats(s.loc[s["group"] == gpos, f].to_numpy(),
                                 s.loc[s["group"] == gneg, f].to_numpy())
                base = {"family": f"SIZE_{tag}", "comparison": cname, "source": src,
                        "feature": f, "stratum": cat, "analysis_role": "PRIMARY"}
                if st is None:
                    size_rows.append({**base, "status": "NOT_ESTIMABLE",
                                      "reason": f"n_pos={int((s['group']==gpos).sum())} "
                                                f"n_neg={int((s['group']==gneg).sum())}"})
                    continue
                ci = bootstrap_patient(s, f, gpos, gneg, n_boot=args.n_boot)
                size_rows.append({**base, "status": "OK", **st, **ci})
        for arch in sorted(sub["architecture"].unique()):
            a = sub[sub["architecture"] == arch]
            for f in feats:
                st = point_stats(a.loc[a["group"] == gpos, f].to_numpy(),
                                 a.loc[a["group"] == gneg, f].to_numpy())
                base = {"family": f"ARCH_{tag}", "comparison": cname, "feature": f,
                        "architecture": arch, "stratum": "pooled", "analysis_role": "ROBUSTNESS"}
                if st is None:
                    auc_rows.append({**base, "status": "NOT_ESTIMABLE"}); continue
                auc_rows.append({**base, "status": "OK", **st})
        if not args.skip_adjusted:
            for f in feats:
                r = size_adjusted(sub, f, gpos, gneg, n_boot=args.n_boot)
                adj_rows.append({"family": f"ADJ_{tag}", "comparison": cname, "source": src,
                                 "feature": f, "analysis_role": "SECONDARY_ADJUSTED",
                                 "model_size": "logit(y) ~ log1p(diam_mm)",
                                 "model_size_filter": "logit(y) ~ log1p(diam_mm) + feature", **r})

    dp = pd.DataFrame(pool); da = pd.DataFrame(auc_rows); ds = pd.DataFrame(size_rows)
    dj = pd.DataFrame(adj_rows) if adj_rows else pd.DataFrame(columns=["family", "p_raw", "status"])

    fam_counts = {}
    for d in (dp, ds, dj, da):
        if d.empty or "family" not in d.columns:
            continue
        for col in ("p_holm", "p_bh"):
            if col not in d.columns:
                d[col] = np.nan
        if "significant_holm" not in d.columns:
            d["significant_holm"] = pd.Series([pd.NA] * len(d), dtype="object")
        else:
            d["significant_holm"] = d["significant_holm"].astype(object)
        for fam in sorted(d["family"].dropna().unique()):
            m = (d["family"] == fam) & d.get("status", "OK").eq("OK") & d["p_raw"].notna()
            if not m.any():
                continue
            p = d.loc[m, "p_raw"].to_numpy(float)
            d.loc[m, "p_holm"] = holm(p); d.loc[m, "p_bh"] = bh(p)
            d.loc[m, "significant_holm"] = (d.loc[m, "p_holm"].to_numpy(float) < ALPHA).tolist()
            fam_counts[fam] = int(m.sum())
    expected_fams = {f"{p}_{t}" for p in ("POOL", "SIZE", "ARCH", "ADJ") for t in ("A", "B")}
    if not args.features_subset and not args.skip_adjusted and set(fam_counts) != expected_fams:
        fail_output(f"famílias Holm {sorted(fam_counts)} != 8 esperadas")

    for name, d in (("pool", dp), ("size", ds), ("auc", da), ("adj", dj)):
        if d.empty:
            continue
        ok = d.get("status", pd.Series(["OK"] * len(d))) == "OK"
        for c in ("auroc", "average_precision", "p_raw", "p_holm", "p_bh",
                  "ap_size_only", "ap_size_filter", "auroc_size_only", "auroc_size_filter"):
            if c in d.columns:
                v = pd.to_numeric(d.loc[ok, c], errors="coerce").dropna()
                if len(v) and (v.min() < 0 or v.max() > 1):
                    fail_output(f"{name}.{c} fora de [0,1]")
        if "p_holm" in d.columns and "p_raw" in d.columns:
            v = d.loc[ok, ["p_raw", "p_holm"]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(v) and (v["p_holm"] < v["p_raw"] - 1e-12).any():
                fail_output(f"{name}: p_holm < p_raw")
        for lo, est, hi in (("auroc_ci_lo", "auroc", "auroc_ci_hi"),
                            ("average_precision_ci_lo", "average_precision", "average_precision_ci_hi"),
                            ("median_diff_ci_lo", "median_diff", "median_diff_ci_hi")):
            if lo in d.columns and hi in d.columns:
                v = d.loc[ok, [lo, hi]].apply(pd.to_numeric, errors="coerce").dropna()
                if len(v) and (v[lo] > v[hi]).any():
                    fail_output(f"{name}: IC invertido em {lo}/{hi}")
        if "n_pos" in d.columns:
            v = d.loc[ok, ["n_pos", "n_neg"]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(v) and ((v["n_pos"] <= 0) | (v["n_neg"] <= 0)).any():
                fail_output(f"{name}: n_pos/n_neg <= 0 em linha OK")

    p_comp = os.path.join(figd, "filter_bd1_group_comparisons.csv")
    p_auc  = os.path.join(figd, "filter_bd1_auc.csv")
    p_size = os.path.join(figd, "filter_bd1_size_strata.csv")
    p_adj  = os.path.join(anad, "filter_bd1_size_adjusted.csv")
    dp.to_csv(p_comp, index=False); da.to_csv(p_auc, index=False)
    ds.to_csv(p_size, index=False); dj.to_csv(p_adj, index=False)

    man = {
        "script": "16_analyze_filters_bd1.py", "script_sha256": sha256(os.path.abspath(__file__)),
        "run_at": t0.isoformat(), "end_at": datetime.datetime.now().astimezone().isoformat(),
        "source": {"features": args.features, "features_sha256": h, "script15_run": args.source_run},
        "population": {"analysed": "OOF only", "rows": len(df), "patients": int(df["patient_id"].nunique()),
                       "internal_discarded": n_int,
                       "internal_use": "NONE — not in tests, AUROC, AP, bootstrap or any decision"},
        "comparisons": {"A": "TP_pred vs FP_pred (prediction-defined)",
                        "B": "TP_gt vs FN_gt (reference-defined)",
                        "forbidden": ["FP_pred vs FN_gt", "TP_pred vs TP_gt"]},
        "size_strata": {"edges_mm": SIZE_EDGES[:-1] + ["inf"], "labels": SIZE_CATS,
                        "diameter": "diam_mm of the object's own source; never recomputed"},
        "hierarchy": {"POOL": "SECONDARY_DESCRIPTIVE (size-confounded)", "SIZE": "PRIMARY",
                      "ARCH": "ROBUSTNESS", "ADJ": "SECONDARY_ADJUSTED"},
        "features_analysed": feats,
        "features_excluded": {k: {"status": v[0], "reason": v[1]} for k, v in EXCLUDED.items()},
        "direction": "raw_feature; positive class = TP; AUROC<0.5 is a valid inverse association",
        "bootstrap": {"unit": "patient_id", "n": args.n_boot, "seed": SEED,
                      "note": "one draw of patients per replicate, all rows of that patient in both classes"},
        "mann_whitney": "object-level U; patient-level resampling used for all CIs",
        "precision_recall_summary": {"function": "sklearn.metrics.average_precision_score",
                                     "metric_name": "average_precision", "abbrev": "AP",
                                     "positive_class": "TP", "score_direction": "raw_feature",
                                     "note": "not a trapezoidal PR integration; ties handled by AP definition"},
        "multiplicity": {"method": "Holm (primary), BH (supplementary)", "families": sorted(fam_counts),
                         "tests_per_family": fam_counts, "alpha": ALPHA,
                         "not_corrected": ["AUROC", "average_precision", "delta_AUROC", "delta_AP"]},
        "counts_expected_frozen": EXPECTED,
        "outputs_sha256": {os.path.basename(p): sha256(p) for p in (p_comp, p_auc, p_size, p_adj)},
        "rows": {"group_comparisons": len(dp), "auc": len(da), "size_strata": len(ds), "size_adjusted": len(dj)},
    }
    json.dump(man, open(os.path.join(anad, "filter_bd1_analysis_manifest.json"), "w"), indent=2)
    log.info(f"Salvo: {p_comp} ({len(dp)}), {p_auc} ({len(da)}), {p_size} ({len(ds)}), {p_adj} ({len(dj)})")
    log.info(f"Famílias Holm: {fam_counts}")
    log.info("Script 16 concluído.")


if __name__ == "__main__":
    main()
