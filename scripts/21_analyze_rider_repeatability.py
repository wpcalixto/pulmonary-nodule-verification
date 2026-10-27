#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    21_analyze_rider_repeatability.py

File type:
    Script

Purpose:
    Repeatability analysis on RIDER (Bland-Altman, bootstrap confidence intervals, ICC(3,1), repeatability coefficient).

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
    - gates.json
    - manifest.json

Dependencies:
    - numpy
    - pandas
    - scipy

Execution:
    PULMONARY_BASE=<project_dir> python scripts/21_analyze_rider_repeatability.py

Portability notes:
    All paths are resolved relative to the project base directory given by the PULMONARY_BASE environment variable. No machine-specific path is required.

Reproducibility notes:
    Environment pinned in requirements.txt and environment/. Seeds and frozen parameters are recorded in the run manifests under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) are not included and must be obtained from their original repositories.
"""

import argparse
import datetime
import hashlib
import json
import logging
import os

import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr, spearmanr, wilcoxon

BASE = os.environ.get("PULMONARY_BASE") or os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(BASE, "analysis")
FIGDATA_DIR = os.path.join(BASE, "figdata")
LOGS_DIR = os.path.join(BASE, "logs")

SEED = 27
B_VALID = 2000
MAX_ATTEMPTS = 10000
ALPHA = 0.05
Z = 1.96

S20_RUN_ESPERADO = "s20_20260923T211225_698236"
N_PARES_ESPERADO = 2274
N_PACIENTES_ESPERADO = 31
ARQ_ESTIMAVEIS = {"2D": 251, "2D_REL": 165, "2D_REM": 226, "3D_CF": 371,
                  "3D_CFREM": 341, "3D_F": 283, "3D_FREM": 195,
                  "F100_fullres": 357, "F100_lowres": 85}
ARQ_SEM_PARES = "F100_ResEncM"
ARQUITECTURAS = sorted(ARQ_ESTIMAVEIS) + [ARQ_SEM_PARES]

FEATURES_16 = [
    "log_max", "log_mean", "log_p95", "log_scale_mm",
    "blob_max", "blob_mean", "blob_p95", "blob_scale_mm",
    "vessel_max", "vessel_mean", "vessel_p95", "vessel_scale_mm",
    "contrast_hu", "centroid_peak_dist_mm",
    "pleura_min_dist_mm", "pleura_centroid_dist_mm",
]
PROIBIDAS = {"frac_high_log"}
MEDIDAS = ([("volume_mm3", "volume_ct1_mm3", "volume_ct2_mm3"),
            ("diameter_mm", "diameter_ct1_mm", "diameter_ct2_mm")]
           + [(f, f"{f}_ct1", f"{f}_ct2") for f in FEATURES_16])
CATEGORIAS = ["lt3mm", "3to6mm", "6to15mm", "ge15mm"]
CHAVE = ["patient_id", "architecture", "component_id_ct1", "component_id_ct2_native"]

FAMILIAS = ["WILCOXON", "PROPORTIONAL_BIAS",
            "MORPH_VOLUME_ASSOC_PRIMARY", "MORPH_VOLUME_ASSOC_RELATIVE_SENSITIVITY"]

os.makedirs(LOGS_DIR, exist_ok=True)
log = logging.getLogger("s21")
log.setLevel(logging.INFO)
if not log.handlers:
    _f = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for _h in (logging.StreamHandler(),
               logging.FileHandler(os.path.join(LOGS_DIR, "analyze_rider_repeatability.log"))):
        _h.setFormatter(_f); log.addHandler(_h)
log.propagate = False


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


def icc31(x, y):
    """ICC(3,1): dois factores, efeitos mistos, consistência, medida única.

    Invariante a deslocamento constante entre aquisições (é consistência, não
    concordância absoluta)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    if n < 2:
        return float("nan")
    grand = np.mean(np.concatenate([x, y]))
    linha = (x + y) / 2.0
    SSR = 2.0 * np.sum((linha - grand) ** 2)
    SSW = np.sum((x - linha) ** 2) + np.sum((y - linha) ** 2)
    SSC = n * ((np.mean(x) - grand) ** 2 + (np.mean(y) - grand) ** 2)
    SSE = SSW - SSC
    MSR = SSR / (n - 1)
    MSE = SSE / max(n - 1, 1)
    if MSR + MSE == 0:
        return float("nan")
    return float((MSR - MSE) / (MSR + MSE))


def icc31_anova_independente(x, y):
    """Segunda implementação, por decomposição ANOVA explícita — só para o
    ICC31_PARITY_GATE. Não é usada no cálculo científico."""
    m = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    n, k = m.shape
    if n < 2:
        return float("nan")
    gm = m.mean()
    ms_linhas = k * ((m.mean(axis=1) - gm) ** 2).sum() / (n - 1)
    ms_colunas = n * ((m.mean(axis=0) - gm) ** 2).sum() / (k - 1)
    ss_total = ((m - gm) ** 2).sum()
    ss_erro = (ss_total - k * ((m.mean(axis=1) - gm) ** 2).sum()
               - n * ((m.mean(axis=0) - gm) ** 2).sum())
    ms_erro = ss_erro / ((n - 1) * (k - 1))
    den = ms_linhas + (k - 1) * ms_erro
    if den == 0:
        return float("nan")
    return float((ms_linhas - ms_erro) / den)


def ba_pontual(delta):
    """Bland–Altman ao nível do objecto + coeficiente de repetibilidade."""
    n = len(delta)
    vies = float(np.mean(delta))
    sd = float(np.std(delta, ddof=1)) if n > 1 else float("nan")
    return {"n_pairs": n, "bias": vies, "sd_difference": sd,
            "lower_loa": vies - Z * sd, "upper_loa": vies + Z * sd,
            "within_subject_sd": sd / np.sqrt(2.0),
            "repeatability_coefficient": Z * sd}


def beta1_ols(mx, delta):
    """Declive de delta ~ 1 + média do par, por forma fechada."""
    v = np.var(mx, ddof=1)
    if not np.isfinite(v) or v == 0:
        return float("nan"), float("nan")
    b1 = float(np.cov(mx, delta, ddof=1)[0, 1] / v)
    b0 = float(np.mean(delta) - b1 * np.mean(mx))
    return b0, b1


def beta1_cluster(mx, delta, cod_pac):
    """Declive com erro-padrão agrupado por paciente (sanduíche CR0 + correcção
    de amostra finita), e p bilateral normal."""
    n = len(mx)
    X = np.column_stack([np.ones(n), mx])
    XtX = X.T @ X
    if np.linalg.matrix_rank(XtX) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    XtXi = np.linalg.inv(XtX)
    beta = XtXi @ (X.T @ delta)
    resid = delta - X @ beta
    grupos = np.unique(cod_pac)
    meio = np.zeros((2, 2))
    for g in grupos:
        s = cod_pac == g
        u = X[s].T @ resid[s]
        meio += np.outer(u, u)
    G = len(grupos)
    ajuste = (G / max(G - 1, 1)) * ((n - 1) / max(n - 2, 1))
    V = XtXi @ meio @ XtXi * ajuste
    se = float(np.sqrt(V[1, 1])) if V[1, 1] > 0 else float("nan")
    b1 = float(beta[1])
    p = float(2.0 * norm.sf(abs(b1 / se))) if np.isfinite(se) and se > 0 else float("nan")
    sst = float(np.sum((delta - delta.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / sst if sst > 0 else float("nan")
    r2a = 1.0 - (1.0 - r2) * (n - 1) / (n - 2) if np.isfinite(r2) and n > 2 else float("nan")
    return b1, se, p, r2a, float(beta[0])


def holm(pvals):
    pv = np.asarray(pvals, float)
    out = np.full(pv.shape, np.nan)
    val = np.isfinite(pv)
    if val.sum() == 0:
        return out
    idx = np.where(val)[0]; p = pv[val]
    o = np.argsort(p); m = len(p)
    aj = np.minimum(1.0, p[o] * (m - np.arange(m)))
    aj = np.maximum.accumulate(aj)
    inv = np.empty(m); inv[o] = aj
    out[idx] = inv
    return out


def ci_percentil(arr):
    a = np.asarray(arr, float); a = a[np.isfinite(a)]
    if len(a) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def sorteios_por_paciente(pacientes_por_linha, semente=SEED):
    """Devolve (lista de índices por réplica, tentativas). A unidade é o
    paciente: todos os objectos do paciente sorteado entram juntos."""
    rng = np.random.default_rng(semente)
    unicos = np.unique(pacientes_por_linha)
    grupos = [np.where(pacientes_por_linha == p)[0] for p in unicos]
    n = len(unicos)
    reps, tent = [], 0
    while len(reps) < B_VALID and tent < MAX_ATTEMPTS:
        tent += 1
        d = rng.integers(0, n, size=n)
        idx = np.concatenate([grupos[i] for i in d])
        if len(np.unique(pacientes_por_linha[idx])) < 2:
            continue
        reps.append(idx)
    return reps, tent


def analisa(sub, nome, c1, c2, reps):
    x = sub[c1].to_numpy(float); y = sub[c2].to_numpy(float)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        fail_output(f"valores não finitos em {nome}")
    delta = y - x
    mx = (x + y) / 2.0
    pac = sub["patient_id"].to_numpy()
    r = {"n_pairs": len(sub), "n_patients": int(pd.unique(pac).size),
         "median_ct1": float(np.median(x)), "iqr_ct1": float(np.subtract(*np.percentile(x, [75, 25]))),
         "median_ct2": float(np.median(y)), "iqr_ct2": float(np.subtract(*np.percentile(y, [75, 25])))}
    if np.std(delta, ddof=1) == 0 and np.std(x, ddof=1) == 0:
        r["status"] = "NOT_ESTIMABLE_DEGENERATE_VARIANCE"
        return r
    r.update(ba_pontual(delta))
    r["icc31"] = icc31(x, y)
    r["pearson_r"] = float(pearsonr(x, y)[0]) if np.std(x, ddof=1) > 0 and np.std(y, ddof=1) > 0 \
        else float("nan")
    b1, se, p_b1, r2a, b0 = beta1_cluster(mx, delta, pd.factorize(pac)[0])
    r.update({"proportional_bias_beta0": b0, "proportional_bias_beta": b1,
              "proportional_bias_se_cluster": se, "proportional_bias_p_raw": p_b1,
              "proportional_bias_adj_r2": r2a})

    bb, bl, bu, bi, bp, bt = [], [], [], [], [], []
    for idx in reps:
        xb, yb = x[idx], y[idx]
        db = yb - xb
        if len(db) < 3:
            continue
        sdb = np.std(db, ddof=1)
        vb = float(np.mean(db))
        bb.append(vb); bl.append(vb - Z * sdb); bu.append(vb + Z * sdb)
        bi.append(icc31(xb, yb))
        bp.append(pearsonr(xb, yb)[0] if np.std(xb, ddof=1) > 0 and np.std(yb, ddof=1) > 0
                  else np.nan)
        bt.append(beta1_ols((xb + yb) / 2.0, db)[1])
    for chave, arr in (("bias", bb), ("lower_loa", bl), ("upper_loa", bu),
                       ("icc31", bi), ("pearson_r", bp), ("proportional_bias", bt)):
        lo, hi = ci_percentil(arr)
        r[f"{chave}_ci_lo"], r[f"{chave}_ci_hi"] = lo, hi
    r["bootstrap_valid"] = len(bb)

    medias = pd.DataFrame({"p": pac, "d": delta}).groupby("p")["d"].mean().to_numpy()
    nz = int(np.sum(medias != 0))
    r["n_patients_wilcoxon"] = len(medias)
    r["n_nonzero_patient_differences"] = nz
    if nz == 0:
        r["wilcoxon_p_raw"] = float("nan")
        r["wilcoxon_status"] = "NOT_ESTIMABLE_ALL_ZERO_DIFFERENCES"
    else:
        r["wilcoxon_p_raw"] = float(wilcoxon(medias, alternative="two-sided",
                                             zero_method="wilcox", correction=False,
                                             method="approx")[1])
        r["wilcoxon_status"] = "ESTIMATED"
    r["status"] = "ESTIMATED"
    return r


def linha_nao_estimavel(arch, medida):
    d = {"architecture": arch, "measure": medida, "n_pairs": 0, "n_patients": 0,
         "status": "NOT_ESTIMABLE_NO_MATCHED_PAIRS"}
    for c in ("median_ct1", "iqr_ct1", "median_ct2", "iqr_ct2", "bias", "bias_ci_lo",
              "bias_ci_hi", "sd_difference", "within_subject_sd", "lower_loa",
              "lower_loa_ci_lo", "lower_loa_ci_hi", "upper_loa", "upper_loa_ci_lo",
              "upper_loa_ci_hi", "repeatability_coefficient", "icc31", "icc31_ci_lo",
              "icc31_ci_hi", "pearson_r", "pearson_r_ci_lo", "pearson_r_ci_hi",
              "wilcoxon_p_raw", "wilcoxon_p_holm", "proportional_bias_beta0",
              "proportional_bias_beta", "proportional_bias_se_cluster",
              "proportional_bias_ci_lo", "proportional_bias_ci_hi",
              "proportional_bias_p_raw", "proportional_bias_p_holm",
              "proportional_bias_adj_r2", "bootstrap_valid",
              "n_patients_wilcoxon", "n_nonzero_patient_differences"):
        d[c] = float("nan")
    d["wilcoxon_status"] = "NOT_ESTIMABLE_NO_MATCHED_PAIRS"
    return d


def wilson(k, n):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    c = (p + Z ** 2 / (2 * n)) / (1 + Z ** 2 / n)
    m = Z * np.sqrt(p * (1 - p) / n + Z ** 2 / (4 * n ** 2)) / (1 + Z ** 2 / n)
    return float(c - m), float(c + m)


def sccr(sub, reps):
    linhas = []
    conc = (sub["size_category_ct1"].to_numpy() == sub["size_category_ct2"].to_numpy())
    pac = sub["patient_id"].to_numpy()
    cat1 = sub["size_category_ct1"].to_numpy()
    for nivel in ["Overall"] + CATEGORIAS:
        sel = np.ones(len(sub), bool) if nivel == "Overall" else (cat1 == nivel)
        n = int(sel.sum()); k = int(conc[sel].sum())
        base = {"category": nivel, "n_pairs": n,
                "n_patients": int(pd.unique(pac[sel]).size), "n_concordant": k}
        if n == 0:
            linhas.append({**base, "sccr": float("nan"), "bootstrap_ci_lo": float("nan"),
                           "bootstrap_ci_hi": float("nan"), "wilson_ci_lo": float("nan"),
                           "wilson_ci_hi": float("nan"),
                           "status": "NOT_ESTIMABLE_NO_BASELINE_CATEGORY"})
            continue
        vals = []
        for idx in reps:
            s = sel[idx]
            if s.sum() == 0:
                continue
            vals.append(conc[idx][s].mean())
        lo, hi = ci_percentil(vals)
        wlo, whi = wilson(k, n)
        linhas.append({**base, "sccr": k / n, "bootstrap_ci_lo": lo, "bootstrap_ci_hi": hi,
                       "wilson_ci_lo": wlo, "wilson_ci_hi": whi,
                       "bootstrap_valid": len(vals), "status": "ESTIMATED"})
    return linhas


def transicoes(sub):
    linhas = []
    for a in CATEGORIAS:
        s = sub[sub["size_category_ct1"] == a]
        for b in CATEGORIAS:
            n = int((s["size_category_ct2"] == b).sum())
            linhas.append({"ct1_category": a, "ct2_category": b, "count": n,
                           "proportion_within_ct1": (n / len(s)) if len(s) else float("nan"),
                           "n_ct1_category": len(s)})
    return linhas


def associacao(sub, xcol, reps):
    x = sub[xcol].to_numpy(float)
    y = sub["abs_delta_log_max"].to_numpy(float)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        fail_output(f"valores não finitos na associação {xcol}")
    rho, p = spearmanr(x, y)
    vals = []
    for idx in reps:
        xb, yb = x[idx], y[idx]
        if len(np.unique(xb)) < 3 or len(np.unique(yb)) < 3:
            continue
        vals.append(spearmanr(xb, yb)[0])
    lo, hi = ci_percentil(vals)
    return {"x": xcol, "y": "abs_delta_log_max", "statistic": "spearman_rho",
            "n_pairs": len(sub), "n_patients": int(sub.patient_id.nunique()),
            "rho": float(rho), "rho_ci_lo": lo, "rho_ci_hi": hi,
            "p_raw": float(p), "bootstrap_valid": len(vals), "status": "ESTIMATED"}


def proporcoes(sub, reps):
    linhas = []
    for col, rot in (("abs_rel_delta_volume", "volume"), ("abs_rel_delta_diameter", "diameter")):
        v = sub[col].to_numpy(float)
        for lim in (0.15, 0.25):
            ind = (v > lim)
            vals = [ind[idx].mean() for idx in reps]
            lo, hi = ci_percentil(vals)
            linhas.append({"measure": rot, "threshold": lim, "n_pairs": len(sub),
                           "n_above": int(ind.sum()), "proportion": float(ind.mean()),
                           "bootstrap_ci_lo": lo, "bootstrap_ci_hi": hi,
                           "n_patients": int(sub.patient_id.nunique()), "status": "ESTIMATED"})
    return linhas


def gate_rc():
    """RC = 1,96·SD(delta); SD intra = SD/√2; RC = 2,77·SD intra."""
    rng = np.random.default_rng(1)
    d = rng.normal(3.0, 7.0, 500)
    s = ba_pontual(d)
    sd = float(np.std(d, ddof=1))
    testes = {
        "rc_igual_1.96_sd": abs(s["repeatability_coefficient"] - 1.96 * sd) < 1e-12,
        "sem_raiz_de_2_extra": abs(s["repeatability_coefficient"] - 1.96 * np.sqrt(2) * sd) > 1e-6,
        "sd_intra_igual_sd_sobre_raiz2": abs(s["within_subject_sd"] - sd / np.sqrt(2)) < 1e-12,
        "rc_igual_2.77_sd_intra": abs(s["repeatability_coefficient"]
                                      - 2.7718585822512662 * s["within_subject_sd"]) < 1e-9,
        "loa_ordenadas": s["lower_loa"] < s["bias"] < s["upper_loa"],
        "amplitude_loa_igual_2x1.96sd": abs((s["upper_loa"] - s["lower_loa"]) - 2 * 1.96 * sd) < 1e-12,
    }
    return {k: bool(v) for k, v in testes.items()}


def gate_icc():
    rng = np.random.default_rng(2)
    testes = {}
    difs = []
    for _ in range(200):
        x = rng.normal(size=40); y = x * 0.8 + rng.normal(scale=0.5, size=40)
        difs.append(abs(icc31(x, y) - icc31_anova_independente(x, y)))
    testes["paridade_anova_independente"] = max(difs) < 1e-9
    x = rng.normal(size=50)
    testes["identidade_da_1"] = abs(icc31(x, x) - 1.0) < 1e-9
    testes["consistencia_invariante_a_deslocamento"] = abs(icc31(x, x + 5.0) - 1.0) < 1e-9
    testes["dominio_valido"] = -1.0 <= icc31(x, -x) <= 1.0
    out = {k: bool(v) for k, v in testes.items() if k != "max_dif_paridade"}
    out["max_dif_paridade"] = float(max(difs))
    return out


def main():
    global B_VALID
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--features-sha256", required=True)
    ap.add_argument("--script20-manifest", required=True)
    ap.add_argument("--script20-run-id", default=S20_RUN_ESPERADO)
    ap.add_argument("--n-boot", type=int, default=B_VALID)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--gates-only", action="store_true")
    args = ap.parse_args()
    t0 = datetime.datetime.now().astimezone()
    B_VALID = args.n_boot

    g_rc, g_icc = gate_rc(), gate_icc()
    if not all(bool(v) for v in g_rc.values()):
        fail_output(f"REPEATABILITY_COEFFICIENT_GATE falhou: {g_rc}")
    if not all(bool(v) for k, v in g_icc.items() if k != "max_dif_paridade"):
        fail_output(f"ICC31_PARITY_GATE falhou: {g_icc}")
    log.info(f"RC gate OK · ICC paridade OK (máx {g_icc['max_dif_paridade']:.2e})")
    if args.gates_only:
        print(json.dumps({"REPEATABILITY_COEFFICIENT_GATE": "PASS",
                          "ICC31_PARITY_GATE": "PASS", "detalhe_rc": g_rc,
                          "detalhe_icc": g_icc}, indent=1, default=str))
        return

    if not os.path.isfile(args.script20_manifest):
        fail_input(f"manifesto do Script 20 ausente: {args.script20_manifest}")
    man20 = json.load(open(args.script20_manifest))
    if man20["run_id"] != args.script20_run_id:
        fail_input(f"run do Script 20 {man20['run_id']} != canónico {args.script20_run_id}")
    if not os.path.isfile(args.features):
        fail_input(f"matriz de pares ausente: {args.features}")
    h = sha256(args.features)
    if h != args.features_sha256:
        fail_input(f"INPUT_AUTHORITY: SHA {h[:12]} != {args.features_sha256[:12]}")
    oh_p = os.path.join(os.path.dirname(args.script20_manifest), "output_hashes.csv")
    if not os.path.isfile(oh_p):
        fail_input(f"output_hashes.csv do Script 20 ausente: {oh_p}")
    oh = pd.read_csv(oh_p)
    linha = oh[oh["path"].str.endswith("rider_pair_features.csv")]
    if len(linha) != 1 or linha["sha256"].iloc[0] != h:
        fail_input("SHA da matriz não confere com o output_hashes.csv do Script 20")

    df = pd.read_csv(args.features)
    if len(df) != N_PARES_ESPERADO:
        fail_input(f"ROW_COUNT: {len(df)} != {N_PARES_ESPERADO}")
    if df.duplicated(CHAVE).any():
        fail_input("KEY_IDENTITY: chave canónica repetida")
    if df["patient_id"].nunique() != N_PACIENTES_ESPERADO:
        fail_input(f"pacientes {df['patient_id'].nunique()} != {N_PACIENTES_ESPERADO}")
    obtidas = dict(df["architecture"].value_counts())
    if obtidas != ARQ_ESTIMAVEIS:
        fail_input(f"ARCHITECTURE_COMPLETENESS: {obtidas} != esperado")
    if (df["architecture"] == ARQ_SEM_PARES).any():
        fail_input(f"{ARQ_SEM_PARES} não devia ter linhas")
    precisas = [c for _, a, b in MEDIDAS for c in (a, b)] + \
               ["size_category_ct1", "size_category_ct2", "abs_rel_delta_volume",
                "abs_rel_delta_diameter", "abs_delta_volume_mm3", "abs_delta_log_max"]
    falta = [c for c in precisas if c not in df.columns]
    if falta:
        fail_input(f"colunas em falta na entrada: {falta}")
    for c in precisas:
        if df[c].dtype.kind in "fi" and not np.isfinite(df[c].to_numpy(float)).all():
            fail_input(f"valor não finito na coluna de entrada {c}")
    cats = set(df["size_category_ct1"]) | set(df["size_category_ct2"])
    if not cats <= set(CATEGORIAS):
        fail_input(f"categorias fora das quatro congeladas: {cats}")
    for p in PROIBIDAS:
        if p in [m[0] for m in MEDIDAS]:
            fail_input(f"variável proibida entre as medidas: {p}")
    log.info(f"Script 20 canónico {man20['run_id']}: {len(df)} pares, "
             f"{df.patient_id.nunique()} pacientes, {len(obtidas)} arquitecturas estimáveis")

    run_id = f"s21_{t0:%Y%m%dT%H%M%S}_{os.getpid()}"
    out_root = os.path.join(args.out_root or os.path.join(ANALYSIS_DIR, "rider_repeatability"), run_id)
    os.makedirs(out_root, exist_ok=True)

    linhas, sccr_linhas, trans_linhas, assoc_linhas, prop_linhas = [], [], [], [], []
    tentativas = {}
    for arch in ARQUITECTURAS:
        if arch == ARQ_SEM_PARES:
            for nome, _, _ in MEDIDAS:
                linhas.append(linha_nao_estimavel(arch, nome))
            for nivel in ["Overall"] + CATEGORIAS:
                sccr_linhas.append({"architecture": arch, "category": nivel, "n_pairs": 0,
                                    "n_patients": 0, "n_concordant": 0, "sccr": float("nan"),
                                    "bootstrap_ci_lo": float("nan"), "bootstrap_ci_hi": float("nan"),
                                    "wilson_ci_lo": float("nan"), "wilson_ci_hi": float("nan"),
                                    "status": "NOT_ESTIMABLE_NO_MATCHED_PAIRS"})
            for a in CATEGORIAS:
                for b in CATEGORIAS:
                    trans_linhas.append({"architecture": arch, "ct1_category": a,
                                         "ct2_category": b, "count": 0,
                                         "proportion_within_ct1": float("nan"),
                                         "n_ct1_category": 0,
                                         "status": "NOT_ESTIMABLE_NO_MATCHED_PAIRS"})
            for xcol, fam in (("abs_delta_volume_mm3", FAMILIAS[2]),
                              ("abs_rel_delta_volume", FAMILIAS[3])):
                assoc_linhas.append({"architecture": arch, "family": fam, "x": xcol,
                                     "y": "abs_delta_log_max", "statistic": "spearman_rho",
                                     "n_pairs": 0, "n_patients": 0, "rho": float("nan"),
                                     "rho_ci_lo": float("nan"), "rho_ci_hi": float("nan"),
                                     "p_raw": float("nan"), "p_holm": float("nan"),
                                     "status": "NOT_ESTIMABLE_NO_MATCHED_PAIRS"})
            for rot in ("volume", "diameter"):
                for lim in (0.15, 0.25):
                    prop_linhas.append({"architecture": arch, "measure": rot, "threshold": lim,
                                        "n_pairs": 0, "n_above": 0, "proportion": float("nan"),
                                        "bootstrap_ci_lo": float("nan"),
                                        "bootstrap_ci_hi": float("nan"), "n_patients": 0,
                                        "status": "NOT_ESTIMABLE_NO_MATCHED_PAIRS"})
            continue

        sub = df[df["architecture"] == arch].reset_index(drop=True)
        reps, tent = sorteios_por_paciente(sub["patient_id"].to_numpy())
        tentativas[arch] = {"valid": len(reps), "attempts": tent}
        if len(reps) != B_VALID:
            fail_output(f"BOOTSTRAP_COUNT: {arch} obteve {len(reps)} réplicas válidas "
                        f"de {B_VALID} em {tent} tentativas")
        log.info(f"  {arch}: {len(sub)} pares, {sub.patient_id.nunique()} pacientes, "
                 f"{len(reps)} réplicas")
        for nome, c1, c2 in MEDIDAS:
            linhas.append({"architecture": arch, "measure": nome,
                           **analisa(sub, f"{arch}/{nome}", c1, c2, reps)})
        for l in sccr(sub, reps):
            sccr_linhas.append({"architecture": arch, **l})
        for l in transicoes(sub):
            trans_linhas.append({"architecture": arch, **l, "status": "ESTIMATED"})
        for xcol, fam in (("abs_delta_volume_mm3", FAMILIAS[2]),
                          ("abs_rel_delta_volume", FAMILIAS[3])):
            assoc_linhas.append({"architecture": arch, "family": fam,
                                 **associacao(sub, xcol, reps)})
        for l in proporcoes(sub, reps):
            prop_linhas.append({"architecture": arch, **l})

    st = pd.DataFrame(linhas)
    est = st["status"] == "ESTIMATED"
    st["wilcoxon_p_holm"] = np.nan
    st.loc[est, "wilcoxon_p_holm"] = holm(st.loc[est, "wilcoxon_p_raw"].to_numpy())
    st["proportional_bias_p_holm"] = np.nan
    st.loc[est, "proportional_bias_p_holm"] = holm(st.loc[est, "proportional_bias_p_raw"].to_numpy())
    st["wilcoxon_family"] = np.where(est, FAMILIAS[0], "")
    st["proportional_bias_family"] = np.where(est, FAMILIAS[1], "")
    assoc = pd.DataFrame(assoc_linhas)
    assoc["p_holm"] = np.nan
    for fam in (FAMILIAS[2], FAMILIAS[3]):
        s = (assoc["family"] == fam) & (assoc["status"] == "ESTIMATED")
        assoc.loc[s, "p_holm"] = holm(assoc.loc[s, "p_raw"].to_numpy())
    familias = {FAMILIAS[0]: int(est.sum()), FAMILIAS[1]: int(est.sum()),
                FAMILIAS[2]: int(((assoc.family == FAMILIAS[2]) & (assoc.status == "ESTIMATED")).sum()),
                FAMILIAS[3]: int(((assoc.family == FAMILIAS[3]) & (assoc.status == "ESTIMATED")).sum())}

    sc = pd.DataFrame(sccr_linhas); tr = pd.DataFrame(trans_linhas)
    pr = pd.DataFrame(prop_linhas)
    ft = st[st["measure"].isin(FEATURES_16)].copy()

    g = {}
    g["INPUT_AUTHORITY_GATE"] = "PASS"
    g["ROW_COUNT_GATE"] = f"{len(df)}/{N_PARES_ESPERADO}"
    g["KEY_IDENTITY_GATE"] = "PASS"
    g["ARCHITECTURE_COMPLETENESS_GATE"] = f"{len(ARQ_ESTIMAVEIS)} estimáveis + 1 não estimável"
    n_esperado = len(ARQUITECTURAS) * len(MEDIDAS)
    if len(st) != n_esperado:
        fail_output(f"OUTPUT_SCHEMA: {len(st)} linhas != {n_esperado}")
    g["OUTPUT_SCHEMA_GATE"] = f"{len(st)} linhas = {len(ARQUITECTURAS)}×{len(MEDIDAS)}"
    ne = st[st.architecture == ARQ_SEM_PARES]
    if len(ne) != len(MEDIDAS) or not (ne["status"] == "NOT_ESTIMABLE_NO_MATCHED_PAIRS").all():
        fail_output("F100_NOT_ESTIMABLE_GATE falhou")
    g["F100_NOT_ESTIMABLE_GATE"] = f"{len(ne)} linhas NOT_ESTIMABLE, nenhum zero fabricado"
    e = st[est]
    if not ((e.lower_loa <= e.bias) & (e.bias <= e.upper_loa)).all():
        fail_output("BA_ORDER_GATE: LoA fora de ordem")
    g["BA_ORDER_GATE"] = "PASS"
    if not np.allclose(e.upper_loa - e.lower_loa, 2 * Z * e.sd_difference, rtol=0, atol=1e-9):
        fail_output("BA_IDENTITY_GATE: amplitude das LoA != 2·1,96·SD")
    g["BA_IDENTITY_GATE"] = "PASS"
    for c in ("bias", "lower_loa", "upper_loa", "icc31", "pearson_r", "proportional_bias"):
        v = e[c] if c != "proportional_bias" else e["proportional_bias_beta"]
        mal = int(((e[f"{c}_ci_lo"] > v) | (v > e[f"{c}_ci_hi"])).sum())
        if mal:
            fail_output(f"BA_CI_ORDER_GATE: {mal} IC não contêm a estimativa em {c}")
    g["BA_CI_ORDER_GATE"] = "PASS"
    if not np.allclose(e.repeatability_coefficient, 1.96 * e.sd_difference, rtol=0, atol=1e-9):
        fail_output("REPEATABILITY_COEFFICIENT_GATE: RC != 1,96·SD")
    g["REPEATABILITY_COEFFICIENT_GATE"] = "RC = 1.96 · SD(delta); sem √2"
    if not ((e.icc31 >= -1) & (e.icc31 <= 1)).all():
        fail_output("ICC_DOMAIN_GATE: ICC fora de [-1,1]")
    g["ICC_DOMAIN_GATE"] = "PASS"
    g["ICC31_PARITY_GATE"] = f"PASS (máx dif {g_icc['max_dif_paridade']:.2e})"
    if not (e.bootstrap_valid == B_VALID).all():
        fail_output("BOOTSTRAP_COUNT_GATE: réplicas válidas insuficientes")
    g["BOOTSTRAP_COUNT_GATE"] = f"{B_VALID} válidas em todas as {int(est.sum())} linhas"
    g["PATIENT_BOOTSTRAP_GATE"] = "unidade = patient_id; objectos do paciente entram juntos"
    if not (e.n_patients_wilcoxon <= e.n_patients).all():
        fail_output("WILCOXON_PATIENT_LEVEL_GATE")
    g["WILCOXON_PATIENT_LEVEL_GATE"] = "teste sobre médias por paciente"
    g["HOLM_FAMILY_GATE"] = familias
    g["PROPORTIONAL_BIAS_CLUSTER_GATE"] = "erros agrupados por paciente (CR0 com correcção)"
    se = sc[sc.status == "ESTIMATED"]
    if not ((se.sccr >= 0) & (se.sccr <= 1)).all():
        fail_output("SCCR_DOMAIN_GATE")
    g["SCCR_DOMAIN_GATE"] = "PASS"
    if sorted(sc.category.unique()) != sorted(["Overall"] + CATEGORIAS):
        fail_output("SCCR_FOUR_CATEGORY_GATE")
    g["SCCR_FOUR_CATEGORY_GATE"] = "Overall + lt3mm/3to6mm/6to15mm/ge15mm"
    g["SCCR_BOOTSTRAP_GATE"] = "IC principal = bootstrap por paciente; Wilson só diagnóstico"
    g["RELATIVE_THRESHOLD_GATE"] = "> 0,15 e > 0,25, estrito, só volume e diâmetro"
    g["FILTER_SET_16_GATE"] = f"{len(FEATURES_16)} variáveis; frac_high_log excluída"
    if len(ft) != len(ARQUITECTURAS) * len(FEATURES_16):
        fail_output(f"FEATURE_REPEATABILITY_COMPLETENESS_GATE: {len(ft)} linhas")
    g["FEATURE_REPEATABILITY_COMPLETENESS_GATE"] = f"{len(ft)} linhas = 10×16"
    g["MORPH_VOLUME_ASSOC_GATE"] = ("primária: abs_delta_volume_mm3 × abs_delta_log_max "
                                    "(Spearman); sensibilidade: abs_rel_delta_volume")
    num = st.select_dtypes(include=[np.number])
    nf = int(((~np.isfinite(num.to_numpy())) & (~num.isna().to_numpy())).sum())
    if nf:
        fail_output(f"NONFINITE_GATE: {nf} valores não finitos que não são NaN declarado")
    g["NONFINITE_GATE"] = "PASS"

    figd = os.path.join(out_root, "figdata"); os.makedirs(figd, exist_ok=True)
    prods = {}
    def grava(nome, frame, pasta=out_root):
        p = os.path.join(pasta, nome)
        frame.to_csv(p, index=False)
        prods[os.path.relpath(p, out_root)] = p

    grava("rider_repeatability_stats.csv", st)
    grava("rider_filter_repeatability_stats.csv", ft)
    grava("rider_filter_volume_association_stats.csv", assoc)
    grava("rider_size_transition_matrix.csv", tr)
    grava("rider_sccr.csv", sc)
    grava("rider_relative_thresholds.csv", pr)
    ba = st[["architecture", "measure", "status", "n_pairs", "n_patients", "bias",
             "bias_ci_lo", "bias_ci_hi", "sd_difference", "within_subject_sd",
             "lower_loa", "lower_loa_ci_lo", "lower_loa_ci_hi", "upper_loa",
             "upper_loa_ci_lo", "upper_loa_ci_hi", "repeatability_coefficient"]]
    grava("rider_bland_altman.csv", ba)

    ident = CHAVE
    grava("volumes_RIDER_iou.csv", df[ident + ["volume_ct1_mm3", "volume_ct2_mm3",
                                               "delta_volume_mm3", "abs_rel_delta_volume",
                                               "size_category_ct1", "size_category_ct2"]], figd)
    grava("diametros_RIDER_iou.csv", df[ident + ["diameter_ct1_mm", "diameter_ct2_mm",
                                                 "delta_diameter_mm", "abs_rel_delta_diameter",
                                                 "size_category_ct1", "size_category_ct2"]], figd)
    grava("rider_icc.csv", st[["architecture", "measure", "status", "n_pairs", "n_patients",
                               "icc31", "icc31_ci_lo", "icc31_ci_hi"]], figd)
    grava("rider_sccr.csv", sc, figd)
    cols_ft = ident + [c for f in FEATURES_16 for c in (f"{f}_ct1", f"{f}_ct2", f"delta_{f}")]
    grava("rider_filter_repeatability.csv", df[cols_ft], figd)
    grava("rider_filter_volume_associations.csv",
          df[ident + ["abs_delta_volume_mm3", "abs_rel_delta_volume", "abs_delta_log_max"]], figd)

    pd.DataFrame([{"path": os.path.relpath(p, out_root), "sha256": sha256(p)}
                  for p in prods.values()]).to_csv(
        os.path.join(out_root, "output_hashes.csv"), index=False)
    pd.DataFrame([{"path": os.path.relpath(args.features, BASE), "sha256": h,
                   "script20_run": man20["run_id"],
                   "script20_script_sha256": man20["script_sha256"]}]).to_csv(
        os.path.join(out_root, "input_manifest.csv"), index=False)
    json.dump(g, open(os.path.join(out_root, "gates.json"), "w"), indent=1, default=str)
    man = {
        "script": "21_analyze_rider_repeatability.py", "run_id": run_id,
        "script_sha256": sha256(os.path.abspath(__file__)),
        "run_at": t0.isoformat(), "end_at": datetime.datetime.now().astimezone().isoformat(),
        "source": {"script20_run": man20["run_id"], "features": os.path.relpath(args.features, BASE),
                   "features_sha256": h, "script20_script_sha256": man20["script_sha256"]},
        "definitions": {
            "delta": "CT2 - CT1", "pair_mean": "(CT1 + CT2)/2",
            "repeatability_coefficient": "1.96 * SD(delta)",
            "within_subject_sd": "SD(delta)/sqrt(2)",
            "limits_of_agreement": "bias +/- 1.96 * SD(delta)",
            "icc": "ICC(3,1) two-way mixed, consistency, single measurement",
            "correlation": "Pearson, descriptive only, not an agreement metric",
            "wilcoxon": "on patient-level mean differences; two-sided, zero_method=wilcox, "
                        "correction=False, method=approx",
            "proportional_bias": "delta ~ 1 + pair_mean, OLS point estimate, "
                                 "cluster-robust SE by patient_id",
            "morphology_volume_association": {
                "primary": "spearman(abs_delta_volume_mm3, abs_delta_log_max)",
                "sensitivity": "spearman(abs_rel_delta_volume, abs_delta_log_max)",
                "prespecification": "log_max was the deterministic first log_* variable in the "
                                    "previous Script 21; NOT re-selected after seeing Script 16"},
            "relative_thresholds": "abs_rel_delta > 0.15 and > 0.25, strict, volume and diameter only",
            "size_categories": CATEGORIAS,
            "analytic_features": FEATURES_16,
            "frac_high_log": "EXCLUDED_DEGENERATE_BY_CONSTRUCTION — no Script 21 statistic uses it"},
        "bootstrap": {"requested": B_VALID, "seed": SEED, "unit": "patient_id",
                      "max_attempts": MAX_ATTEMPTS, "per_architecture": tentativas},
        "holm_families": familias,
        "counts": {"pairs": len(df), "patients": int(df.patient_id.nunique()),
                   "architectures_total": len(ARQUITECTURAS),
                   "architectures_estimable": len(ARQ_ESTIMAVEIS),
                   "measures": len(MEDIDAS), "stats_rows": len(st),
                   "filter_stats_rows": len(ft), "sccr_rows": len(sc),
                   "transition_rows": len(tr)},
        "gates": g,
        "promotion": "NOT PROMOTED — os produtos ficam no directório do run até os gates fecharem",
    }
    json.dump(man, open(os.path.join(out_root, "manifest.json"), "w"), indent=2, default=str)
    log.info(f"{len(st)} linhas de estatística · {len(ft)} de características · "
             f"{len(sc)} de SCCR · {len(tr)} de transição")
    log.info(f"Saída: {out_root}")
    log.info("Script 21 concluído.")


if __name__ == "__main__":
    main()
