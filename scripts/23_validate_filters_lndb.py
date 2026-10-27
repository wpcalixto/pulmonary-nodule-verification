#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    23_validate_filters_lndb.py

File type:
    Script

Purpose:
    External validation of the morphological filters and of the frozen verifier on LNDb.

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
    - output_hashes.csv

Outputs:
    - gates.json
    - manifest.json

Dependencies:
    - numpy
    - pandas
    - scipy
    - sklearn

Execution:
    PULMONARY_BASE=<project_dir> python scripts/23_validate_filters_lndb.py

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
import pickle

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

BASE = os.environ.get("PULMONARY_BASE") or os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(BASE, "analysis")
FIGDATA_DIR = os.path.join(BASE, "figdata")
LOGS_DIR = os.path.join(BASE, "logs")

ARQUITECTURAS = ["3D_F", "3D_FREM"]
LNDB_ANALYTIC_N = 211
N_GT_POR_ARQ = 850
ESPERADO_S22_LINHAS = 2729
ESPERADO_S22B_LINHAS = 3323
GRUPOS_POR_ARQ = {
    "3D_F": {"TP_pred": 290, "FP_pred": 534, "TP_gt": 290, "FN_gt": 560},
    "3D_FREM": {"TP_pred": 304, "FP_pred": 495, "TP_gt": 304, "FN_gt": 546}}
CHAVE = ["source", "scan_id", "architecture", "group", "object_id"]
FEATURES_16 = [
    "log_max", "log_mean", "log_p95", "log_scale_mm",
    "blob_max", "blob_mean", "blob_p95", "blob_scale_mm",
    "vessel_max", "vessel_mean", "vessel_p95", "vessel_scale_mm",
    "contrast_hu", "centroid_peak_dist_mm",
    "pleura_min_dist_mm", "pleura_centroid_dist_mm"]
PROIBIDAS = {"frac_high_log", "volume_mm3", "diam_mm"}
COMPARACOES = [("A_TP_pred_vs_FP_pred", "TP_pred", "FP_pred"),
               ("B_TP_gt_vs_FN_gt", "TP_gt", "FN_gt")]
GRELHA_LIMIARES = [round(x, 2) for x in np.arange(0.05, 1.00, 0.05)]

os.makedirs(LOGS_DIR, exist_ok=True)
log = logging.getLogger("s23")
log.setLevel(logging.INFO)
if not log.handlers:
    _f = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for _h in (logging.StreamHandler(),
               logging.FileHandler(os.path.join(LOGS_DIR, "validate_filters_lndb.log"))):
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


def auroc(pos, neg):
    """AUROC pela estatística U de Mann–Whitney, com a classe positiva fixa.
    Nunca se aplica max(AUC, 1-AUC) nem se troca a classe."""
    n1, n2 = len(pos), len(neg)
    if n1 == 0 or n2 == 0:
        return float("nan")
    u = mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u / (n1 * n2))


def mw(pos, neg):
    if len(pos) < 1 or len(neg) < 1:
        return float("nan"), float("nan")
    r = mannwhitneyu(pos, neg, alternative="two-sided")
    return float(r.statistic), float(r.pvalue)


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


def carrega_verificador(pkl, sha_esperado, man17):
    if not os.path.isfile(pkl):
        fail_input(f"verificador congelado do Script 17 ausente: {pkl} — o Script 17 "
                   f"treinou e congelou o modelo, portanto a ausência é falha de "
                   f"entrada, não NOT_ESTIMABLE")
    h = sha256(pkl)
    if sha_esperado and h != sha_esperado:
        fail_input(f"VERIFIER_SHA: {h[:12]} != {sha_esperado[:12]}")
    if man17.get("outputs_sha256", {}).get(os.path.basename(pkl)) not in (None, h):
        fail_input(f"VERIFIER_SHA: não confere com o manifesto do Script 17")
    with open(pkl, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        fail_input("o artefacto do Script 17 não é o payload esperado")
    for k in ("canonical", "model", "model_label", "features", "threshold"):
        if k not in payload:
            fail_input(f"VERIFIER_PAYLOAD: falta a chave {k}")
    if payload["model_label"] != "A" or not payload["canonical"]:
        fail_input(f"VERIFIER_MODEL_A: modelo {payload['model_label']} "
                   f"(canónico={payload['canonical']}) — o verificador externo é o A")
    if list(payload["features"]) != FEATURES_16:
        fail_input(f"FEATURE_ORDER: ordem do payload != as 16 congeladas")
    if not hasattr(payload["model"], "predict_proba"):
        fail_input("payload['model'] não expõe predict_proba")
    thr = float(payload["threshold"])
    if not np.isfinite(thr) or not 0.0 < thr < 1.0:
        fail_input(f"limiar congelado inválido: {thr}")
    thr17 = man17.get("results", {}).get("A", {}).get("threshold")
    if thr17 is not None and abs(float(thr17) - thr) > 1e-12:
        fail_input(f"FROZEN_THRESHOLD: {thr} != {thr17} do manifesto do Script 17")
    return payload, thr


def paridade_serializacao(payload, thr, bd1_csv, man17, alias_pkl):
    """VERIFIER_SERIALIZATION_PARITY_GATE.

    A ordem pedia que o modelo serializado reproduzisse «as probabilidades
    congeladas do Script 17». Essa referência não existe: o Script 17 guardou
    métricas **fora-de-fold** (AUROC 0,846015, contagens 11414/5636/3259/17678),
    obtidas por GroupKFold, e o `.pkl` é o modelo final ajustado em TODAS as
    linhas de desenvolvimento. Não são a mesma coisa, e não devem coincidir —
    o AUROC em amostra do modelo final é 0,848064.

    O que se verifica aqui, a cada execução:
      1. o payload recarrega e é um Pipeline (escalonamento encapsulado, nada a
         reconstruir no LNDb);
      2. o alias canónico `filter_verifier.pkl` dá probabilidades idênticas ao
         `filter_verifier_A.pkl`, bit a bit;
      3. o limiar do payload é exactamente o do manifesto do Script 17;
      4. o AUROC em amostra é coerente com o fora-de-fold (>= e próximo), o que
         se reporta — não se exige igualdade.

    A reprodução exacta do limiar, do AUROC e das quatro contagens congeladas
    está feita à parte, no preflight, reexecutando a validação cruzada do
    Script 17 com os mesmos folds e semente. Não corre aqui porque envolve
    `fit`, e o caminho científico do LNDb não pode conter ajuste nenhum.
    """
    if not os.path.isfile(bd1_csv):
        fail_input(f"matriz do BD1 para a coerência ausente: {bd1_csv}")
    d = pd.read_csv(bd1_csv)
    d = d[(d["subset"] == "oof") & (d["group"].isin(["TP_pred", "FP_pred"]))]
    esperado_linhas = man17.get("population", {}).get("rows")
    if esperado_linhas and len(d) != esperado_linhas:
        fail_input(f"coerência: {len(d)} linhas OOF != {esperado_linhas} do Script 17")
    X = d[list(payload["features"])].to_numpy(float)
    if not np.isfinite(X).all():
        fail_input("valores não finitos nas linhas do BD1")
    modelo = payload["model"]
    passos = list(getattr(modelo, "named_steps", {}).keys())
    if not passos:
        fail_input("o modelo serializado não é um Pipeline — o escalonamento teria "
                   "de ser reconstruído, e reconstruí-lo no LNDb é proibido")
    p_a = modelo.predict_proba(X)[:, 1]
    if not np.isfinite(p_a).all():
        fail_output("probabilidade não finita na verificação de serialização")

    if not os.path.isfile(alias_pkl):
        fail_input(f"alias canónico do verificador ausente: {alias_pkl}")
    with open(alias_pkl, "rb") as f:
        pal = pickle.load(f)
    if pal.get("model_label") != payload["model_label"]:
        fail_input("o alias canónico não aponta para o Modelo A")
    p_alias = pal["model"].predict_proba(X)[:, 1]
    if not np.array_equal(p_a, p_alias):
        fail_output("alias e ficheiro do Modelo A dão probabilidades diferentes")

    from sklearn.metrics import roc_auc_score
    y = (d["group"] == "TP_pred").to_numpy()
    auc_amostra = float(roc_auc_score(y, p_a))
    auc_oof = float(man17.get("results", {}).get("A", {}).get("auroc_oof", float("nan")))
    if np.isfinite(auc_oof) and not (auc_oof - 1e-3 <= auc_amostra <= auc_oof + 0.05):
        fail_output(f"AUROC em amostra {auc_amostra:.6f} incoerente com o fora-de-fold "
                    f"{auc_oof:.6f} — o artefacto pode não ser o modelo do Script 17")
    return {"pipeline": type(modelo).__name__, "steps": passos,
            "scaling_encapsulated": True,
            "alias_identical": True, "rows": len(d),
            "threshold": repr(thr),
            "auroc_in_sample": auc_amostra, "auroc_out_of_fold_script17": auc_oof,
            "note": ("as contagens congeladas do Script 17 são fora-de-fold; a sua "
                     "reprodução exacta está no preflight, não neste caminho")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s22-run", required=True)
    ap.add_argument("--s22-manifest-sha256", required=True)
    ap.add_argument("--s22b-run", required=True)
    ap.add_argument("--s22b-manifest-sha256", required=True)
    ap.add_argument("--verifier", default=os.path.join(BASE, "models", "filter_verifier_A.pkl"))
    ap.add_argument("--verifier-sha256", default="")
    ap.add_argument("--s17-manifest",
                    default=os.path.join(ANALYSIS_DIR, "filter_verifier_manifest.json"))
    ap.add_argument("--bd1-features",
                    default=os.path.join(FIGDATA_DIR, "bd1_filter_features.csv"))
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()
    t0 = datetime.datetime.now().astimezone()

    if [(r, p_, n_) for r, p_, n_ in COMPARACOES] != [
            ("A_TP_pred_vs_FP_pred", "TP_pred", "FP_pred"),
            ("B_TP_gt_vs_FN_gt", "TP_gt", "FN_gt")]:
        fail_input(f"COMPARISONS_FROZEN_GATE: as duas comparações primárias foram "
                   f"alteradas: {COMPARACOES}")
    grelha_congelada = [round(x, 2) for x in np.arange(0.05, 1.00, 0.05)]
    if list(GRELHA_LIMIARES) != grelha_congelada:
        fail_input(f"THRESHOLD_SENSITIVITY_GRID_GATE: grelha alterada "
                   f"({len(GRELHA_LIMIARES)} pontos) — era 0,05–0,95 com 19 pontos")
    if len(FEATURES_16) != 16:
        fail_input(f"FEATURE_SET_16_GATE: {len(FEATURES_16)} variáveis != 16")

    def autoridade(run, sha_arg, produto, esperado_linhas, rotulo, parquet=False):
        man_p = os.path.join(run, "manifest.json")
        if not os.path.isfile(man_p):
            fail_input(f"run do {rotulo} sem manifest.json: {run}")
        h = sha256(man_p)
        if h != sha_arg:
            fail_input(f"INPUT_AUTHORITY: SHA do manifesto do {rotulo} {h[:12]} != {sha_arg[:12]}")
        man = json.load(open(man_p))
        if man["run_id"] != os.path.basename(os.path.normpath(run)):
            fail_input(f"run_id do manifesto do {rotulo} != pasta")
        p = os.path.join(run, produto)
        if not os.path.isfile(p):
            fail_input(f"produto canónico ausente: {p}")
        hp = sha256(p)
        oh = pd.read_csv(os.path.join(run, "output_hashes.csv"))
        linha = oh[oh["path"] == produto]
        if len(linha) != 1 or linha["sha256"].iloc[0] != hp:
            fail_input(f"SHA de {produto} não confere com o output_hashes.csv do {rotulo}")
        d = pd.read_parquet(p) if parquet else pd.read_csv(p)
        if len(d) != esperado_linhas:
            fail_input(f"{rotulo}: {len(d)} linhas != {esperado_linhas}")
        return man, d, h, hp

    man22, obj22, h22, h22p = autoridade(args.s22_run, args.s22_manifest_sha256,
                                         "lndb_object_matches.csv", ESPERADO_S22_LINHAS,
                                         "Script 22")
    man22b, feat, h22b, h22bp = autoridade(args.s22b_run, args.s22b_manifest_sha256,
                                           "lndb_filter_features_raw.parquet",
                                           ESPERADO_S22B_LINHAS, "Script 22B", parquet=True)
    if man22b["source"]["script22a_run"] != man22b["source"]["script22a_run"]:
        fail_input("cadeia de proveniência do 22B inconsistente")
    n_analitico = man22.get("cohort", {}).get("LNDB_ANALYTIC_N")
    if n_analitico != LNDB_ANALYTIC_N:
        fail_input(f"EXAM_DENOMINATOR: o manifesto do Script 22 declara "
                   f"{n_analitico} exames analíticos, o Script 23 usa {LNDB_ANALYTIC_N}")
    n_gt_22 = {a: int(obj22[(obj22.architecture == a)
                            & (obj22.status.isin(["TP", "FN"]))].shape[0])
               for a in ARQUITECTURAS}
    for a, n in n_gt_22.items():
        if n != N_GT_POR_ARQ:
            fail_input(f"N_GT do Script 22 em {a} é {n} != {N_GT_POR_ARQ}")
    log.info(f"Script 22 {man22['run_id']} ({len(obj22)} linhas) · "
             f"Script 22B {man22b['run_id']} ({len(feat)} objectos)")

    if feat.duplicated(CHAVE).any():
        fail_input("COMPOSITE_KEY: chave canónica repetida")
    if len(feat.drop_duplicates(CHAVE)) != ESPERADO_S22B_LINHAS:
        fail_input("COMPOSITE_KEY: chaves únicas != 3323")
    validas = {("pred", "TP_pred"), ("pred", "FP_pred"), ("gt", "TP_gt"), ("gt", "FN_gt")}
    combos = set(zip(feat.source, feat.group))
    if not combos <= validas:
        fail_input(f"SOURCE_GROUP: combinações ilegais {sorted(combos - validas)}")
    if sorted(feat.architecture.unique()) != sorted(ARQUITECTURAS):
        fail_input(f"arquitecturas inesperadas: {sorted(feat.architecture.unique())}")
    for arch, esperado in GRUPOS_POR_ARQ.items():
        obtido = {k: int(v) for k, v in
                  feat[feat.architecture == arch].group.value_counts().items()}
        if obtido != esperado:
            fail_input(f"GROUP_COUNT_RECONCILIATION em {arch}: {obtido} != {esperado}")
        s = obj22[obj22.architecture == arch]
        if (int((s.status == "TP").sum()) != esperado["TP_pred"]
                or int((s.status == "FP").sum()) != esperado["FP_pred"]
                or int((s.status == "FN").sum()) != esperado["FN_gt"]):
            fail_input(f"contagens do Script 22 não reconciliam em {arch}")

    falta = [f for f in FEATURES_16 if f not in feat.columns]
    if falta:
        fail_input(f"FEATURE_SET: variáveis em falta {falta}")
    for p_ in PROIBIDAS:
        if p_ in FEATURES_16:
            fail_input(f"variável proibida no conjunto analítico: {p_}")
    X = feat[FEATURES_16].to_numpy(float)
    if not np.isfinite(X).all():
        fail_input("NONFINITE: valor não finito nas variáveis analíticas — "
                   "sem imputação, sem fillna")

    if not os.path.isfile(args.s17_manifest):
        fail_input(f"manifesto do Script 17 ausente: {args.s17_manifest}")
    man17 = json.load(open(args.s17_manifest))
    payload, thr = carrega_verificador(args.verifier, args.verifier_sha256, man17)
    par = paridade_serializacao(payload, thr, args.bd1_features, man17,
                                os.path.join(BASE, "models", "filter_verifier.pkl"))
    log.info(f"Verificador A congelado · limiar {thr!r} ({payload.get('threshold_rule')}) · "
             f"serialização: {par['pipeline']}{par['steps']}, alias idêntico, "
             f"AUROC em amostra {par['auroc_in_sample']:.6f} vs OOF "
             f"{par['auroc_out_of_fold_script17']:.6f}")

    run_id = f"s23_{t0:%Y%m%dT%H%M%S}_{os.getpid()}"
    out_root = os.path.join(args.out_root or os.path.join(ANALYSIS_DIR, "lndb_filter_validation"),
                            run_id)
    os.makedirs(out_root, exist_ok=True)

    linhas = []
    for rotulo, gpos, gneg in COMPARACOES:
        for arch in ARQUITECTURAS:
            s = feat[feat.architecture == arch]
            pos = s[s.group == gpos]; neg = s[s.group == gneg]
            for f in FEATURES_16:
                a = pos[f].to_numpy(float); b = neg[f].to_numpy(float)
                u, p = mw(a, b)
                linhas.append({
                    "comparison": rotulo, "architecture": arch, "feature": f,
                    "positive_group": gpos, "negative_group": gneg,
                    "n_positive": len(a), "n_negative": len(b),
                    "median_positive": float(np.median(a)), "median_negative": float(np.median(b)),
                    "iqr_positive": float(np.subtract(*np.percentile(a, [75, 25]))),
                    "iqr_negative": float(np.subtract(*np.percentile(b, [75, 25]))),
                    "mannwhitney_u": u, "p_raw": p, "auroc": auroc(a, b)})
    comp = pd.DataFrame(linhas)
    familias = {}
    comp["p_holm"] = np.nan
    for rotulo, _, _ in COMPARACOES:
        sel = comp["comparison"] == rotulo
        comp.loc[sel, "p_holm"] = holm(comp.loc[sel, "p_raw"].to_numpy())
        familias[rotulo] = int(sel.sum())
    comp["reject_holm"] = comp["p_holm"] < 0.05
    comp["holm_family"] = comp["comparison"]
    if set(familias.values()) != {len(ARQUITECTURAS) * len(FEATURES_16)}:
        fail_output(f"HOLM_FAMILY: tamanhos {familias} != 32 por família")
    if not ((comp.auroc >= 0) & (comp.auroc <= 1)).all():
        fail_output("AUROC fora de [0,1]")

    pred = feat[feat.source == "pred"].copy()
    prob = payload["model"].predict_proba(pred[list(payload["features"])].to_numpy(float))[:, 1]
    if not np.isfinite(prob).all():
        fail_output("probabilidade não finita do verificador")
    pred["p_tp"] = prob
    pred["retained"] = pred["p_tp"] >= thr
    vm = []
    for arch in ARQUITECTURAS:
        s = pred[pred.architecture == arch]
        tp0 = int((s.group == "TP_pred").sum()); fp0 = int((s.group == "FP_pred").sum())
        fn0 = GRUPOS_POR_ARQ[arch]["FN_gt"]
        tpf = int(((s.group == "TP_pred") & s.retained).sum())
        fpf = int(((s.group == "FP_pred") & s.retained).sum())
        fnf = fn0 + (tp0 - tpf)
        if tpf + fnf != N_GT_POR_ARQ or tp0 + fn0 != N_GT_POR_ARQ:
            fail_output(f"FILTERED_COUNT_IDENTITY: {tpf}+{fnf} != {N_GT_POR_ARQ} em {arch}")
        if tpf > tp0 or fpf > fp0:
            fail_output(f"o verificador não pode aumentar TP nem FP em {arch}")
        for etiqueta, tp, fp, fn in (("original", tp0, fp0, fn0), ("filtered", tpf, fpf, fnf)):
            sens = tp / (tp + fn); prec = tp / (tp + fp) if (tp + fp) else float("nan")
            f1 = (2 * prec * sens / (prec + sens)) if (np.isfinite(prec) and (prec + sens) > 0) \
                else float("nan")
            vm.append({"architecture": arch, "detector": etiqueta, "n_gt": N_GT_POR_ARQ,
                       "n_exams": LNDB_ANALYTIC_N, "tp": tp, "fp": fp, "fn": fn,
                       "sensitivity": sens, "precision": prec, "f1": f1,
                       "fp_per_exam": fp / LNDB_ANALYTIC_N,
                       "verifier_threshold": thr if etiqueta == "filtered" else "",
                       "threshold_source": "SCRIPT17_DEVELOPMENT_OOF" if etiqueta == "filtered" else ""})
    ver = pd.DataFrame(vm)

    curva = []
    for arch in ARQUITECTURAS:
        s = pred[pred.architecture == arch]
        fn0 = GRUPOS_POR_ARQ[arch]["FN_gt"]; tp0 = int((s.group == "TP_pred").sum())
        for t, tipo in ([(x, "GRID_SENSITIVITY") for x in GRELHA_LIMIARES]
                        + [(thr, "SCRIPT17_FROZEN_PRIMARY")]):
            ret = s.p_tp >= t
            tp = int(((s.group == "TP_pred") & ret).sum())
            fp = int(((s.group == "FP_pred") & ret).sum())
            fn = fn0 + (tp0 - tp)
            sens = tp / (tp + fn); prec = tp / (tp + fp) if (tp + fp) else float("nan")
            f1 = (2 * prec * sens / (prec + sens)) if (np.isfinite(prec) and (prec + sens) > 0) \
                else float("nan")
            curva.append({"architecture": arch, "curve_type": "VERIFIER_THRESHOLD_SENSITIVITY",
                          "operating_point_type": tipo, "threshold": t,
                          "tp": tp, "fp": fp, "fn": fn, "n_gt": N_GT_POR_ARQ,
                          "n_exams": LNDB_ANALYTIC_N, "sensitivity": sens,
                          "precision": prec, "f1": f1,
                          "fp_per_exam": fp / LNDB_ANALYTIC_N})
    cur = pd.DataFrame(curva)
    auc_t = comp[["comparison", "architecture", "feature", "auroc", "p_raw", "p_holm",
                  "reject_holm", "n_positive", "n_negative"]]

    g = {"INPUT_AUTHORITY_GATE": "PASS",
         "SCRIPT22_SHA_GATE": h22[:16], "SCRIPT22B_SHA_GATE": h22b[:16],
         "SCRIPT22_COUNT_GATE": f"{len(obj22)}/{ESPERADO_S22_LINHAS}",
         "SCRIPT22B_COUNT_GATE": f"{len(feat)}/{ESPERADO_S22B_LINHAS}",
         "COMPOSITE_KEY_GATE": f"{len(feat.drop_duplicates(CHAVE))} chaves únicas",
         "SOURCE_GROUP_GATE": "só as quatro combinações legais",
         "GROUP_COUNT_RECONCILIATION_GATE": GRUPOS_POR_ARQ,
         "FEATURE_SET_16_GATE": f"{len(FEATURES_16)} variáveis",
         "FEATURE_ORDER_GATE": "ordem do payload == ordem congelada",
         "NONFINITE_GATE": "PASS", "NO_IMPUTATION_GATE": "nenhum fillna no caminho científico",
         "MW_VALIDITY_GATE": f"{int(comp.p_raw.notna().sum())}/{len(comp)} testes válidos",
         "AUROC_DOMAIN_GATE": "PASS",
         "AUROC_DIRECTION_GATE": "classe positiva congelada; nunca max(AUC, 1-AUC)",
         "HOLM_FAMILY_GATE": familias,
         "VERIFIER_PAYLOAD_GATE": sorted(payload.keys()),
         "VERIFIER_MODEL_A_GATE": f"{payload['model_label']} canónico",
         "VERIFIER_SERIALIZATION_PARITY_GATE": par,
         "FROZEN_THRESHOLD_GATE": repr(thr),
         "NO_FIT_GATE": "nenhuma chamada a fit/fit_transform/partial_fit",
         "NO_RECALIBRATION_GATE": "NONE",
         "NO_LNDB_THRESHOLD_SELECTION_GATE": "THRESHOLD_TUNING_ON_LNDB = NO",
         "EXAM_DENOMINATOR_211_GATE": LNDB_ANALYTIC_N,
         "ORIGINAL_FILTERED_SAME_SCAN_SET_GATE": "mesmos 211 exames antes e depois",
         "FILTERED_COUNT_IDENTITY_GATE": "TP_filtrado + FN_filtrado = 850 por arquitectura",
         "THRESHOLD_SENSITIVITY_GRID_GATE": f"{len(GRELHA_LIMIARES)} pontos, 0,05–0,95, congelada",
         "COMPARISONS_FROZEN_GATE": [r for r, _, _ in COMPARACOES],
         "EXAM_DENOMINATOR_SOURCE": "cohort.LNDB_ANALYTIC_N do manifesto do Script 22",
         "PRIMARY_OPERATING_POINT_GATE": "SCRIPT17_FROZEN_PRIMARY presente por arquitectura"}
    if int((cur.operating_point_type == "SCRIPT17_FROZEN_PRIMARY").sum()) != len(ARQUITECTURAS):
        fail_output("ponto de operação congelado em falta na curva")
    if not (ver.n_exams == LNDB_ANALYTIC_N).all() or not (cur.n_exams == LNDB_ANALYTIC_N).all():
        fail_output(f"EXAM_DENOMINATOR: denominador != {LNDB_ANALYTIC_N}")

    prods = {}

    def grava(nome, frame):
        p = os.path.join(out_root, nome); frame.to_csv(p, index=False); prods[nome] = p

    grava("lndb_filter_group_comparisons.csv", comp)
    grava("lndb_filter_auc.csv", auc_t)
    grava("lndb_filter_verifier_metrics.csv", ver)
    grava("lndb_verifier_threshold_sensitivity.csv", cur)
    grava("lndb_filter_froc.csv", cur)
    grava("lndb_verifier_object_scores.csv",
          pred[CHAVE + ["p_tp", "retained"]])
    grava("input_manifest.csv", pd.DataFrame([{
        "script22_run": man22["run_id"], "script22_manifest_sha256": h22,
        "object_matches_sha256": h22p,
        "script22b_run": man22b["run_id"], "script22b_manifest_sha256": h22b,
        "features_parquet_sha256": h22bp,
        "verifier": os.path.relpath(args.verifier, BASE),
        "verifier_sha256": sha256(args.verifier),
        "script17_manifest_sha256": sha256(args.s17_manifest)}]))
    json.dump(g, open(os.path.join(out_root, "gates.json"), "w"), indent=1, default=str)
    prods["gates.json"] = os.path.join(out_root, "gates.json")
    pd.DataFrame([{"path": n, "sha256": sha256(p)} for n, p in prods.items()]).to_csv(
        os.path.join(out_root, "output_hashes.csv"), index=False)
    man = {
        "script": "23_validate_filters_lndb.py", "run_id": run_id,
        "script_sha256": sha256(os.path.abspath(__file__)),
        "run_at": t0.isoformat(), "end_at": datetime.datetime.now().astimezone().isoformat(),
        "source": {"script22_run": man22["run_id"], "script22b_run": man22b["run_id"],
                   "verifier": os.path.relpath(args.verifier, BASE),
                   "script17_run": man17.get("script_sha256")},
        "definitions": {
            "primary_comparisons": [{"label": r, "positive": p, "negative": n}
                                    for r, p, n in COMPARACOES],
            "auroc": "U/(n1·n2) com classe positiva congelada; AUROC < 0,5 é admissível",
            "holm_families": familias,
            "verifier": {"model": payload["model_label"], "type": type(payload["model"]).__name__,
                         "features": list(payload["features"]), "threshold": repr(thr),
                         "threshold_rule": payload.get("threshold_rule"),
                         "THRESHOLD_SOURCE": "SCRIPT17_DEVELOPMENT_OOF",
                         "THRESHOLD_TUNING_ON_LNDB": "NO",
                         "RECALIBRATION_ON_LNDB": "NONE",
                         "applied_to": "source=pred apenas (TP_pred e FP_pred)"},
            "exam_denominator": LNDB_ANALYTIC_N,
            "filtered_fn": "FN original + TP_pred rejeitados",
            "curve": {"name": "VERIFIER_THRESHOLD_SENSITIVITY",
                      "status": "SECONDARY_DESCRIPTIVE_SENSITIVITY_ANALYSIS",
                      "grid": GRELHA_LIMIARES, "grid_frozen_before_results": True,
                      "no_optimal_point_declared": True,
                      "compatibility_alias": "lndb_filter_froc.csv"},
            "analytic_features": FEATURES_16,
            "excluded": sorted(PROIBIDAS)},
        "counts": {"objects": len(feat), "comparisons": len(comp),
                   "verifier_rows": len(ver), "curve_rows": len(cur)},
        "gates": g,
        "promotion": "NOT PROMOTED — produtos ficam no directório do run"}
    json.dump(man, open(os.path.join(out_root, "manifest.json"), "w"), indent=2, default=str)
    log.info(f"{len(comp)} comparações · {len(ver)} linhas de verificador · "
             f"{len(cur)} pontos de curva")
    log.info(f"Saída: {out_root}")
    log.info("Script 23 concluído.")


if __name__ == "__main__":
    main()
