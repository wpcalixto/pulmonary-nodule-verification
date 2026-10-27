#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    24_integrated_statistics.py

File type:
    Script

Purpose:
    Integrates all results: patient-level bootstrap confidence intervals and Holm correction across comparisons.

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
    - analysis/bd1_object_matches.csv
    - analysis/bd1_detection_metrics.csv
    - analysis/bd1_size_stratified_metrics.csv
    - figdata/filter_bd1_auc.csv
    - lndb_filter_group_comparisons.csv
    - lndb_external_metrics.csv
    - lndb_object_matches.csv
    - rider_repeatability_stats.csv

Outputs:
    - analysis/report_12.json
    - gates.json
    - manifest.json

Dependencies:
    - numpy
    - pandas
    - scipy

Execution:
    PULMONARY_BASE=<project_dir> python scripts/24_integrated_statistics.py

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
from scipy.stats import wilcoxon

BASE = os.environ.get("PULMONARY_BASE") or os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(BASE, "analysis")
CONFIG_DIR = os.path.join(BASE, "config")
LOGS_DIR = os.path.join(BASE, "logs")

SEED = 27
B_BOOT = 2000
LIMIARES = [0.10, 0.25, 0.50]
IOU_PRIMARIO = 0.10
ARQ_10 = ["2D", "2D_REL", "2D_REM", "3D_CF", "3D_CFREM", "3D_F", "3D_FREM",
          "F100_ResEncM", "F100_fullres", "F100_lowres"]
ARQ_REF = ["3D_F", "3D_FREM"]
SUBSETS = ["oof", "internal"]
ROSTER_ESPERADO = {"oof": 718, "internal": 158}
ESTRATOS_COMUNS = [("lt6mm", 0.0, 6.0), ("6to15mm", 6.0, 15.0), ("ge15mm", 15.0, np.inf)]
ESTRATOS_11B = {"small": (3.0, 6.0), "medium": (6.0, 15.0), "large": (15.0, np.inf)}
FEATURES_16 = [
    "log_max", "log_mean", "log_p95", "log_scale_mm",
    "blob_max", "blob_mean", "blob_p95", "blob_scale_mm",
    "vessel_max", "vessel_mean", "vessel_p95", "vessel_scale_mm",
    "contrast_hu", "centroid_peak_dist_mm",
    "pleura_min_dist_mm", "pleura_centroid_dist_mm"]
LNDB_GT_POR_ARQ = 850
LNDB_ARQ = ["3D_F", "3D_FREM"]

os.makedirs(LOGS_DIR, exist_ok=True)
log = logging.getLogger("s24")
log.setLevel(logging.INFO)
if not log.handlers:
    _f = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for _h in (logging.StreamHandler(),
               logging.FileHandler(os.path.join(LOGS_DIR, "integrated_statistics.log"))):
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


ENTRADAS = []


def autoridade(produtor, raiz, run_id_esperado, produtos):
    """Verifica um run a montante nos dois modos de proveniência reais:
    LEGACY_CANONICAL (SHA256SUMS.txt) e MODERN_CANONICAL (manifest.json +
    output_hashes.csv). Se ambos existirem, os dois têm de concordar."""
    if not os.path.isdir(raiz):
        fail_input(f"{produtor}: run canónico ausente: {raiz}")
    run_id = os.path.basename(os.path.normpath(raiz))
    if run_id != run_id_esperado:
        fail_input(f"{produtor}: run_id {run_id} != canónico {run_id_esperado}")

    legado = os.path.join(raiz, "SHA256SUMS.txt")
    moderno = os.path.join(raiz, "output_hashes.csv")
    man = os.path.join(raiz, "manifest.json")
    tem_legado = os.path.isfile(legado)
    tem_moderno = os.path.isfile(moderno) and os.path.isfile(man)
    if not (tem_legado or tem_moderno):
        fail_input(f"{produtor}: sem autoridade de proveniência em {raiz} — "
                   f"nem SHA256SUMS.txt nem manifest.json + output_hashes.csv")
    modo = ("LEGACY_AND_MODERN" if (tem_legado and tem_moderno)
            else ("LEGACY_CANONICAL" if tem_legado else "MODERN_CANONICAL"))
    esperados_legado, esperados_moderno = {}, {}
    if tem_legado:
        for linha in open(legado):
            partes = linha.split(None, 1)
            if len(partes) == 2:
                esperados_legado[partes[1].strip().lstrip("./")] = partes[0]
    if tem_moderno:
        oh = pd.read_csv(moderno)
        esperados_moderno = dict(zip(oh["path"], oh["sha256"]))
        mj = json.load(open(man))
        if mj.get("run_id") != run_id_esperado:
            fail_input(f"{produtor}: run_id do manifest.json != {run_id_esperado}")

    saidas = {}
    for rel in produtos:
        p = os.path.join(raiz, rel)
        if not os.path.isfile(p):
            fail_input(f"{produtor}: produto canónico ausente: {p}")
        h = sha256(p)
        base = os.path.basename(rel)
        esperado = None
        for tabela in (esperados_legado, esperados_moderno):
            for chave in (rel, base, "./" + rel):
                if chave in tabela:
                    if esperado is not None and tabela[chave] != esperado:
                        fail_input(f"{produtor}: as duas autoridades discordam em {rel}")
                    esperado = tabela[chave]
        if esperado is None:
            fail_input(f"{produtor}: {rel} não consta de nenhuma autoridade de hash")
        if h != esperado:
            fail_input(f"{produtor}: SHA de {rel} {h[:12]} != {esperado[:12]} da autoridade")
        ENTRADAS.append({"producer_script": produtor, "canonical_run_id": run_id,
                         "provenance_mode": modo,
                         "authority_file": ("SHA256SUMS.txt" if rel in esperados_legado
                                            or base in esperados_legado else "output_hashes.csv"),
                         "product": rel, "expected_sha256": esperado,
                         "actual_sha256": h, "sha256_verified": True})
        saidas[rel] = p
    log.info(f"  {produtor} {run_id} [{modo}]: {len(produtos)} produtos verificados")
    return saidas


def metricas(tp, fn, fp):
    tot = tp + fn
    s = tp / tot if tot > 0 else 0.0
    pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * pr * s / (pr + s) if (pr + s) > 0 else 0.0
    return s, pr, f1


def bootstrap(tp_v, fn_v, fp_v, b=B_BOOT, semente=SEED):
    """Reamostragem de PACIENTES com reposição, na mesma ordem de consumo do
    gerador que o Script 11 usa: uma extracção de tamanho n por réplica."""
    rng = np.random.default_rng(semente)
    n = len(tp_v)
    s_b = np.empty(b); p_b = np.empty(b); f_b = np.empty(b)
    validas = 0
    for i in range(b):
        idx = rng.integers(0, n, size=n)
        s, pr, f1 = metricas(int(tp_v[idx].sum()), int(fn_v[idx].sum()), int(fp_v[idx].sum()))
        if not (np.isfinite(s) and np.isfinite(pr) and np.isfinite(f1)):
            fail_output("réplica de bootstrap não finita — a convenção do Script 11 "
                        "devolve 0,0 para denominador nulo, nunca NaN")
        s_b[i], p_b[i], f_b[i] = s, pr, f1
        validas += 1
    if validas != b:
        fail_output(f"BOOTSTRAP_COUNT: {validas} réplicas válidas != {b}")
    ic = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return ic(s_b), ic(p_b), ic(f_b), validas


def holm(pvals):
    pv = np.asarray(pvals, float)
    if not np.isfinite(pv).all():
        fail_output("HOLM: p não finito na família — nenhum teste pode desaparecer")
    o = np.argsort(pv); m = len(pv)
    aj = np.minimum(1.0, pv[o] * (m - np.arange(m)))
    aj = np.maximum.accumulate(aj)
    out = np.empty(m); out[o] = aj
    return out


def main():
    ap = argparse.ArgumentParser()
    U = os.path.join(ANALYSIS_DIR, "upstream_canonical")
    ap.add_argument("--s11b-run", default=os.path.join(
        U, "script11_final", "s11B_20260921T164539_27883"))
    ap.add_argument("--s12-run", default=os.path.join(
        U, "script12_final", "s12_20260921T223557_82479"))
    ap.add_argument("--s16-run", default=os.path.join(
        U, "script16_final", "s16_20260923T094823_494840"))
    ap.add_argument("--s21-run", default=os.path.join(
        ANALYSIS_DIR, "rider_repeatability", "s21_20260923T215458_708846"))
    ap.add_argument("--s22-run", default=os.path.join(
        ANALYSIS_DIR, "lndb_external", "s22_20260924T084718_826234"))
    ap.add_argument("--s23-run", default=os.path.join(
        ANALYSIS_DIR, "lndb_filter_validation", "s23_20260924T153750_909396"))
    ap.add_argument("--n-boot", type=int, default=B_BOOT)
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()
    t0 = datetime.datetime.now().astimezone()

    if SEED != 27:
        fail_input(f"SEED_GATE: semente {SEED} != 27")
    if LIMIARES != [0.10, 0.25, 0.50]:
        fail_input(f"IOU_THRESHOLD_GATE: limiares {LIMIARES} != [0.10, 0.25, 0.50]")
    if SUBSETS != ["oof", "internal"]:
        fail_input(f"SUBSET_GATE: subconjuntos {SUBSETS} != ['oof', 'internal']")
    if sorted(ARQ_10) != sorted(["2D", "2D_REL", "2D_REM", "3D_CF", "3D_CFREM", "3D_F",
                                 "3D_FREM", "F100_ResEncM", "F100_fullres", "F100_lowres"]):
        fail_input(f"ARCHITECTURE_GATE: conjunto de arquitecturas alterado")
    if ARQ_REF != ["3D_F", "3D_FREM"]:
        fail_input(f"REFERENCE_PAIR_GATE: par de referência {ARQ_REF} alterado")
    if args.n_boot != B_BOOT:
        log.warning(f"bootstrap com {args.n_boot} réplicas — o run canónico exige {B_BOOT}")

    log.info("Autoridade de entrada:")
    p11 = autoridade("Script11B", args.s11b_run, "s11B_20260921T164539_27883",
                     ["analysis/bd1_object_matches.csv", "analysis/bd1_detection_metrics.csv",
                      "analysis/bd1_size_stratified_metrics.csv",
                      "analysis/bd1_segmentation_metrics.csv"])
    p12 = autoridade("Script12", args.s12_run, "s12_20260921T223557_82479",
                     ["analysis/report_12.json"])
    p16 = autoridade("Script16", args.s16_run, "s16_20260923T094823_494840",
                     ["figdata/filter_bd1_auc.csv"])
    p21 = autoridade("Script21", args.s21_run, "s21_20260923T215458_708846",
                     ["rider_repeatability_stats.csv", "rider_sccr.csv"])
    p22 = autoridade("Script22", args.s22_run, "s22_20260924T084718_826234",
                     ["lndb_external_metrics.csv", "lndb_object_matches.csv"])
    p23 = autoridade("Script23", args.s23_run, "s23_20260924T153750_909396",
                     ["lndb_filter_group_comparisons.csv",
                      "lndb_filter_verifier_metrics.csv"])

    split_p = os.path.join(CONFIG_DIR, "bd1_split.csv")
    if not os.path.isfile(split_p):
        fail_input(f"partição canónica ausente: {split_p}")
    split = pd.read_csv(split_p)
    roster = {"oof": sorted(split.loc[split.split.str.strip().str.lower() == "development",
                                      "patient_id"].str.strip()),
              "internal": sorted(split.loc[split.split.str.strip().str.lower() == "internal",
                                           "patient_id"].str.strip())}
    for s, n in ROSTER_ESPERADO.items():
        if len(roster[s]) != n:
            fail_input(f"ROSTER: {s} tem {len(roster[s])} pacientes != {n}")
    if len(set(roster["oof"]) & set(roster["internal"])):
        fail_input("ROSTER: paciente em dois subconjuntos")
    log.info(f"Roster canónico: oof {len(roster['oof'])}, internal {len(roster['internal'])}")

    obj = pd.read_csv(p11["analysis/bd1_object_matches.csv"])
    det = pd.read_csv(p11["analysis/bd1_detection_metrics.csv"])
    siz = pd.read_csv(p11["analysis/bd1_size_stratified_metrics.csv"])
    auc16 = pd.read_csv(p16["figdata/filter_bd1_auc.csv"])
    comp23 = pd.read_csv(p23["lndb_filter_group_comparisons.csv"])
    met22 = pd.read_csv(p22["lndb_external_metrics.csv"])
    obj22 = pd.read_csv(p22["lndb_object_matches.csv"])
    if sorted(obj.model.unique()) != sorted(ARQ_10):
        fail_input(f"arquitecturas do Script 11B inesperadas: {sorted(obj.model.unique())}")
    for s in SUBSETS:
        extra = set(obj[obj.subset == s].patient_id) - set(roster[s])
        if extra:
            fail_input(f"pacientes fora do roster em {s}: {sorted(extra)[:5]}")

    run_id = f"s24_{t0:%Y%m%dT%H%M%S}_{os.getpid()}"
    out_root = os.path.join(args.out_root or os.path.join(ANALYSIS_DIR, "integrated_statistics"),
                            run_id)
    os.makedirs(out_root, exist_ok=True)
    g = {}

    linhas_a = []
    for arch in ARQ_10:
        for sub in SUBSETS:
            pids = roster[sub]
            s = obj[(obj.model == arch) & (obj.subset == sub)]
            tp = s[s.status == "TP"].groupby("patient_id").size()
            fn = s[s.status == "FN"].groupby("patient_id").size()
            fp = s[s.status == "FP"].groupby("patient_id").size()
            tp_v = np.array([tp.get(p, 0) for p in pids], float)
            fn_v = np.array([fn.get(p, 0) for p in pids], float)
            fp_v = np.array([fp.get(p, 0) for p in pids], float)
            TP, FN, FP = int(tp_v.sum()), int(fn_v.sum()), int(fp_v.sum())
            ref = det[(det.model == arch) & (det.subset == sub)
                      & (np.isclose(det.threshold, IOU_PRIMARIO))]
            if len(ref) != 1:
                fail_input(f"contagens canónicas ausentes para {arch}/{sub}")
            r = ref.iloc[0]
            if (TP, FN, FP) != (int(r.TP), int(r.FN), int(r.FP)):
                fail_output(f"POINT_ESTIMATE_PARITY: {arch}/{sub} dá {TP}/{FN}/{FP} "
                            f"mas o Script 11B regista {int(r.TP)}/{int(r.FN)}/{int(r.FP)}")
            sens, prec, f1 = metricas(TP, FN, FP)
            (sl, sh), (pl, ph), (fl, fh), nv = bootstrap(tp_v, fn_v, fp_v, args.n_boot)
            for nome, est, lo, hi in (("sensitivity", sens, sl, sh),
                                      ("precision", prec, pl, ph), ("f1", f1, fl, fh)):
                if not (lo <= est <= hi):
                    fail_output(f"CI_ORDER: {nome} {est} fora de [{lo}, {hi}] em {arch}/{sub}")
            if nv != args.n_boot:
                fail_output(f"BOOTSTRAP_COUNT: {nv} réplicas válidas != {args.n_boot} pedidas")
            if args.n_boot == B_BOOT:
                for nome, lo, hi, cl, ch in (("sensitivity", sl, sh, "sens_lo", "sens_hi"),
                                             ("precision", pl, ph, "prec_lo", "prec_hi"),
                                             ("f1", fl, fh, "f1_lo", "f1_hi")):
                    if (round(lo, 4) != float(r[cl])) or (round(hi, 4) != float(r[ch])):
                        fail_output(f"CI_PARITY: {nome} de {arch}/{sub} dá "
                                    f"[{round(lo,4)}, {round(hi,4)}] mas o Script 11B "
                                    f"regista [{r[cl]}, {r[ch]}]")
            linhas_a.append({
                "architecture": arch, "subset": sub, "matching_iou": IOU_PRIMARIO,
                "n_patients": len(pids),
                "n_patients_with_objects": int(s.patient_id.nunique()),
                "n_patients_zero_cluster": len(pids) - int(s.patient_id.nunique()),
                "TP": TP, "FN": FN, "FP": FP,
                "sensitivity": sens, "sensitivity_ci_lo": sl, "sensitivity_ci_hi": sh,
                "precision": prec, "precision_ci_lo": pl, "precision_ci_hi": ph,
                "f1": f1, "f1_ci_lo": fl, "f1_ci_hi": fh,
                "fp_per_exam": FP / len(pids),
                "n_boot_requested": args.n_boot, "n_boot_valid": nv,
                "bootstrap_unit": "patient_id", "seed": SEED})
    boot = pd.DataFrame(linhas_a)
    if len(boot) != len(ARQ_10) * len(SUBSETS):
        fail_output(f"bd1_bootstrap_metrics: {len(boot)} != 20 linhas")
    g["BD1_BOOTSTRAP_ROWS"] = f"{len(boot)} = {len(ARQ_10)}×{len(SUBSETS)}"
    g["POINT_ESTIMATE_PARITY_GATE"] = "TP/FN/FP idênticos ao Script 11B em 20/20"
    g["ROSTER_GATE"] = ROSTER_ESPERADO
    g["ZERO_CLUSTER_PATIENTS"] = int(boot.n_patients_zero_cluster.sum())
    if int(boot.n_boot_valid.min()) != args.n_boot or int(boot.n_boot_requested.min()) != args.n_boot:
        fail_output("BOOTSTRAP_COUNT_GATE: réplicas pedidas e válidas não coincidem")
    g["BOOTSTRAP_GATE"] = {"requested": args.n_boot, "valid": int(boot.n_boot_valid.min()),
                           "unit": "patient_id", "seed": SEED}
    g["SEED_GATE"] = SEED
    g["IOU_THRESHOLD_GATE"] = LIMIARES
    g["SUBSET_GATE"] = SUBSETS
    g["CI_PARITY_GATE"] = ("IC reproduzem os do Script 11B a 4 casas em 20/20"
                           if args.n_boot == B_BOOT else "NAO_APLICAVEL — B != 2000")

    bd1 = auc16[(auc16.architecture.isin(ARQ_REF))
                & (auc16.family.isin(["ARCH_A", "ARCH_B"]))].copy()
    bd1["comparison_key"] = np.where(bd1.family == "ARCH_A", "A", "B")
    bd1 = bd1[["comparison_key", "architecture", "feature", "auroc", "p_raw"]]
    bd1["dataset"] = "BD1_OOF"
    ln = comp23[comp23.architecture.isin(ARQ_REF)].copy()
    ln["comparison_key"] = np.where(ln.comparison.str.startswith("A_"), "A", "B")
    ln = ln[["comparison_key", "architecture", "feature", "auroc", "p_raw"]]
    ln["dataset"] = "LNDB"
    cross = pd.concat([bd1, ln], ignore_index=True)
    if set(cross.feature.unique()) != set(FEATURES_16):
        fail_input(f"variáveis comuns != as 16 congeladas: "
                   f"{sorted(set(cross.feature.unique()) ^ set(FEATURES_16))}")
    cross["p_holm_cross_dataset"] = np.nan
    familias = {}
    for k in ("A", "B"):
        sel = cross.comparison_key == k
        if int(sel.sum()) != 64:
            fail_output(f"CROSS_DATASET_{k}: {int(sel.sum())} testes != 64")
        cross.loc[sel, "p_holm_cross_dataset"] = holm(cross.loc[sel, "p_raw"].to_numpy())
        familias[f"CROSS_DATASET_{k}"] = int(sel.sum())
    cross["reject_holm_cross_dataset"] = cross.p_holm_cross_dataset < 0.05
    cross["source_run"] = np.where(cross.dataset == "BD1_OOF",
                                   "s16_20260923T094823_494840",
                                   "s23_20260924T153750_909396")
    cross["source_sha"] = np.where(
        cross.dataset == "BD1_OOF", sha256(p16["figdata/filter_bd1_auc.csv"])[:16],
        sha256(p23["lndb_filter_group_comparisons.csv"])[:16])
    if len(cross) != 128:
        fail_output(f"multiple_comparisons_corrected: {len(cross)} != 128 linhas")
    g["CROSS_DATASET_HOLM_GATE"] = familias
    g["CROSS_DATASET_RAW_P_ONLY"] = "p_holm dos Scripts 16 e 23 nunca reutilizado"

    gt = obj[obj.status.isin(["TP", "FN"])].copy()
    if gt.diam_mm_gt.isna().any():
        fail_input("objecto de referência sem diâmetro no Script 11B")
    linhas_c = []
    for arch in ARQ_10:
        for sub in SUBSETS:
            s = gt[(gt.model == arch) & (gt.subset == sub)]
            total = 0
            for nome, lo, hi in ESTRATOS_COMUNS:
                q = s[(s.diam_mm_gt >= lo) & (s.diam_mm_gt < hi)]
                TP = int(q.tp_010.sum()); n = len(q); FN = n - TP
                total += n
                linhas_c.append({"dataset": "BD1", "architecture": arch, "subset": sub,
                                 "matching_iou": IOU_PRIMARIO, "stratum": nome,
                                 "n_gt": n, "TP": TP, "FN": FN,
                                 "sensitivity": TP / n if n else 0.0})
            if total != len(s):
                fail_output(f"estratos não particionam o GT de {arch}/{sub}: {total} != {len(s)}")
    difs = []
    for arch in ARQ_10:
        for sub in SUBSETS:
            s = gt[(gt.model == arch) & (gt.subset == sub)]
            for rot, (lo, hi) in ESTRATOS_11B.items():
                q = s[(s.diam_mm_gt >= lo) & (s.diam_mm_gt < hi)]
                ref = siz[(siz.model == arch) & (siz.subset == sub)
                          & (siz.size_category == rot) & (np.isclose(siz.threshold, IOU_PRIMARIO))]
                if len(ref) != 1:
                    fail_input(f"tabela estratificada do 11B sem {arch}/{sub}/{rot}")
                r = ref.iloc[0]
                if (len(q), int(q.tp_010.sum())) != (int(r.n_gt), int(r.TP)):
                    difs.append((arch, sub, rot, len(q), int(r.n_gt)))
    if difs:
        fail_output(f"BD1_SIZE_REDERIVATION_PARITY falhou em {len(difs)} casos: {difs[:3]}")
    g["BD1_SIZE_REDERIVATION_PARITY_GATE"] = ("3-6/6-15/>=15 reproduzem o Script 11B "
                                              "em 60/60 combinações")
    sub3 = int((gt.diam_mm_gt < 3.0).sum())
    g["SUB3MM_RECOVERED"] = (f"{sub3} objectos de referência abaixo de 3 mm, ausentes da "
                             f"tabela estratificada histórica, entram agora no estrato <6 mm")
    mapa = {"small_lt6mm": "lt6mm", "medium_6to15mm": "6to15mm", "large_ge15mm": "ge15mm"}
    ln22 = met22[met22.stratum != "Overall"].copy()
    for _, r in ln22.iterrows():
        linhas_c.append({"dataset": "LNDB", "architecture": r.architecture, "subset": "external",
                         "matching_iou": IOU_PRIMARIO, "stratum": mapa[r.stratum],
                         "n_gt": int(r.n_gt), "TP": int(r.n_tp), "FN": int(r.n_fn),
                         "sensitivity": float(r.sensitivity) if pd.notna(r.sensitivity) else 0.0})
    size = pd.DataFrame(linhas_c)
    for arch in LNDB_ARQ:
        n = int(size[(size.dataset == "LNDB") & (size.architecture == arch)].n_gt.sum())
        if n != LNDB_GT_POR_ARQ:
            fail_output(f"LNDb {arch}: estratos somam {n} != {LNDB_GT_POR_ARQ}")
    if len(size) != len(ARQ_10) * len(SUBSETS) * 3 + len(LNDB_ARQ) * 3:
        fail_output(f"size_stratified_integrated: {len(size)} != 66 linhas")
    g["SIZE_INTEGRATED_ROWS"] = f"{len(size)} = 60 BD1 + 6 LNDb"

    linhas_d = []
    for arch in ARQ_10:
        for sub in SUBSETS:
            n_ex = ROSTER_ESPERADO[sub]
            for t in LIMIARES:
                ref = det[(det.model == arch) & (det.subset == sub)
                          & (np.isclose(det.threshold, t))]
                if len(ref) != 1:
                    fail_input(f"contagens canónicas ausentes: {arch}/{sub}/θ={t}")
                r = ref.iloc[0]
                TP, FN, FP = int(r.TP), int(r.FN), int(r.FP)
                sens, prec, f1 = metricas(TP, FN, FP)
                linhas_d.append({"architecture": arch, "subset": sub, "matching_iou": t,
                                 "n_exams": n_ex, "TP": TP, "FN": FN, "FP": FP,
                                 "n_gt": TP + FN, "n_pred": TP + FP,
                                 "sensitivity": sens, "precision": prec, "f1": f1,
                                 "fp_per_exam": FP / n_ex,
                                 "source": "SCRIPT11B_THRESHOLD_AWARE_HUNGARIAN"})
    sens_iou = pd.DataFrame(linhas_d)
    for arch in ARQ_10:
        for sub in SUBSETS:
            s = sens_iou[(sens_iou.architecture == arch) & (sens_iou.subset == sub)]
            if s.n_gt.nunique() != 1:
                fail_output(f"N_GT varia com o limiar em {arch}/{sub} — "
                            f"o emparelhamento não pode mudar o número de objectos de referência")
    if len(sens_iou) != len(ARQ_10) * len(SUBSETS) * len(LIMIARES):
        fail_output(f"sensitivity_iou: {len(sens_iou)} != 60 linhas")
    g["SENSITIVITY_IOU_ROWS"] = f"{len(sens_iou)} = 10×2×3"
    g["NO_THIRD_MATCHER"] = "contagens lidas do Script 11B; nenhum emparelhamento aqui"

    linhas_e = []
    for sub, papel in (("internal", "primary"), ("oof", "secondary")):
        pids = roster[sub]
        taxas = {}
        for arch in ARQ_REF:
            s = obj[(obj.model == arch) & (obj.subset == sub) & (obj.status.isin(["TP", "FN"]))]
            tp = s[s.status == "TP"].groupby("patient_id").size()
            gtn = s.groupby("patient_id").size()
            taxas[arch] = {p: (tp.get(p, 0) / gtn[p]) for p in gtn.index}
        com_gt = sorted(set(taxas[ARQ_REF[0]]) & set(taxas[ARQ_REF[1]]))
        a = np.array([taxas[ARQ_REF[0]][p] for p in com_gt])
        b = np.array([taxas[ARQ_REF[1]][p] for p in com_gt])
        dif = b - a
        nz = int((dif != 0).sum())
        if nz == 0:
            p = float("nan"); est = "NOT_ESTIMABLE_ALL_ZERO_DIFFERENCES"
        else:
            p = float(wilcoxon(a, b, alternative="two-sided", zero_method="wilcox",
                               correction=False, method="approx").pvalue)
            est = "ESTIMATED"
        linhas_e.append({
            "comparison": "3D_F_vs_3D_FREM", "subset": sub, "role": papel,
            "endpoint": "patient_level_detection_rate_TP_over_TP_plus_FN",
            "matching_iou": IOU_PRIMARIO,
            "n_total": len(pids), "n_with_gt": len(com_gt),
            "n_excluded_no_gt": len(pids) - len(com_gt), "n_nonzero_pairs": nz,
            "median_3D_F": float(np.median(a)), "median_3D_FREM": float(np.median(b)),
            "median_difference": float(np.median(dif)),
            "mean_difference": float(np.mean(dif)),
            "wilcoxon_p_raw": p, "status": est})
    ref_arch = pd.DataFrame(linhas_e)
    est_sel = ref_arch.status == "ESTIMATED"
    ref_arch["holm_family_size"] = int(est_sel.sum())
    ref_arch["wilcoxon_p_holm"] = np.nan
    if int(est_sel.sum()):
        ref_arch.loc[est_sel, "wilcoxon_p_holm"] = holm(
            ref_arch.loc[est_sel, "wilcoxon_p_raw"].to_numpy())
    if len(ref_arch) != 2:
        fail_output(f"reference_arch_comparison: {len(ref_arch)} != 2 linhas")
    g["REFERENCE_PAIR_GATE"] = "só 3D_F vs 3D_FREM; nenhuma comparação secundária"
    g["REFERENCE_HOLM_FAMILY"] = int(est_sel.sum())

    r12 = json.load(open(p12["analysis/report_12.json"]))
    s21 = pd.read_csv(p21["rider_repeatability_stats.csv"])
    sccr = pd.read_csv(p21["rider_sccr.csv"])
    v23 = pd.read_csv(p23["lndb_filter_verifier_metrics.csv"])
    if len(s21) != 180 or len(sccr) != 50:
        fail_input(f"produtos do Script 21 inesperados: {len(s21)} e {len(sccr)} linhas")
    if int(obj22[obj22.status.isin(["TP", "FN"])].groupby("architecture").size().min()) != LNDB_GT_POR_ARQ:
        fail_input("o Script 22 não confirma 850 objectos de referência por arquitectura")
    if not (v23.n_exams == 211).all():
        fail_input("o Script 23 não confirma o denominador de 211 exames")
    g["UPSTREAM_CONSISTENCY_GATE"] = {
        "script12": "manifesto lido; nenhuma estatística interobservador recomputada",
        "script21": f"{len(s21)} linhas de repetibilidade e {len(sccr)} de SCCR verificadas",
        "script22": f"{LNDB_GT_POR_ARQ} objectos de referência por arquitectura",
        "script23": "denominador de 211 exames confirmado",
        "recomputation": "NONE"}
    g["NO_INTERMEDIATE_ROUNDING"] = "nenhum round antes de IC, Holm, Wilcoxon ou identidades"

    prods = {}

    def grava(nome, frame):
        p = os.path.join(out_root, nome); frame.to_csv(p, index=False); prods[nome] = p

    grava("bd1_bootstrap_metrics.csv", boot)
    grava("multiple_comparisons_corrected.csv", cross)
    grava("size_stratified_integrated.csv", size)
    grava("sensitivity_iou.csv", sens_iou)
    grava("reference_arch_comparison.csv", ref_arch)
    grava("input_manifest.csv", pd.DataFrame(ENTRADAS))
    json.dump(g, open(os.path.join(out_root, "gates.json"), "w"), indent=1, default=str)
    prods["gates.json"] = os.path.join(out_root, "gates.json")
    pd.DataFrame([{"path": n, "sha256": sha256(p)} for n, p in prods.items()]).to_csv(
        os.path.join(out_root, "output_hashes.csv"), index=False)
    man = {
        "script": "24_integrated_statistics.py", "run_id": run_id,
        "script_sha256": sha256(os.path.abspath(__file__)),
        "run_at": t0.isoformat(), "end_at": datetime.datetime.now().astimezone().isoformat(),
        "sources": {e["producer_script"]: e["canonical_run_id"] for e in ENTRADAS},
        "definitions": {
            "roster": "config/bd1_split.csv — 718 development + 158 internal; exames sem "
                      "objectos permanecem como agregados de contagem zero",
            "bootstrap": {"unit": "patient_id", "B": args.n_boot, "seed": SEED,
                          "null_denominator": "0.0 (convenção do Script 11)"},
            "matching_counts": "Script 11B, Húngaro consciente do limiar; sem novo emparelhamento",
            "common_size_strata": [e[0] for e in ESTRATOS_COMUNS],
            "bd1_lt6_rule": "derivado de diam_mm_gt ao nível do objecto, incluindo os <3 mm "
                            "que a tabela estratificada histórica excluía",
            "cross_dataset_holm": {"families": familias, "input": "p_raw apenas",
                                   "original_holm": "Scripts 16 e 23 permanecem intactos"},
            "reference_comparison": "3D_F vs 3D_FREM, pareado por paciente; "
                                    "INTERNAL primária, OOF secundária",
            "no_pooling": "OOF e INTERNAL nunca agrupados"},
        "counts": {"bootstrap_rows": len(boot), "cross_dataset_rows": len(cross),
                   "size_rows": len(size), "iou_rows": len(sens_iou),
                   "reference_rows": len(ref_arch)},
        "gates": g,
        "promotion": "NOT PROMOTED — produtos ficam no directório do run"}
    json.dump(man, open(os.path.join(out_root, "manifest.json"), "w"), indent=2, default=str)
    log.info(f"{len(boot)} bootstrap · {len(cross)} Holm transversal · {len(size)} estratos · "
             f"{len(sens_iou)} IoU · {len(ref_arch)} comparação de referência")
    log.info(f"Saída: {out_root}")
    log.info("Script 24 concluído.")


if __name__ == "__main__":
    main()
