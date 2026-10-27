#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    22_evaluate_lndb.py

File type:
    Script

Purpose:
    External evaluation of the nnU-Net configurations on LNDb without retraining (matching, TP, FP, FN and derived metrics).

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
    - input_manifest.csv
    - gates.json
    - manifest.json

Dependencies:
    - SimpleITK
    - numpy
    - pandas
    - scipy

Execution:
    PULMONARY_BASE=<project_dir> python scripts/22_evaluate_lndb.py

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
import importlib.util
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import scipy.ndimage as nd
import SimpleITK as sitk
from scipy.optimize import linear_sum_assignment

BASE = os.environ.get("PULMONARY_BASE") or os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.join(BASE, "analysis")
LOGS_DIR = os.path.join(BASE, "logs")
CONSENSUS_DIR = os.path.join(BASE, "annotations", "consensus", "lndb")
PRED_DIR = os.path.join(BASE, "predictions", "lndb")
MANIFESTO_LNDB = os.path.join(BASE, "config", "lndb_annotations_manifest.json")

ARQUITECTURAS = ["3D_F", "3D_FREM"]
PRIMARY_MATCHING_IOU = 0.10
LIMIARES_IOU = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75]
LNDB_NOMINAL_N = 236
LNDB_ANALYTIC_N = 211
ESTADOS_ANALITICOS = ("ANALYTIC_UNIT", "ANALYTIC_REPRESENTATIVE")
ESTADOS_EXCLUIDOS = {"EXCLUDED_REFERENCE_READER_DISCORDANCE": 13,
                     "EXCLUDED_DUPLICATE_IMAGE_PRESERVED": 12}
ESTRATOS = [("small_lt6mm", 0.0, 6.0), ("medium_6to15mm", 6.0, 15.0),
            ("large_ge15mm", 15.0, float("inf"))]
STRUCT_6 = nd.generate_binary_structure(3, 1)

os.makedirs(LOGS_DIR, exist_ok=True)
log = logging.getLogger("s22")
log.setLevel(logging.INFO)
if not log.handlers:
    _f = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for _h in (logging.StreamHandler(),
               logging.FileHandler(os.path.join(LOGS_DIR, "evaluate_lndb.log"))):
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


def _carregar(nome, ficheiro):
    p = os.path.join(SCRIPTS_DIR, ficheiro)
    if not os.path.isfile(p):
        fail_input(f"módulo canónico ausente: {p}")
    sp = importlib.util.spec_from_file_location(nome, p)
    m = importlib.util.module_from_spec(sp)
    sys.modules[nome] = m
    sp.loader.exec_module(m)
    return m, sha256(p)


S11, SHA11 = _carregar("s11_canon", "11_match_and_score_bd1.py")
if not np.array_equal(S11.STRUCT_6, STRUCT_6):
    fail_input("STRUCT_6 do Script 11 diverge da 6-conectividade explícita")


def emparelhar_por_limiar(mat, n_gt, n_pred, theta):
    """Emparelhador canónico do Script 11 (linhas 259-261), copiado à letra.

    Só arestas com IoU >= θ são admissíveis; o peso `1000 + IoU` maximiza o
    NÚMERO de pares válidos e desempata pela soma de IoU. É diferente de
    «Húngaro global seguido de filtro», que pode gastar uma atribuição numa
    aresta inadmissível e perder um par válido."""
    if n_gt == 0 or n_pred == 0:
        return set()
    w = np.where(mat >= theta, 1000.0 + mat, 0.0)
    rows, cols = linear_sum_assignment(-w)
    return {(int(r), int(c)) for r, c in zip(rows, cols) if mat[r, c] >= theta}


def emparelhar_global_depois_filtrar(mat, n_gt, n_pred, theta):
    """A forma NÃO canónica — existe só para o contra-exemplo do gate."""
    if n_gt == 0 or n_pred == 0:
        return set()
    rows, cols = linear_sum_assignment(-mat)
    return {(int(r), int(c)) for r, c in zip(rows, cols) if mat[r, c] >= theta}


def encaixa_no_referencial(pred_img, ref_img, rotulo):
    """A predição vive na grelha recortada do Script 06, que é um SUB-BLOCO
    alinhado a voxel da grelha da referência: mesma direcção, mesmo espaçamento
    e deslocamento de índice inteiro. Devolve o deslocamento (z0, y0, x0).

    Não há reamostragem nem interpolação: é um encaixe por índice. Qualquer
    nódulo de referência fora do recorte pulmonar fica, honestamente, FN."""
    if not np.allclose(pred_img.GetSpacing(), ref_img.GetSpacing(), rtol=0, atol=1e-6):
        fail_input(f"NATIVE_GEOMETRY: espaçamento != referência em {rotulo}")
    if not np.allclose(pred_img.GetDirection(), ref_img.GetDirection(), rtol=0, atol=1e-6):
        fail_input(f"NATIVE_GEOMETRY: direcção != referência em {rotulo}")
    ci = np.array(ref_img.TransformPhysicalPointToContinuousIndex(pred_img.GetOrigin()))
    if np.abs(ci - np.round(ci)).max() > 1e-3:
        fail_input(f"NATIVE_GEOMETRY: deslocamento de índice não inteiro em {rotulo}: {ci}")
    o = np.round(ci).astype(int)
    sz = np.array(pred_img.GetSize()); rs = np.array(ref_img.GetSize())
    if not ((o >= 0).all() and (o + sz <= rs).all()):
        fail_input(f"NATIVE_GEOMETRY: recorte fora da referência em {rotulo}")
    return int(o[2]), int(o[1]), int(o[0])


def le_mascara(caminho, rotulo, ref=None):
    if not os.path.isfile(caminho):
        fail_input(f"{rotulo} ausente: {caminho}")
    img = sitk.ReadImage(caminho)
    arr = sitk.GetArrayFromImage(img)
    if not np.isin(arr, (0, 1)).all():
        fail_input(f"{rotulo} não binária: {caminho}")
    sp = img.GetSpacing()
    if not all(np.isfinite(sp)) or min(sp) <= 0:
        fail_input(f"espaçamento inválido {sp} em {caminho}")
    binario = (arr > 0).astype(np.uint8)
    rotulado, n = nd.label(binario, structure=STRUCT_6)
    return img, binario, rotulado, int(n), sp


def categoria_lndb(d):
    if not np.isfinite(d) or d <= 0:
        fail_output(f"diâmetro inválido: {d}")
    for nome, lo, hi in ESTRATOS:
        if lo <= d < hi:
            return nome
    fail_output(f"diâmetro {d} não cai em nenhum estrato congelado — "
                f"a partição {[e[0] for e in ESTRATOS]} não é exaustiva")


def dice_componentes(a, b):
    s = int(a.sum()) + int(b.sum())
    return (2.0 * float(np.logical_and(a, b).sum()) / s) if s > 0 else 0.0


def avalia(scan_id, arch, gt_path, pred_path):
    gt_img, gt_bin, gt_lab, n_gt, sp = le_mascara(gt_path, f"referência {scan_id}")
    pr_img, pr_bin_c, pr_lab_c, n_pred, _ = le_mascara(pred_path, f"predição {arch}/{scan_id}")
    z0, y0, x0 = encaixa_no_referencial(pr_img, gt_img, f"{arch}/{scan_id}")
    Z, Y, X = pr_lab_c.shape
    pr_lab = np.zeros(gt_lab.shape, np.int32)
    pr_lab[z0:z0 + Z, y0:y0 + Y, x0:x0 + X] = pr_lab_c
    dentro = np.zeros(gt_lab.shape, bool)
    dentro[z0:z0 + Z, y0:y0 + Y, x0:x0 + X] = True
    gt_fora_vox = int((gt_bin.astype(bool) & ~dentro).sum())
    gt_fora_comp = sum(1 for c in range(1, n_gt + 1) if not ((gt_lab == c) & dentro).any())
    mat = S11.build_iou_matrix(gt_lab, pr_lab, n_gt, n_pred)
    pares = {t: emparelhar_por_limiar(mat, n_gt, n_pred, t) for t in LIMIARES_IOU}

    contagens = {}
    for t, s in pares.items():
        if len({r for r, _ in s}) != len(s) or len({c for _, c in s}) != len(s):
            fail_output(f"emparelhamento não um-para-um: {scan_id}/{arch}/θ={t}")
        if any(mat[r, c] < t for r, c in s):
            fail_output(f"TP retido com IoU < θ: {scan_id}/{arch}/θ={t}")
        tp = len(s)
        contagens[t] = (tp, n_gt - tp, n_pred - tp)
        if tp + (n_gt - tp) != n_gt or tp + (n_pred - tp) != n_pred:
            fail_output(f"identidade aritmética violada: {scan_id}/{arch}/θ={t}")

    principal = pares[PRIMARY_MATCHING_IOU]
    gt_de_pred = {c: r for r, c in principal}
    pred_de_gt = {r: c for r, c in principal}
    linhas = []
    for gi in range(n_gt):
        gc = (gt_lab == gi + 1)
        d_gt = S11.diam_mm(gc.astype(np.uint8), sp)
        base = {"scan_id": scan_id, "architecture": arch,
                "gt_component_id": gi + 1, "diam_mm_gt": d_gt,
                "size_category_gt": categoria_lndb(d_gt),
                "n_voxels_gt": int(gc.sum())}
        if gi in pred_de_gt:
            pj = pred_de_gt[gi]
            pc = (pr_lab == pj + 1)
            d_pr = S11.diam_mm(pc.astype(np.uint8), sp)
            linhas.append({**base, "pred_component_id": pj + 1, "status": "TP",
                           "iou": float(mat[gi, pj]), "dice": dice_componentes(gc, pc),
                           "diam_mm_pred": d_pr, "size_category_pred": categoria_lndb(d_pr),
                           "n_voxels_pred": int(pc.sum())})
        else:
            linhas.append({**base, "pred_component_id": "", "status": "FN",
                           "iou": 0.0, "dice": 0.0, "diam_mm_pred": float("nan"),
                           "size_category_pred": "", "n_voxels_pred": ""})
    for pj in range(n_pred):
        if pj in gt_de_pred:
            continue
        pc = (pr_lab == pj + 1)
        d_pr = S11.diam_mm(pc.astype(np.uint8), sp)
        linhas.append({"scan_id": scan_id, "architecture": arch, "gt_component_id": "",
                       "diam_mm_gt": float("nan"), "size_category_gt": "",
                       "n_voxels_gt": "", "pred_component_id": pj + 1, "status": "FP",
                       "iou": 0.0, "dice": float("nan"), "diam_mm_pred": d_pr,
                       "size_category_pred": categoria_lndb(d_pr),
                       "n_voxels_pred": int(pc.sum())})
    return linhas, contagens, n_gt, n_pred, (z0, y0, x0), gt_fora_vox, gt_fora_comp


NA = "NOT_APPLICABLE_FP_HAS_NO_GT_SIZE"


def metricas(obj, arch, n_exames):
    sub = obj[obj.architecture == arch]
    linhas = []
    tp = sub[sub.status == "TP"]; fn = sub[sub.status == "FN"]; fp = sub[sub.status == "FP"]
    n_tp, n_fn, n_fp = len(tp), len(fn), len(fp)
    n_gt = n_tp + n_fn
    sens = n_tp / n_gt if n_gt else float("nan")
    prec = n_tp / (n_tp + n_fp) if (n_tp + n_fp) else float("nan")
    f1 = (2 * prec * sens / (prec + sens)) if (np.isfinite(prec) and np.isfinite(sens)
                                               and (prec + sens) > 0) else float("nan")
    soma_dice = float(tp["dice"].sum())
    linhas.append({
        "architecture": arch, "stratum": "Overall", "n_gt": n_gt, "n_tp": n_tp,
        "n_fn": n_fn, "n_fp": n_fp, "n_exams": n_exames,
        "sensitivity": sens, "precision": prec, "f1": f1,
        "dice_global": soma_dice / n_gt if n_gt else float("nan"),
        "dice_cond": float(tp["dice"].mean()) if n_tp else float("nan"),
        "fp_per_exam": n_fp / n_exames,
        "status_precision": "ESTIMATED", "status_f1": "ESTIMATED",
        "status_fp_per_exam": "ESTIMATED"})
    for nome, _, _ in ESTRATOS:
        t = tp[tp.size_category_gt == nome]; f = fn[fn.size_category_gt == nome]
        ngt = len(t) + len(f)
        sd = float(t["dice"].sum())
        linhas.append({
            "architecture": arch, "stratum": nome, "n_gt": ngt, "n_tp": len(t),
            "n_fn": len(f), "n_fp": "", "n_exams": n_exames,
            "sensitivity": len(t) / ngt if ngt else float("nan"),
            "precision": float("nan"), "f1": float("nan"),
            "dice_global": sd / ngt if ngt else float("nan"),
            "dice_cond": float(t["dice"].mean()) if len(t) else float("nan"),
            "fp_per_exam": float("nan"),
            "status_precision": NA, "status_f1": NA, "status_fp_per_exam": NA})
    return linhas


def gate_emparelhador():
    """(1) paridade com o código canónico do Script 11, extraído do FICHEIRO;
    (2) contra-exemplo em que «global + filtro» perde um par válido."""
    fonte = open(os.path.join(SCRIPTS_DIR, "11_match_and_score_bd1.py")).read()
    marca = "w = np.where(mat >= t, 1000.0 + mat, 0.0)"
    if marca not in fonte:
        fail_output("bloco canónico de emparelhamento não encontrado no Script 11")
    bloco = fonte[fonte.index(marca):]
    bloco = bloco[:bloco.index("\n", bloco.index("tp_by_theta[t] = "))]
    bloco = "\n".join(l.strip() for l in bloco.splitlines())

    rng = np.random.default_rng(27)
    difs = 0
    for _ in range(500):
        ng, npd = int(rng.integers(1, 6)), int(rng.integers(1, 6))
        mat = rng.random((ng, npd))
        mat[mat < 0.4] = 0.0
        for t in LIMIARES_IOU:
            ns = {"np": np, "linear_sum_assignment": linear_sum_assignment,
                  "mat": mat, "t": t, "tp_by_theta": {}}
            exec(bloco, ns)
            if ns["tp_by_theta"][t] != emparelhar_por_limiar(mat, ng, npd, t):
                difs += 1
    ce = np.array([[0.99, 0.41], [0.40, 0.00]])
    t = 0.40
    canon = emparelhar_por_limiar(ce, 2, 2, t)
    naive = emparelhar_global_depois_filtrar(ce, 2, 2, t)
    return {"paridade_com_script11_divergencias": int(difs),
            "paridade_com_script11": difs == 0,
            "contra_exemplo_canonico": sorted(canon),
            "contra_exemplo_global_depois_filtro": sorted(naive),
            "canonico_retem_mais": len(canon) > len(naive),
            "matriz_do_contra_exemplo": ce.tolist(), "theta": t}


def gate_conectividade():
    a = np.zeros((9, 9, 9), np.uint8)
    a[2:4, 2:4, 2:4] = 1; a[4:6, 4:6, 4:6] = 1
    b = np.zeros((9, 9, 9), np.uint8)
    b[2:4, 2:4, 2:4] = 1; b[2:4, 4:6, 4:6] = 1
    n6a = int(nd.label(a, structure=STRUCT_6)[1]); n26a = int(nd.label(a, structure=np.ones((3, 3, 3)))[1])
    n6b = int(nd.label(b, structure=STRUCT_6)[1]); n26b = int(nd.label(b, structure=np.ones((3, 3, 3)))[1])
    return {"vertice_6": n6a, "vertice_26": n26a, "aresta_6": n6b, "aresta_26": n26b,
            "distintos_com_6": n6a == 2 and n6b == 2,
            "fundidos_com_26": n26a == 1 and n26b == 1}


def gate_diametro():
    """A definição canónica é o maior diâmetro axial por bbox de corte, do
    Script 11 — nunca a esfera equivalente."""
    m = np.zeros((6, 12, 12), np.uint8)
    m[2, 3:9, 4:6] = 1
    sp = (1.0, 2.0, 3.0)
    axial = S11.diam_mm(m, sp)
    vol = int(m.sum()) * sp[0] * sp[1] * sp[2]
    esfera = (6.0 * vol / np.pi) ** (1 / 3)
    return {"axial_bbox_mm": axial, "esperado": 6 * sp[1],
            "esfera_equivalente_mm": esfera,
            "correcto": abs(axial - 6 * sp[1]) < 1e-9,
            "difere_da_esfera": abs(axial - esfera) > 1e-6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lndb-manifest", default=MANIFESTO_LNDB)
    ap.add_argument("--manifest-sha256", default="")
    ap.add_argument("--limit-scans", type=int, default=0)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--gates-only", action="store_true")
    args = ap.parse_args()
    t0 = datetime.datetime.now().astimezone()

    if PRIMARY_MATCHING_IOU != 0.10:
        fail_input(f"PRIMARY_MATCHING_IOU congelado em 0.10; encontrado "
                   f"{PRIMARY_MATCHING_IOU} — THRESHOLD_TUNING_ON_LNDB = NO")
    if PRIMARY_MATCHING_IOU not in LIMIARES_IOU:
        fail_input("o limiar primário tem de estar entre os limiares da curva")
    bordas = [(lo, hi) for _, lo, hi in ESTRATOS]
    if bordas[0][0] != 0.0 or bordas[-1][1] != float("inf") or \
            any(bordas[i][1] != bordas[i + 1][0] for i in range(len(bordas) - 1)):
        fail_input(f"estratos de tamanho não cobrem [0, inf) de forma contígua: {bordas}")

    g_emp = gate_emparelhador()
    g_con = gate_conectividade()
    g_dia = gate_diametro()
    if not g_emp["paridade_com_script11"]:
        fail_output(f"HUNGARIAN_PARITY_WITH_SCRIPT11 falhou: {g_emp}")
    if not g_emp["canonico_retem_mais"]:
        fail_output("contra-exemplo do emparelhador não discrimina as duas formas")
    if not (g_con["distintos_com_6"] and g_con["fundidos_com_26"]):
        fail_output(f"CONNECTIVITY_6_GATE falhou: {g_con}")
    if not (g_dia["correcto"] and g_dia["difere_da_esfera"]):
        fail_output(f"DIAMETER_PARITY_GATE falhou: {g_dia}")
    log.info(f"emparelhador: 0 divergências em 500×10 matrizes; contra-exemplo "
             f"canónico {g_emp['contra_exemplo_canonico']} vs global+filtro "
             f"{g_emp['contra_exemplo_global_depois_filtro']}")
    if args.gates_only:
        print(json.dumps({"HUNGARIAN_PARITY_WITH_SCRIPT11": "PASS",
                          "CONNECTIVITY_6_GATE": "PASS", "DIAMETER_PARITY_GATE": "PASS",
                          "emparelhador": g_emp, "conectividade": g_con,
                          "diametro": g_dia}, indent=1, default=str))
        return

    if not os.path.isfile(args.lndb_manifest):
        fail_input(f"manifesto do Script 05 ausente: {args.lndb_manifest}")
    hman = sha256(args.lndb_manifest)
    if args.manifest_sha256 and hman != args.manifest_sha256:
        fail_input(f"SHA do manifesto {hman[:12]} != {args.manifest_sha256[:12]}")
    man = json.load(open(args.lndb_manifest))
    for c in ("rows", "SCRIPT05_NOMINAL_COUNT", "SCRIPT05_ANALYTIC_COUNT",
              "downstream_discovery_mode"):
        if c not in man:
            fail_input(f"manifesto malformado: falta {c}")
    if man["downstream_discovery_mode"] != "MANIFEST_ONLY":
        fail_input(f"modo de descoberta {man['downstream_discovery_mode']} != MANIFEST_ONLY")
    linhas_man = man["rows"]
    if len(linhas_man) != LNDB_NOMINAL_N or man["SCRIPT05_NOMINAL_COUNT"] != LNDB_NOMINAL_N:
        fail_input(f"LNDB_NOMINAL_COUNT_GATE: {len(linhas_man)} / "
                   f"{man['SCRIPT05_NOMINAL_COUNT']} != {LNDB_NOMINAL_N}")
    part = {}
    for r in linhas_man:
        part[r["analytic_status"]] = part.get(r["analytic_status"], 0) + 1
    analiticos = sorted(r["lndb_id"] for r in linhas_man
                        if r["analytic_status"] in ESTADOS_ANALITICOS)
    if len(analiticos) != LNDB_ANALYTIC_N or man["SCRIPT05_ANALYTIC_COUNT"] != LNDB_ANALYTIC_N:
        fail_input(f"LNDB_ANALYTIC_COUNT_GATE: {len(analiticos)} != {LNDB_ANALYTIC_N}")
    if len(set(analiticos)) != len(analiticos):
        fail_input("identificadores analíticos repetidos")
    if sum(part.values()) != LNDB_NOMINAL_N:
        fail_input("partição de analytic_status não exaustiva")
    for est, n in ESTADOS_EXCLUIDOS.items():
        if part.get(est) != n:
            fail_input(f"LNDB_STATUS_PARTITION_GATE: {est} = {part.get(est)} != {n}")
    nao_analiticos = LNDB_NOMINAL_N - LNDB_ANALYTIC_N
    log.info(f"Coorte LNDb: nominal {LNDB_NOMINAL_N}, analítica {LNDB_ANALYTIC_N}, "
             f"não analíticos {nao_analiticos} · partição {part}")

    escolhidos = analiticos[: args.limit_scans] if args.limit_scans else analiticos
    inv_ref = {r["lndb_id"]: {"n_vox": r["n_vox_consensus"],
                              "n_comp": r["n_components_consensus"],
                              "vazio": r["consensus_empty"],
                              "sha_fonte": r.get("sha256_gt_source", "")}
               for r in linhas_man}

    entradas = []
    for arch in ARQUITECTURAS:
        d = os.path.join(PRED_DIR, arch)
        if not os.path.isdir(d):
            fail_input(f"arquitectura obrigatória ausente: {d}")
        gp = os.path.join(d, "BLOCK_GATE.json")
        if not os.path.isfile(gp):
            fail_input(f"prediction_gate ausente: {gp}")
        bg = json.load(open(gp))
        for k, v in (("BLOCK_INFERENCE_RC", 0), ("EXPECTED_IDS", LNDB_ANALYTIC_N),
                     ("OBSERVED_IDS", LNDB_ANALYTIC_N), ("MISSING_IDS", 0), ("EXTRA_IDS", 0)):
            if bg.get(k) != v:
                fail_input(f"BLOCK_GATE de {arch}: {k} = {bg.get(k)} != {v}")
        presentes = {f[:-7] for f in os.listdir(d)
                     if f.endswith(".nii.gz") and not f.endswith("_cc.nii.gz")}
        if presentes != set(analiticos):
            fail_input(f"conjunto de predições de {arch} != os {LNDB_ANALYTIC_N} analíticos "
                       f"(em falta {len(set(analiticos) - presentes)}, "
                       f"a mais {len(presentes - set(analiticos))})")
    for sid in escolhidos:
        gtp = os.path.join(CONSENSUS_DIR, sid, "consensus.nii.gz")
        if not os.path.isfile(gtp):
            fail_input(f"referência do Script 05 ausente: {gtp}")
        h = sha256(gtp)
        inv = inv_ref[sid]
        arr = sitk.GetArrayFromImage(sitk.ReadImage(gtp))
        if not np.isin(arr, (0, 1)).all():
            fail_input(f"referência não binária: {gtp}")
        nv = int((arr > 0).sum())
        nc = int(nd.label(arr > 0, structure=STRUCT_6)[1])
        if nv != inv["n_vox"]:
            fail_input(f"REFERENCE_CONTENT: {sid} tem {nv} voxels != "
                       f"{inv['n_vox']} do manifesto do Script 05")
        if nc != inv["n_comp"]:
            fail_input(f"REFERENCE_CONTENT: {sid} tem {nc} componentes a 6-conectividade "
                       f"!= {inv['n_comp']} do manifesto do Script 05")
        if bool(inv["vazio"]) != (nv == 0):
            fail_input(f"REFERENCE_CONTENT: consensus_empty de {sid} incoerente")
        linha = {"scan_id": sid, "reference_path": os.path.relpath(gtp, BASE),
                 "reference_sha256": h, "reference_source_sha256": inv["sha_fonte"],
                 "reference_voxels": nv, "reference_components": nc}
        for arch in ARQUITECTURAS:
            pp = os.path.join(PRED_DIR, arch, f"{sid}.nii.gz")
            if not os.path.isfile(pp):
                fail_input(f"predição ausente: {pp}")
            linha[f"prediction_{arch}_path"] = os.path.relpath(pp, BASE)
            linha[f"prediction_{arch}_sha256"] = sha256(pp)
        entradas.append(linha)
    log.info(f"Entradas verificadas: {len(entradas)} varrimentos × {len(ARQUITECTURAS)} "
             f"arquitecturas, referências e predições com SHA registado")

    run_id = f"s22_{t0:%Y%m%dT%H%M%S}_{os.getpid()}"
    out_root = os.path.join(args.out_root or os.path.join(ANALYSIS_DIR, "lndb_external"), run_id)
    os.makedirs(out_root, exist_ok=True)

    objectos, auditoria, sens_rows = [], [], []
    acumulado = {a: {t: [0, 0, 0] for t in LIMIARES_IOU} for a in ARQUITECTURAS}
    for i, sid in enumerate(escolhidos):
        gtp = os.path.join(CONSENSUS_DIR, sid, "consensus.nii.gz")
        for arch in ARQUITECTURAS:
            pp = os.path.join(PRED_DIR, arch, f"{sid}.nii.gz")
            linhas, cont, n_gt, n_pred, off, fora_vox, fora_comp = avalia(sid, arch, gtp, pp)
            objectos += linhas
            tp, fn, fp = cont[PRIMARY_MATCHING_IOU]
            auditoria.append({"scan_id": sid, "architecture": arch,
                              "n_gt_components": n_gt, "n_pred_components": n_pred,
                              "matching_iou_threshold": PRIMARY_MATCHING_IOU,
                              "tp": tp, "fn": fn, "fp": fp,
                              "identity_gt_ok": int(tp + fn == n_gt),
                              "identity_pred_ok": int(tp + fp == n_pred),
                              "crop_offset_zyx": json.dumps(list(off)),
                              "gt_voxels_outside_crop": fora_vox,
                              "gt_components_outside_crop": fora_comp})
            for t in LIMIARES_IOU:
                a_ = acumulado[arch][t]
                for k in range(3):
                    a_[k] += cont[t][k]
        if (i + 1) % 25 == 0 or i + 1 == len(escolhidos):
            log.info(f"  [{i+1}/{len(escolhidos)}] {sid}")

    obj = pd.DataFrame(objectos)
    aud = pd.DataFrame(auditoria)
    n_exames = len(escolhidos)
    met = pd.DataFrame([l for a in ARQUITECTURAS for l in metricas(obj, a, n_exames)])
    for arch in ARQUITECTURAS:
        for t in LIMIARES_IOU:
            tp, fn, fp = acumulado[arch][t]
            sens_rows.append({"architecture": arch, "curve_type": "MATCHING_IOU_SENSITIVITY",
                              "matching_iou_threshold": t, "tp": tp, "fn": fn, "fp": fp,
                              "n_exams_evaluated": n_exames,
                              "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
                              "fp_per_exam": fp / n_exames})
    sen = pd.DataFrame(sens_rows)

    g = {"LNDB_NOMINAL_COUNT_GATE": LNDB_NOMINAL_N,
         "LNDB_ANALYTIC_COUNT_GATE": f"{len(analiticos)}/{LNDB_ANALYTIC_N}",
         "LNDB_STATUS_PARTITION_GATE": part,
         "MANIFEST_LOADED": man.get("manifest_id"),
         "MANIFEST_STATUS_PARTITION": "exaustiva e disjunta",
         "ANALYTIC_ID_SET": f"{len(analiticos)} identificadores do manifesto",
         "ANALYTIC_REFERENCES_COMPLETE": f"{len(entradas)}/{len(escolhidos)}",
         "REFERENCE_CONTENT_GATE": ("n_vox_consensus e n_components_consensus (6-conn) "
                                    "conferidos contra o manifesto do Script 05 em "
                                    f"{len(entradas)}/{len(escolhidos)}"),
         "EXTRA_REFERENCE_IDS": 0, "NOMINAL_COUNT": LNDB_NOMINAL_N,
         "ANALYTIC_COUNT": LNDB_ANALYTIC_N,
         "HUNGARIAN_PARITY_WITH_SCRIPT11": "PASS (0 divergências em 5000 casos)",
         "THRESHOLD_AWARE_COUNTEREXAMPLE": {
             "canonico": g_emp["contra_exemplo_canonico"],
             "global_depois_filtro": g_emp["contra_exemplo_global_depois_filtro"]},
         "CONNECTIVITY_6_GATE": "PASS (vértice e aresta permanecem distintos)",
         "DIAMETER_PARITY_GATE": "PASS (axial bbox; difere da esfera equivalente)",
         "MASK_BINARY_GATE": "PASS", "NATIVE_GEOMETRY_GATE": "PASS"}
    esperado_aud = len(escolhidos) * len(ARQUITECTURAS)
    if len(aud) != esperado_aud:
        fail_output(f"auditoria: {len(aud)} != {esperado_aud}")
    if (aud.identity_gt_ok == 0).any() or (aud.identity_pred_ok == 0).any():
        fail_output("identidade aritmética violada na auditoria")
    g["SCAN_AUDIT_ROWS"] = f"{len(aud)} = {len(escolhidos)}×{len(ARQUITECTURAS)}"
    g["GRID_EMBEDDING_GATE"] = ("predição = sub-bloco alinhado a voxel da grelha da "
                                "referência; encaixe por índice, sem reamostragem")
    g["GT_OUTSIDE_CROP"] = {"voxels": int(aud["gt_voxels_outside_crop"].sum()),
                            "componentes": int(aud["gt_components_outside_crop"].sum()),
                            "tratamento": "referência fora do recorte pulmonar conta como FN"}
    g["ONE_TO_ONE_GT"] = "PASS"; g["ONE_TO_ONE_PRED"] = "PASS"
    g["TP_IOU_THRESHOLD_GATE"] = f"nenhum TP com IoU < {PRIMARY_MATCHING_IOU}"
    tp_all = obj[obj.status == "TP"]
    if len(tp_all) and (tp_all["iou"] < PRIMARY_MATCHING_IOU).any():
        fail_output("TP com IoU abaixo do limiar primário")
    if tp_all.duplicated(["scan_id", "architecture", "gt_component_id"]).any() \
            or tp_all.duplicated(["scan_id", "architecture", "pred_component_id"]).any():
        fail_output("componente repetido entre os TP")
    for arch in ARQUITECTURAS:
        s = aud[aud.architecture == arch]
        o = obj[obj.architecture == arch]
        for est, col in (("TP", "tp"), ("FN", "fn"), ("FP", "fp")):
            if int((o.status == est).sum()) != int(s[col].sum()):
                fail_output(f"agregado {est} de {arch} não reproduz a soma por varrimento")
    g["AGGREGATE_MATCHES_SCAN_SUMS"] = "PASS"
    ov = met[met.stratum == "Overall"]
    for _, r in ov.iterrows():
        soma = met[(met.architecture == r.architecture) & (met.stratum != "Overall")]["n_gt"].sum()
        if int(soma) != int(r.n_gt):
            fail_output(f"estratos não particionam o GT de {r.architecture}: {soma} != {r.n_gt}")
        for c in ("sensitivity", "precision", "f1", "dice_global", "dice_cond"):
            v = r[c]
            if np.isfinite(v) and not (0.0 <= v <= 1.0):
                fail_output(f"{c} fora de [0,1] em {r.architecture}")
    est = met[met.stratum != "Overall"]
    for c, st in (("precision", "status_precision"), ("f1", "status_f1"),
                  ("fp_per_exam", "status_fp_per_exam")):
        if not (est[st] == NA).all():
            fail_output(f"SIZE_STRATIFIED_METRIC_GATE: {st} estimado num estrato de tamanho")
        if est[c].notna().any():
            fail_output(f"SIZE_STRATIFIED_METRIC_GATE: {c} com valor num estrato de tamanho")
    g["SIZE_STRATIFIED_METRIC_GATE"] = f"{NA} em precisão, F1 e FP/exame"
    for _, r2 in ov.iterrows():
        n_av = int(aud[aud.architecture == r2.architecture].scan_id.nunique())
        if int(r2.n_exams) != n_av:
            fail_output(f"FP_PER_EXAM_DENOMINATOR: {r2.n_exams} != {n_av} exames avaliados")
        if n_av == LNDB_NOMINAL_N and LNDB_NOMINAL_N != len(escolhidos):
            fail_output("denominador nominal (236) usado como denominador analítico")
        if abs(float(r2.fp_per_exam) - int(r2.n_fp) / n_av) > 1e-12:
            fail_output(f"FP/exame não reproduz FP/{n_av} em {r2.architecture}")
    g["SIZE_STRATUM_PARTITION_GATE"] = "PASS"
    g["METRIC_DOMAIN_GATE"] = "PASS"
    g["SIZE_STRATIFIED_PRECISION"] = NA
    g["PRIMARY_MATCHING_IOU"] = PRIMARY_MATCHING_IOU
    g["THRESHOLD_TUNING_ON_LNDB"] = "NO"
    if len(sen) != len(ARQUITECTURAS) * len(LIMIARES_IOU):
        fail_output(f"sensibilidade: {len(sen)} linhas != {len(ARQUITECTURAS)*len(LIMIARES_IOU)}")
    if not (sen.n_exams_evaluated == n_exames).all():
        fail_output("varrimentos avaliados variam entre limiares")
    g["MATCHING_IOU_SENSITIVITY_ROWS"] = f"{len(sen)} = {len(ARQUITECTURAS)}×{len(LIMIARES_IOU)}"
    g["CURVE_NAMING_GATE"] = ("MATCHING_IOU_SENSITIVITY — varia o limiar de IoU de "
                              "correspondência, não um limiar de confiança; não é FROC")
    g["FP_PER_EXAM_DENOMINATOR"] = f"{n_exames} exames analíticos (nunca {LNDB_NOMINAL_N})"
    num = met.select_dtypes(include=[np.number])
    if ((~np.isfinite(num.to_numpy())) & (~num.isna().to_numpy())).sum():
        fail_output("valor não finito não declarado nas métricas")
    g["NONFINITE_GATE"] = "PASS"

    prods = {}

    def grava(nome, frame):
        p = os.path.join(out_root, nome); frame.to_csv(p, index=False); prods[nome] = p

    grava("lndb_object_matches.csv", obj)
    grava("lndb_scan_audit.csv", aud)
    grava("lndb_external_metrics.csv", met)
    grava("lndb_matching_iou_sensitivity.csv", sen)
    grava("lndb_froc.csv", sen)
    pd.DataFrame(entradas).to_csv(os.path.join(out_root, "input_manifest.csv"), index=False)
    json.dump(g, open(os.path.join(out_root, "gates.json"), "w"), indent=1, default=str)
    pd.DataFrame([{"path": n, "sha256": sha256(p)} for n, p in prods.items()]).to_csv(
        os.path.join(out_root, "output_hashes.csv"), index=False)
    manifesto = {
        "script": "22_evaluate_lndb.py", "run_id": run_id,
        "script_sha256": sha256(os.path.abspath(__file__)),
        "run_at": t0.isoformat(), "end_at": datetime.datetime.now().astimezone().isoformat(),
        "purpose": "EXTERNAL_TRANSFER_EVALUATION_WITHOUT_RETRAINING",
        "cohort": {"LNDB_NOMINAL_N": LNDB_NOMINAL_N, "LNDB_ANALYTIC_N": LNDB_ANALYTIC_N,
                   "evaluated": n_exames, "status_partition": part,
                   "source": os.path.relpath(args.lndb_manifest, BASE),
                   "source_sha256": hman, "manifest_id": man.get("manifest_id"),
                   "SUPERSEDED": "236 como denominador analítico — população nominal apenas"},
        "architectures": ARQUITECTURAS,
        "definitions": {
            "connectivity": "6 (generate_binary_structure(3,1)), explícita",
            "matching": "Húngaro consciente do limiar, importado literalmente do Script 11 "
                        "(arestas IoU >= theta; peso 1000+IoU)",
            "PRIMARY_MATCHING_IOU": PRIMARY_MATCHING_IOU,
            "THRESHOLD_TUNING_ON_LNDB": "NO",
            "diameter": "MAXIMUM_AXIAL_DIAMETER_MM (bbox por corte), função do Script 11; "
                        "esfera equivalente SUPERSEDED",
            "size_strata": [e[0] for e in ESTRATOS],
            "size_stratified_precision_f1_fp": NA,
            "dice_global": "soma dos Dice dos TP / (TP + FN), em precisão completa",
            "dice_cond": "média dos Dice dos TP",
            "fp_per_exam_denominator": n_exames,
            "curve": {"name": "MATCHING_IOU_SENSITIVITY",
                      "varies": "matching IoU threshold",
                      "does_not_vary": "prediction confidence",
                      "not_a_froc": True,
                      "compatibility_alias": "lndb_froc.csv contém os mesmos dados"},
            "no_retraining": True, "no_recalibration": True},
        "canonical_modules": {"script11": SHA11},
        "counts": {"scans": n_exames, "object_rows": len(obj), "audit_rows": len(aud),
                   "metric_rows": len(met), "sensitivity_rows": len(sen)},
        "gates": g,
        "promotion": "NOT PROMOTED — produtos ficam no directório do run",
    }
    json.dump(manifesto, open(os.path.join(out_root, "manifest.json"), "w"), indent=2, default=str)
    log.info(f"{len(obj)} objectos · {len(aud)} linhas de auditoria · {len(met)} métricas · "
             f"{len(sen)} pontos de sensibilidade")
    log.info(f"Saída: {out_root}")
    log.info("Script 22 concluído.")


if __name__ == "__main__":
    main()
