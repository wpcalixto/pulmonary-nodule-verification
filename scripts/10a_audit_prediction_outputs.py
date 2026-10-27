#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    10a_audit_prediction_outputs.py

File type:
    Script

Purpose:
    Audits the completeness and integrity of the exported prediction products before the post-detection analyses.

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
    - invalidate_downstream.txt
    - prediction_gate.json

Outputs:
    - See docs/file_level_documentation.md.

Dependencies:
    - SimpleITK
    - numpy
    - scipy

Execution:
    PULMONARY_BASE=<project_dir> python scripts/10a_audit_prediction_outputs.py

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
import argparse
import logging
import platform
import hashlib
from datetime import datetime

from pulmonary_base import BASE

PRED_DIR     = os.path.join(BASE, "predictions")
ANALYSIS_DIR = os.path.join(BASE, "analysis")
CONFIG_DIR   = os.path.join(BASE, "config")
LOGS_DIR     = os.path.join(BASE, "logs")
RUNS_DIR     = os.path.join(BASE, "runs")

CURRENT_CROPS     = os.path.join(RUNS_DIR, "CURRENT.json")
CURRENT_INFERENCE = os.path.join(RUNS_DIR, "CURRENT_INFERENCE.json")

ALVOS = {
    "bd1_oof":      {"canonico": 718, "historico": 821},
    "bd1_internal": {"canonico": 158, "historico": 189},
    "rider":        {"canonico": 62,  "historico": 62},
    "lndb":         {"canonico": 211, "historico": 224},
}
INTERNAL_CANONICAL_INPUTS   = 158
INTERNAL_REFERENCE_EVALUABLE = 158
INTERNAL_REFERENCE_PENDING   = 0

LNDB_ARCH_NAMES = ["3D_F", "3D_FREM"]

CAB_INV = ["run_id", "dataset", "subset", "architecture", "fold", "patient_id",
           "expected", "mask_exists", "prob_exists", "objects_exists",
           "readable", "geometry_ok", "binary_ok", "probability_ok",
           "objects_reproducible", "positive_voxels", "n_components",
           "integrity_status", "functional_status", "failure_reason"]

RUN_ID = "s10a_%s_%d" % (datetime.now().strftime("%Y%m%dT%H%M%S"), os.getpid())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(os.path.join(LOGS_DIR, "10a_%s.log" % RUN_ID)),
              logging.StreamHandler()])
log = logging.getLogger(__name__)


def _sha256(caminho, bloco=1 << 20):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for p in iter(lambda: f.read(bloco), b""):
            h.update(p)
    return h.hexdigest()


def carregar_esperados():
    """Quem DEVIA estar lá, por subconjunto — dos manifestos, não do disco.

    Comparar o disco consigo próprio não audita nada: se faltarem 40 casos em
    todas as arquiteturas, um inventário que se baseie no disco conclui que
    está tudo certo. Os IDs esperados vêm dos manifestos congelados.
    """
    esperados = {}

    split_csv = os.path.join(CONFIG_DIR, "bd1_split.csv")
    if os.path.isfile(split_csv):
        dev, interno = [], []
        with open(split_csv) as f:
            for row in csv.DictReader(f):
                if row["split"] == "development":
                    dev.append(row["patient_id"])
                elif row["split"] == "internal":
                    interno.append(row["patient_id"])
        esperados["bd1_oof"] = sorted(dev)
        esperados["bd1_internal"] = sorted(interno)
    else:
        log.error("bd1_split.csv ausente: %s — bd1_* nao auditavel", split_csv)

    folds_json = os.path.join(CONFIG_DIR, "bd1_folds.json")
    folds = {}
    if os.path.isfile(folds_json):
        d = json.load(open(folds_json))
        for i in range(5):
            k = "fold_%d" % i
            if k in d.get("folds", {}):
                folds[i] = sorted(d["folds"][k]["val"])
    else:
        log.error("bd1_folds.json ausente — OOF por fold nao auditavel")

    raiz = None
    if os.path.isfile(CURRENT_CROPS):
        p = json.load(open(CURRENT_CROPS))
        man = p.get("manifest", "")
        if os.path.isfile(man):
            raiz = os.path.dirname(man)
    for banco, chave in (("rider", "rider"), ("lndb", "lndb")):
        if raiz and os.path.isdir(os.path.join(raiz, banco, "images")):
            d = os.path.join(raiz, banco, "images")
            esperados[chave] = sorted(
                x for x in os.listdir(d)
                if os.path.isdir(os.path.join(d, x)) and not x.startswith("."))
        else:
            log.error("recortes de %s indisponiveis — %s nao auditavel",
                      banco, chave)

    return esperados, folds, raiz


def auditar_ficheiro(mask_path, ct_path, recalcular_cc):
    """Abre a máscara e devolve as medidas. Nada aqui é inferido do nome."""
    import numpy as np
    import SimpleITK as sitk

    r = {"readable": False, "geometry_ok": "", "binary_ok": "",
         "positive_voxels": "", "n_components": "",
         "objects_reproducible": "", "failure_reason": ""}
    try:
        img = sitk.ReadImage(mask_path)
        arr = sitk.GetArrayFromImage(img)
    except Exception as e:
        r["failure_reason"] = "mascara ilegivel: %s: %s" % (type(e).__name__, e)
        return r
    r["readable"] = True

    vals = np.unique(arr)
    r["binary_ok"] = bool(set(vals.tolist()) <= {0, 1})
    if not r["binary_ok"]:
        r["failure_reason"] = "mascara nao binaria: valores %s" % vals[:6].tolist()
    r["positive_voxels"] = int((arr > 0).sum())

    if ct_path and os.path.isfile(ct_path):
        try:
            ct = sitk.ReadImage(ct_path)
            mesma = (ct.GetSize() == img.GetSize()
                     and np.allclose(ct.GetSpacing(), img.GetSpacing(), atol=1e-4)
                     and np.allclose(ct.GetOrigin(), img.GetOrigin(), atol=1e-3))
            r["geometry_ok"] = bool(mesma)
            if not mesma:
                r["failure_reason"] = (
                    "geometria difere da CT: size %s vs %s, spacing %s vs %s"
                    % (img.GetSize(), ct.GetSize(),
                       tuple(round(x, 4) for x in img.GetSpacing()),
                       tuple(round(x, 4) for x in ct.GetSpacing())))
        except Exception as e:
            r["geometry_ok"] = False
            r["failure_reason"] = "CT ilegivel para comparar geometria: %s" % e
    else:
        r["geometry_ok"] = ""
        r["failure_reason"] = r["failure_reason"] or "CT correspondente nao encontrada"

    cc_path = mask_path.replace(".nii.gz", "_cc.nii.gz")
    if recalcular_cc:
        try:
            from scipy import ndimage as nd
            _, n = nd.label((arr > 0).astype(np.uint8))
            r["n_components"] = int(n)
            if os.path.isfile(cc_path):
                cc = sitk.GetArrayFromImage(sitk.ReadImage(cc_path))
                n_tab = int(cc.max())
                r["objects_reproducible"] = bool(n_tab == n)
                if n_tab != n:
                    r["failure_reason"] = (
                        "componentes divergem: tabela=%d recalculado=%d"
                        % (n_tab, n))
            else:
                r["objects_reproducible"] = ""
        except Exception as e:
            r["failure_reason"] = "falha a recalcular componentes: %s" % e
    elif os.path.isfile(cc_path):
        try:
            cc = sitk.GetArrayFromImage(sitk.ReadImage(cc_path))
            r["n_components"] = int(cc.max())
        except Exception as e:
            r["failure_reason"] = "tabela de componentes ilegivel: %s" % e
    return r


def auditar_prob(prob_path):
    """Probabilidades finitas em [0,1]. Passo 4."""
    if not os.path.isfile(prob_path):
        return "", ""
    try:
        import numpy as np
        with np.load(prob_path) as z:
            chave = z.files[0]
            a = z[chave]
            ok = bool(np.isfinite(a).all() and a.min() >= -1e-6 and a.max() <= 1 + 1e-6)
        return ok, "" if ok else "probabilidades fora de [0,1] ou nao finitas"
    except Exception as e:
        return False, "npz ilegivel: %s: %s" % (type(e).__name__, e)


def classificar(r, esperado, mask_exists):
    """Devolve `(integridade, funcional)`.

    A regra que este script existe para aplicar:

      ficheiro ausente        → integridade FAIL_OUTPUT   · funcional NOT_ESTIMABLE
      ficheiro presente, 0 vx → integridade PASS          · funcional COMPLETE_DEGENERATE
      ficheiro presente, >0vx → integridade PASS          · funcional FUNCTIONAL

    A segunda linha é a que se perde quando alguém escreve «não há nódulos
    detectados» a partir de um `ls` que não encontrou nada.
    """
    if esperado and not mask_exists:
        return "FAIL_OUTPUT", "NOT_ESTIMABLE"
    if not esperado and mask_exists:
        return "FAIL_OUTPUT", "NOT_APPLICABLE"
    if not r["readable"]:
        return "FAIL_OUTPUT", "NOT_ESTIMABLE"
    if r["binary_ok"] is False or r["geometry_ok"] is False:
        return "FAIL_OUTPUT", "NOT_ESTIMABLE"
    if r["objects_reproducible"] is False:
        return "FAIL_OUTPUT", "NOT_ESTIMABLE"

    integridade = "PASS"
    if r["geometry_ok"] == "" or r["failure_reason"]:
        integridade = "PASS_WITH_FLAG"

    pv = r["positive_voxels"]
    if pv == "" or pv is None:
        return integridade, "NOT_ESTIMABLE"
    funcional = "COMPLETE_DEGENERATE" if int(pv) == 0 else "FUNCTIONAL"
    return integridade, funcional


def ids_em(pasta):
    if not os.path.isdir(pasta):
        return set()
    out = set()
    for f in os.listdir(pasta):
        if f.startswith(".") or f.startswith("_"):
            continue
        if f.endswith("_cc.nii.gz") or not f.endswith(".nii.gz"):
            continue
        out.add(f[:-len(".nii.gz")])
    return out


def ct_de(raiz_recortes, subset, pid):
    """Caminho da CT que corresponde a esta predição."""
    if not raiz_recortes:
        return None
    banco = {"bd1_oof": "lidc_idri", "bd1_internal": "lidc_idri",
             "rider": "rider", "lndb": "lndb"}.get(subset)
    if not banco:
        return None
    return os.path.join(raiz_recortes, banco, "images", pid, "ct.nii.gz")


def varrer(args):
    esperados, folds, raiz = carregar_esperados()
    linhas, falhas = [], []

    subsets = args.subset or ["bd1_oof", "bd1_internal", "rider", "lndb"]
    for subset in subsets:
        base_sub = os.path.join(PRED_DIR, subset)
        if not os.path.isdir(base_sub):
            log.warning("subconjunto ausente em disco: %s", base_sub)
            continue
        arqs = sorted(a for a in os.listdir(base_sub)
                      if os.path.isdir(os.path.join(base_sub, a))
                      and not a.startswith("_") and not a.startswith("."))
        if args.arq:
            arqs = [a for a in arqs if a in set(args.arq)]

        for arq in arqs:
            if subset == "bd1_oof":
                for fold, val_ids in sorted(folds.items()):
                    d = os.path.join(base_sub, arq, "fold_%d" % fold)
                    linhas += auditar_pasta(d, subset, arq, str(fold), val_ids,
                                            raiz, args, falhas)
            else:
                d = os.path.join(base_sub, arq)
                esp = esperados.get(subset, [])
                if subset == "lndb" and arq not in LNDB_ARCH_NAMES:
                    log.warning("LNDb com arquitetura nao pre-especificada: %s", arq)
                linhas += auditar_pasta(d, subset, arq, "", esp, raiz, args, falhas)
    return linhas, falhas, esperados, folds


def auditar_pasta(d, subset, arq, fold, esperados_ids, raiz, args, falhas):
    obtidos = ids_em(d)
    universo = sorted(set(esperados_ids) | obtidos)
    linhas = []
    log.info("  %s/%s%s: esperados=%d obtidos=%d",
             subset, arq, ("/fold_" + fold) if fold else "",
             len(esperados_ids), len(obtidos))

    for pid in universo:
        esperado = pid in set(esperados_ids)
        mask = os.path.join(d, "%s.nii.gz" % pid)
        prob = os.path.join(d, "%s.npz" % pid)
        cc   = os.path.join(d, "%s_cc.nii.gz" % pid)
        mask_exists = os.path.isfile(mask)

        if mask_exists:
            r = auditar_ficheiro(mask, ct_de(raiz, subset, pid),
                                 not args.sem_recalcular_cc)
            prob_ok, prob_motivo = auditar_prob(prob)
        else:
            r = {"readable": False, "geometry_ok": "", "binary_ok": "",
                 "positive_voxels": "", "n_components": "",
                 "objects_reproducible": "",
                 "failure_reason": "predicao ausente" if esperado else ""}
            prob_ok, prob_motivo = "", ""

        integridade, funcional = classificar(r, esperado, mask_exists)
        motivo = r["failure_reason"] or prob_motivo
        if prob_ok is False and integridade == "PASS":
            integridade = "FAIL_OUTPUT"

        linha = {
            "run_id": RUN_ID, "dataset": "Dataset001_PulmonaryNodules",
            "subset": subset, "architecture": arq, "fold": fold,
            "patient_id": pid, "expected": esperado,
            "mask_exists": mask_exists, "prob_exists": os.path.isfile(prob),
            "objects_exists": os.path.isfile(cc),
            "readable": r["readable"], "geometry_ok": r["geometry_ok"],
            "binary_ok": r["binary_ok"], "probability_ok": prob_ok,
            "objects_reproducible": r["objects_reproducible"],
            "positive_voxels": r["positive_voxels"],
            "n_components": r["n_components"],
            "integrity_status": integridade, "functional_status": funcional,
            "failure_reason": motivo,
        }
        linhas.append(linha)
        if integridade != "PASS":
            falhas.append(linha)
    return linhas


def resumir(linhas, esperados, folds):
    """Um registo por (subconjunto, arquitetura). Totais = soma das linhas."""
    chaves, resumo = {}, []
    for l in linhas:
        k = (l["subset"], l["architecture"])
        chaves.setdefault(k, []).append(l)
    for (subset, arq), ls in sorted(chaves.items()):
        esperado_n = sum(1 for x in ls if x["expected"])
        obtido_n = sum(1 for x in ls if x["mask_exists"] and x["expected"])
        extra_n = sum(1 for x in ls if x["mask_exists"] and not x["expected"])
        alvo = ALVOS.get(subset, {})
        ids_unicos = {x["patient_id"] for x in ls if x["expected"]}
        resumo.append({
            "run_id": RUN_ID, "subset": subset, "architecture": arq,
            "n_expected": esperado_n, "n_unique_expected": len(ids_unicos),
            "n_obtained": obtido_n, "n_missing": esperado_n - obtido_n,
            "n_extra": extra_n,
            "target_canonical": alvo.get("canonico", ""),
            "target_historical": alvo.get("historico", ""),
            "n_pass": sum(1 for x in ls if x["integrity_status"] == "PASS"),
            "n_flag": sum(1 for x in ls if x["integrity_status"] == "PASS_WITH_FLAG"),
            "n_fail": sum(1 for x in ls if x["integrity_status"] == "FAIL_OUTPUT"),
            "n_functional": sum(1 for x in ls if x["functional_status"] == "FUNCTIONAL"),
            "n_degenerate": sum(1 for x in ls
                                if x["functional_status"] == "COMPLETE_DEGENERATE"),
            "empty_masks_present": sum(1 for x in ls
                                       if x["mask_exists"] and x["positive_voxels"] == 0),
        })
    return resumo


def gate(linhas, resumo):
    """`prediction_gate.json` — reprova por ausente, extra, ilegível,
    incompatível ou não rastreável no escopo consumido."""
    motivos = []
    for r in resumo:
        onde = "%s/%s" % (r["subset"], r["architecture"])
        if r["n_missing"]:
            motivos.append("%s: %d predicoes ausentes" % (onde, r["n_missing"]))
        if r["n_extra"]:
            motivos.append("%s: %d IDs extra" % (onde, r["n_extra"]))
        if r["n_fail"]:
            motivos.append("%s: %d com integridade FAIL_OUTPUT" % (onde, r["n_fail"]))
        alvo = r["target_canonical"]
        if alvo and r["n_unique_expected"] and r["n_unique_expected"] != alvo:
            motivos.append("%s: %d IDs unicos esperados, alvo canonico %d"
                           % (onde, r["n_unique_expected"], alvo))

    vistos = {}
    for l in linhas:
        k = (l["subset"], l["architecture"], l["fold"], l["patient_id"])
        vistos[k] = vistos.get(k, 0) + 1
    dups = [k for k, n in vistos.items() if n > 1]
    if dups:
        motivos.append("%d IDs duplicados (ex.: %s)" % (len(dups), dups[:3]))

    if sum(r["n_pass"] + r["n_flag"] + r["n_fail"] for r in resumo) != len(linhas):
        motivos.append("totais do resumo nao batem com a soma das linhas")

    return {"run_id": RUN_ID, "status": "PASS" if not motivos else "FAIL",
            "n_rows": len(linhas), "n_groups": len(resumo),
            "reasons": motivos,
            "internal_layers": {
                "INTERNAL_CANONICAL_INPUTS": INTERNAL_CANONICAL_INPUTS,
                "INTERNAL_REFERENCE_EVALUABLE": INTERNAL_REFERENCE_EVALUABLE,
                "INTERNAL_REFERENCE_PENDING": INTERNAL_REFERENCE_PENDING},
            "published_at": datetime.now().isoformat(),
            "host": platform.node(), "pulmonary_base": BASE}


def descendentes_a_invalidar(resumo):
    """Quem consome o que reprovou — para não ficar resultado velho por baixo."""
    mapa = {"bd1_oof": ["11", "13", "14", "15", "16", "17", "24", "25"],
            "bd1_internal": ["11", "17A", "24", "25"],
            "rider": ["18", "19", "20", "21", "24", "25"],
            "lndb": ["22", "22A", "22B", "23", "24", "25"]}
    fora = set()
    for r in resumo:
        if r["n_missing"] or r["n_fail"] or r["n_extra"]:
            fora.update(mapa.get(r["subset"], []))
    return sorted(fora)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", action="append", default=None)
    ap.add_argument("--arq", action="append", default=None)
    ap.add_argument("--sem-recalcular-cc", action="store_true",
                    help="salta o passo 6; use so em varrimento exploratorio")
    args = ap.parse_args()

    saida = os.path.join(ANALYSIS_DIR, "10a_%s" % RUN_ID)
    os.makedirs(saida, exist_ok=True)
    log.info("Script 10A · run_id=%s · BASE=%s", RUN_ID, BASE)
    if args.sem_recalcular_cc:
        log.warning("--sem-recalcular-cc: o passo 6 NAO corre; o gate deste "
                    "varrimento nao autoriza o Script 11.")

    linhas, falhas, esperados, folds = varrer(args)
    resumo = resumir(linhas, esperados, folds)
    g = gate(linhas, resumo)
    if args.sem_recalcular_cc and g["status"] == "PASS":
        g["status"] = "FAIL"
        g["reasons"].append("varrimento sem recalculo de componentes")

    def escrever(nome, cab, rows):
        p = os.path.join(saida, nome)
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cab, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        return p

    escrever("prediction_inventory.csv", CAB_INV, linhas)
    escrever("prediction_failures.csv", CAB_INV, falhas)
    if resumo:
        escrever("prediction_summary.csv", list(resumo[0].keys()), resumo)

    invalidar = descendentes_a_invalidar(resumo)
    with open(os.path.join(saida, "invalidate_downstream.txt"), "w") as f:
        f.write("\n".join(invalidar) + ("\n" if invalidar else ""))
    with open(os.path.join(saida, "prediction_gate.json"), "w") as f:
        json.dump(g, f, indent=2, ensure_ascii=False)

    log.info("─" * 70)
    for r in resumo:
        log.info("  %-13s %-14s esperados=%4d obtidos=%4d faltam=%3d "
                 "FAIL=%3d vazias=%3d",
                 r["subset"], r["architecture"], r["n_expected"],
                 r["n_obtained"], r["n_missing"], r["n_fail"],
                 r["empty_masks_present"])
    log.info("─" * 70)
    log.info("GATE: %s", g["status"])
    for m in g["reasons"]:
        log.error("  %s", m)
    if invalidar:
        log.warning("descendentes a invalidar: %s", ", ".join(invalidar))
    log.info("Saidas em %s", os.path.relpath(saida, BASE))

    return 0 if g["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
