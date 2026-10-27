#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    07_draw_bd1_partition.py

File type:
    Script

Purpose:
    Draws the BD1 partition into development and internal-validation sets and the five cross-validation folds.

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
    - sklearn

Execution:
    PULMONARY_BASE=<project_dir> python scripts/07_draw_bd1_partition.py

Portability notes:
    All paths are resolved relative to the project base directory given by the PULMONARY_BASE environment variable. No machine-specific path is required.

Reproducibility notes:
    Environment pinned in requirements.txt and environment/. Seeds and frozen parameters are recorded in the run manifests under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) are not included and must be obtained from their original repositories.
"""

import os
import sys
import csv
import json
import collections
import time
import zipfile
import hashlib
import logging
import argparse
import xml.etree.ElementTree as ET

from sklearn.model_selection import KFold, train_test_split

BASE = os.environ["PULMONARY_BASE"]
CONFIG = os.path.join(BASE, "config")
REFDIR = os.path.join(BASE, "data", "reference")
PLANILHA = os.path.join(REFDIR, "lidc-idri-nodule-counts-6-23-2015.xlsx")
PONTEIRO_ANOT = os.path.join(BASE, "runs", "CURRENT_LIDC_ANNOTATIONS.json")
RUNS = os.path.join(BASE, "runs")
PONTEIRO = os.path.join(RUNS, "CURRENT_BD1_PARTITION.json")

SEED = 27
N_FOLDS = 5
TEST_SIZE = 0.2
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _exigir(cond, msg):
    if not cond:
        raise SystemExit("[FAIL] %s" % msg)


def sha(p, bs=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def ler_planilha(p):
    """Lê o XLSX sem openpyxl: é um zip com XML."""
    z = zipfile.ZipFile(p)
    ss = [t.text or "" for t in
          ET.fromstring(z.read("xl/sharedStrings.xml")).iter(NS + "t")]
    out = {}
    for r in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(NS + "row"):
        v = []
        for c in r:
            e = c.find(NS + "v")
            x = e.text if e is not None else ""
            if c.get("t") == "s" and x:
                x = ss[int(x)]
            v.append(x)
        if v and str(v[0]).startswith("LIDC-IDRI"):
            out[v[0]] = {"total": v[1], "ge3mm": v[2]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--allow-different", action="store_true",
                    help="aceita uma particao que NAO reproduza a publicada; so "
                         "com mudanca deliberada da coorte, e invalida tudo o que "
                         "dependia da anterior")
    ap.add_argument("--supersede", action="store_true",
                    help="substitui deliberadamente uma particao ja publicada; "
                         "invalida tudo o que dela dependa")
    args = ap.parse_args()

    run_id = "s07draw_%s_%d" % (time.strftime("%Y%m%dT%H%M%S"), os.getpid())
    dest = os.path.join(RUNS, run_id)
    _exigir(not os.path.exists(dest), "destino ja existe")

    if os.path.isfile(PONTEIRO) and not args.supersede:
        _p = json.load(open(PONTEIRO))
        _exigir(False,
                "ja existe particao publicada: %s (%s dev + %s interno). "
                "Nao se resorteia por cima. Para substituir, remova "
                "runs/CURRENT_BD1_PARTITION.json deliberadamente ou use "
                "--supersede, sabendo que invalida tudo o que vem a jusante."
                % (_p.get("run_id"), _p.get("split_development"),
                   _p.get("split_internal")))

    for legado, rot in ((os.path.join(CONFIG, "bd1_split.csv"), "bd1_split.csv"),
                        (os.path.join(CONFIG, "bd1_folds.json"), "bd1_folds.json")):
        _exigir(not os.path.exists(legado),
                "%s ainda esta em config/. A particao e sorteada nesta execucao; "
                "nenhum resultado anterior pode estar no caminho." % rot)

    _exigir(os.path.isfile(PLANILHA),
            "planilha de contagens ausente: %s (recurso de ORIGEM do LIDC-IDRI)"
            % PLANILHA)
    _exigir(os.path.isfile(PONTEIRO_ANOT),
            "ponteiro das anotacoes ausente: %s. O Script 04 tem de correr antes."
            % PONTEIRO_ANOT)
    _ptr = json.load(open(PONTEIRO_ANOT))
    _mp = os.path.join(BASE, _ptr["annotations_manifest"])
    _exigir(os.path.isfile(_mp), "manifesto do 04 ausente: %s" % _mp)
    _man = json.load(open(_mp))
    _exigir(_man.get("manifest_sha256") == _ptr.get("manifest_sha256"),
            "hash do manifesto das anotacoes diverge do ponteiro")
    _exigir(_man.get("SCRIPT04_OUTPUT_STATUS") == "PASS",
            "Script 04 em %s" % _man.get("SCRIPT04_OUTPUT_STATUS"))

    sha_pl = sha(PLANILHA)
    contagens = ler_planilha(PLANILHA)
    universo = sorted(r["patient_id"] for r in _man["rows"] if r["has_consensus"])
    _tipo_rotulo = {r["patient_id"]: ("documented_negative"
                                      if r["status_04"] == "NEGATIVE_DOCUMENTED"
                                      else "annotated")
                    for r in _man["rows"] if r["has_consensus"]}

    log.info("Script 07 — sorteio da particao BD1 | run_id=%s" % run_id)
    log.info("  universo: %d casos com consenso publicado pelo 04 (run %s)"
             % (len(universo), _ptr["run_id"]))
    log.info("  planilha 2015: %d pacientes · sha %s…" % (len(contagens), sha_pl[:16]))
    log.info("  criterio da planilha: UNIAO entre leitores (>=1 de 4), NAO consenso")

    sem = [p for p in universo if p not in contagens]
    _exigir(not sem, "%d pacientes sem contagem na planilha: %s" % (len(sem), sem[:5]))

    estrato = {p: "%s_%s" % (contagens[p]["total"], contagens[p]["ge3mm"])
               for p in universo}
    freq = {}
    for e in estrato.values():
        freq[e] = freq.get(e, 0) + 1
    unicos = sorted(p for p in universo if freq[estrato[p]] == 1)
    resto = sorted(p for p in universo if freq[estrato[p]] > 1)
    log.info("  estratos distintos: %d · membro unico: %d · para sorteio: %d"
             % (len(freq), len(unicos), len(resto)))

    tr, va = train_test_split([[p] for p in resto], test_size=TEST_SIZE,
                              stratify=[estrato[p] for p in resto],
                              random_state=SEED)
    dev = sorted({x[0] for x in tr} | set(unicos))
    interno = sorted({x[0] for x in va})
    _exigir(not (set(dev) & set(interno)), "desenvolvimento e interno intersectam-se")
    _exigir(set(dev) | set(interno) == set(universo), "a divisao nao cobre o universo")
    log.info("  divisao: %d = desenvolvimento %d + interno %d"
             % (len(universo), len(dev), len(interno)))

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = {}
    for i, (itr, iva) in enumerate(kf.split(dev)):
        folds["fold_%d" % i] = {"train": sorted(dev[j] for j in itr),
                                "val": sorted(dev[j] for j in iva)}
        log.info("    fold_%d: train %d · val %d"
                 % (i, len(folds["fold_%d" % i]["train"]),
                    len(folds["fold_%d" % i]["val"])))
    vistos = [p for f in folds.values() for p in f["val"]]
    _exigir(len(vistos) == len(set(vistos)) == len(dev),
            "cobertura OOF nao e exata: %d validacoes para %d casos"
            % (len(vistos), len(dev)))
    for i in range(N_FOLDS):
        f = folds["fold_%d" % i]
        _exigir(not (set(f["train"]) & set(f["val"])),
                "fold_%d: sobreposicao treino/validacao" % i)
        _exigir(set(f["train"]) | set(f["val"]) == set(dev),
                "fold_%d: uniao != desenvolvimento" % i)

    gates = {
        "SEED": SEED, "N_FOLDS": N_FOLDS, "TEST_SIZE": TEST_SIZE,
    "SINGLE_MEMBER_STRATA_POLICY": (
            "estrato com um so membro nao pode ser estratificado: vai INTEIRO "
            "para desenvolvimento. Por isso a fraccao interna observada fica "
            "abaixo de TEST_SIZE, e a diferenca e essa e nao outra."),
        "TEST_SIZE_REQUESTED": TEST_SIZE,
        "UNIVERSE_SOURCE": "CURRENT_LIDC_ANNOTATIONS.json (Script 04)",
        "ANNOTATION_RUN_ID": _ptr["run_id"],
        "ANNOTATION_MANIFEST_SHA256": _man.get("manifest_sha256"),
        "UNIVERSE_COUNT": len(universo),
        "NODULE_COUNTS_SOURCE": os.path.relpath(PLANILHA, BASE),
        "NODULE_COUNTS_SHA256": sha_pl,
        "NODULE_COUNTS_CRITERION": "UNION_OVER_READERS_AT_LEAST_ONE_OF_FOUR",
        "NOT_SCRIPT04_CONSENSUS": True,
        "STRATA_DISTINCT": len(freq),
        "SINGLE_MEMBER_STRATA": len(unicos),
        "SPLIT_DEVELOPMENT": len(dev), "SPLIT_INTERNAL": len(interno),
        "FOLDS_COUNTS": "+".join(str(len(folds["fold_%d" % i]["val"]))
                                 for i in range(N_FOLDS)),
        "OOF_COVERAGE_EXACT": True,
        "NEW_DRAW_PERFORMED": True,
        "REUSED_PRIOR_PARTITION": False,
        "DOCUMENTED_NEGATIVES": sum(1 for v in _tipo_rotulo.values()
                                    if v == "documented_negative"),
        "LABEL_KIND_RULE": (
            "documented_negative = referencia vazia com evidencia POSITIVA de "
            "ausencia do alvo, adjudicada pelo Script 04 com duas fontes "
            "concordantes. Nunca ausencia de anotacao convertida em fundo."),
        "PARTITION_FROZEN": "YES",
        "PRODUCT_STATUS": "PASS",
        "SCRIPT07_OUTPUT_STATUS": "PASS",
    }
    for k in ("UNIVERSE_COUNT", "SINGLE_MEMBER_STRATA", "SPLIT_DEVELOPMENT",
              "SPLIT_INTERNAL", "FOLDS_COUNTS", "NEW_DRAW_PERFORMED",
              "PARTITION_FROZEN"):
        log.info("  %-26s %s" % (k, gates[k]))

    reproducao = {"compared": False}
    if os.path.isfile(PONTEIRO):
        _pa = json.load(open(PONTEIRO))
        _csv = _pa.get("split_csv")
        _fj = _pa.get("folds_json")
        _csv = _csv if os.path.isabs(_csv or "") else os.path.join(BASE, _csv or "")
        _fj = _fj if os.path.isabs(_fj or "") else os.path.join(BASE, _fj or "")
        if os.path.isfile(_csv) and os.path.isfile(_fj):
            _ant_split = {}
            with open(_csv) as f:
                for r in csv.DictReader(f):
                    _ant_split[r["patient_id"]] = r["split"]
            _ant_folds = json.load(open(_fj)).get("folds", {})
            _novo_split = {p: ("development" if p in set(dev) else "internal")
                           for p in universo}
            _dif_uni = (set(_ant_split) ^ set(_novo_split))
            _dif_lado = {p for p in set(_ant_split) & set(_novo_split)
                         if _ant_split[p] != _novo_split[p]}
            _dif_folds = {}
            for k in sorted(set(_ant_folds) | set(folds)):
                _a, _b = _ant_folds.get(k) or {}, folds.get(k) or {}
                for lado in ("train", "val"):
                    a = set(_a.get(lado) or [])
                    b = set(_b.get(lado) or [])
                    if a != b:
                        _dif_folds.setdefault(k, {})[lado] = {
                            "so_no_publicado": sorted(a - b)[:5],
                            "so_no_novo": sorted(b - a)[:5],
                            "n_diferentes": len(a ^ b)}
            reproducao = {"compared": True,
                          "previous_run": _pa.get("run_id"),
                          "universe_symmetric_difference": len(_dif_uni),
                          "cases_changing_side": len(_dif_lado),
                          "folds_differing": sorted(_dif_folds),
                          "fold_detail": _dif_folds,
                          "identical": not (_dif_uni or _dif_lado or _dif_folds)}
            if reproducao["identical"]:
                log.info("  REPRODUZ a particao publicada (%s): mesmo universo, "
                         "mesmos lados, mesmos folds", _pa.get("run_id"))
            else:
                log.error("  NAO reproduz a particao publicada (%s)", _pa.get("run_id"))
                log.error("    universo difere em %d casos", len(_dif_uni))
                log.error("    mudam de lado: %d", len(_dif_lado))
                log.error("    folds diferentes: %s", sorted(_dif_folds))
                _exigir(args.allow_different,
                        "o sorteio NAO reproduz a particao publicada. Com a mesma "
                        "semente isso significa que o UNIVERSO mudou a montante — "
                        "e um achado, nao um detalhe. Investigue antes de "
                        "substituir; se a mudanca for legitima e deliberada, use "
                        "--allow-different, sabendo que invalida todo o treino, "
                        "OOF e avaliacao que dependam da particao anterior.")
    gates["REPRODUCES_PUBLISHED"] = reproducao.get("identical")
    gates["REPRODUCTION_CHECK"] = reproducao

    if args.plan_only:
        log.info("  PLAN_STATUS = PASS · nada escrito")
        return

    os.makedirs(dest, exist_ok=False)
    sp = os.path.join(dest, "bd1_split.csv")
    with open(sp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "split", "label_kind"])
        for p in universo:
            w.writerow([p, "development" if p in set(dev) else "internal",
                        _tipo_rotulo.get(p, "annotated")])
    fo = os.path.join(dest, "bd1_folds.json")
    json.dump({"seed": SEED, "n_folds": N_FOLDS, "n_dev_total": len(dev),
               "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "source": "07_draw_bd1_partition.py",
               "method": "KFold(n_splits=5, shuffle=True, random_state=27)",
               "new_draw_performed": True, "folds": folds},
              open(fo, "w"), indent=1, sort_keys=True)

    m = {"manifest_id": "bd1_splits_analytic_%s" % run_id, "run_id": run_id,
         "script": "07_DRAW_BD1_PARTITION", "script_version": "2026-08-07.draw.v2",
         "generated": time.strftime("%Y-%m-%dT%H:%M:%S BRT"),
         "producer_sha256": sha(os.path.abspath(__file__)),
         "method_recovered_from": "Material_Antigo/codigos/make_subset.ipynb (celula 9)",
         "gates": gates,
         "SCRIPT07_OUTPUT_STATUS": "PASS",
         "split_csv_sha256": sha(sp), "folds_json_sha256": sha(fo),
         "annotation_pointer_sha256": sha(PONTEIRO_ANOT),
         "annotation_run_id": _ptr["run_id"]}
    corpo = hashlib.sha256(json.dumps(m, indent=1, sort_keys=True).encode()).hexdigest()
    m["manifest_body_sha256"] = corpo
    pm = os.path.join(dest, "manifest.json")
    json.dump(m, open(pm, "w"), indent=1, sort_keys=True)

    p = {"run_id": run_id, "script": "07",
         "script_version": m["script_version"], "manifest": pm,
         "pointer_schema": "pulmonary/v2",
         "manifest_body_sha256": corpo, "manifest_file_sha256": sha(pm),
         "split_csv": os.path.relpath(sp, BASE),
         "folds_json": os.path.relpath(fo, BASE),
         "split_csv_sha256": m["split_csv_sha256"],
         "folds_json_sha256": m["folds_json_sha256"],
         "universe_count": len(universo),
         "split_development": len(dev),
         "split_internal": len(interno),
         "published_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    tmp = PONTEIRO + ".tmp"
    json.dump(p, open(tmp, "w"), indent=1, sort_keys=True)
    os.replace(tmp, PONTEIRO)
    log.info("  publicado em runs/%s" % run_id)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except SystemExit:
        raise
    except Exception as e:
        log.exception("ERRO NAO PREVISTO: %s: %s" % (type(e).__name__, e))
        sys.exit(1)
