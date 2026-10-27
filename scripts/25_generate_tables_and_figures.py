#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    25_generate_tables_and_figures.py

File type:
    Script

Purpose:
    Generates the tables and figures of the manuscript from the stage outputs.

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
    - manifest_figures.csv
    - manifest_tables.csv
    - input_manifest.csv
    - gates.json
    - manifest.json

Dependencies:
    - pandas

Execution:
    PULMONARY_BASE=<project_dir> python scripts/25_generate_tables_and_figures.py

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
import shutil
import tempfile

import pandas as pd

BASE = os.environ.get("PULMONARY_BASE") or os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(BASE, "analysis")
LOGS_DIR = os.path.join(BASE, "logs")

os.makedirs(LOGS_DIR, exist_ok=True)
log = logging.getLogger("s25")
log.setLevel(logging.INFO)
if not log.handlers:
    _f = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for _h in (logging.StreamHandler(),
               logging.FileHandler(os.path.join(LOGS_DIR, "consolidate_final_matrices.log"))):
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


U = os.path.join(ANALYSIS_DIR, "upstream_canonical")
RUNS = {
    "Script11B": os.path.join(U, "script11_final", "s11B_20260921T164539_27883"),
    "Script14": os.path.join(U, "script14_final", "s14b_20260922T203413_344007"),
    "Script16": os.path.join(U, "script16_final", "s16_20260923T094823_494840"),
    "Script21": os.path.join(ANALYSIS_DIR, "rider_repeatability", "s21_20260923T215458_708846"),
    "Script22": os.path.join(ANALYSIS_DIR, "lndb_external", "s22_20260924T084718_826234"),
    "Script23": os.path.join(ANALYSIS_DIR, "lndb_filter_validation",
                             "s23_20260924T153750_909396"),
    "Script24": os.path.join(ANALYSIS_DIR, "integrated_statistics",
                             "s24_20260924T171351_929519"),
    "Script17A": os.path.join(ANALYSIS_DIR, "internal_verifier_application",
                              "s17a_20260924T225525_1009246")}

FIGDATA = [
    ("Script11B", "analysis/bd1_detection_metrics.csv",
     "bd1_detection_metrics.csv", "DIRECT_COPY", ""),
    ("Script11B", "analysis/bd1_size_stratified_metrics.csv",
     "bd1_size_stratified_metrics.csv", "DIRECT_COPY", ""),
    ("Script11B", "analysis/bd1_froc.csv",
     "bd1_matching_iou_sensitivity.csv", "LEGACY_RENAME_AND_ANNOTATE",
     "renomeado: o ficheiro varia o limiar de IoU de correspondência, não uma "
     "confiança preditiva; acrescentada a coluna constante curve_type="
     "MATCHING_IOU_SENSITIVITY e renomeada threshold→matching_iou_threshold"),
    ("Script16", "figdata/filter_bd1_auc.csv", "filter_bd1_auc.csv", "DIRECT_COPY", ""),
    ("Script16", "figdata/filter_bd1_group_comparisons.csv",
     "filter_bd1_group_comparisons.csv", "DIRECT_COPY", ""),
    ("Script16", "figdata/filter_bd1_size_strata.csv",
     "filter_bd1_size_strata.csv", "DIRECT_COPY", ""),
    ("Script21", "figdata/rider_icc.csv", "rider_icc.csv", "DIRECT_COPY", ""),
    ("Script21", "figdata/rider_sccr.csv", "rider_sccr.csv", "DIRECT_COPY", ""),
    ("Script21", "figdata/rider_filter_repeatability.csv",
     "rider_filter_repeatability.csv", "DIRECT_COPY", ""),
    ("Script21", "figdata/rider_filter_volume_associations.csv",
     "rider_filter_volume_associations.csv", "DIRECT_COPY", ""),
    ("Script21", "figdata/volumes_RIDER_iou.csv", "volumes_RIDER_iou.csv", "DIRECT_COPY", ""),
    ("Script21", "figdata/diametros_RIDER_iou.csv", "diametros_RIDER_iou.csv", "DIRECT_COPY", ""),
    ("Script21", "rider_bland_altman.csv", "rider_bland_altman.csv", "DIRECT_COPY",
     "substitui o cálculo local de Bland–Altman da versão anterior do Script 25"),
    ("Script21", "rider_repeatability_stats.csv",
     "rider_repeatability_stats.csv", "DIRECT_COPY", ""),
    ("Script22", "lndb_external_metrics.csv", "lndb_external_metrics.csv", "DIRECT_COPY", ""),
    ("Script22", "lndb_matching_iou_sensitivity.csv",
     "lndb_matching_iou_sensitivity.csv", "DIRECT_COPY", ""),
    ("Script23", "lndb_filter_group_comparisons.csv",
     "lndb_filter_group_comparisons.csv", "DIRECT_COPY", ""),
    ("Script23", "lndb_filter_auc.csv", "lndb_filter_auc.csv", "DIRECT_COPY", ""),
    ("Script23", "lndb_filter_verifier_metrics.csv",
     "lndb_filter_verifier_metrics.csv", "DIRECT_COPY", ""),
    ("Script23", "lndb_verifier_threshold_sensitivity.csv",
     "lndb_verifier_threshold_sensitivity.csv", "DIRECT_COPY", ""),
    ("Script24", "bd1_bootstrap_metrics.csv", "bd1_bootstrap_metrics.csv", "DIRECT_COPY", ""),
    ("Script24", "multiple_comparisons_corrected.csv",
     "multiple_comparisons_corrected.csv", "DIRECT_COPY", ""),
    ("Script24", "size_stratified_integrated.csv",
     "size_stratified_integrated.csv", "DIRECT_COPY", ""),
    ("Script24", "sensitivity_iou.csv", "bd1_sensitivity_iou_integrated.csv",
     "LEGACY_RENAME", "renomeado para explicitar que varia o limiar de IoU"),
    ("Script17A", "internal_verifier_metrics.csv",
     "internal_verifier_metrics.csv", "DIRECT_COPY", ""),
]
TABDATA = [
    ("Script11B", "analysis/bd1_detection_metrics.csv",
     "table_bd1_detection.csv", "DIRECT_COPY", ""),
    ("Script22", "lndb_external_metrics.csv", "table_lndb_external.csv", "DIRECT_COPY", ""),
    ("Script23", "lndb_filter_verifier_metrics.csv",
     "table_lndb_verifier.csv", "DIRECT_COPY", ""),
    ("Script21", "rider_bland_altman.csv", "table_rider_bland_altman.csv", "DIRECT_COPY", ""),
    ("Script24", "multiple_comparisons_corrected.csv",
     "table_cross_dataset_holm.csv", "DIRECT_COPY", ""),
    ("Script24", "reference_arch_comparison.csv",
     "table_reference_arch.csv", "DIRECT_COPY", ""),
    ("Script17A", "internal_verifier_metrics.csv",
     "table_internal_verifier.csv", "DIRECT_COPY", ""),
]
EXEMPLOS = ["LIDC-IDRI-0001_2D_oof_TP_pred_c1", "LIDC-IDRI-0002_3D_CFREM_oof_TP_pred_c6",
            "LIDC-IDRI-0003_2D_oof_TP_pred_c4", "LIDC-IDRI-0004_2D_oof_TP_pred_c2",
            "LIDC-IDRI-0005_2D_internal_TP_pred_c1"]
ACTIVOS = ["ct_crop.nii.gz", "log.nii.gz", "blobness.nii.gz", "vesselness.nii.gz"]
CURVE_TYPE_ANOTADO = "MATCHING_IOU_SENSITIVITY"

BLOQUEADOS = [
    {"planned_product": "dice_bd1.tex", "renderer": "Script29",
     "status": "BLOCKED_SOURCE", "reason": "ANALYTIC_ESTIMAND_NOT_FROZEN",
     "detail": "não está congelado qual Dice a tabela resume (dice_global por paciente "
               "ou dice por objecto pareado) nem a unidade inferencial; a versão anterior "
               "fazia groupby.agg(mean,std) ao nível do objecto"},
    {"planned_product": "loss-<config>.pdf / dice-<config>.pdf", "renderer": "Script28",
     "status": "BLOCKED_SOURCE", "reason": "TRAINING_LOG_PROVENANCE_NOT_ESTABLISHED",
     "detail": "os únicos logs de treino preservados são os da ronda de Julho em "
               "quarentena, cujos pesos foram apagados por coorte contaminada; não foram "
               "localizados os logs correspondentes aos checkpoints que produziram as "
               "predições canónicas do run s10_20260918T114645_2953858"},
]


def main():
    """A consolidação corre dentro de `_consolidar`; a área de trabalho
    temporária é removida em qualquer saída, incluindo falha de gate."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--flowchart-spec", default="")
    args = ap.parse_args()
    t0 = datetime.datetime.now().astimezone()

    autoridades = {}
    for nome, raiz in RUNS.items():
        if not os.path.isdir(raiz):
            fail_input(f"{nome}: run canónico ausente: {raiz}")
        if os.path.basename(os.path.normpath(raiz)) != os.path.basename(raiz):
            fail_input(f"{nome}: caminho irregular")
        legado = os.path.join(raiz, "SHA256SUMS.txt")
        moderno = os.path.join(raiz, "output_hashes.csv")
        tabela, modo = {}, None
        if os.path.isfile(legado):
            modo = "LEGACY_CANONICAL"
            for linha in open(legado):
                p = linha.split(None, 1)
                if len(p) == 2:
                    tabela[p[1].strip().lstrip("./")] = p[0]
        if os.path.isfile(moderno):
            modo = "LEGACY_AND_MODERN" if modo else "MODERN_CANONICAL"
            oh = pd.read_csv(moderno)
            for k, v in zip(oh["path"], oh["sha256"]):
                if k in tabela and tabela[k] != v:
                    fail_input(f"{nome}: autoridades discordam em {k}")
                tabela[k] = v
        if not tabela:
            fail_input(f"{nome}: sem autoridade de proveniência em {raiz}")
        autoridades[nome] = (raiz, tabela, modo,
                             os.path.basename(os.path.normpath(raiz)))
        log.info(f"  {nome} {autoridades[nome][3]} [{modo}]: "
                 f"{len(tabela)} entradas de hash")

    def resolve(produtor, rel):
        raiz, tabela, modo, run_id = autoridades[produtor]
        p = os.path.join(raiz, rel)
        if not os.path.isfile(p):
            fail_input(f"{produtor}: produto canónico ausente: {p}")
        h = sha256(p)
        esperado = tabela.get(rel) or tabela.get(os.path.basename(rel))
        if esperado is None:
            fail_input(f"{produtor}: {rel} não consta da autoridade de hash")
        if h != esperado:
            fail_input(f"{produtor}: SHA de {rel} {h[:12]} != {esperado[:12]}")
        return p, h, modo, run_id

    run_id = f"s25_{t0:%Y%m%dT%H%M%S}_{os.getpid()}"
    destino_final = os.path.join(args.out_root or os.path.join(ANALYSIS_DIR, "final_matrices"),
                                 run_id)
    if os.path.exists(destino_final):
        fail_output(f"o directório do run já existe: {destino_final}")
    staging = tempfile.mkdtemp(prefix="s25_staging_")
    try:
        return _consolidar(args, t0, run_id, destino_final, staging, autoridades, resolve)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _consolidar(args, t0, run_id, destino_final, staging, autoridades, resolve):
    out_root = os.path.join(staging, run_id)
    figdata = os.path.join(out_root, "figdata")
    tabdata = os.path.join(out_root, "tabdata")
    manifests = os.path.join(out_root, "manifests")
    for d in (figdata, tabdata, manifests, os.path.join(figdata, "filter_examples")):
        os.makedirs(d, exist_ok=True)

    entradas, produtos = [], []

    def consolida(produtor, rel, destino, classe, regra, area):
        p, h, modo, rid = resolve(produtor, rel)
        alvo = os.path.join(figdata if area == "figdata" else tabdata, destino)
        n_in = n_out = ""
        if classe in ("DIRECT_COPY", "LEGACY_RENAME"):
            shutil.copyfile(p, alvo)
            if sha256(alvo) != h:
                fail_output(f"cópia byte a byte falhou em {destino}")
            n_in = n_out = len(pd.read_csv(alvo))
        elif classe == "LEGACY_RENAME_AND_ANNOTATE":
            d = pd.read_csv(p)
            n_in = len(d)
            if "threshold" in d.columns:
                d = d.rename(columns={"threshold": "matching_iou_threshold"})
            d["curve_type"] = CURVE_TYPE_ANOTADO
            d.to_csv(alvo, index=False)
            n_out = len(d)
            if n_out != n_in:
                fail_output(f"anotação alterou o número de linhas em {destino}")
        else:
            fail_output(f"classe desconhecida: {classe}")
        entradas.append({"producer_script": produtor, "canonical_run_id": rid,
                         "provenance_mode": modo, "source_path": rel,
                         "expected_sha256": h, "actual_sha256": h,
                         "sha256_verified": True})
        produtos.append({"area": area, "product": destino, "transform_class": classe,
                         "transform_rule": regra, "producer_script": produtor,
                         "canonical_run_id": rid, "source_path": rel,
                         "source_sha256": h, "rows_in": n_in, "rows_out": n_out,
                         "destination_sha256": sha256(alvo), "status": "CONSOLIDATED"})

    log.info("figdata:")
    for produtor, rel, destino, classe, regra in FIGDATA:
        consolida(produtor, rel, destino, classe, regra, "figdata")
    log.info("tabdata:")
    for produtor, rel, destino, classe, regra in TABDATA:
        consolida(produtor, rel, destino, classe, regra, "tabdata")

    linhas_ex = []
    for ex in EXEMPLOS:
        destino_dir = os.path.join(figdata, "filter_examples", ex)
        os.makedirs(destino_dir, exist_ok=True)
        for activo in ACTIVOS:
            rel = f"figdata/examples/{ex}/{activo}"
            p, h, modo, rid = resolve("Script14", rel)
            alvo = os.path.join(destino_dir, activo)
            shutil.copyfile(p, alvo)
            hd = sha256(alvo)
            if hd != h:
                fail_output(f"cópia do activo {rel} não é byte a byte")
            linhas_ex.append({"example_id": ex, "asset_role": activo.split(".")[0],
                              "source_run": rid, "selection_rule": "EXAMPLE_SELECTION_RULE_V2",
                              "source_path": rel, "source_sha256": h,
                              "destination_path": f"figdata/filter_examples/{ex}/{activo}",
                              "destination_sha256": hd, "sha256_verified": True})
            entradas.append({"producer_script": "Script14", "canonical_run_id": rid,
                             "provenance_mode": modo, "source_path": rel,
                             "expected_sha256": h, "actual_sha256": h,
                             "sha256_verified": True})
    ex_df = pd.DataFrame(linhas_ex)
    ex_p = os.path.join(figdata, "filter_examples", "manifest_examples.csv")
    ex_df.to_csv(ex_p, index=False)
    produtos.append({"area": "figdata", "product": "filter_examples/",
                     "transform_class": "CANONICAL_EXAMPLE_ASSET",
                     "transform_rule": "EXAMPLE_SELECTION_RULE_V2; ficheiros nomeados "
                                       "explicitamente, sem glob nem reselecção",
                     "producer_script": "Script14",
                     "canonical_run_id": autoridades["Script14"][3],
                     "source_path": "figdata/examples/", "source_sha256": "",
                     "rows_in": len(EXEMPLOS) * len(ACTIVOS), "rows_out": len(ex_df),
                     "destination_sha256": sha256(ex_p), "status": "CONSOLIDATED"})

    if args.flowchart_spec:
        if not os.path.isfile(args.flowchart_spec):
            fail_input(f"especificação do fluxograma ausente: {args.flowchart_spec}")
        try:
            spec = json.load(open(args.flowchart_spec))
        except Exception as e:
            fail_input(f"especificação do fluxograma ilegível: {e}")
        for ch in ("spec_id", "nodes", "edges", "visual_groups", "mandatory_text",
                   "forbidden_in_figure", "rendering_script"):
            if ch not in spec:
                fail_input(f"FLOWCHART_SPEC: falta a chave {ch}")
        if spec["spec_id"] != "PULMONARY_METHODOLOGY_FLOWCHART_V1":
            fail_input(f"FLOWCHART_SPEC: spec_id {spec['spec_id']} não é o congelado")
        ids = [n["id"] for n in spec["nodes"]]
        if len(ids) != len(set(ids)):
            fail_input("FLOWCHART_SPEC: identificador de nó repetido")
        soltos = [e for e in spec["edges"]
                  if e["from"] not in set(ids) or e["to"] not in set(ids)]
        if soltos:
            fail_input(f"FLOWCHART_SPEC: aresta para nó inexistente: {soltos[:3]}")
        cobertos = set().union(*[set(v) for v in spec["visual_groups"].values()])
        if cobertos != set(ids):
            fail_input(f"FLOWCHART_SPEC: nós sem grupo visual: "
                       f"{sorted(set(ids) - cobertos)} / fora dos nós: "
                       f"{sorted(cobertos - set(ids))}")
        fora = [k for k in spec["mandatory_text"] if k not in set(ids)]
        if fora:
            fail_input(f"FLOWCHART_SPEC: texto obrigatório para nó inexistente: {fora}")
        orfaos = [i for i in ids if not any(e["from"] == i or e["to"] == i
                                            for e in spec["edges"])]
        if orfaos:
            fail_input(f"FLOWCHART_SPEC: nós sem qualquer ligação: {orfaos}")
        alvo = os.path.join(figdata, "methodology_flowchart_spec.json")
        shutil.copyfile(args.flowchart_spec, alvo)
        produtos.append({"area": "figdata", "product": "methodology_flowchart_spec.json",
                         "transform_class": "STATIC_SPEC",
                         "transform_rule": "especificação congelada de nós, ligações e textos",
                         "producer_script": "STATIC", "canonical_run_id": "",
                         "source_path": args.flowchart_spec,
                         "source_sha256": sha256(args.flowchart_spec),
                         "rows_in": len(spec["nodes"]), "rows_out": len(spec["edges"]),
                         "destination_sha256": sha256(alvo), "status": "CONSOLIDATED"})
        g_spec = {"spec_id": spec["spec_id"], "nodes": len(spec["nodes"]),
                  "edges": len(spec["edges"]), "groups_cover_all_nodes": True,
                  "mandatory_text_nodes": sorted(spec["mandatory_text"])}
    else:
        BLOQUEADOS.append({
            "planned_product": "methodology_flowchart.pdf", "renderer": "Script28",
            "status": "BLOCKED_SOURCE", "reason": "STATIC_SPEC_NOT_PROVIDED",
            "detail": "a figura conceptual precisa de uma especificação congelada em "
                      "figdata/methodology_flowchart_spec.json; passe --flowchart-spec"})

    for b in BLOQUEADOS:
        produtos.append({"area": "figdata" if b["renderer"] == "Script28" else "tabdata",
                         "product": b["planned_product"], "transform_class": b["reason"],
                         "transform_rule": b["detail"], "producer_script": "",
                         "canonical_run_id": "", "source_path": "", "source_sha256": "",
                         "rows_in": "", "rows_out": "", "destination_sha256": "",
                         "status": "BLOCKED_SOURCE"})

    prod = pd.DataFrame(produtos)
    mf = prod[prod.area == "figdata"]
    mt = prod[prod.area == "tabdata"]
    mf.to_csv(os.path.join(manifests, "manifest_figures.csv"), index=False)
    mt.to_csv(os.path.join(manifests, "manifest_tables.csv"), index=False)
    pd.DataFrame(entradas).to_csv(os.path.join(out_root, "input_manifest.csv"), index=False)

    g = {"RUNS": {k: v[3] for k, v in autoridades.items()},
         "PROVENANCE_MODES": {k: v[2] for k, v in autoridades.items()},
         "INPUTS_VERIFIED": f"{len(entradas)}/{len(entradas)}",
         "FIGDATA_CONSOLIDATED": int((mf.status == "CONSOLIDATED").sum()),
         "TABDATA_CONSOLIDATED": int((mt.status == "CONSOLIDATED").sum()),
         "BLOCKED": prod[prod.status == "BLOCKED_SOURCE"][
             ["product", "transform_class"]].to_dict("records"),
         "NO_RECOMPUTATION_GATE": "nenhuma estatística é calculada neste script",
         "NO_GLOB_GATE": "exemplos enumerados por nome; nenhuma descoberta por padrão",
         "NO_FROC_NAME_GATE": "nenhum produto final chamado froc",
         "BYTE_COPY_GATE": "todas as cópias directas conferidas por SHA no destino",
         "SINGLE_SOURCE_PER_PRODUCT": "cada produto declara exactamente uma fonte"}
    g["FLOWCHART_SPEC_GATE"] = g_spec if args.flowchart_spec else "NAO_FORNECIDA"
    for r in produtos:
        if r["transform_class"] == "LEGACY_RENAME_AND_ANNOTATE":
            d = pd.read_csv(os.path.join(figdata, r["product"]))
            if "curve_type" not in d.columns or d.curve_type.nunique() != 1 \
                    or d.curve_type.iloc[0] != "MATCHING_IOU_SENSITIVITY":
                fail_output(f"CURVE_TYPE_GATE: {r['product']} sem o rótulo declarado")
    g["CURVE_TYPE_GATE"] = "MATCHING_IOU_SENSITIVITY constante nos produtos anotados"
    for b in prod[prod.status == "BLOCKED_SOURCE"]["product"]:
        for raiz_area in (figdata, tabdata):
            if os.path.exists(os.path.join(raiz_area, b)):
                fail_output(f"BLOCKED_NOT_WRITTEN_GATE: {b} está bloqueado mas foi escrito")
    g["BLOCKED_NOT_WRITTEN_GATE"] = "nenhum produto bloqueado existe no disco"
    for nome in list(mf["product"]) + list(mt["product"]):
        if "froc" in nome.lower():
            fail_output(f"NO_FROC_NAME_GATE: produto final com nome proibido: {nome}")
    if prod.duplicated(["area", "product"]).any():
        fail_output("SINGLE_SOURCE_PER_PRODUCT: produto repetido")
    cons = prod[prod.status == "CONSOLIDATED"]
    if cons.source_path.eq("").any() and not (cons.transform_class == "STATIC_SPEC").all():
        pass
    if not all(os.path.isfile(os.path.join(figdata, p)) for p in
               mf[(mf.status == "CONSOLIDATED") & (~mf["product"].str.endswith("/"))]["product"]):
        fail_output("produto de figdata declarado mas ausente no disco")
    if not all(os.path.isfile(os.path.join(tabdata, p)) for p in
               mt[mt.status == "CONSOLIDATED"]["product"]):
        fail_output("produto de tabdata declarado mas ausente no disco")

    saidas = {}
    for raiz_area, nome_area in ((figdata, "figdata"), (tabdata, "tabdata"),
                                 (manifests, "manifests")):
        for dp, _, fs in os.walk(raiz_area):
            for f in fs:
                p = os.path.join(dp, f)
                saidas[os.path.relpath(p, out_root)] = sha256(p)
    for f in ("input_manifest.csv",):
        saidas[f] = sha256(os.path.join(out_root, f))
    json.dump(g, open(os.path.join(out_root, "gates.json"), "w"), indent=1, default=str)
    saidas["gates.json"] = sha256(os.path.join(out_root, "gates.json"))
    pd.DataFrame([{"path": k, "sha256": v} for k, v in sorted(saidas.items())]).to_csv(
        os.path.join(out_root, "output_hashes.csv"), index=False)
    man = {
        "script": "25_generate_tables_and_figures.py", "run_id": run_id,
        "script_sha256": sha256(os.path.abspath(__file__)),
        "run_at": t0.isoformat(), "end_at": datetime.datetime.now().astimezone().isoformat(),
        "role": "PURE_DATA_CONSOLIDATOR",
        "renders": "NOTHING — figuras no Script 28, tabelas no Script 29",
        "sources": {k: v[3] for k, v in autoridades.items()},
        "definitions": {
            "allowed_transforms": ["DIRECT_COPY", "LEGACY_RENAME",
                                   "LEGACY_RENAME_AND_ANNOTATE",
                                   "CANONICAL_EXAMPLE_ASSET", "STATIC_SPEC"],
            "forbidden": ["qualquer estatística", "glob", "selecção por mtime",
                          "reselecção de exemplos", "nome froc em produto final"],
            "renamed": {"bd1_froc.csv": "bd1_matching_iou_sensitivity.csv",
                        "sensitivity_iou.csv": "bd1_sensitivity_iou_integrated.csv"},
            "not_promoted": ["lndb_froc.csv", "lndb_filter_froc.csv"],
            "example_selection": "EXAMPLE_SELECTION_RULE_V2, congelada no Script 14"},
        "counts": {"inputs_verified": len(entradas),
                   "figdata_products": int((mf.status == "CONSOLIDATED").sum()),
                   "tabdata_products": int((mt.status == "CONSOLIDATED").sum()),
                   "blocked": int((prod.status == "BLOCKED_SOURCE").sum()),
                   "example_assets": len(ex_df)},
        "gates": g,
        "promotion": "NOT PROMOTED — produtos ficam no directório do run"}
    json.dump(man, open(os.path.join(out_root, "manifest.json"), "w"), indent=2, default=str)
    os.makedirs(os.path.dirname(destino_final), exist_ok=True)
    shutil.move(out_root, destino_final)
    shutil.rmtree(staging, ignore_errors=True)
    out_root = destino_final
    log.info(f"figdata {g['FIGDATA_CONSOLIDATED']} · tabdata {g['TABDATA_CONSOLIDATED']} · "
             f"activos de exemplo {len(ex_df)} · bloqueados {len(g['BLOCKED'])}")
    for b in g["BLOCKED"]:
        log.info(f"  BLOCKED_SOURCE: {b['product']} — {b['transform_class']}")
    log.info(f"Saída: {out_root}")
    log.info("Script 25 concluído.")


if __name__ == "__main__":
    main()
