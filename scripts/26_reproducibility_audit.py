#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    26_reproducibility_audit.py

File type:
    Script

Purpose:
    Reproducibility audit: arithmetic identities, leakage checks, mask alignment, seeds, frozen parameters and manuscript consistency.

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
    - rider_iou_audit.csv
    - lndb_scan_audit.csv
    - final_audit_report.md

Outputs:
    - reproducibility_checks.csv
    - input_manifest.csv
    - gates.json
    - manifest.json

Dependencies:
    - SimpleITK
    - numpy
    - pandas

Execution:
    PULMONARY_BASE=<project_dir> python scripts/26_reproducibility_audit.py

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
import re
import shutil
import tempfile

import numpy as np
import pandas as pd
import SimpleITK as sitk

BASE = os.environ.get("PULMONARY_BASE") or os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_DIR = os.path.join(BASE, "analysis")
CONFIG_DIR = os.path.join(BASE, "config")
LOGS_DIR = os.path.join(BASE, "logs")
U = os.path.join(ANALYSIS_DIR, "upstream_canonical")

SEMENTE = 27
TAU_LIDC_PROTOCOLO = 0.60
LIDC_ANALITICO = 1009
LIDC_AVALIADOS = 876
LIDC_DEV, LIDC_INT = 718, 158
LIDC_FORA = 133
MOTIVOS_FORA = {"SUBTHRESHOLD_NODULES_ONLY": 112, "FAIL_CLUSTER_AMBIGUOUS": 14,
                "FAIL_READER_ASSIGNMENT": 6,
                "HOLD_ANNOTATION_OUTSIDE_RETAINED_BLOCK": 1}
LNDB_NOMINAL, LNDB_ANALITICO = 236, 211
RIDER_PARES, RIDER_CTS = 31, 62
N_ARQUITECTURAS, N_MODELOS = 10, 50
ARQ_10 = ["2D", "2D_REL", "2D_REM", "3D_CF", "3D_CFREM", "3D_F", "3D_FREM",
          "F100_ResEncM", "F100_fullres", "F100_lowres"]
PRODUTOS_BLOQUEADOS = ["dice_bd1.tex", "loss-", "dice-"]

RUNS = {
    "Script11B": os.path.join(U, "script11_final", "s11B_20260921T164539_27883"),
    "Script14": os.path.join(U, "script14_final", "s14b_20260922T203413_344007"),
    "Script16": os.path.join(U, "script16_final", "s16_20260923T094823_494840"),
    "Script19": os.path.join(ANALYSIS_DIR, "rider_matching", "s19_20260923T200344_675042"),
    "Script20": os.path.join(BASE, "features", "rider_morphometry",
                             "s20_20260923T211225_698236"),
    "Script21": os.path.join(ANALYSIS_DIR, "rider_repeatability", "s21_20260923T215458_708846"),
    "Script22": os.path.join(ANALYSIS_DIR, "lndb_external", "s22_20260924T084718_826234"),
    "Script23": os.path.join(ANALYSIS_DIR, "lndb_filter_validation",
                             "s23_20260924T153750_909396"),
    "Script24": os.path.join(ANALYSIS_DIR, "integrated_statistics",
                             "s24_20260924T171351_929519"),
    "Script25": os.path.join(ANALYSIS_DIR, "final_matrices",
                             "s25_20260924T231056_1013291"),
    "Script17A": os.path.join(ANALYSIS_DIR, "internal_verifier_application",
                              "s17a_20260924T225525_1009246")}

SCRIPT25_SUPERSEDIDO = "s25_20260924T192315_957346"
SCRIPT17A_CODE_SHA = \
    "d955c91b6865e9c4a9efd870d353582162edc2dd0bc5e7a1997fe34c4308931e"
S17A_N_DECISOES = 5573
S17A_N_LINHAS_METRICAS = 20
S17A_N_GT_POR_ARQ = 412
S17A_LIMIAR = 0.3417980765530837

TABELA_ESCRITA_PELO_PRODUTOR = {
    "Script17A": {"SHA256SUMS.txt", "output_hashes.csv"},
    "Script18": {"output_hashes.csv"}, "Script19": {"output_hashes.csv"},
    "Script20": {"output_hashes.csv"}, "Script21": {"output_hashes.csv"},
    "Script22": {"output_hashes.csv"}, "Script23": {"output_hashes.csv"},
    "Script24": {"output_hashes.csv"}, "Script25": {"output_hashes.csv"}}

os.makedirs(LOGS_DIR, exist_ok=True)
log = logging.getLogger("s26")
log.setLevel(logging.INFO)
if not log.handlers:
    _f = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for _h in (logging.StreamHandler(),
               logging.FileHandler(os.path.join(LOGS_DIR, "reproducibility_audit.log"))):
        _h.setFormatter(_f); log.addHandler(_h)
log.propagate = False

VERIFICACOES = []


def registo(nome, estado, detalhe="", evidencia=None):
    """Um único sítio por onde passa cada veredicto. `estado` nunca é booleano:
    é um dos estados explícitos, para que ausência não vire aprovação."""
    if estado not in ("PASS", "FAIL_INPUT", "FAIL_SCHEMA", "FAIL_PROVENANCE",
                      "FAIL_EMPTY_SUPPORT", "FAIL_VALUE", "NOT_ESTIMABLE"):
        raise SystemExit(f"estado inválido em {nome}: {estado}")
    icone = {"PASS": "✓", "NOT_ESTIMABLE": "?"}.get(estado, "✗")
    log.info(f"  [{icone}] {nome}: {estado} — {detalhe}")
    VERIFICACOES.append({"check": nome, "status": estado, "detail": detalhe,
                         "evidence": json.dumps(evidencia, default=str) if evidencia else ""})
    return estado == "PASS"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def le(caminho, nome, colunas=()):
    """Leitura com estado explícito: ausência e esquema incompleto são falhas
    distintas, nunca silêncio."""
    if not os.path.isfile(caminho):
        registo(nome, "FAIL_INPUT", f"ausente: {caminho}")
        return None
    d = pd.read_parquet(caminho) if caminho.endswith(".parquet") else pd.read_csv(caminho)
    falta = [c for c in colunas if c not in d.columns]
    if falta:
        registo(nome, "FAIL_SCHEMA", f"colunas ausentes {falta} em {os.path.basename(caminho)}")
        return None
    if len(d) == 0:
        registo(nome, "FAIL_EMPTY_SUPPORT", f"vazio: {os.path.basename(caminho)}")
        return None
    return d


TOPO = ("config", "analysis", "figdata", "tabdata", "features", "models",
        "predictions", "runs", "logs")


def reancorar(absoluto):
    """Um caminho absoluto gravado noutra máquina, ou noutra ancoragem, aponta
    para fora da árvore que estamos a auditar. Reancora-se em `PULMONARY_BASE`
    pelo primeiro directório de topo conhecido — mesma regra do Script 06 — e o
    que não reancorar devolve `None`, nunca o caminho de origem."""
    if os.path.commonpath([os.path.realpath(absoluto), os.path.realpath(BASE)]) \
            == os.path.realpath(BASE) and os.path.exists(absoluto):
        return absoluto
    partes = os.path.normpath(absoluto).split(os.sep)
    for i, p in enumerate(partes):
        if p in TOPO:
            cand = os.path.join(BASE, *partes[i:])
            return cand if os.path.isfile(cand) else None
    return None


def cabecalho(p):
    i = sitk.ReadImage(p)
    return (tuple(i.GetSize()), tuple(round(x, 6) for x in i.GetSpacing()),
            tuple(round(x, 4) for x in i.GetOrigin()),
            tuple(round(x, 6) for x in i.GetDirection()))


def audita_proveniencia():
    resolvidos, entradas = {}, []
    for nome, raiz in RUNS.items():
        if not os.path.isdir(raiz):
            registo(f"provenance_{nome}", "FAIL_INPUT", f"run canónico ausente: {raiz}")
            continue
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
            tabela.update(dict(zip(oh["path"], oh["sha256"])))
        if not tabela:
            registo(f"provenance_{nome}", "FAIL_PROVENANCE",
                    "sem SHA256SUMS.txt nem output_hashes.csv")
            continue
        maus, conferidos, irresolutos = [], 0, []
        for rel, esperado in tabela.items():
            p = next((c for c in (os.path.join(raiz, rel), os.path.join(BASE, rel))
                      if os.path.isfile(c)), None)
            if p is None:
                irresolutos.append(rel); continue
            conferidos += 1
            if sha256(p) != esperado:
                maus.append(rel)
        if maus or irresolutos:
            registo(f"provenance_{nome}", "FAIL_PROVENANCE",
                    f"{len(maus)} SHA divergentes {maus[:3]} · "
                    f"{len(irresolutos)} caminhos não resolvidos {irresolutos[:3]}")
            continue
        if conferidos == 0:
            registo(f"provenance_{nome}", "FAIL_PROVENANCE",
                    f"tabela com {len(tabela)} entradas e nenhum ficheiro conferido")
            continue
        registo(f"provenance_{nome}", "PASS",
                f"{os.path.basename(raiz)} [{modo}] — {conferidos} ficheiros conferidos")
        resolvidos[nome] = raiz
        entradas.append({"producer_script": nome, "canonical_run_id": os.path.basename(raiz),
                         "provenance_mode": modo, "files_verified": conferidos})
    return resolvidos, entradas


def audita_coorte_lidc():
    rec_p = os.path.join(CONFIG_DIR, "cohort", "COORTE_RECONCILIACAO.csv")
    cols = ["patient_id", "script07_subset", "exclusion_reason", "consensus_written",
            "final_cohort_status"]
    d = le(rec_p, "cohort_reconciliation_input", cols)
    if d is None:
        registo("cohort_row_level_reconciliation", "FAIL_INPUT",
                "sem a reconciliação ao nível do paciente — ver o estado de "
                "`cohort_reconciliation_input` para a causa")
        return
    if len(d) != LIDC_ANALITICO or d.patient_id.nunique() != LIDC_ANALITICO:
        registo("cohort_total_1009", "FAIL_VALUE",
                f"{len(d)} linhas / {d.patient_id.nunique()} únicos != {LIDC_ANALITICO}")
        return
    registo("cohort_total_1009", "PASS", f"{LIDC_ANALITICO} pacientes únicos")

    dev = set(d.loc[d.script07_subset == "development", "patient_id"])
    inte = set(d.loc[d.script07_subset == "internal", "patient_id"])
    fora = set(d.loc[d.script07_subset == "NOT_DRAWN", "patient_id"])
    ok = (len(dev) == LIDC_DEV and len(inte) == LIDC_INT and len(fora) == LIDC_FORA)
    registo("cohort_split_counts", "PASS" if ok else "FAIL_VALUE",
            f"development {len(dev)} · internal {len(inte)} · fora {len(fora)}")
    registo("cohort_disjoint", "PASS" if not (dev & inte) and not ((dev | inte) & fora)
            else "FAIL_VALUE",
            f"intersecções: dev∩int {len(dev & inte)}, avaliados∩fora {len((dev | inte) & fora)}")
    uniao = dev | inte | fora
    registo("cohort_union_complete", "PASS" if uniao == set(d.patient_id) else "FAIL_VALUE",
            f"união {len(uniao)} de {LIDC_ANALITICO}")
    registo("cohort_arithmetic", "PASS"
            if len(dev) + len(inte) + len(fora) == LIDC_ANALITICO else "FAIL_VALUE",
            f"{len(dev)} + {len(inte)} + {len(fora)} = {len(dev)+len(inte)+len(fora)}")

    motivos = d.loc[d.script07_subset == "NOT_DRAWN", "exclusion_reason"].value_counts().to_dict()
    registo("cohort_exclusion_reasons", "PASS" if motivos == MOTIVOS_FORA else "FAIL_VALUE",
            f"{motivos}", {"esperado": MOTIVOS_FORA})
    registo("cohort_exclusion_arithmetic",
            "PASS" if sum(MOTIVOS_FORA.values()) == LIDC_FORA else "FAIL_VALUE",
            f"{' + '.join(map(str, MOTIVOS_FORA.values()))} = {sum(MOTIVOS_FORA.values())}")

    sp = le(os.path.join(CONFIG_DIR, "bd1_split.csv"), "bd1_split_input",
            ["patient_id", "split"])
    if sp is None:
        return dev, inte
    sp_dev = set(sp.loc[sp.split.str.strip().str.lower() == "development", "patient_id"])
    sp_int = set(sp.loc[sp.split.str.strip().str.lower() == "internal", "patient_id"])
    registo("cohort_split_file_matches_reconciliation",
            "PASS" if (sp_dev == dev and sp_int == inte) else "FAIL_VALUE",
            f"development iguais: {sp_dev == dev} · internal iguais: {sp_int == inte}")
    registo("cohort_consensus_written",
            "PASS" if int((d.consensus_written.astype(str) == "True").sum()) == LIDC_AVALIADOS
            else "FAIL_VALUE",
            f"consenso escrito em {int((d.consensus_written.astype(str) == 'True').sum())} "
            f"de {LIDC_AVALIADOS}")
    return dev, inte


def audita_tau():
    p = os.path.join(CONFIG_DIR, "lidc_annotations_manifest.json")
    if not os.path.isfile(p):
        registo("lidc_consensus_threshold_provenance", "FAIL_INPUT",
                f"manifesto do produtor ausente ({p}); o valor do protocolo não "
                f"substitui prova de execução")
        return
    m = json.load(open(p))
    regra = m.get("CONSENSUS_RULE", "")
    achados = re.findall(r"(\d+\.\d+)", str(regra))
    if not achados:
        registo("lidc_consensus_threshold_provenance", "FAIL_SCHEMA",
                f"CONSENSUS_RULE sem valor numérico: {regra!r}")
        return
    tau = float(achados[0])
    congelado = m.get("SCRIPT04_OUTPUT_MANIFEST_FROZEN") == "YES"
    estado = m.get("SCRIPT04_OUTPUT_STATUS")
    ok = (tau == TAU_LIDC_PROTOCOLO and congelado and estado == "PASS")
    registo("lidc_consensus_threshold_provenance", "PASS" if ok else "FAIL_PROVENANCE",
            f"τ={tau} de {m.get('producer')} ({m.get('manifest_id')}); "
            f"congelado={congelado}, estado={estado}; protocolo={TAU_LIDC_PROTOCOLO}",
            {"consensus_rule": regra,
             "authority_file_sha256": sha256(p),
             "declared_manifest_sha256_NOT_INDEPENDENTLY_REPRODUCIBLE":
                 m.get("manifest_sha256"),
             "n_consensus_published": m.get("n_consensus_published")})
    registo("lidc_consensus_published_876",
            "PASS" if m.get("n_consensus_published") == LIDC_AVALIADOS else "FAIL_VALUE",
            f"{m.get('n_consensus_published')} consensos publicados")


def audita_lndb_rider(runs):
    p = os.path.join(CONFIG_DIR, "lndb_annotations_manifest.json")
    if not os.path.isfile(p):
        registo("lndb_cohort", "FAIL_INPUT", "manifesto do Script 05 ausente")
    else:
        m = json.load(open(p))
        linhas = m.get("rows")
        if not isinstance(linhas, list):
            registo("lndb_cohort", "FAIL_SCHEMA", "manifesto do Script 05 sem `rows`")
        elif not linhas:
            registo("lndb_cohort", "FAIL_EMPTY_SUPPORT",
                    "manifesto do Script 05 com `rows` vazio")
        elif any("analytic_status" not in r for r in linhas):
            registo("lndb_cohort", "FAIL_SCHEMA", "linhas sem `analytic_status`")
        else:
            part = {}
            for r in linhas:
                part[r["analytic_status"]] = part.get(r["analytic_status"], 0) + 1
            analitico = sum(v for k, v in part.items() if k.startswith("ANALYTIC"))
            ok = (len(linhas) == LNDB_NOMINAL and analitico == LNDB_ANALITICO
                  and sum(part.values()) == LNDB_NOMINAL
                  and part.get("EXCLUDED_REFERENCE_READER_DISCORDANCE") == 13
                  and part.get("EXCLUDED_DUPLICATE_IMAGE_PRESERVED") == 12)
            registo("lndb_cohort", "PASS" if ok else "FAIL_VALUE",
                    f"nominal {len(linhas)} · analítico {analitico} · partição {part}")
            decl = (m.get("SCRIPT05_NOMINAL_COUNT"), m.get("SCRIPT05_ANALYTIC_COUNT"),
                    m.get("n_reference_reader_discordance_excluded"),
                    m.get("SCRIPT05_EXCLUDED_PRESERVED"))
            if None in decl:
                registo("lndb_declared_counts", "FAIL_SCHEMA",
                        f"contadores do Script 05 ausentes: {decl}")
            else:
                coerente = (decl == (LNDB_NOMINAL, LNDB_ANALITICO, 13, 12)
                            and decl[0] == decl[1] + decl[2] + decl[3])
                registo("lndb_declared_counts", "PASS" if coerente else "FAIL_VALUE",
                        f"{decl[0]} = {decl[1]} + {decl[2]} + {decl[3]} "
                        f"= {decl[1] + decl[2] + decl[3]}")
            registo("lndb_no_tau_applied",
                    "PASS" if m.get("TAU_LNDB_APPLIED") is False else "FAIL_VALUE",
                    f"TAU_LNDB_APPLIED={m.get('TAU_LNDB_APPLIED')} — o LNDb não tem "
                    f"consenso voxel a voxel; nenhum τ pode ter sido aplicado")
    if "Script19" in runs:
        a = le(os.path.join(runs["Script19"], "rider_iou_audit.csv"), "rider_audit_input",
               ["patient_id", "architecture"])
        if a is not None:
            registo("rider_cohort", "PASS" if a.patient_id.nunique() == RIDER_PARES
                    else "FAIL_VALUE",
                    f"{a.patient_id.nunique()} pares ({RIDER_PARES} esperados), "
                    f"{a.patient_id.nunique() * 2} CTs ({RIDER_CTS} esperados)")
    else:
        registo("rider_cohort", "FAIL_INPUT", "run do Script 19 não resolvido")

    tm = os.path.join(CONFIG_DIR, "MANIFESTO_TERMINAIS_50.csv")
    d = le(tm, "fold_models_input", ["architecture", "fold", "terminal_checkpoint"])
    if d is not None:
        arqs = sorted(d.architecture.unique())
        pares = d.groupby("architecture").fold.nunique().to_dict()
        ok = (len(d) == N_MODELOS and arqs == sorted(ARQ_10)
              and all(v == 5 for v in pares.values()))
        registo("architectures_and_models", "PASS" if ok else "FAIL_VALUE",
                f"{len(d)} modelos terminais ({N_MODELOS} esperados) · "
                f"{len(arqs)} arquitecturas ({N_ARQUITECTURAS} esperadas) · "
                f"folds por arquitectura {sorted(set(pares.values()))}",
                {"arquitecturas": arqs})


def audita_identidades(runs):
    for produtor, dependentes in (
            ("Script11B", ("bd1_ngt_invariant", "bd1_objects_match_aggregates")),
            ("Script22", ("lndb_arithmetic_identities",)),
            ("Script19", ("rider_arithmetic_identities",))):
        if produtor not in runs:
            for c in dependentes:
                registo(c, "FAIL_INPUT", f"run do {produtor} não resolvido na secção A")
    if "Script11B" in runs:
        det = le(os.path.join(runs["Script11B"], "analysis", "bd1_detection_metrics.csv"),
                 "bd1_detection_input", ["model", "subset", "threshold", "TP", "FN", "FP"])
        if det is not None:
            mau = []
            for (mo, su), g in det.groupby(["model", "subset"]):
                if g.eval("TP + FN").nunique() != 1:
                    mau.append((mo, su, "N_GT varia com o limiar"))
            registo("bd1_ngt_invariant", "PASS" if not mau else "FAIL_VALUE",
                    f"{len(det)} linhas · {len(mau)} violações", mau[:3])
            obj = le(os.path.join(runs["Script11B"], "analysis", "bd1_object_matches.csv"),
                     "bd1_objects_input", ["model", "subset", "status", "tp_010"])
            if obj is not None:
                dif = []
                for (mo, su), g in obj.groupby(["model", "subset"]):
                    tp = int((g.status == "TP").sum()); fn = int((g.status == "FN").sum())
                    fp = int((g.status == "FP").sum())
                    r = det[(det.model == mo) & (det.subset == su)
                            & (np.isclose(det.threshold, 0.10))]
                    if len(r) != 1 or (tp, fn, fp) != (int(r.TP.iloc[0]), int(r.FN.iloc[0]),
                                                       int(r.FP.iloc[0])):
                        dif.append((mo, su))
                registo("bd1_objects_match_aggregates", "PASS" if not dif else "FAIL_VALUE",
                        f"{len(det[np.isclose(det.threshold, 0.10)])} combinações · "
                        f"{len(dif)} divergentes", dif[:3])
    if "Script22" in runs:
        a = le(os.path.join(runs["Script22"], "lndb_scan_audit.csv"), "lndb_audit_input",
               ["n_gt_components", "n_pred_components", "tp", "fn", "fp"])
        if a is not None:
            v1 = int((a.tp + a.fn != a.n_gt_components).sum())
            v2 = int((a.tp + a.fp != a.n_pred_components).sum())
            registo("lndb_arithmetic_identities", "PASS" if v1 == v2 == 0 else "FAIL_VALUE",
                    f"{len(a)} linhas · TP+FN≠N_GT em {v1} · TP+FP≠N_pred em {v2}")
    if "Script19" in runs:
        a = le(os.path.join(runs["Script19"], "rider_iou_audit.csv"), "rider_identity_input",
               ["n_ct1_native", "n_ct2_native", "n_pairs", "n_ct1_only", "n_ct2_only"])
        if a is not None:
            v1 = int((a.n_ct1_native != a.n_pairs + a.n_ct1_only).sum())
            v2 = int((a.n_ct2_native != a.n_pairs + a.n_ct2_only).sum())
            registo("rider_arithmetic_identities", "PASS" if v1 == v2 == 0 else "FAIL_VALUE",
                    f"{len(a)} linhas · violações CT1 {v1} · CT2 {v2}")


def audita_separacao(runs, dev, inte):
    m = os.path.join(ANALYSIS_DIR, "filter_bd1_analysis_manifest.json")
    if not os.path.isfile(m):
        registo("bd1_filters_oof_only", "FAIL_INPUT", f"manifesto do Script 16 ausente: {m}")
    else:
        j = json.load(open(m))
        pop = j.get("population")
        if not isinstance(pop, dict) or "analysed" not in pop or "patients" not in pop:
            registo("bd1_filters_oof_only", "FAIL_SCHEMA",
                    "manifesto do Script 16 sem `population.analysed`/`population.patients`")
        else:
            ok = (pop.get("analysed") == "OOF only" and pop.get("patients") == LIDC_DEV
                  and str(pop.get("internal_use", "")).startswith("NONE"))
            registo("bd1_filters_oof_only", "PASS" if ok else "FAIL_VALUE",
                    f"analisado={pop.get('analysed')!r} · pacientes={pop.get('patients')} "
                    f"({LIDC_DEV} esperados) · internos descartados="
                    f"{pop.get('internal_discarded')} · uso interno="
                    f"{str(pop.get('internal_use'))[:40]!r}")

    a = os.path.join(ANALYSIS_DIR, "filter_verifier_manifest.json")
    if not os.path.isfile(a):
        registo("verifier_trained_on_oof_only", "FAIL_INPUT", "manifesto do Script 17 ausente")
    else:
        j = json.load(open(a))
        pop = j.get("population")
        obrigatorias = ("subset", "patients", "internal_rows_in_training")
        if not isinstance(pop, dict) or any(k not in pop for k in obrigatorias):
            registo("verifier_trained_on_oof_only", "FAIL_SCHEMA",
                    f"`population` do Script 17 sem {obrigatorias}")
        else:
            ok = (pop["subset"] == "oof" and pop["internal_rows_in_training"] == 0
                  and pop["patients"] == LIDC_DEV)
            registo("verifier_trained_on_oof_only", "PASS" if ok else "FAIL_VALUE",
                    f"subset={pop['subset']!r} · pacientes={pop['patients']} · "
                    f"linhas internas no treino={pop['internal_rows_in_training']}")
        pkl = os.path.join(BASE, "models", "filter_verifier_A.pkl")
        esperado = (j.get("outputs_sha256") or {}).get("filter_verifier_A.pkl")
        if not os.path.isfile(pkl):
            registo("verifier_binary_identity", "FAIL_INPUT", f"ausente: {pkl}")
        elif not esperado:
            registo("verifier_binary_identity", "FAIL_SCHEMA",
                    "`outputs_sha256.filter_verifier_A.pkl` ausente do manifesto do 17")
        else:
            obtido = sha256(pkl)
            registo("verifier_binary_identity",
                    "PASS" if obtido == esperado else "FAIL_PROVENANCE",
                    f"SHA do .pkl {obtido[:12]} vs declarado {esperado[:12]}")


def e_estocastico(caminho_script):
    """Sorteios e sementes deste código, por AST — `random` numa docstring ou num
    nome de coluna não é sorteio, e uma procura de texto não sabe a diferença.

    Devolve (sorteia, exemplo, constante, literais, sem_argumento).

    `constante` é o valor da constante congelada do módulo (`SEED`/`SEMENTE`) —
    a única semente que a AST pode resolver com honestidade, porque nestes
    scripts o valor viaja por parâmetro (`default_rng(seed)`) e segui-lo até à
    chamada exigiria interpretar o programa. `literais` são as sementes escritas
    à mão nos construtores, registadas como evidência para leitura humana, e
    `sem_argumento` são geradores construídos sem semente nenhuma — esses são
    defeito de reprodutibilidade em qualquer circunstância.
    """
    alvos = {"choice", "permutation", "shuffle", "sample", "resample", "randint",
             "rand", "randn", "random_sample"}
    construtores = {"default_rng", "RandomState"}
    arvore = ast.parse(open(caminho_script, encoding="utf-8").read())
    constante = None
    for n in arvore.body:
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id in ("SEED", "SEMENTE"):
                    constante = n.value.value
    sorteia, exemplo, literais, sem_argumento = False, None, set(), []
    for n in ast.walk(arvore):
        if not isinstance(n, ast.Call):
            continue
        f, partes = n.func, []
        while isinstance(f, ast.Attribute):
            partes.append(f.attr); f = f.value
        if isinstance(f, ast.Name):
            partes.append(f.id)
        partes.reverse()
        if not partes:
            continue
        nome = ".".join(partes)
        if partes[-1] in construtores:
            if not n.args and not n.keywords:
                sem_argumento.append(nome)
            elif n.args and isinstance(n.args[0], ast.Constant) \
                    and isinstance(n.args[0].value, int):
                literais.add(n.args[0].value)
            sorteia, exemplo = True, exemplo or nome
        elif partes[-1] in alvos or "random" in partes:
            sorteia, exemplo = True, exemplo or nome
    return sorteia, exemplo, constante, literais, sem_argumento


def audita_semente_e_ajuste(runs):
    """A semente exige-se a quem sorteia. Quem não sorteia não é dispensado por
    omissão: prova-se pelo código que não há sorteio."""
    a = os.path.join(ANALYSIS_DIR, "filter_verifier_manifest.json")
    fontes = [("Script16", os.path.join(ANALYSIS_DIR, "filter_bd1_analysis_manifest.json")),
              ("Script17", a)]
    for nome in ("Script19", "Script21", "Script22", "Script23", "Script24", "Script25"):
        if nome in runs:
            fontes.append((nome, os.path.join(runs[nome], "manifest.json")))
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    j17 = json.load(open(a)) if os.path.isfile(a) else None

    sementes, dispensados, problemas, outras = {}, {}, [], {}
    for nome, caminho in fontes:
        if not os.path.isfile(caminho):
            problemas.append(f"{nome}: manifesto ausente"); continue
        j = json.load(open(caminho))
        fonte = os.path.join(scripts_dir, str(j.get("script", "")))
        if not j.get("script") or not os.path.isfile(fonte):
            problemas.append(f"{nome}: manifesto não identifica o código que correu")
            continue
        if j.get("script_sha256") and sha256(fonte) != j["script_sha256"]:
            problemas.append(f"{nome}: o código em disco não é o que produziu o run")
            continue
        sorteia, onde, constante, literais, sem_argumento = e_estocastico(fonte)
        d = j.get("bootstrap") or (j.get("definitions") or {}).get("bootstrap") or {}
        if not sorteia:
            dispensados[nome] = "sem sorteio no código"
            if d.get("seed") is not None:
                problemas.append(f"{nome}: declara seed sem passo estocástico")
            continue
        if sem_argumento:
            problemas.append(f"{nome}: gerador sem semente em {sem_argumento[:2]}")
            continue
        fontes_semente = {}
        if constante is not None:
            fontes_semente["codigo"] = int(constante)
        if "seed" in d:
            fontes_semente["manifesto"] = int(d["seed"])
        if not fontes_semente and len(literais) == 1:
            fontes_semente["literal"] = int(next(iter(literais)))
        if not fontes_semente:
            problemas.append(f"{nome}: sorteia em {onde} e não fixa semente nem na "
                             f"constante do módulo, nem no manifesto, nem numa "
                             f"única literal ({sorted(literais) or 'nenhuma'})")
            continue
        if len(set(fontes_semente.values())) > 1:
            problemas.append(f"{nome}: sementes discordantes {fontes_semente}")
            continue
        sementes[nome] = next(iter(fontes_semente.values()))
        if literais - {SEMENTE}:
            outras.setdefault(nome, sorted(literais - {SEMENTE}))
    if j17 is not None and "random_state" in (j17.get("logistic") or {}):
        sementes["Script17_logistic"] = j17["logistic"]["random_state"]

    if problemas:
        registo("seed_27_everywhere", "FAIL_SCHEMA",
                f"sementes {sementes} · dispensados {dispensados} · problemas {problemas}")
    elif not sementes:
        registo("seed_27_everywhere", "FAIL_EMPTY_SUPPORT",
                "nenhum produtor estocástico encontrado — não há o que verificar, "
                "e isso contradiz o bootstrap por paciente do protocolo")
    else:
        erradas = {k: v for k, v in sementes.items() if int(v) != SEMENTE}
        registo("seed_27_everywhere", "PASS" if not erradas else "FAIL_VALUE",
                (f"{len(sementes)} produtores estocásticos, todos com semente "
                 f"{SEMENTE}: {sementes} · sem sorteio, semente não aplicável: "
                 f"{sorted(dispensados)} · correntes auxiliares fora de {SEMENTE} "
                 f"(auto-testes, não reprovam): {outras or 'nenhuma'}")
                if not erradas else f"sementes fora de {SEMENTE}: {erradas}",
                {"sementes": sementes, "dispensados": dispensados,
                 "correntes_auxiliares": outras})

    if "Script23" not in runs:
        registo("no_lndb_tuning", "FAIL_INPUT", "run do Script 23 não resolvido")
        return
    mp = os.path.join(runs["Script23"], "manifest.json")
    if not os.path.isfile(mp):
        registo("no_lndb_tuning", "FAIL_INPUT", "manifesto do Script 23 ausente")
        return
    if j17 is None:
        registo("no_lndb_tuning", "FAIL_INPUT", "manifesto do Script 17 ausente")
        return
    thr17 = ((j17.get("results") or {}).get("A") or {}).get("threshold")
    j = json.load(open(mp))
    v = (j.get("definitions") or {}).get("verifier") or {}
    if thr17 is None or "threshold" not in v:
        registo("no_lndb_tuning", "FAIL_SCHEMA",
                f"limiar ausente: results.A.threshold={thr17}, "
                f"definitions.verifier.threshold={'threshold' in v}")
        return
    thr23 = str(v["threshold"]).strip("'\"")
    mesmo = repr(float(thr23)) == repr(float(thr17)) and thr23 == repr(float(thr17))
    decl = (v.get("THRESHOLD_TUNING_ON_LNDB") == "NO"
            and v.get("RECALIBRATION_ON_LNDB") == "NONE"
            and v.get("THRESHOLD_SOURCE") == "SCRIPT17_DEVELOPMENT_OOF")
    try:
        anterior = (datetime.datetime.fromisoformat(j17["end_at"])
                    < datetime.datetime.fromisoformat(j["run_at"]))
    except (KeyError, ValueError) as e:
        registo("no_lndb_tuning", "FAIL_SCHEMA", f"carimbos temporais ilegíveis: {e}")
        return
    ok = mesmo and decl and anterior
    registo("no_lndb_tuning", "PASS" if ok else "FAIL_VALUE",
            f"limiar do 17 {thr17!r} == usado no 23 {thr23!r}: {mesmo} · "
            f"declarações do 23: {decl} · 17 termina antes de o 23 começar: {anterior}",
            {"script17_end": j17.get("end_at"), "script23_start": j.get("run_at")})
    ct = os.path.join(CONFIG_DIR, "filter_verifier_threshold.json")
    if not os.path.isfile(ct):
        registo("verifier_threshold_frozen", "FAIL_INPUT", f"ausente: {ct}")
    else:
        c = json.load(open(ct))
        ma = c.get("model_A") or {}
        if "threshold" not in ma:
            registo("verifier_threshold_frozen", "FAIL_SCHEMA",
                    "`model_A.threshold` ausente da configuração congelada")
        else:
            igual = (repr(float(ma["threshold"])) == repr(float(thr17))
                     and c.get("canonical_model") == "A")
            registo("verifier_threshold_frozen", "PASS" if igual else "FAIL_VALUE",
                    f"config {ma['threshold']!r} · Script 17 {thr17!r} · "
                    f"modelo canónico {c.get('canonical_model')!r}")
    if "Script19" in runs:
        pares = le(os.path.join(runs["Script19"], "rider_iou_pairs.csv"), "rider_pairs_input")
        if pares is not None:
            proibidas = [c for c in pares.columns if c.upper() in ("TP", "FP", "FN", "LABEL")]
            registo("rider_has_no_detection_labels",
                    "PASS" if not proibidas else "FAIL_VALUE",
                    f"colunas proibidas: {proibidas or 'nenhuma'}")


def audita_geometria(runs, dev, inte, limite=0):
    """LAYER B: leitura independente dos cabeçalhos, sem carregar um único voxel.

    Não se confia em nenhum `identity_*_ok` a montante — cada quarteto
    (Size, Spacing, Origin, Direction) é relido aqui e comparado entre a CT
    recortada e tudo o que tem de partilhar o seu referencial. `limite` só
    existe para o ensaio; no run canónico é 0 e a varredura é exaustiva.
    """
    from caminhos_canonicos import raiz_recortes, ct_de, mascara_pulmonar_de
    try:
        raiz = raiz_recortes()
    except Exception as e:
        for n in ("geometry_bd1", "geometry_rider", "geometry_lndb"):
            registo(n, "FAIL_INPUT", f"raiz dos recortes não resolve: {e}")
        return

    def varre(nome, casos, banco, extras, esperado):
        """`extras(banco, caso)` devolve os ficheiros que têm de partilhar o
        referencial da CT. Ausência é FAIL_INPUT, nunca ausência de problema."""
        if not casos:
            registo(nome, "FAIL_EMPTY_SUPPORT", "sem casos a verificar")
            return
        maus, ausentes, lidos = [], [], 0
        for caso in casos:
            try:
                c = ct_de(raiz, banco, caso)
            except Exception as e:
                ausentes.append((caso, f"ct: {e}")); continue
            if not os.path.isfile(c):
                ausentes.append((caso, "ct ausente")); continue
            try:
                hc = cabecalho(c)
            except Exception as e:
                maus.append((caso, f"ct ilegível: {e}")); continue
            lidos += 1
            for etiqueta, p in extras(banco, caso):
                if p is None or not os.path.isfile(p):
                    ausentes.append((caso, etiqueta)); continue
                try:
                    h = cabecalho(p)
                except Exception as e:
                    maus.append((caso, f"{etiqueta} ilegível: {e}")); continue
                lidos += 1
                if h != hc:
                    difs = [k for k, x, y in zip(("Size", "Spacing", "Origin", "Direction"),
                                                 hc, h) if x != y]
                    maus.append((caso, etiqueta, difs))
        estado = ("FAIL_VALUE" if maus else
                  ("FAIL_INPUT" if ausentes else
                   ("PASS" if len(casos) == esperado or limite else "FAIL_EMPTY_SUPPORT")))
        registo(nome, estado,
                f"{len(casos)} casos ({esperado} esperados) · {lidos} cabeçalhos lidos · "
                f"divergentes {len(maus)} · ausentes {len(ausentes)}",
                {"divergentes": maus[:5], "ausentes": ausentes[:5]})

    def com_mascara_e_consenso(banco, caso):
        return [("lung_mask_crop", mascara_pulmonar_de(raiz, banco, caso)),
                ("consensus", os.path.join(raiz, banco, "labels", caso, "consensus.nii.gz"))]

    def so_mascara(banco, caso):
        return [("lung_mask_crop", mascara_pulmonar_de(raiz, banco, caso))]

    ids = sorted((dev or set()) | (inte or set()))
    varre("geometry_bd1", ids[:limite] if limite else ids, "lidc_idri",
          com_mascara_e_consenso, LIDC_AVALIADOS)

    if "Script19" in runs:
        a = pd.read_csv(os.path.join(runs["Script19"], "rider_iou_audit.csv"))
        exames = sorted(f"{p}_{k}" for p in a.patient_id.unique() for k in (1, 2))
        varre("geometry_rider", exames[:limite] if limite else exames, "rider",
              so_mascara, RIDER_CTS)
    else:
        registo("geometry_rider", "FAIL_INPUT", "run do Script 19 não resolvido")

    if "Script22" in runs:
        a = pd.read_csv(os.path.join(runs["Script22"], "lndb_scan_audit.csv"))
        sids = sorted(a.scan_id.unique())

        def lndb_extras(banco, caso):
            itens = com_mascara_e_consenso(banco, caso)
            for arch in ("3D_F", "3D_FREM"):
                itens.append((f"pred_{arch}", os.path.join(
                    BASE, "predictions", "lndb", arch, f"{caso}.nii.gz")))
            return itens

        varre("geometry_lndb", sids[:limite] if limite else sids, "lndb",
              lndb_extras, LNDB_ANALITICO)
    else:
        registo("geometry_lndb", "FAIL_INPUT", "run do Script 22 não resolvido")


DEPENDENTES_S25 = ("final_package_integrity", "blocked_products_absent",
                   "no_froc_in_final_names", "final_package_runs_are_audited")


def audita_pacote_final(runs):
    if "Script25" not in runs:
        for c in DEPENDENTES_S25:
            registo(c, "FAIL_INPUT", "run do Script 25 não resolvido na secção A")
        return None
    r = runs["Script25"]
    cols = ["area", "product", "status", "transform_class", "source_path",
            "source_sha256", "destination_sha256", "producer_script",
            "canonical_run_id"]
    mf = le(os.path.join(r, "manifests", "manifest_figures.csv"),
            "manifest_figures_input", cols)
    mt = le(os.path.join(r, "manifests", "manifest_tables.csv"),
            "manifest_tables_input", cols)
    if mf is None or mt is None:
        for c in DEPENDENTES_S25:
            registo(c, "FAIL_INPUT",
                    "manifestos do pacote final ilegíveis — ver "
                    "`manifest_figures_input`/`manifest_tables_input`")
        return None
    todos = pd.concat([mf, mt], ignore_index=True)
    cons = todos[todos.status == "CONSOLIDATED"]
    bloq = todos[todos.status == "BLOCKED_SOURCE"]
    maus, ficheiros, pastas, origens = [], 0, 0, 0
    for _, x in cons.iterrows():
        alvo = os.path.join(r, str(x["area"]), str(x["product"]))
        if str(x["product"]).endswith("/"):
            if not os.path.isdir(alvo):
                maus.append((x["product"], "pasta ausente"))
            elif not os.listdir(alvo):
                maus.append((x["product"], "pasta vazia"))
            else:
                pastas += 1
            continue
        if not os.path.isfile(alvo):
            maus.append((x["product"], "ausente")); continue
        if not isinstance(x["destination_sha256"], str) or not x["destination_sha256"]:
            maus.append((x["product"], "sem SHA de destino no manifesto")); continue
        if sha256(alvo) != x["destination_sha256"]:
            maus.append((x["product"], "SHA de destino divergente")); continue
        ficheiros += 1
        sp = str(x["source_path"])
        produtor = str(x["producer_script"])
        if os.path.isabs(sp):
            origem = reancorar(sp)
        else:
            origem = os.path.join(RUNS[produtor], sp) if produtor in RUNS else None
        if origem is None or not os.path.isfile(origem):
            maus.append((x["product"], f"origem não resolve: {produtor}:{sp}")); continue
        o_sha = sha256(origem)
        if not isinstance(x["source_sha256"], str) or not x["source_sha256"]:
            maus.append((x["product"], "sem SHA de origem no manifesto")); continue
        if o_sha != x["source_sha256"]:
            maus.append((x["product"], "SHA de origem divergente")); continue
        if x["transform_class"] == "DIRECT_COPY" and o_sha != x["destination_sha256"]:
            maus.append((x["product"], "DIRECT_COPY com origem != destino")); continue
        origens += 1
    rastreados = ficheiros + pastas
    if len(cons) == 0:
        registo("final_package_integrity", "FAIL_EMPTY_SUPPORT",
                f"{len(todos)} produtos declarados e nenhum CONSOLIDATED a rastrear")
    elif rastreados == 0:
        registo("final_package_integrity", "FAIL_PROVENANCE",
                f"{len(cons)} produtos declarados e nenhum rastreado")
    else:
        registo("final_package_integrity", "PASS" if not maus else "FAIL_PROVENANCE",
                f"{len(cons)} consolidados · {ficheiros} destinos com SHA conferido · "
                f"{origens} com origem resolvida e conferida · {pastas} pastas não "
                f"vazias · {len(maus)} problemas", maus[:5])

    if len(bloq) == 0:
        registo("blocked_products_absent", "FAIL_EMPTY_SUPPORT",
                "nenhum produto BLOCKED_SOURCE no manifesto — o Script 25 fechou com "
                "dois bloqueios conhecidos; a sua ausência é ela própria uma anomalia")
    else:
        presentes = []
        for area in ("figdata", "tabdata"):
            d = os.path.join(r, area)
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if any(padrao in f for padrao in PRODUTOS_BLOQUEADOS):
                    presentes.append(os.path.join(area, f))
        registo("blocked_products_absent", "PASS" if not presentes else "FAIL_VALUE",
                f"{len(bloq)} bloqueados declarados · ficheiros que correspondem a um "
                f"padrão bloqueado: {presentes or 'nenhum'}",
                {"bloqueados": bloq["product"].tolist()})

    nomes = [p for p in todos["product"] if "froc" in str(p).lower()]
    no_disco = [f for area in ("figdata", "tabdata")
                if os.path.isdir(os.path.join(r, area))
                for f in os.listdir(os.path.join(r, area)) if "froc" in f.lower()]
    registo("no_froc_in_final_names",
            "PASS" if not nomes and not no_disco else "FAIL_VALUE",
            f"no manifesto: {nomes or 'nenhum'} · no disco: {no_disco or 'nenhum'}")

    validos = {os.path.basename(v) for v in runs.values()}
    orfaos = sorted(set(todos["canonical_run_id"].dropna().astype(str)) - validos - {""})
    registo("final_package_runs_are_audited",
            "PASS" if not orfaos else "FAIL_PROVENANCE",
            f"runs citados pelo Script 25 e não validados na secção A: {orfaos or 'nenhum'}")
    return bloq


def audita_manuscrito(caminho, sha_esperado, bloq):
    if not caminho:
        registo("manuscript_crosscheck", "NOT_ESTIMABLE",
                "nenhum manuscrito canónico declarado; passe --manuscript e "
                "--manuscript-sha256 para o run canónico")
        return
    if not os.path.isfile(caminho):
        registo("manuscript_crosscheck", "FAIL_INPUT", f"manuscrito ausente: {caminho}")
        return
    h = sha256(caminho)
    if sha_esperado and h != sha_esperado:
        registo("manuscript_crosscheck", "FAIL_PROVENANCE",
                f"SHA {h[:12]} != {sha_esperado[:12]}")
        return
    texto = open(caminho, encoding="utf-8", errors="replace").read()
    if not texto.strip():
        registo("manuscript_crosscheck", "FAIL_EMPTY_SUPPORT",
                f"manuscrito vazio: {caminho}")
        return
    registo("manuscript_crosscheck", "PASS",
            f"manuscrito lido: {os.path.basename(caminho)} · SHA {h[:12]} · "
            f"{len(texto)} caracteres · SHA declarado conferido: {bool(sha_esperado)}")
    citados = [b for b in PRODUTOS_BLOQUEADOS if b in texto]
    registo("manuscript_does_not_cite_blocked",
            "PASS" if not citados else "FAIL_VALUE",
            f"produtos bloqueados citados: {citados or 'nenhum'}")
    obsoletos = {"821": "split histórico pré-fusão", "189": "split histórico pré-fusão",
                 "822": "contagem obsoleta", "813": "coorte órfã de Julho",
                 "709": "cruzamento que não é coorte"}
    achados = {k: v for k, v in obsoletos.items()
               if re.search(rf"(?<![\d.]){k}(?![\d])", texto)}
    registo("manuscript_no_obsolete_counts",
            "PASS" if not achados else "FAIL_VALUE",
            f"números obsoletos presentes: {achados or 'nenhum'}")
    esperados = {"1009": LIDC_ANALITICO, "718": LIDC_DEV, "158": LIDC_INT,
                 "211": LNDB_ANALITICO, "236": LNDB_NOMINAL}
    em_falta = [k for k in esperados
                if not re.search(rf"(?<![\d.]){k}(?![\d])", texto.replace(".", ""))]
    registo("manuscript_declares_canonical_counts",
            "PASS" if not em_falta else "FAIL_VALUE",
            f"contagens canónicas ausentes do texto: {em_falta or 'nenhuma'}")
    registo("manuscript_no_froc_term",
            "PASS" if not re.search(r"\bFROC\b", texto) else "FAIL_VALUE",
            "termo FROC presente" if re.search(r"\bFROC\b", texto) else "termo FROC ausente")


DEPENDENTES_S17A = ("script17a_provenance", "script17a_cohort",
                    "script17a_object_support", "script17a_metrics",
                    "script17a_arithmetic", "script17a_denominators",
                    "script17a_degenerate_case", "script17a_inference_scope")


def audita_script17a(runs, dev, inte):
    """Oito gates separados, de propósito. Um `PASS` composto esconde a falha de
    um dos membros; aqui cada afirmação cai sozinha."""
    if "Script17A" not in runs:
        for c in DEPENDENTES_S17A:
            registo(c, "FAIL_INPUT", "run do Script 17A não resolvido na secção A")
        return None
    r = runs["Script17A"]
    man = os.path.join(r, "manifest.json")
    if not os.path.isfile(man):
        for c in DEPENDENTES_S17A:
            registo(c, "FAIL_INPUT", "manifesto do Script 17A ausente")
        return None
    j = json.load(open(man))
    v = (j.get("definitions") or {}).get("verifier") or {}

    pkl = os.path.join(BASE, "models", "filter_verifier_A.pkl")
    cfg_p = os.path.join(CONFIG_DIR, "filter_verifier_threshold.json")
    problemas = []
    if j.get("script_sha256") != SCRIPT17A_CODE_SHA:
        problemas.append(f"SHA do código {str(j.get('script_sha256'))[:12]} != "
                         f"autorizado {SCRIPT17A_CODE_SHA[:12]}")
    if not os.path.isfile(pkl):
        problemas.append("modelo ausente")
    elif (j.get("sources") or {}).get("verifier_sha256") != sha256(pkl):
        problemas.append("SHA do modelo não confere com o declarado pelo run")
    if "threshold" not in v:
        problemas.append("limiar ausente do manifesto")
    elif repr(float(str(v["threshold"]).strip("'\""))) != repr(S17A_LIMIAR):
        problemas.append(f"limiar {v['threshold']!r} != {S17A_LIMIAR!r}")
    if os.path.isfile(cfg_p):
        cfg = json.load(open(cfg_p))
        if list(v.get("features") or []) != list((cfg.get("model_A") or {}).get("features") or []):
            problemas.append("ordem das variáveis != configuração publicada")
    else:
        problemas.append("configuração do limiar ausente")
    if v.get("feature_order_enforced_by") != "POSITIONAL_FROM_PAYLOAD_FEATURES":
        problemas.append("o run não declara como impôs a ordem das variáveis")
    registo("script17a_provenance", "PASS" if not problemas else "FAIL_PROVENANCE",
            f"código {str(j.get('script_sha256'))[:12]} · modelo "
            f"{str((j.get('sources') or {}).get('verifier_sha256'))[:12]} · limiar "
            f"{v.get('threshold')} · {len(v.get('features') or [])} variáveis · "
            f"problemas: {problemas or 'nenhum'}")

    dec = le(os.path.join(r, "internal_verifier_object_decisions.csv"),
             "script17a_decisions_input",
             ["patient_id", "architecture", "component_id", "status_original",
              "decision", "status_filtered", "verifier_probability"])
    met = le(os.path.join(r, "internal_verifier_metrics.csv"),
             "script17a_metrics_input",
             ["architecture", "detector", "tp", "fp", "fn", "n_gt", "n_pred",
              "n_exams", "sensitivity", "precision", "f1", "fp_per_exam"])
    if dec is None or met is None:
        for c in DEPENDENTES_S17A[1:]:
            registo(c, "FAIL_INPUT", "produtos do Script 17A ilegíveis")
        return None

    internos = inte or set()
    pac = set(dec.patient_id)
    fora_oof = pac & (dev or set())
    ok = (len(pac) == LIDC_INT and pac == internos and not fora_oof
          and (met.n_exams == LIDC_INT).all())
    registo("script17a_cohort", "PASS" if ok else "FAIL_VALUE",
            f"{len(pac)} pacientes ({LIDC_INT} esperados) · conjunto igual ao "
            f"interno canónico: {pac == internos} · pacientes de desenvolvimento "
            f"presentes: {len(fora_oof)} · n_exams declarado: "
            f"{sorted(met.n_exams.unique().tolist())}")

    chave = ["patient_id", "architecture", "component_id"]
    dup = int(dec.duplicated(chave).sum())
    correspondem = "NÃO VERIFICADO"
    if "Script11B" in runs:
        O = pd.read_csv(os.path.join(runs["Script11B"], "analysis",
                                     "bd1_object_matches.csv"))
        O = O[(O.subset == "internal") & (O.status.isin(["TP", "FP"]))]
        a = set(zip(dec.patient_id, dec.architecture,
                    pd.to_numeric(dec.component_id, errors="coerce")))
        b = set(zip(O.patient_id, O.model,
                    pd.to_numeric(O.pred_component_id, errors="coerce")))
        correspondem = (a == b)
    ok = (len(dec) == S17A_N_DECISOES and dup == 0 and correspondem is True)
    registo("script17a_object_support", "PASS" if ok else "FAIL_VALUE",
            f"{len(dec)} decisões ({S17A_N_DECISOES} esperadas) · chaves "
            f"duplicadas {dup} · conjunto idêntico aos objectos preditos "
            f"internos do Script 11B: {correspondem}")

    estados = set(met.detector.unique())
    ok = (len(met) == S17A_N_LINHAS_METRICAS
          and met.architecture.nunique() == N_ARQUITECTURAS
          and estados == {"original", "filtered"})
    registo("script17a_metrics", "PASS" if ok else "FAIL_VALUE",
            f"{len(met)} linhas ({S17A_N_LINHAS_METRICAS}) · "
            f"{met.architecture.nunique()} arquitecturas ({N_ARQUITECTURAS}) · "
            f"estados {sorted(estados)}")

    maus = []
    for arch in sorted(met.architecture.unique()):
        s_ = dec[dec.architecture == arch]
        for et, sub in (("original", s_),
                        ("filtered", s_[s_.decision == "ACCEPT"])):
            linha = met[(met.architecture == arch) & (met.detector == et)]
            if len(linha) != 1:
                maus.append((arch, et, "linha ausente")); continue
            x = linha.iloc[0]
            tp = int((sub.status_original == "TP").sum())
            fp = int((sub.status_original == "FP").sum())
            fn = int(x.n_gt) - tp
            if (tp, fp, fn) != (int(x.tp), int(x.fp), int(x.fn)):
                maus.append((arch, et, f"{(tp, fp, fn)} != "
                                       f"{(int(x.tp), int(x.fp), int(x.fn))}"))
                continue
            sens = tp / (tp + fn) if (tp + fn) else float("nan")
            prec = tp / (tp + fp) if (tp + fp) else float("nan")
            f1 = (2 * prec * sens / (prec + sens)) \
                if (np.isfinite(prec) and (prec + sens) > 0) else float("nan")
            for nome_m, meu, dele in (("sensitivity", sens, x.sensitivity),
                                      ("precision", prec, x.precision),
                                      ("f1", f1, x.f1),
                                      ("fp_per_exam", fp / LIDC_INT, x.fp_per_exam)):
                if np.isnan(meu) and pd.isna(dele):
                    continue
                if not np.isclose(meu, float(dele), rtol=0, atol=1e-12):
                    maus.append((arch, et, f"{nome_m}: {meu} != {dele}"))
    registo("script17a_arithmetic", "PASS" if not maus else "FAIL_VALUE",
            f"{len(met)} linhas reconstruídas das decisões por objecto · "
            f"divergências: {len(maus)}", maus[:5])

    total_orig = int(met[met.detector == "original"].n_pred.sum())
    ok = ((met.n_gt == S17A_N_GT_POR_ARQ).all() and total_orig == S17A_N_DECISOES
          and (met.tp + met.fn == met.n_gt).all()
          and (met.tp + met.fp == met.n_pred).all())
    registo("script17a_denominators", "PASS" if ok else "FAIL_VALUE",
            f"N_GT {sorted(met.n_gt.unique().tolist())} ({S17A_N_GT_POR_ARQ} "
            f"esperado) · soma de N_pred no original {total_orig} "
            f"({S17A_N_DECISOES}) · TP+FN=N_GT e TP+FP=N_pred em todas as linhas")

    deg = met[met.architecture == "F100_ResEncM"]
    ok = (len(deg) == 2 and (deg.tp == 0).all() and (deg.fp == 0).all()
          and (deg.n_pred == 0).all() and deg.precision.isna().all()
          and deg.f1.isna().all() and (deg.sensitivity == 0).all())
    registo("script17a_degenerate_case", "PASS" if ok else "FAIL_VALUE",
            f"F100_ResEncM: N_pred {deg.n_pred.tolist()} · precisão NaN "
            f"{deg.precision.isna().tolist()} · F1 NaN {deg.f1.isna().tolist()} · "
            f"sensibilidade {deg.sensitivity.tolist()} (denominador "
            f"{S17A_N_GT_POR_ARQ}, é zero medido e não indefinido)")

    g = j.get("gates") or {}
    declara = (g.get("INFERENCE") == "POINT_ESTIMATES_ONLY"
               and g.get("CONFIDENCE_INTERVALS") == "NOT_RUN"
               and g.get("HYPOTHESIS_TESTS") == "NOT_RUN"
               and g.get("BOOTSTRAP") == "NOT_RUN"
               and "INFERENCE_NOT_PRESPECIFIED" in g)
    suspeitas = [c for c in list(met.columns) + list(dec.columns)
                 if any(t in c.lower() for t in ("_ci_", "ci_lo", "ci_hi",
                                                 "p_value", "p_raw", "p_holm",
                                                 "boot"))]
    registo("script17a_inference_scope",
            "PASS" if declara and not suspeitas else "FAIL_VALUE",
            f"declarações do manifesto coerentes: {declara} · colunas "
            f"inferenciais nos produtos: {suspeitas or 'nenhuma'}")
    return j


def audita_integracao_17a_no_25(runs, man17a):
    """SCRIPT25_INTEGRATION_OF_17A — o pacote final consolidou o produto do 17A
    sem alterar um byte, e cita o run certo."""
    if "Script25" not in runs or "Script17A" not in runs or man17a is None:
        registo("script25_integration_of_script17a", "FAIL_INPUT",
                "run do 25 ou do 17A não resolvido")
        return
    fonte = os.path.join(runs["Script17A"], "internal_verifier_metrics.csv")
    if not os.path.isfile(fonte):
        registo("script25_integration_of_script17a", "FAIL_INPUT",
                "métricas do 17A ausentes")
        return
    s_fonte = sha256(fonte)
    r = runs["Script25"]
    achados, problemas = [], []
    for f in ("manifest_figures.csv", "manifest_tables.csv"):
        p = os.path.join(r, "manifests", f)
        if not os.path.isfile(p):
            problemas.append(f"{f} ausente"); continue
        m = pd.read_csv(p)
        for _, x in m[m.producer_script == "Script17A"].iterrows():
            achados.append(str(x["product"]))
            alvo = os.path.join(r, str(x["area"]), str(x["product"]))
            if x["transform_class"] != "DIRECT_COPY":
                problemas.append(f"{x['product']}: classe {x['transform_class']}")
            if x["canonical_run_id"] != os.path.basename(runs["Script17A"]):
                problemas.append(f"{x['product']}: run {x['canonical_run_id']}")
            if x["source_sha256"] != s_fonte or x["destination_sha256"] != s_fonte:
                problemas.append(f"{x['product']}: SHA != fonte do 17A")
            if not os.path.isfile(alvo) or sha256(alvo) != s_fonte:
                problemas.append(f"{x['product']}: ficheiro consolidado difere da fonte")
    if len(achados) != 2:
        problemas.append(f"{len(achados)} produtos do 17A no pacote final, 2 esperados")
    registo("script25_integration_of_script17a",
            "PASS" if not problemas else "FAIL_PROVENANCE",
            f"produtos: {achados} · fonte {s_fonte[:12]} · problemas: "
            f"{problemas or 'nenhum'}")
    citados = set()
    for f in ("manifest_figures.csv", "manifest_tables.csv"):
        p = os.path.join(r, "manifests", f)
        if os.path.isfile(p):
            citados |= set(pd.read_csv(p).canonical_run_id.dropna().astype(str))
    registo("script25_is_the_superseding_run",
            "PASS" if SCRIPT25_SUPERSEDIDO not in citados
            and os.path.basename(r) != SCRIPT25_SUPERSEDIDO else "FAIL_PROVENANCE",
            f"run auditado {os.path.basename(r)} · o substituído "
            f"{SCRIPT25_SUPERSEDIDO} é citado: {SCRIPT25_SUPERSEDIDO in citados}")


def audita_semantica_das_atestacoes(runs):
    """PROVENANCE_ATTESTATION_SEMANTICS.

    Uma tabela de hashes escrita pelo run e outra acrescentada no arquivamento
    comprovam coisas diferentes: a primeira é contemporânea, a segunda atesta os
    bytes encontrados mais tarde. Ambas podem ser válidas; o que não pode é a
    segunda ser apresentada como produto do run. Um run com apenas
    `output_hashes.csv` **não** é reprovado — só se exige que a origem de cada
    tabela seja nomeada correctamente.
    """
    linhas, problemas = [], []
    for nome, raiz in sorted(runs.items()):
        presentes = {t for t in ("SHA256SUMS.txt", "output_hashes.csv")
                     if os.path.isfile(os.path.join(raiz, t))}
        do_produtor = TABELA_ESCRITA_PELO_PRODUTOR.get(nome, set())
        retrospectivas = presentes - do_produtor
        em_falta = do_produtor - presentes
        if em_falta:
            problemas.append(f"{nome}: o produtor escreve {sorted(do_produtor)} "
                             f"e falta(m) {sorted(em_falta)}")
        linhas.append({
            "producer": nome,
            "tables_present": sorted(presentes),
            "producer_written": sorted(do_produtor) or ["NONE"],
            "producer_table_type": ("CONTEMPORANEOUS_PRODUCER_OUTPUT"
                                    if do_produtor and not (do_produtor - presentes)
                                    else "NONE"),
            "archive_attestation": sorted(retrospectivas) or [],
            "archive_attestation_type": ("RETROSPECTIVE_DAS_SHA256SUMS"
                                         if retrospectivas else "")})
    n_retro = sum(1 for x in linhas if x["archive_attestation"])
    registo("provenance_attestation_semantics",
            "PASS" if not problemas else "FAIL_PROVENANCE",
            f"{len(linhas)} runs · {n_retro} com atestação acrescentada no "
            f"arquivamento, nomeada como tal e não como produto do run · "
            f"problemas: {problemas or 'nenhum'}", linhas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manuscript", default="")
    ap.add_argument("--manuscript-sha256", default="")
    ap.add_argument("--geometry-limit", type=int, default=0,
                    help="só para ensaio; 0 = exaustivo")
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()
    t0 = datetime.datetime.now().astimezone()

    log.info("A. proveniência dos runs canónicos")
    runs, entradas = audita_proveniencia()
    log.info("B. coorte LIDC ao nível do paciente")
    r = audita_coorte_lidc()
    dev, inte = r if r else (set(), set())
    log.info("C. limiar de consenso do LIDC")
    audita_tau()
    log.info("D. LNDb, RIDER, arquitecturas e modelos")
    audita_lndb_rider(runs)
    log.info("E. identidades aritméticas")
    audita_identidades(runs)
    log.info("F. separação de subconjuntos")
    audita_separacao(runs, dev, inte)
    log.info("G. semente e ausência de ajuste no LNDb")
    audita_semente_e_ajuste(runs)
    log.info("H. geometria por cabeçalho")
    audita_geometria(runs, dev, inte, args.geometry_limit)
    log.info("I. pacote final do Script 25")
    bloq = audita_pacote_final(runs)
    log.info("K. Script 17A — verificador congelado no BD1 interno")
    man17a = audita_script17a(runs, dev, inte)
    audita_integracao_17a_no_25(runs, man17a)
    audita_semantica_das_atestacoes(runs)
    log.info("J. cruzamento com o manuscrito")
    audita_manuscrito(args.manuscript, args.manuscript_sha256, bloq)

    df = pd.DataFrame(VERIFICACOES)
    resumo = df.status.value_counts().to_dict()
    falhas = df[df.status.str.startswith("FAIL")]
    nao_estimaveis = df[df.status == "NOT_ESTIMABLE"]
    veredicto = ("PASS" if len(falhas) == 0 and len(nao_estimaveis) == 0
                 else ("BLOCKED_BY_FAILURE" if len(falhas)
                       else "INCOMPLETE_NOT_ESTIMABLE"))

    run_id = f"s26_{t0:%Y%m%dT%H%M%S}_{os.getpid()}"
    destino = os.path.join(args.out_root or os.path.join(ANALYSIS_DIR, "reproducibility_audit"),
                           run_id)
    staging = tempfile.mkdtemp(prefix="s26_staging_")
    try:
        out = os.path.join(staging, run_id)
        os.makedirs(out)
        df.to_csv(os.path.join(out, "reproducibility_checks.csv"), index=False)
        pd.DataFrame(entradas).to_csv(os.path.join(out, "input_manifest.csv"), index=False)
        with open(os.path.join(out, "final_audit_report.md"), "w") as f:
            f.write(f"# Auditoria de reprodutibilidade — {run_id}\n\n")
            f.write(f"**Veredicto:** `{veredicto}`\n\n")
            f.write(f"| estado | n |\n|---|---|\n")
            for k, v in sorted(resumo.items()):
                f.write(f"| {k} | {v} |\n")
            f.write("\n## Verificações\n\n| verificação | estado | detalhe |\n|---|---|---|\n")
            for _, x in df.iterrows():
                f.write(f"| {x['check']} | `{x['status']}` | {x['detail']} |\n")
            if len(falhas):
                f.write("\n## Falhas — bloqueiam 27, 28 e 29\n\n")
                for _, x in falhas.iterrows():
                    f.write(f"- **{x['check']}** — `{x['status']}`: {x['detail']}\n")
            if len(nao_estimaveis):
                f.write("\n## Não estimáveis — sem autoridade disponível\n\n")
                for _, x in nao_estimaveis.iterrows():
                    f.write(f"- **{x['check']}**: {x['detail']}\n")
        g = {"verdict": veredicto, "summary": resumo,
             "failures": falhas["check"].tolist(),
             "not_estimable": nao_estimaveis["check"].tolist(),
             "runs_resolved": {k: os.path.basename(v) for k, v in runs.items()},
             "PASS_BY_MISSING_DATA": "PROHIBITED — estados explícitos por verificação",
             "downstream": ("Scripts 27, 28 e 29 BLOQUEADOS" if veredicto != "PASS"
                            else "Scripts 27, 28 e 29 desbloqueados pela auditoria")}
        json.dump(g, open(os.path.join(out, "gates.json"), "w"), indent=1, default=str)
        saidas = {f: sha256(os.path.join(out, f)) for f in sorted(os.listdir(out))}
        pd.DataFrame([{"path": k, "sha256": v} for k, v in saidas.items()]).to_csv(
            os.path.join(out, "output_hashes.csv"), index=False)
        man = {"script": "26_reproducibility_audit.py", "run_id": run_id,
               "script_sha256": sha256(os.path.abspath(__file__)),
               "run_at": t0.isoformat(),
               "end_at": datetime.datetime.now().astimezone().isoformat(),
               "role": "AUDIT_ONLY — não recalcula estatística",
               "frozen_truths": {"LIDC_ANALYTIC": LIDC_ANALITICO,
                                 "LIDC_EVALUATED": LIDC_AVALIADOS,
                                 "development": LIDC_DEV, "internal": LIDC_INT,
                                 "outside_split": LIDC_FORA,
                                 "LNDB_NOMINAL": LNDB_NOMINAL,
                                 "LNDB_ANALYTIC": LNDB_ANALITICO,
                                 "RIDER_pairs": RIDER_PARES, "RIDER_CTs": RIDER_CTS,
                                 "architectures": N_ARQUITECTURAS, "fold_models": N_MODELOS,
                                 "seed": SEMENTE, "tau_lidc": TAU_LIDC_PROTOCOLO},
               "manuscript": {"path": args.manuscript or None,
                              "sha256": (sha256(args.manuscript)
                                         if args.manuscript and os.path.isfile(args.manuscript)
                                         else None)},
               "geometry_limit": args.geometry_limit,
               "counts": {"checks": len(df), **resumo},
               "gates": g,
               "promotion": "NOT PROMOTED — produtos ficam no directório do run"}
        json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=2, default=str)
        os.makedirs(os.path.dirname(destino), exist_ok=True)
        if os.path.exists(destino):
            raise SystemExit(f"FAIL_OUTPUT: o directório do run já existe: {destino}")
        shutil.move(out, destino)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    log.info(f"Veredicto: {veredicto} · {resumo}")
    if len(falhas):
        log.error(f"{len(falhas)} falhas bloqueiam os Scripts 27, 28 e 29: "
                  f"{falhas['check'].tolist()}")
    if len(nao_estimaveis):
        log.info(f"{len(nao_estimaveis)} não estimáveis: {nao_estimaveis['check'].tolist()}")
    log.info(f"Saída: {destino}")
    log.info("Script 26 concluído.")


if __name__ == "__main__":
    main()
