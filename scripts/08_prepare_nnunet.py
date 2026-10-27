#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    08_prepare_nnunet.py

File type:
    Script

Purpose:
    Prepares the nnU-Net v2 dataset tree for the canonical cohort under the adjudicated-or-negative-confirmed labelling policy.

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
    - dataset.json
    - internal_pending.csv
    - INVALID.json

Outputs:
    - .stdout.log
    - .stderr.log

Dependencies:
    - SimpleITK

Execution:
    PULMONARY_BASE=<project_dir> python scripts/08_prepare_nnunet.py

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
import time
import shutil
import hashlib
import logging
import argparse
import subprocess
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Set

from pulmonary_base import (BASE, check_base, sha256_file, verify_nifti,
                            nifti_info, ESQUEMA_PONTEIRO_ACTUAL)


SCRIPT08_VERSION      = "2026-08-09.nnunet.v7.pointer_cohort"
SCRIPT08_LABEL_POLICY = "ADJUDICATED_OR_NEGATIVE_CONFIRMED"
SPLITS_FINAL_POLICY   = "PROJECTED_FROM_FROZEN_COHORT_FOLDS_WITHOUT_REDRAW"

DECISAO_ADJUDICADO = "INCLUDE_ADJUDICATED_REFERENCE"
DECISAO_NEGATIVO   = "INCLUDE_NEGATIVE_CONFIRMED"
XML_NEGATIVO       = "NEGATIVE_CONFIRMED"

N_FOLDS      = 5
DATASET_ID   = 1
DATASET_NAME = "Dataset%03d_PulmonaryNodules" % DATASET_ID

_MAPA_PLANEADOR = {
    "nnUNetPlans": None,
    "nnUNetResEncUNetMPlans": "nnUNetPlannerResEncM",
    "nnUNetResEncUNetLPlans": "nnUNetPlannerResEncL",
}

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import treino_core as _tc

ARCHITECTURES = []
for _a in _tc.ARQUITETURAS:
    _pl = _MAPA_PLANEADOR.get(_a["plans_id"], "__DESCONHECIDO__")
    ARCHITECTURES.append({"name": _a["name"], "config": _a["config"],
                          "plans_id": _a["plans_id"], "planner": _pl})
_desconhecidos = sorted({a["plans_id"] for a in ARCHITECTURES
                         if a["planner"] == "__DESCONHECIDO__"})
if _desconhecidos:
    raise SystemExit("[FAIL] plano sem planeador conhecido em _MAPA_PLANEADOR: %s"
                     % _desconhecidos)

EXPECTED_ARCHITECTURES = len(ARCHITECTURES)
EXPECTED_PREPROCESSING_INVOCATIONS = len({(a["config"], a["planner"])
                                          for a in ARCHITECTURES})
_por_alvo = {}
for _a in ARCHITECTURES:
    _por_alvo.setdefault((_a["config"], _a["planner"]), []).append(_a["name"])
EXPECTED_SHARED = {k: sorted(v) for k, v in _por_alvo.items() if len(v) > 1}

PREFIXO_MANIFEST_ID = {"06": "script06_", "07": "bd1_splits_analytic_"}


def _trabalhadores():
    """
    Trabalhadores do pre-processamento, derivados da RAM e dos nucleos.

    Regra: ~3,5 GB por trabalhador, limitado pelos nucleos e por um tecto de 8.

    NAO BAIXAR. Em 13/08/2026 experimentei 1,6 GB, com base numa medicao feita
    durante o alvo `2d`, onde os trabalhadores ocupavam 0,8 a 1,3 GB. Com 8
    trabalhadores o alvo `3d_fullres` do ResEncM morreu aos 222 de 718 casos, e o
    OOM killer do nucleo registou o processo morto com **3,5 GB de RSS**:

        Out of memory: Killed process 3312337 (python) anon-rss:3588636kB

    Ou seja: o consumo depende da configuracao, e o 2d nao serve de medida para o
    3d. Um patch de 96x128x192 em float32, mais o alvo e os buffers de
    reamostragem, chega aos 3,5 GB por trabalhador. A regra original estava certa
    e o meu «acerto» custou 45 minutos de preprocessamento perdidos.
    """
    import multiprocessing
    try:
        import os as _os
        gb = _os.sysconf("SC_PAGE_SIZE") * _os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
    except (ValueError, OSError, AttributeError):
        gb = 8.0
    n = max(1, min(int(gb // 3.5), multiprocessing.cpu_count(), 8))
    return n


N_WORKERS = _trabalhadores()
PLANS_ID = {None: "nnUNetPlans",
            "nnUNetPlannerResEncM": "nnUNetResEncUNetMPlans",
            "nnUNetPlannerResEncL": "nnUNetResEncUNetLPlans"}

PONTEIRO_ANOT = os.path.join(BASE, "runs", "CURRENT_LIDC_ANNOTATIONS.json")
if os.environ.get("SCRIPT08_LABEL_OVERLAY"):
    raise SystemExit("[FAIL] SCRIPT08_LABEL_OVERLAY foi eliminado. Os rotulos vem "
                     "somente do produto do Script 06 apontado por CURRENT.json.")

CURRENT_CROPS  = os.path.join(BASE, "runs", "CURRENT.json")
CURRENT_BD1    = os.path.join(BASE, "runs", "CURRENT_BD1_PARTITION.json")
PONTEIRO_04    = os.path.join(BASE, "runs", "CURRENT_LIDC_ANNOTATIONS.json")
MANIFESTO_04   = os.path.join(BASE, "config", "lidc_annotations_manifest.json")

STAGING_ROOT = os.path.join(BASE, "staging")
RUNS_ROOT    = os.path.join(BASE, "runs")
CURRENT_NNUNET           = os.path.join(RUNS_ROOT, "CURRENT_NNUNET.json")
CURRENT_NNUNET_STRUCTURE = os.path.join(RUNS_ROOT, "CURRENT_NNUNET_STRUCTURE.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


class FalhaMetodologica(Exception):
    """Invalida o produto. Nunca absorvida; nunca produz codigo de saida zero."""


def _exigir(c, m):
    if not c:
        raise FalhaMetodologica(m)


def _ler_ids(p):
    _exigir(os.path.isfile(p), "lista de identificadores ausente: %s" % p)
    ids = [x.strip() for x in open(p) if x.strip()]
    dup = [k for k, v in Counter(ids).items() if v > 1]
    _exigir(not dup, "%s com identificadores repetidos: %s" % (os.path.basename(p), dup[:5]))
    return set(ids)


def _sha_lista(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


@dataclass(frozen=True)
class Contexto:
    args: Any
    run_id: str
    staging: str
    ponteiro_06: Dict
    ponteiro_07: Dict
    raiz_06: str
    verificacao_entradas: Dict
    ancora_04: Dict
    coorte: Dict
    treino: Set
    adjudicados: Set
    negativos: Set
    interno: Set
    ref_avaliaveis: Set
    pend_int: Set
    excl_dev: Set
    excl_int: Set
    splits: List
    alvos: List
    mapa_alvos: Dict


def _corpo_sha(m):
    corpo = dict((k, v) for k, v in m.items()
                 if k not in ("manifest_sha256", "manifest_body_sha256"))
    return hashlib.sha256(json.dumps(corpo, indent=1, sort_keys=True).encode()).hexdigest()


RESOLUCOES = []


def _reancorar(abs_registado, rotulo, run_id_esperado=None):
    """
    Resolve o caminho de um manifesto quando o ponteiro guarda um caminho
    ABSOLUTO de outra maquina.

    A identidade de um manifesto e o seu SHA-256, nao o caminho onde calhou
    estar. O caminho e localizacao; o hash e identidade. Por isso:

        1  tenta-se o caminho absoluto registado
        2  se existir, exige-se o hash esperado — resolvido nao dispensa conferido
        3  se nao existir, constroi-se PULMONARY_BASE / parte_relativa
        4  o resultado tem de ficar DENTRO de PULMONARY_BASE
        5  o hash tem de conferir na nova localizacao

    A parte relativa vem do nome do diretorio de BASE, que tem de aparecer
    EXATAMENTE UMA VEZ no caminho registado. Zero ou mais do que uma vez e
    ambiguidade, e ambiguidade aborta.

    NAO ha busca recursiva por nome, NAO se escolhe o ficheiro mais recente e
    NAO se casa apenas pelo nome-base. Qualquer um dos tres poderia devolver um
    ficheiro parecido em vez do certo.
    """
    if os.path.isfile(abs_registado):
        return abs_registado, "ABSOLUTE_PATH_AS_REGISTERED"
    partes = os.path.normpath(abs_registado).split(os.sep)
    _exigir(run_id_esperado, "%s: ponteiro sem run_id; nao ha parte relativa "
                             "deterministica" % rotulo)
    n = partes.count(run_id_esperado)
    _exigir(n == 1,
            "%s: parte relativa AMBIGUA — o run_id %r aparece %d vezes em %s"
            % (rotulo, run_id_esperado, n, abs_registado))
    i = partes.index(run_id_esperado)
    _exigir(i >= 1 and partes[i - 1] == "runs",
            "%s: o run_id nao esta sob 'runs/' no caminho registado: %s"
            % (rotulo, abs_registado))
    rel = os.path.join(*partes[i - 1:])
    resolvido = os.path.join(BASE, rel)
    dentro = (os.path.commonpath([os.path.realpath(os.path.dirname(resolvido)),
                                  os.path.realpath(BASE)]) == os.path.realpath(BASE))
    _exigir(dentro, "%s: caminho resolvido ESCAPA de PULMONARY_BASE: %s"
            % (rotulo, resolvido))
    _exigir(os.path.isfile(resolvido),
            "%s: manifesto ausente no caminho registado (%s) e no reancorado (%s)"
            % (rotulo, abs_registado, resolvido))
    return resolvido, "REANCHORED_TO_PULMONARY_BASE"


def _carregar_ponteiro(path, script, rotulo):
    """Amarra ponteiro, manifesto e diretorio a MESMA execucao; confere os hashes."""
    _exigir(os.path.isfile(path),
            "%s ausente: %s. O Script %s ainda nao publicou." % (rotulo, path, script))
    p = json.load(open(path))
    _exigir(p.get("script") == script,
            "%s: ponteiro.script=%r != %r" % (rotulo, p.get("script"), script))
    _campo = next((k for k in ("manifest", "annotations_manifest",
                               "lidc_final_manifest", "manifest_path")
                   if p.get(k)), None)
    _exigir(_campo, "%s: ponteiro nao declara caminho do manifesto" % rotulo)
    registado = p[_campo]
    man, modo = _reancorar(registado, rotulo, p.get("run_id"))
    m = json.load(open(man))
    _exigir(isinstance(m, dict), "%s: manifesto nao e um objecto JSON" % rotulo)
    calculado_bytes = sha256_file(man)
    calculado_corpo = _corpo_sha(m)
    _declarados = {k: p[k] for k in ("manifest_file_sha256", "manifest_body_sha256",
                                     "manifest_sha256") if p.get(k)}
    _papel = {k: ("FILE" if v == calculado_bytes else
                  "BODY" if v == calculado_corpo else "UNRECOGNISED")
              for k, v in _declarados.items()}
    esperado = "|".join("%s=%s" % (k, _papel[k]) for k in sorted(_declarados))
    bate = bool(_declarados) and all(v != "UNRECOGNISED" for v in _papel.values())
    RESOLUCOES.append({
        "label": rotulo, "script": script,
        "resolution_mode": modo,
        "original_absolute_path": registado,
        "resolved_cluster_path": man,
        "expected_sha256": esperado,
        "calculated_sha256_bytes": calculado_bytes,
        "calculated_sha256_body": calculado_corpo,
        "hash_status": "MATCH" if bate else "MISMATCH",
        "pointer_convention": ("BYTES_AND_BODY" if "manifest_body_sha256" in p
                               else "BODY_ONLY_LEGACY")})
    _exigir(bate, "%s: hash declarado no ponteiro nao corresponde nem ao ficheiro "
                  "(%s) nem ao corpo (%s) do manifesto %s. papeis=%s"
                  % (rotulo, calculado_bytes[:12], calculado_corpo[:12], man, _papel))

    dir_nome = os.path.basename(os.path.dirname(man))
    _exigir(p.get("run_id") == m.get("run_id") == dir_nome,
            "%s: execucoes misturadas — ponteiro=%r manifesto=%r diretorio=%r"
            % (rotulo, p.get("run_id"), m.get("run_id"), dir_nome))
    _exigir(m.get("script_version"), "%s: manifesto sem script_version" % rotulo)
    chave = "SCRIPT%s_OUTPUT_STATUS" % script
    _exigir(chave in m, "%s: manifesto sem %s" % (rotulo, chave))
    prefixo = PREFIXO_MANIFEST_ID.get(script)
    _exigir(prefixo and str(m.get("manifest_id", "")).startswith(prefixo),
            "%s: manifest_id=%r nao comeca por %r" % (rotulo, m.get("manifest_id"), prefixo))
    _exigir(str(m.get("manifest_id", "")).endswith(p["run_id"]),
            "%s: manifest_id nao termina no run_id do ponteiro" % rotulo)

    recalc = _corpo_sha(m)
    interno = m.get("manifest_body_sha256") or m.get("manifest_sha256")
    _exigir(interno is not None,
            "%s: manifesto sem assinatura do proprio corpo "
            "(manifest_body_sha256)" % rotulo)
    _exigir(interno == recalc,
            "%s: corpo nao confere. declarado=%s recalculado=%s" % (rotulo, interno, recalc))

    bytes_sha = calculado_bytes
    _tem_ficheiro = "FILE" in _papel.values()
    _tem_corpo = "BODY" in _papel.values()
    if _tem_ficheiro and _tem_corpo:
        convencao, integridade = "FILE_AND_BODY", "FULL"
    elif _tem_corpo:
        convencao, integridade = "BODY_ONLY_LEGACY", "PARTIAL_BODY_ONLY"
    else:
        convencao, integridade = "FILE_ONLY_LEGACY", "PARTIAL_FILE_ONLY"
    _exigir(recalc == interno,
            "%s: o manifesto nao assina o proprio corpo. declarado=%s "
            "recalculado=%s" % (rotulo, interno, recalc))
    if _tem_corpo:
        _corpo_no_ponteiro = next(v for k, v in _declarados.items()
                                  if _papel[k] == "BODY")
        _exigir(_corpo_no_ponteiro == interno,
                "%s: ponteiro e manifesto discordam no corpo" % rotulo)

    veredicto = ("PASS_FULL_INTEGRITY" if integridade == "FULL"
                 else "PASS_" + integridade)
    (log.info if integridade == "FULL" else log.warning)(
        "  %-22s run_id=%s | %s | %s" % (rotulo, p["run_id"], convencao, veredicto))
    return p, m, {"pointer_convention": convencao, "INTEGRITY_LEVEL": integridade,
                  "VERIFICATION_VERDICT": veredicto,
                  "manifest_body_sha256": interno,
                  "manifest_bytes_sha256_observed": bytes_sha,
                  "bytes_anchored_in_pointer": integridade == "FULL",
                  "producer_script_version": m.get("script_version")}


def _ancorar_manifesto_04():
    """O manifesto do 04 determina quem tem alvo adjudicado, logo determina os 704."""
    _exigir(os.path.isfile(PONTEIRO_04), "ponteiro do 04 ausente: %s" % PONTEIRO_04)
    pt = json.load(open(PONTEIRO_04))
    manif = os.path.join(BASE, pt["annotations_manifest"])
    _exigir(os.path.isfile(manif), "manifesto do 04 ausente: %s" % manif)
    m = json.load(open(manif))
    interno = m.get("manifest_body_sha256") or m.get("manifest_sha256")
    recalc = _corpo_sha(m)
    _exigir(interno == recalc, "manifesto do 04: corpo nao confere")
    _exigir(interno == (pt.get("manifest_body_sha256") or pt.get("manifest_sha256")),
            "ponteiro do 04 aponta para sha %s, manifesto tem %s"
            % (pt.get("manifest_body_sha256") or pt.get("manifest_sha256"), interno))
    _exigir(m.get("SCRIPT04_OUTPUT_STATUS") == "PASS",
            "Script 04 em %s" % m.get("SCRIPT04_OUTPUT_STATUS"))
    _exigir(m.get("SCRIPT04_OUTPUT_MANIFEST_FROZEN") == "YES", "manifesto do 04 nao congelado")
    linhas = m["rows"]
    ids = [r["patient_id"] for r in linhas]
    dup = [k for k, v in Counter(ids).items() if v > 1]
    _exigir(not dup, "manifesto do 04 com identificadores repetidos: %s" % dup[:5])
    nao_bool = [r["patient_id"] for r in linhas
                if not isinstance(r.get("has_consensus"), bool)]
    _exigir(not nao_bool, "has_consensus nao booleano em %d casos" % len(nao_bool))
    log.info("  manifesto do 04 ancorado | corpo=%s... | %d casos"
             % (interno[:16], len(linhas)))
    return ({"body_sha256": interno, "file_sha256": sha256_file(manif),
             "script04_run_id": pt.get("run_id"),
             "n_rows": len(linhas), "duplicate_ids": 0, "has_consensus_all_bool": True},
            dict((r["patient_id"], r["has_consensus"]) for r in linhas))


def _entradas():
    p6, m6, h6 = _carregar_ponteiro(CURRENT_CROPS, "06", "publicacao do 06")
    _exigir(m6.get("SCRIPT06_OUTPUT_STATUS") == "PASS", "Script 06 nao esta em PASS")
    _exigir(m6.get("SCRIPT06_REFERENCE_GUIDED_CROP") == "NO",
            "Script 06 com recorte guiado por rotulo")
    _exigir(int(m6.get("SCRIPT06_TRUNCATED_VOXELS", -1)) == 0,
            "Script 06 com truncamento: %s" % m6.get("SCRIPT06_TRUNCATED_VOXELS"))
    p7, m7, h7 = _carregar_ponteiro(CURRENT_BD1, "07", "publicacao do 07")
    _g7 = m7.get("gates", m7)
    _exigir(_g7.get("PRODUCT_STATUS") == "PASS", "Script 07 nao esta em PASS")
    _exigir(_g7.get("PARTITION_FROZEN") == "YES", "particao do 07 nao esta congelada")
    _exigir(_g7.get("OOF_COVERAGE_EXACT") is True, "cobertura OOF do 07 nao e exacta")
    r6 = [x for x in RESOLUCOES if x["script"] == "06"][-1]["resolved_cluster_path"]
    return p6, m6, p7, m7, os.path.dirname(r6), {"script06": h6, "script07": h7}


def _carregar_coorte(tem_alvo):
    """
    Monta a coorte a partir de quem a produz HOJE: o Script 04 (quem tem
    referencia publicada) e o Script 07 (quem vai para desenvolvimento e quem
    fica no conjunto interno).

    A versao anterior lia oito ficheiros de `config/cohort/` — `training_ids_813.txt`,
    `internal_ids_187.txt`, `cohort_folds_813.json`, `cohort_decision_manifest.csv`
    e companhia — e exigia que a coorte corrente coubesse dentro dos 813
    congelados em julho. Nenhum script do pipeline produz esses ficheiros: o
    estagio nao arrancava, e mesmo criando-os as comportas falhariam, porque a
    coorte mudou. Verdade congelada em ficheiro sem produtor e a mesma doenca do
    `run002` e do sha do manifesto do 04.
    """
    _exigir(os.path.isfile(PONTEIRO_ANOT),
            "ponteiro das anotacoes ausente: %s" % PONTEIRO_ANOT)
    _ptr = json.load(open(PONTEIRO_ANOT))
    _mp = os.path.join(BASE, _ptr["annotations_manifest"])
    _exigir(os.path.isfile(_mp), "manifesto do 04 ausente: %s" % _mp)
    _mpub = json.load(open(_mp))
    _exigir(_mpub.get("SCRIPT04_OUTPUT_STATUS") == "PASS",
            "Script 04 em %s" % _mpub.get("SCRIPT04_OUTPUT_STATUS"))
    linhas = _mpub["rows"]
    ids = [r["patient_id"] for r in linhas]
    _exigir(len(ids) == len(set(ids)), "manifesto do 04 com identificadores repetidos")

    publicados = {r["patient_id"] for r in linhas if r["has_consensus"]}
    excluidos04 = {r["patient_id"] for r in linhas if not r["has_consensus"]}
    _exigir(publicados, "o 04 nao publicou nenhum consenso")

    _exigir(os.path.isfile(CURRENT_BD1), "ponteiro da particao ausente: %s" % CURRENT_BD1)
    _p7 = json.load(open(CURRENT_BD1))
    _split = os.path.join(BASE, _p7["split_csv"])
    _exigir(os.path.isfile(_split), "bd1_split.csv ausente: %s" % _split)
    _exigir(sha256_file(_split) == _p7["split_csv_sha256"],
            "bd1_split.csv diverge do hash declarado no ponteiro do 07")
    part = {r["patient_id"]: r["split"] for r in csv.DictReader(open(_split, newline=""))}

    universo07 = set(part)
    _exigir(universo07 <= publicados | excluidos04,
            "particao do 07 tem casos fora do universo do 04: %s"
            % sorted(universo07 - (publicados | excluidos04))[:5])

    dev = {p for p, v in part.items() if v == "development"}
    interno = {p for p, v in part.items() if v == "internal"}
    _exigir(dev and interno, "particao sem desenvolvimento ou sem interno")
    _exigir(not (dev & interno), "caso em desenvolvimento e interno ao mesmo tempo")

    treino = dev & publicados
    adjudicados = set(treino)
    _adjudicados_negativos = {r["patient_id"] for r in linhas
                              if r.get("status_04") == "NEGATIVE_DOCUMENTED"}
    negativos = _adjudicados_negativos & treino
    adjudicados = adjudicados - negativos
    _exigir(not (adjudicados & negativos),
            "caso classificado como adjudicado e negativo ao mesmo tempo")
    _exigir(adjudicados | negativos == treino,
            "a soma de adjudicados e negativos nao cobre o treino: %d + %d != %d"
            % (len(adjudicados), len(negativos), len(treino)))
    log.info("  negativos documentados pelo 04: %d, dos quais %d no treino"
             % (len(_adjudicados_negativos), len(negativos)))
    log.info("  treino %d = adjudicados %d + negativos documentados %d"
             % (len(treino), len(adjudicados), len(negativos)))
    excl_dev = dev - treino
    excl_int = interno - publicados
    ref_aval = interno & publicados
    pend_int = interno - publicados

    log.info("  coorte via produtores vigentes | 04=%s | 07=%s" %
             (_ptr.get("run_id"), _p7.get("run_id")))
    log.info("  desenvolvimento %d = treino %d + sem referencia %d"
             % (len(dev), len(treino), len(excl_dev)))
    log.info("  interno %d = avaliavel %d + pendente %d"
             % (len(interno), len(ref_aval), len(pend_int)))

    sem_alvo = sorted(p for p in adjudicados if not tem_alvo.get(p))
    _exigir(not sem_alvo,
            "em treino mas sem consenso no manifesto do 04: %s" % sem_alvo[:5])

    _st04 = {r["patient_id"]: r.get("status_04") for r in linhas}
    dec = {}
    for p in universo07:
        if p in negativos:
            d, x = DECISAO_NEGATIVO, XML_NEGATIVO
        elif p in treino:
            d, x = DECISAO_ADJUDICADO, None
        else:
            d, x = "EXCLUDE_NO_PUBLISHED_REFERENCE", None
        dec[p] = {"case_id": p, "training_decision": d, "xml_audit_status": x,
                  "status_04": _st04.get(p), "cohort": part.get(p, ""),
                  "policy_version": SCRIPT08_LABEL_POLICY}
    _n_aut = sum(1 for v in dec.values()
                 if v["training_decision"] == DECISAO_NEGATIVO
                 and v["xml_audit_status"] == XML_NEGATIVO)
    _exigir(_n_aut == len(negativos),
            "tabela de decisao autoriza %d mascaras vazias, mas ha %d negativos"
            % (_n_aut, len(negativos)))

    meta = {
        "COHORT_SOURCE": "runs/CURRENT_LIDC_ANNOTATIONS.json + runs/CURRENT_BD1_PARTITION.json",
        "LEGACY_COHORT_FILES": "REMOVED — config/cohort/* nao tinha produtor no pipeline",
        "ANNOTATION_POINTER_SHA256": sha256_file(PONTEIRO_ANOT),
        "ANNOTATION_RUN_ID": _ptr.get("run_id"),
        "ANNOTATION_MANIFEST_SHA256": (_mpub.get("manifest_body_sha256")
                                       or _mpub.get("manifest_sha256")),
        "PARTITION_POINTER_SHA256": sha256_file(CURRENT_BD1),
        "PARTITION_RUN_ID": _p7.get("run_id"),
        "SPLIT_CSV_SHA256": _p7.get("split_csv_sha256"),
        "SOURCE_UNIVERSE_N": len(linhas),
        "PUBLISHED_REFERENCE_N": len(publicados),
        "TRAINING_COHORT_N": len(treino),
        "ADJUDICATED_REFERENCE_N": len(adjudicados),
        "NEGATIVE_CONFIRMED_N": len(negativos),
        "NEGATIVE_CONFIRMED_ADJUDICATED_TOTAL": len(_adjudicados_negativos),
        "NEGATIVE_CONFIRMED_POLICY": (
            "quem adjudica e o Script 04, com duas fontes independentes e "
            "concordantes; este estagio LE `status_04 == NEGATIVE_DOCUMENTED` e "
            "cria mascara vazia SO para essa classe. Ausencia de anotacao nunca "
            "vira negativo aqui. Decisao do utilizador de 2026-08-11, no Caderno."),
        "INTERNAL_N": len(interno),
        "INTERNAL_REFERENCE_EVALUABLE_N": len(ref_aval),
        "INTERNAL_PENDING_N": len(pend_int),
        "EXCLUDED_DEVELOPMENT_N": len(excl_dev),
        "EXCLUDED_INTERNAL_N": len(excl_int),
        "training_ids_sha256": _sha_lista(treino),
        "internal_ids_sha256": _sha_lista(interno),
    }
    return (treino, adjudicados, negativos, interno, ref_aval, pend_int,
            excl_dev, excl_int, dec, meta)


def _folds_congelados(treino):
    """
    Le os folds do Script 07, resolvidos pelo ponteiro. NAO sorteia e NAO
    reatribui: restringe cada fold a coorte de treino e confere cobertura OOF
    exacta.
    """
    _p7 = json.load(open(CURRENT_BD1))
    fpath = os.path.join(BASE, _p7["folds_json"])
    _exigir(os.path.isfile(fpath), "bd1_folds.json ausente: %s" % fpath)
    _exigir(sha256_file(fpath) == _p7["folds_json_sha256"],
            "bd1_folds.json diverge do hash declarado no ponteiro do 07")
    j = json.load(open(fpath))
    F = j["folds"]
    _exigir(len(F) == N_FOLDS, "manifesto de folds com %d folds" % len(F))
    saida, vistos = [], []
    for i in range(N_FOLDS):
        f = F["fold_%d" % i]
        t, v = set(f["train"]) & treino, set(f["val"]) & treino
        _exigir(not (t & v), "fold_%d: sobreposicao treino/validacao" % i)
        _exigir(t | v == treino, "fold_%d: uniao != coorte de treino" % i)
        _exigir(t and v, "fold_%d: particao vazia" % i)
        saida.append({"train": sorted(t), "val": sorted(v)})
        vistos += sorted(v)
        log.info("    fold_%d: train %d | val %d" % (i, len(t), len(v)))
    fora = dict((k, n) for k, n in Counter(vistos).items() if n != 1)
    _exigir(not fora, "OOF nao unica para %d pacientes" % len(fora))
    _exigir(set(vistos) == treino, "validacoes nao cobrem a coorte")
    return saida, sha256_file(fpath)


def _alvos_distintos():
    vistos, alvos = {}, []
    for a in ARCHITECTURES:
        chave = (a["config"], a["planner"])
        vistos.setdefault(chave, []).append(a["name"])
        if chave not in alvos:
            alvos.append(chave)
    _exigir(len(alvos) == EXPECTED_PREPROCESSING_INVOCATIONS,
            "%d invocacoes, exigido %d" % (len(alvos), EXPECTED_PREPROCESSING_INVOCATIONS))
    partilhados = {k: sorted(v) for k, v in vistos.items() if len(v) > 1}
    _exigir(partilhados == EXPECTED_SHARED,
            "a partilha de alvos nao e a que o registo canonico implica: "
            "observado %s, derivado do registo %s" % (partilhados, EXPECTED_SHARED))
    mapa = dict(("%s|%s" % (c, p or "ExperimentPlanner"), n) for (c, p), n in vistos.items())
    return alvos, mapa


ROTULOS_RESOLVIDOS = []


def _ausencia_de_rotulo(bl, pid, rotulo):
    """
    Negativo confirmado NAO pode ter consenso em lado nenhum: nem no produto do
    Script 06 nem na sobreposicao. Verificar so um dos dois deixaria passar um
    rotulo escondido no outro.
    """
    caminhos = [os.path.join(bl, pid, "consensus.nii.gz")]
    presentes = [c for c in caminhos if os.path.isfile(c)]
    _exigir(not presentes, "%s: %s mas TEM consensus.nii.gz em %s"
            % (pid, rotulo, presentes))


def _rotulo(bl, pid):
    """
    Caminho do consenso, com precedencia da sobreposicao sobre o produto do 06.

    A sobreposicao existe porque a rota XML alterou algumas mascaras e o Script 08
    consome o consenso RECORTADO, nao o de grade original. Cada resolucao e
    registada; ausencia de sobreposicao nao e inferida por varredura.
    """
    base = os.path.join(bl, pid, "consensus.nii.gz")
    ROTULOS_RESOLVIDOS.append({"case_id": pid, "source": "SCRIPT06", "path": base})
    return base


def _resumo_rotulos():
    """
    Origem do rotulo POR CASO, nao por chamada: `_rotulo()` e invocada na
    validacao profunda e outra vez na construcao, de modo que contar chamadas
    daria o dobro. Uma resolucao incoerente entre chamadas e falha.
    """
    por_caso = {}
    for r in ROTULOS_RESOLVIDOS:
        anterior = por_caso.get(r["case_id"])
        _exigir(anterior is None or anterior == (r["source"], r["path"]),
                "%s: rotulo resolvido de duas maneiras: %s e %s"
                % (r["case_id"], anterior, (r["source"], r["path"])))
        por_caso[r["case_id"]] = (r["source"], r["path"])
    n_ov = sum(1 for v in por_caso.values() if v[0] != "SCRIPT06")
    return {"LABEL_SOURCE_OVERLAY": n_ov,
            "LABEL_SOURCE_SCRIPT06": len(por_caso) - n_ov,
            "LABEL_SOURCE_CASES": len(por_caso),
            "overlay_root": "", "overlay_cases": []}


def _validar_par(ct, lb, pid, exigir_rotulo, abrir_ct=False):
    _exigir(os.path.isfile(ct), "%s: CT ausente em %s" % (pid, ct))
    abertos = 0
    if abrir_ct:
        ok, motivo = verify_nifti(ct)
        _exigir(ok, "%s: CT invalida — %s" % (pid, motivo))
        abertos += 1
    if not exigir_rotulo:
        return abertos
    _exigir(os.path.isfile(lb),
            "%s: declarado com alvo adjudicado mas sem consensus.nii.gz em %s. "
            "Nenhuma mascara e fabricada por ausencia de ficheiro." % (pid, lb))
    ok, motivo = verify_nifti(lb, ref_geom=nifti_info(ct), expect_binary=True)
    _exigir(ok, "%s: rotulo invalido — %s" % (pid, motivo))
    return abertos + 1


def _confirmar_rotulo_vazio(caminho, pid):
    """
    Prova que o rotulo copiado do Script 04 nao tem um unico voxel de alvo.
    Chamar-lhe negativo sem o abrir seria confiar num nome de estado.
    """
    import SimpleITK as _sitk
    a = _sitk.GetArrayFromImage(_sitk.ReadImage(caminho))
    n = int((a != 0).sum())
    _exigir(n == 0,
            "%s: classificado NEGATIVE_DOCUMENTED mas o rotulo tem %d voxels de "
            "alvo. O 04 e o 08 discordam sobre este caso." % (pid, n))
    return {"label_voxels_nonzero": 0, "label_shape": list(a.shape),
            "label_verified_empty": True}


def _autorizar_mascara_vazia(pid, dec):
    """
    UNICO portao para criar mascara vazia. Exige a CONJUNCAO:

        training_decision == INCLUDE_NEGATIVE_CONFIRMED   (manifesto de coorte)
        xml_audit_status  == NEGATIVE_CONFIRMED           (auditoria XML)

    Qualquer outra combinacao — incluindo «nao ha ficheiro SEG» — e FAIL_INPUT.
    Devolve o registo de autorizacao, que vai para o manifesto: a criacao tem de
    ser auditavel caso a caso, nao por contagem agregada.
    """
    r = dec.get(pid)
    _exigir(r is not None, "%s: sem registo no manifesto de decisao" % pid)
    d, x = r.get("training_decision"), r.get("xml_audit_status")
    _exigir(d == DECISAO_NEGATIVO and x == XML_NEGATIVO,
            "%s: mascara vazia NAO autorizada — training_decision=%r xml_audit_status=%r; "
            "exige-se %r E %r. FAIL_INPUT."
            % (pid, d, x, DECISAO_NEGATIVO, XML_NEGATIVO))
    return {"case_id": pid, "training_decision": d, "xml_audit_status": x,
            "authorization": "DUAL_CONDITION_SATISFIED",
            "policy_version": r.get("policy_version", "")}


def _criar_mascara_vazia(ct, dst):
    """
    Escreve mascara toda a zero na geometria EXATA da CT. Le a CT so para copiar
    grade — tamanho, espacamento, origem e direcao — sem reamostrar coisa nenhuma.
    So e chamada depois de `_autorizar_mascara_vazia()`.
    """
    import SimpleITK as sitk
    ref = sitk.ReadImage(ct)
    vazia = sitk.Image(ref.GetSize(), sitk.sitkUInt8)
    vazia.SetSpacing(ref.GetSpacing())
    vazia.SetOrigin(ref.GetOrigin())
    vazia.SetDirection(ref.GetDirection())
    _exigir(dst.endswith(".nii.gz"), "destino sem extensao .nii.gz: %s" % dst)
    tmp = dst[:-len(".nii.gz")] + ".tmp.nii.gz"
    sitk.WriteImage(vazia, tmp, useCompression=True)
    os.replace(tmp, dst)
    ok, motivo = verify_nifti(dst, ref_geom=nifti_info(ct), expect_binary=True)
    _exigir(ok, "mascara vazia criada mas invalida: %s" % motivo)
    return {"geometry_source": "CT", "size": list(ref.GetSize()),
            "spacing": [round(float(v), 6) for v in ref.GetSpacing()]}


def _validacao_profunda(ctx):
    """Abre e valida TODOS os volumes envolvidos. Nao escreve nada."""
    bi = os.path.join(ctx.raiz_06, "lidc_idri", "images")
    bl = os.path.join(ctx.raiz_06, "lidc_idri", "labels")
    casos = ficheiros = 0
    for pid in sorted(ctx.adjudicados):
        ficheiros += _validar_par(os.path.join(bi, pid, "ct.nii.gz"),
                                  _rotulo(bl, pid),
                                  pid, True, abrir_ct=True)
        casos += 1
    for pid in sorted(ctx.negativos):
        _ausencia_de_rotulo(bl, pid, "classificado NEGATIVE_CONFIRMED")
        ficheiros += _validar_par(os.path.join(bi, pid, "ct.nii.gz"), None, pid,
                                  False, abrir_ct=True)
        casos += 1
    for pid in sorted(ctx.ref_avaliaveis):
        ficheiros += _validar_par(os.path.join(bi, pid, "ct.nii.gz"),
                                  _rotulo(bl, pid),
                                  pid, True, abrir_ct=True)
        casos += 1
    for pid in sorted(ctx.pend_int):
        ficheiros += _validar_par(os.path.join(bi, pid, "ct.nii.gz"), None, pid,
                                  False, abrir_ct=True)
        casos += 1
    esperado_c = len(ctx.treino) + len(ctx.interno)
    esperado_f = esperado_c + len(ctx.adjudicados) + len(ctx.ref_avaliaveis)
    _exigir(casos == esperado_c, "casos validados %d != %d" % (casos, esperado_c))
    _exigir(ficheiros == esperado_f, "NIfTI abertos %d != %d" % (ficheiros, esperado_f))
    log.info("  validacao profunda: CASES_VALIDATED=%d NIFTI_FILES_OPENED=%d"
             % (casos, ficheiros))
    return casos, ficheiros


def _copiar(src, dst):
    """Copia real. Nada de symlink: foi a origem dos 1718 links quebrados."""
    _exigir(os.path.isfile(src), "origem ausente: %s" % src)
    tmp = dst + ".tmp"
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _construir(ctx, dec):
    ds = os.path.join(ctx.staging, "nnUNet_raw", DATASET_NAME)
    imgs_tr = os.path.join(ds, "imagesTr")
    lbls_tr = os.path.join(ds, "labelsTr")
    imgs_ts = os.path.join(ds, "imagesTs")
    for d in (imgs_tr, lbls_tr, imgs_ts):
        os.makedirs(d, exist_ok=True)
    bi = os.path.join(ctx.raiz_06, "lidc_idri", "images")
    bl = os.path.join(ctx.raiz_06, "lidc_idri", "labels")
    casos, autorizacoes = [], []
    n_copiadas = n_criadas = 0

    for pid in sorted(ctx.treino):
        ct = os.path.join(bi, pid, "ct.nii.gz")
        lb = _rotulo(bl, pid)
        d_ct = os.path.join(imgs_tr, "%s_0000.nii.gz" % pid)
        d_lb = os.path.join(lbls_tr, "%s.nii.gz" % pid)
        if pid in ctx.adjudicados:
            _validar_par(ct, lb, pid, True)
            _copiar(ct, d_ct)
            _copiar(lb, d_lb)
            origem = "ADJUDICATED_CONSENSUS_COPIED"
            n_copiadas += 1
        else:
            aut = _autorizar_mascara_vazia(pid, dec)
            _validar_par(ct, lb, pid, True)
            _copiar(ct, d_ct)
            _copiar(lb, d_lb)
            aut.update(_confirmar_rotulo_vazio(d_lb, pid))
            aut["sha256_copied_label"] = sha256_file(d_lb)
            aut["label_source"] = "SCRIPT04_CONSENSUS_EMPTY_COPIED"
            autorizacoes.append(aut)
            origem = "AUTHORIZED_EMPTY_MASK_COPIED_FROM_04"
            n_criadas += 1
        casos.append({"patient_id": pid, "role": "training", "label_origin": origem,
                      "reference_evaluable": True,
                      "sha256_image": sha256_file(d_ct),
                      "sha256_label": sha256_file(d_lb)})

    for pid in sorted(ctx.interno):
        avaliavel = pid in ctx.ref_avaliaveis
        ct = os.path.join(bi, pid, "ct.nii.gz")
        lb = _rotulo(bl, pid)
        _validar_par(ct, lb, pid, avaliavel)
        d_ct = os.path.join(imgs_ts, "%s_0000.nii.gz" % pid)
        _copiar(ct, d_ct)
        casos.append({"patient_id": pid, "role": "internal_test",
                      "reference_evaluable": avaliavel,
                      "label_origin": "NOT_IN_TREE",
                      "sha256_image": sha256_file(d_ct),
                      "sha256_reference": sha256_file(lb) if avaliavel else "",
                      "reference_path": lb if avaliavel else "",
                      "status": "" if avaliavel else "NO_SEG_PENDING"})

    obs_tr = {f[:-len("_0000.nii.gz")] for f in os.listdir(imgs_tr) if f.endswith(".nii.gz")}
    obs_lb = {f[:-len(".nii.gz")] for f in os.listdir(lbls_tr) if f.endswith(".nii.gz")}
    obs_ts = {f[:-len("_0000.nii.gz")] for f in os.listdir(imgs_ts) if f.endswith(".nii.gz")}
    _exigir(obs_tr == ctx.treino, "imagesTr != coorte: %s" % sorted(obs_tr ^ ctx.treino)[:5])
    _exigir(obs_lb == ctx.treino, "labelsTr != coorte: %s" % sorted(obs_lb ^ ctx.treino)[:5])
    _exigir(obs_ts == ctx.interno, "imagesTs != interno: %s" % sorted(obs_ts ^ ctx.interno)[:5])

    import SimpleITK as _sitk
    vazios = set()
    for pid in sorted(ctx.treino):
        _a = _sitk.GetArrayFromImage(_sitk.ReadImage(
            os.path.join(lbls_tr, "%s.nii.gz" % pid)))
        if not (_a > 0).any():
            vazios.add(pid)
    _ptr4 = json.load(open(PONTEIRO_04))
    _m4 = json.load(open(os.path.join(BASE, _ptr4["annotations_manifest"])))
    doc_neg = {r["patient_id"] for r in _m4["rows"]
               if r.get("status_04") == "NEGATIVE_DOCUMENTED"} & ctx.treino
    _exigir(vazios == doc_neg,
            "rotulos vazios nao adjudicados em imagesTr: %s | esperados (negativos "
            "documentados): %d, observados: %d"
            % (sorted(vazios - doc_neg)[:5], len(doc_neg), len(vazios)))
    log.info("  rotulos vazios em treino: %d — todos negativos documentados"
             % len(vazios))
    _exigir(not (obs_tr & ctx.excl_dev), "excluidos do desenvolvimento em imagesTr")
    _exigir(not (obs_ts & ctx.excl_int), "excluidos internos em imagesTs")
    _exigir(n_copiadas == len(ctx.adjudicados),
            "mascaras adjudicadas copiadas %d != %d" % (n_copiadas, len(ctx.adjudicados)))
    _exigir(n_criadas == len(ctx.negativos),
            "mascaras vazias criadas %d != %d" % (n_criadas, len(ctx.negativos)))
    _exigir(len(autorizacoes) == n_criadas, "autorizacoes != mascaras criadas")

    portoes = {
        "IMAGES_TR_EXACT_SET_MATCH": "PASS", "LABELS_TR_EXACT_SET_MATCH": "PASS",
        "IMAGES_TS_EXACT_SET_MATCH": "PASS",
        "EXCLUDED_DEVELOPMENT_IN_TRAIN": 0, "EXCLUDED_INTERNAL_IN_TEST": 0,
        "ADJUDICATED_MASKS_COPIED": n_copiadas,
        "AUTHORIZED_EMPTY_MASKS_CREATED": n_criadas,
        "UNAUTHORIZED_EMPTY_MASKS": 0,
        "imagesTr": len(obs_tr), "labelsTr": len(obs_lb), "imagesTs": len(obs_ts),
        "EMPTY_LABELS_IN_TRAINING": len(vazios),
        "EMPTY_LABELS_ALL_ADJUDICATED": True,
        "EMPTY_LABEL_RULE": ("todo rotulo vazio em treino tem de ser negativo "
                             "DOCUMENTADO pelo Script 04; vazio nao adjudicado "
                             "aborta o estagio"),
        "imagesTr_ids_sha256": _sha_lista(obs_tr),
        "imagesTs_ids_sha256": _sha_lista(obs_ts),
    }
    log.info("  imagesTr=%d labelsTr=%d imagesTs=%d | adjudicadas copiadas=%d | "
             "vazias autorizadas criadas=%d | nao autorizadas=0"
             % (len(obs_tr), len(obs_lb), len(obs_ts), n_copiadas, n_criadas))
    return ds, casos, autorizacoes, portoes


def _escrever_auxiliares(ctx, ds, sha_folds):
    with open(os.path.join(ds, "dataset.json"), "w") as f:
        json.dump({"channel_names": {"0": "CT"},
                   "labels": {"background": 0, "nodule": 1},
                   "numTraining": len(ctx.treino),
                   "file_ending": ".nii.gz",
                   "overwrite_image_reader_writer": "SimpleITKIO"},
                  f, indent=2, sort_keys=True)
    pre = os.path.join(ctx.staging, "nnUNet_preprocessed", DATASET_NAME)
    os.makedirs(pre, exist_ok=True)
    p_splits = os.path.join(pre, "splits_final.json")
    with open(p_splits, "w") as f:
        json.dump(ctx.splits, f, indent=2)

    orig = json.load(open(os.path.join(
        BASE, json.load(open(CURRENT_BD1))["folds_json"])))["folds"]
    esc = json.load(open(p_splits))
    removidos = set()
    equivalente = True
    for i in range(N_FOLDS):
        o_tr = set(orig["fold_%d" % i]["train"])
        o_va = set(orig["fold_%d" % i]["val"])
        if (set(esc[i]["train"]) != o_tr & ctx.treino or
                set(esc[i]["val"]) != o_va & ctx.treino):
            equivalente = False
        removidos |= (o_tr | o_va) - ctx.treino
    _exigir(equivalente, "splits_final.json nao preserva a atribuicao de fold da "
                         "coorte corrente: houve reatribuicao, nao apenas remocao")
    _ptr2 = json.load(open(PONTEIRO_ANOT))
    _m2 = json.load(open(os.path.join(BASE, _ptr2["annotations_manifest"])))
    _pub2 = {r["patient_id"] for r in _m2["rows"] if r["has_consensus"]}
    _sp2 = os.path.join(BASE, json.load(open(CURRENT_BD1))["split_csv"])
    excluidos_esperados = {r["patient_id"] for r in csv.DictReader(open(_sp2, newline=""))
                           if r["split"] == "development" and r["patient_id"] not in _pub2}
    _exigir(removidos == excluidos_esperados,
            "removidos dos folds %s != exclusoes declaradas %s"
            % (sorted(removidos)[:5], sorted(excluidos_esperados)[:5]))
    portoes = {"COHORT_FOLDS_INPUT_HASH_MATCH": "PASS",
               "cohort_folds_input_sha256": sha_folds,
               "FOLD_ASSIGNMENTS": "PRESERVED",
               "FOLDS_REMOVED_N": len(removidos),
               "FOLDS_REMOVED_EQUALS_EXCLUSIONS": "PASS",
               "SPLITS_SEMANTIC_EQUIVALENCE": "PASS",
               "SPLITS_FINAL_HASH_RECORDED": "PASS",
               "splits_final_sha256": sha256_file(p_splits)}

    bi = os.path.join(ctx.raiz_06, "lidc_idri", "images")
    with open(os.path.join(ctx.staging, "internal_pending.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "split", "status",
                                          "source_ct", "sha256_source_ct"])
        w.writeheader()
        for pid in sorted(ctx.pend_int):
            src = os.path.join(bi, pid, "ct.nii.gz")
            _exigir(os.path.isfile(src), "%s: CT pendente ausente" % pid)
            w.writerow({"patient_id": pid, "split": "internal", "status": "NO_SEG_PENDING",
                        "source_ct": src, "sha256_source_ct": sha256_file(src)})
    return pre, portoes


INVENTORY_SCOPE = ("todos os ficheiros regulares sob a raiz da execucao no instante "
                   "da chamada: nnUNet_raw, nnUNet_preprocessed, nnUNet_results, "
                   "preprocess_logs e internal_pending.csv")
INVENTORY_EXCLUSIONS = ["manifest.json", "manifest_structure.json",
                        "pointer (CURRENT_NNUNET*.json, fora da arvore)"]


def _sha_do_proprio_script():
    try:
        return sha256_file(os.path.abspath(__file__))
    except Exception as e:
        return "NOT_DETERMINED:%s" % type(e).__name__


def _inventario(raiz):
    itens = []
    for d, _s, fs in os.walk(raiz):
        for nome in sorted(fs):
            ab = os.path.join(d, nome)
            if os.path.islink(ab) or not os.path.isfile(ab):
                continue
            itens.append({"relative_path": os.path.relpath(ab, raiz),
                          "size_bytes": os.path.getsize(ab), "sha256": sha256_file(ab)})
    return sorted(itens, key=lambda x: x["relative_path"])


def _planos_gerados(pre_ds, cfg, planner):
    plans_id = PLANS_ID.get(planner, planner or "nnUNetPlans")
    caminho = os.path.join(pre_ds, "%s.json" % plans_id)
    if not os.path.isfile(caminho):
        return {"plans_file": caminho, "plans_present": False, "configuration_present": False}
    j = json.load(open(caminho))
    cfgs = list(j.get("configurations", {}))
    return {"plans_file": caminho, "plans_present": True,
            "plans_sha256": sha256_file(caminho),
            "configurations_in_plans": cfgs, "configuration_present": cfg in cfgs}


def _resolver_data_identifier(plans, cfg, _visto=None):
    """Segue `inherits_from`: a cascata reutiliza os dados do `3d_fullres`."""
    _visto = _visto or set()
    cfgs = plans.get("configurations", {})
    if cfg not in cfgs:
        return None, "CONFIGURATION_ABSENT"
    if cfg in _visto:
        return None, "INHERITS_CYCLE"
    _visto.add(cfg)
    bloco = cfgs[cfg]
    if bloco.get("data_identifier"):
        return bloco["data_identifier"], "DIRECT"
    pai = bloco.get("inherits_from")
    if pai:
        di, origem = _resolver_data_identifier(plans, pai, _visto)
        return di, ("INHERITED_FROM_%s" % pai if di else origem)
    return None, "NO_DATA_IDENTIFIER"


def _cobertura_preprocessada(pre_ds, data_id, esperados):
    """Confere os IDs REALMENTE pre-processados. Exige igualdade exata do conjunto."""
    d = os.path.join(pre_ds, data_id) if data_id else None
    if not d or not os.path.isdir(d):
        return {"data_identifier": data_id, "preprocessed_dir": d, "dir_present": False,
                "n_observed": 0, "coverage": "0/%d" % len(esperados),
                "COVERAGE_STATUS": "FAIL_DIR_ABSENT", "missing_ids": sorted(esperados)[:20],
                "extra_ids": [], "n_missing": len(esperados), "n_extra": 0,
                "id_list_sha256": ""}
    dados, segs, pkls = set(), set(), set()
    for e in os.listdir(d):
        for ext in (".b2nd", ".npz", ".npy"):
            if e.endswith("_seg" + ext):
                segs.add(e[:-len("_seg" + ext)]); break
            if e.endswith(ext):
                dados.add(e[:-len(ext)]); break
        else:
            if e.endswith(".pkl"):
                pkls.add(e[:-len(".pkl")])
    observados = dados & pkls
    faltam = sorted(esperados - observados)
    sobram = sorted(observados - esperados)
    ok = (not faltam) and (not sobram)
    return {"data_identifier": data_id, "preprocessed_dir": d, "dir_present": True,
            "n_data_files": len(dados), "n_seg_files": len(segs), "n_pkl_files": len(pkls),
            "n_observed": len(observados),
            "coverage": "%d/%d" % (len(observados), len(esperados)),
            "COVERAGE_STATUS": "PASS" if ok else "FAIL_INCOMPLETE_COVERAGE",
            "n_missing": len(faltam), "n_extra": len(sobram),
            "missing_ids": faltam[:20], "extra_ids": sobram[:20],
            "id_list_sha256": _sha_lista(observados)}


def _preprocessar(ctx, raiz_exec, pre_ds):
    logs_dir = os.path.join(raiz_exec, "preprocess_logs")
    os.makedirs(logs_dir, exist_ok=True)
    os.environ["nnUNet_raw"] = os.path.join(raiz_exec, "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = os.path.join(raiz_exec, "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = os.path.join(raiz_exec, "nnUNet_results")
    os.makedirs(os.environ["nnUNet_results"], exist_ok=True)
    registos, falhas, mapeamento = [], [], []

    def _correr(cmd, rotulo):
        base = os.path.join(logs_dir, rotulo)
        t0 = time.time()
        with open(base + ".stdout.log", "w") as fo, open(base + ".stderr.log", "w") as fe:
            r = subprocess.run(cmd, stdout=fo, stderr=fe, text=True)
        return {"label": rotulo, "command": cmd, "returncode": r.returncode,
                "duration_s": round(time.time() - t0, 1),
                "stdout_log": os.path.relpath(base + ".stdout.log", raiz_exec),
                "stderr_log": os.path.relpath(base + ".stderr.log", raiz_exec),
                "stdout_sha256": sha256_file(base + ".stdout.log"),
                "stderr_sha256": sha256_file(base + ".stderr.log")}

    reg = _correr(_executavel_nnunet("nnUNetv2_extract_fingerprint")
                  + ["-d", str(DATASET_ID), "--verify_dataset_integrity"], "fingerprint")
    registos.append(reg)
    _exigir(reg["returncode"] == 0,
            "fingerprint falhou (rc=%s); ver %s" % (reg["returncode"], reg["stderr_log"]))
    _exigir(os.path.isfile(os.path.join(pre_ds, "dataset_fingerprint.json")),
            "dataset_fingerprint.json nao foi gerado")
    log.info("  fingerprint OK (%ss)" % reg["duration_s"])

    identificadores = []
    for cfg, pl in ctx.alvos:
        rotulo = "%s__%s" % (cfg, pl or "ExperimentPlanner")
        cmd = (_executavel_nnunet("nnUNetv2_plan_and_preprocess")
               + ["-d", str(DATASET_ID), "-c", cfg, "--verify_dataset_integrity",
                  "-np", str(N_WORKERS)]
               + (["-pl", pl] if pl else []))
        log.info("  pre-processando %s ..." % rotulo)
        reg = _correr(cmd, rotulo)
        plano = _planos_gerados(pre_ds, cfg, pl)
        if plano.get("plans_file"):
            plano["plans_file"] = os.path.relpath(plano["plans_file"], raiz_exec)
        reg.update(plano)

        if reg["returncode"] == 0 and reg.get("configuration_present"):
            plans = json.load(open(os.path.join(raiz_exec, plano["plans_file"])))
            data_id, origem = _resolver_data_identifier(plans, cfg)
            cob = _cobertura_preprocessada(pre_ds, data_id, ctx.treino)
            cob["data_identifier_origin"] = origem
            if cob.get("preprocessed_dir"):
                cob["preprocessed_dir"] = os.path.relpath(cob["preprocessed_dir"], raiz_exec)
            reg.update(cob)
            if data_id:
                identificadores.append(data_id)
            mapeamento.append({
                "configuration": cfg,
                "planner": pl or "ExperimentPlanner",
                "nnunet_configuration": cfg,
                "resolved_data_identifier": data_id,
                "inherits_from": (json.load(open(os.path.join(raiz_exec, plano["plans_file"])))
                                  .get("configurations", {}).get(cfg, {}).get("inherits_from")),
                "data_identifier_origin": origem,
                "architectures": ctx.mapa_alvos.get(
                    "%s|%s" % (cfg, pl or "ExperimentPlanner"), []),
            })

        registos.append(reg)
        if reg["returncode"] != 0:
            falhas.append("%s: rc=%s" % (rotulo, reg["returncode"]))
            log.error("    FALHOU rc=%s — ver %s" % (reg["returncode"], reg["stderr_log"]))
        elif not reg["configuration_present"]:
            falhas.append("%s: retorno zero mas a configuracao %s nao consta dos planos"
                          % (rotulo, cfg))
            log.error("    rc=0 mas %s ausente dos planos" % cfg)
        elif reg.get("COVERAGE_STATUS") != "PASS":
            falhas.append("%s: cobertura %s — faltam %d, sobram %d"
                          % (rotulo, reg.get("coverage"), reg.get("n_missing", -1),
                             reg.get("n_extra", -1)))
            log.error("    cobertura %s | data_identifier=%s"
                      % (reg.get("coverage"), reg.get("data_identifier")))
        else:
            log.info("    OK (%ss) | cobertura %s | data_identifier=%s (%s)"
                     % (reg["duration_s"], reg["coverage"], reg["data_identifier"],
                        reg["data_identifier_origin"]))

    distintos = sorted(set(identificadores))
    _ids_declarados_nos_planos = set()
    for _a in ARCHITECTURES:
        _pf = os.path.join(pre_ds, "%s.json" % _a["plans_id"])
        if not os.path.isfile(_pf):
            continue
        _cfgs = (json.load(open(_pf)) or {}).get("configurations", {})
        _c = _cfgs.get(_a["config"], {})
        _di = _c.get("data_identifier")
        if _di is None and _c.get("inherits_from"):
            _di = _cfgs.get(_c["inherits_from"], {}).get("data_identifier")
        if _di:
            _ids_declarados_nos_planos.add(_di)
    _exigir(_ids_declarados_nos_planos,
            "nenhum plano declara `data_identifier`: nao ha contra o que verificar")
    invocacoes_obs = {(m["configuration"], m["planner"]) for m in mapeamento}
    invocacoes_esp = {(c, p or "ExperimentPlanner") for c, p in ctx.alvos}
    portoes = {
        "PREPROCESSING_INVOCATIONS": len(ctx.alvos),
        "DISTINCT_DATA_IDENTIFIERS": len(distintos),
        "distinct_data_identifiers": distintos,
        "preprocessing_mapping": mapeamento,
        "PREPROCESSING_MAPPING_SHA256": hashlib.sha256(
            json.dumps(mapeamento, sort_keys=True).encode()).hexdigest(),
        "PREPROCESSING_INVOCATION_SET_MATCH":
            "PASS" if invocacoes_obs == invocacoes_esp else "FAIL",
        "DATA_IDENTIFIER_EXACT_SET_MATCH":
            "PASS" if set(distintos) == set(_ids_declarados_nos_planos) else "FAIL",
        "data_identifiers_declared_by_plans": sorted(_ids_declarados_nos_planos),
    }
    if not falhas:
        _exigir(invocacoes_obs == invocacoes_esp,
                "conjunto de invocacoes divergente: %s"
                % sorted(invocacoes_obs ^ invocacoes_esp))
        _exigir(set(distintos) == set(_ids_declarados_nos_planos),
                "os diretorios de dados nao sao os que os planos declaram — "
                "observado %s, declarado pelos planos %s"
                % (distintos, sorted(_ids_declarados_nos_planos)))
    log.info("  PREPROCESSING_INVOCATIONS=%d | DISTINCT_DATA_IDENTIFIERS=%d | "
             "INVOCATION_SET_MATCH=%s | DATA_IDENTIFIER_EXACT_SET_MATCH=%s"
             % (len(ctx.alvos), len(distintos),
                portoes["PREPROCESSING_INVOCATION_SET_MATCH"],
                portoes["DATA_IDENTIFIER_EXACT_SET_MATCH"]))
    return registos, falhas, os.path.relpath(logs_dir, raiz_exec), portoes


def _executavel_nnunet(nome):
    """
    Devolve o comando que invoca o entry point do nnU-Net PELO INTERPRETADOR que
    executa este script, em vez do script gerado em `venv/bin`.

    Os wrappers de `venv/bin` trazem um shebang com o caminho absoluto de quando
    o ambiente foi criado. No Mac esse caminho aponta para `Programa_Pulmonary`,
    o nome antigo da pasta, e a chamada morre com rc=126 — comando encontrado mas
    nao executavel. Chamar pelo modulo ignora o shebang e funciona igual nas duas
    maquinas, sem recriar ambientes ja validados.

    O entry point resolvido e o MESMO que o comando original chamaria: e lido dos
    metadados do pacote instalado, nao adivinhado.
    """
    import importlib.metadata as _md
    eps = _md.entry_points()
    consoles = (eps.get("console_scripts", []) if hasattr(eps, "get")
                else eps.select(group="console_scripts"))
    alvo = [e for e in consoles if e.name == nome]
    _exigir(alvo, "entry point %s nao existe na instalacao de %s"
                  % (nome, sys.executable))
    modulo, _, funcao = alvo[0].value.partition(":")
    _exigir(modulo.startswith("nnunetv2"),
            "%s resolve para %r, fora do nnunetv2" % (nome, alvo[0].value))
    ENTRYPOINTS_USADOS[nome] = {"value": alvo[0].value,
                                "interpreter": sys.executable}
    return [sys.executable, "-c",
            "import sys; from %s import %s as _f; sys.argv[0]=%r; _f()"
            % (modulo, funcao, nome)]


ENTRYPOINTS_USADOS = {}


def _versao_nnunet():
    try:
        import importlib.metadata as md
        return md.version("nnunetv2")
    except Exception as e:
        return "NOT_DETERMINED:%s" % type(e).__name__


def _publicar(alvo, run_id, destino, manifesto_path, estado, executado, passados):
    pt = {"run_id": run_id, "script": "08", "script_version": SCRIPT08_VERSION,
          "pointer_schema": ESQUEMA_PONTEIRO_ACTUAL,
          "manifest": manifesto_path,
          "manifest_file_sha256": sha256_file(manifesto_path),
          "manifest_body_sha256": _corpo_sha(json.load(open(manifesto_path))),
          "nnUNet_raw": os.path.join(destino, "nnUNet_raw"),
          "nnUNet_preprocessed": os.path.join(destino, "nnUNet_preprocessed"),
          "RUN_STATUS": estado, "PREPROCESS_EXECUTED": executado,
          "PREPROCESSING_TARGETS_PASSED": passados,
          "published_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    tmp = alvo + ".tmp"
    with open(tmp, "w") as f:
        json.dump(pt, f, indent=1, sort_keys=True)
    os.replace(tmp, alvo)
    return pt


def _gravar_manifesto(caminho, corpo):
    corpo["manifest_body_sha256"] = _corpo_sha(corpo)
    tmp = caminho + ".tmp"
    with open(tmp, "w") as f:
        json.dump(corpo, f, indent=1, sort_keys=True)
    os.replace(tmp, caminho)
    return corpo


def main():
    ap = argparse.ArgumentParser()
    modo = ap.add_mutually_exclusive_group()
    modo.add_argument("--plan-only", action="store_true")
    modo.add_argument("--skip-preprocess", action="store_true")
    ap.add_argument("--deep", action="store_true")
    args = ap.parse_args()
    _exigir(not args.deep or args.plan_only, "--deep exige --plan-only")
    check_base(strict=True)

    nome_modo = ("PLAN-ONLY-DEEP" if args.plan_only and args.deep else
                 "PLAN-ONLY" if args.plan_only else
                 "SKIP-PREPROCESS" if args.skip_preprocess else "CANONICO")
    run_id = "s08_%s_%d" % (time.strftime("%Y%m%dT%H%M%S"), os.getpid())
    log.info("Script 08 %s | run_id=%s | modo=%s" % (SCRIPT08_VERSION, run_id, nome_modo))
    log.info("  SCRIPT08_LABEL_POLICY = %s" % SCRIPT08_LABEL_POLICY)

    p6, m6, p7, m7, raiz6, verif = _entradas()
    ancora04, tem_alvo = _ancorar_manifesto_04()
    (treino, adjud, negat, interno, ref_aval, pend_int,
     excl_dev, excl_int, dec, meta) = _carregar_coorte(tem_alvo)
    log.info("  folds congelados:")
    splits, sha_folds = _folds_congelados(treino)
    alvos, mapa = _alvos_distintos()
    log.info("  arquiteturas=%d | invocacoes distintas=%d (do registo do treino_core)"
             % (len(ARCHITECTURES), len(alvos)))

    ctx = Contexto(args=args, run_id=run_id,
                   staging=os.path.join(STAGING_ROOT, run_id),
                   ponteiro_06=p6, ponteiro_07=p7, raiz_06=raiz6,
                   verificacao_entradas=verif, ancora_04=ancora04, coorte=meta,
                   treino=treino, adjudicados=adjud, negativos=negat,
                   interno=interno, ref_avaliaveis=ref_aval, pend_int=pend_int,
                   excl_dev=excl_dev, excl_int=excl_int,
                   splits=splits, alvos=alvos, mapa_alvos=mapa)

    if args.plan_only:
        casos_val, abertos = _validacao_profunda(ctx) if args.deep else (0, 0)
        estado = "PASS_FULL" if args.deep else "PASS_METADATA_ONLY"
        log.info("-" * 68)
        log.info("  PLAN-ONLY — nada escrito")
        for k, v in list(meta.items()) + [("CASES_VALIDATED", casos_val),
                                          ("NIFTI_FILES_OPENED", abertos),
                                          ("PLAN_STATUS", estado)]:
            if not isinstance(v, dict):
                log.info("  %-40s = %s" % (k, v))
        return

    os.makedirs(STAGING_ROOT, exist_ok=True)
    _exigir(not os.path.exists(ctx.staging), "staging preexistente: %s" % ctx.staging)
    os.makedirs(ctx.staging, exist_ok=False)
    try:
        return _executar(ctx, dec, sha_folds)
    finally:
        if os.path.isdir(ctx.staging):
            logs = os.path.join(ctx.staging, "preprocess_logs")
            if os.path.isdir(logs):
                destino_logs = os.path.join(BASE, "quarantine", "script08_failed_runs",
                                            ctx.run_id)
                os.makedirs(os.path.dirname(destino_logs), exist_ok=True)
                shutil.move(logs, destino_logs)
                with open(os.path.join(destino_logs, "INVALID.json"), "w") as fh:
                    json.dump({"record": "INVALID", "run_id": ctx.run_id, "script": "08",
                               "script_version": SCRIPT08_VERSION,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S BRT"),
        "producer_script_sha256": _sha_do_proprio_script(),
                               "status": "FAILED_BEFORE_PUBLICATION",
                               "payload_discarded": True, "logs_preserved": True,
                               "publication": "NEVER_PUBLISHED"}, fh, indent=1, sort_keys=True)
                log.warning("  logs preservados em %s" % destino_logs)
            shutil.rmtree(ctx.staging, ignore_errors=True)
            log.warning("  staging removido: %s" % ctx.staging)


def _executar(ctx, dec, sha_folds):
    ds, casos, autorizacoes, portoes_arvore = _construir(ctx, dec)
    pre_ds_estrutura, portoes_folds = _escrever_auxiliares(ctx, ds, sha_folds)

    base = {
        "run_id": ctx.run_id, "script": "08", "script_version": SCRIPT08_VERSION,
        "nnunet_version": _versao_nnunet(),
        "PATH_RESOLUTION": {
            "policy": "REANCHOR_WITH_SHA256",
            "rule": "caminho absoluto registado; se ausente, "
                    "PULMONARY_BASE/parte_relativa, sempre com hash conferido",
            "no_recursive_search": True, "no_newest_file_heuristic": True,
            "no_basename_only_match": True,
            "resolutions": RESOLUCOES},
        "PYTHON_EXECUTABLE": sys.executable,
        "NNUNET_ENTRYPOINTS_USED": dict(ENTRYPOINTS_USADOS),
        "NNUNET_INVOCATION_MODE": "PYTHON_MODULE_ENTRYPOINT",
        "NNUNET_INVOCATION_REASON": ("os wrappers de venv/bin trazem shebang "
                                     "absoluto do momento da criacao; no Mac "
                                     "apontam para a pasta antiga e falham com "
                                     "rc=126"),
        "PYTHON_INTERPRETER": sys.executable,
        "PREPROCESS_WORKERS": N_WORKERS,
        "PREPROCESS_WORKERS_RULE": ("derivado da RAM (~3,5 GB por trabalhador), "
                                    "limitado por nucleos e tecto de 8"),
        "input_run_id_script06": ctx.ponteiro_06["run_id"],
        "input_run_id_script07": ctx.ponteiro_07["run_id"],
        "input_manifest_verification": ctx.verificacao_entradas,
        "INPUT_INTEGRITY_LEVELS": dict((k, v["INTEGRITY_LEVEL"])
                                       for k, v in ctx.verificacao_entradas.items()),
        "ALL_INPUTS_FULL_INTEGRITY": all(v["INTEGRITY_LEVEL"] == "FULL"
                                         for v in ctx.verificacao_entradas.values()),
        "DATA_INPUT_AUDIT": "PASS_FULL",
        "SCRIPT06_POINTER_AUDIT": ctx.verificacao_entradas["script06"]["VERIFICATION_VERDICT"],
        "SCRIPT07_POINTER_AUDIT": ctx.verificacao_entradas["script07"]["VERIFICATION_VERDICT"],
        "script04_manifest_anchor": ctx.ancora_04,
        "SCRIPT08_LABEL_POLICY": SCRIPT08_LABEL_POLICY,
        "SPLITS_FINAL_POLICY": SPLITS_FINAL_POLICY,
        "cohort": ctx.coorte,
        "label_source_resolution": _resumo_rotulos(),
        "empty_mask_authorizations": autorizacoes,
        "EMPTY_MASK_RULE": ("criacao exige training_decision=%s E xml_audit_status=%s; "
                            "ausencia de SEG isoladamente e FAIL_INPUT"
                            % (DECISAO_NEGATIVO, XML_NEGATIVO)),
        "ARCHITECTURES": len(ARCHITECTURES),
        "PREPROCESSING_INVOCATIONS": len(ctx.alvos),
        "preprocessing_target_map": ctx.mapa_alvos,
        "cases": casos,
    }
    base.update(portoes_arvore)
    base.update(portoes_folds)

    if ctx.args.skip_preprocess:
        estrutura = dict(base)
        estrutura.update({"manifest_id": "nnunet_structure_%s" % ctx.run_id,
                          "RUN_STATUS": "STRUCTURE_ONLY_PASS",
                          "PREPROCESS_EXECUTED": "NO",
                          "PREPROCESSING_TARGETS_PASSED": "0/%d" % len(ctx.alvos),
                          "SCRIPT08_STRUCTURE_STATUS": "PASS",
                          "inventory_scope": INVENTORY_SCOPE,
                          "inventory_exclusions": INVENTORY_EXCLUSIONS,
                          "inventory": _inventario(ctx.staging)})
        _gravar_manifesto(os.path.join(ctx.staging, "manifest_structure.json"), estrutura)
        destino = os.path.join(RUNS_ROOT, ctx.run_id)
        _exigir(not os.path.exists(destino), "destino ja existe: %s" % destino)
        os.makedirs(RUNS_ROOT, exist_ok=True)
        shutil.move(ctx.staging, destino)
        _publicar(CURRENT_NNUNET_STRUCTURE, ctx.run_id, destino,
                  os.path.join(destino, "manifest_structure.json"),
                  "STRUCTURE_ONLY_PASS", "NO", "0/%d" % len(ctx.alvos))
        log.info("-" * 68)
        for k in ("IMAGES_TR_EXACT_SET_MATCH", "LABELS_TR_EXACT_SET_MATCH",
                  "IMAGES_TS_EXACT_SET_MATCH", "ADJUDICATED_MASKS_COPIED",
                  "AUTHORIZED_EMPTY_MASKS_CREATED", "UNAUTHORIZED_EMPTY_MASKS",
                  "SPLITS_SEMANTIC_EQUIVALENCE"):
            log.info("  %-40s = %s" % (k, estrutura[k]))
        log.info("  %-40s = %s" % ("CURRENT_NNUNET_UPDATE", "NO"))
        return

    pre_ds = os.path.join(ctx.staging, "nnUNet_preprocessed", DATASET_NAME)
    registos, falhas, logs_rel, portoes_pre = _preprocessar(ctx, ctx.staging, pre_ds)
    passados = sum(1 for r in registos[1:]
                   if r["returncode"] == 0 and r.get("configuration_present")
                   and r.get("COVERAGE_STATUS") == "PASS")
    if falhas:
        log.error("%d de %d alvos falharam. Nada e publicado." % (len(falhas), len(ctx.alvos)))
        for f_ in falhas:
            log.error("  . %s" % f_)
        sys.exit(1)

    canon = dict(base)
    canon.update({
        "manifest_id": "nnunet_prepared_%s" % ctx.run_id,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S BRT"),
        "producer_script_sha256": _sha_do_proprio_script(),
        "RUN_STATUS": "PREPROCESS_COMPLETE_PASS",
        "PREPROCESS_EXECUTED": "YES",
        "PREPROCESSING_TARGETS_PASSED": "%d/%d" % (passados, len(ctx.alvos)),
        "preprocessing_runs": registos,
        "preprocess_logs_dir": logs_rel,
        "dataset_fingerprint_sha256": sha256_file(
            os.path.join(pre_ds, "dataset_fingerprint.json")),
        "preprocessed_json_sha256": {
            _f: sha256_file(os.path.join(pre_ds, _f))
            for _f in sorted(os.listdir(pre_ds))
            if _f.endswith(".json") and os.path.isfile(os.path.join(pre_ds, _f))},
        "plans_generated": [{"target": r["label"], "plans_file": r.get("plans_file"),
                             "plans_sha256": r.get("plans_sha256"),
                             "configuration_present": r.get("configuration_present"),
                             "data_identifier": r.get("data_identifier"),
                             "data_identifier_origin": r.get("data_identifier_origin"),
                             "coverage": r.get("coverage"),
                             "COVERAGE_STATUS": r.get("COVERAGE_STATUS"),
                             "id_list_sha256": r.get("id_list_sha256")}
                            for r in registos[1:]],
        "EXPECTED_COVERAGE_PER_TARGET": "%d/%d" % (len(ctx.treino), len(ctx.treino)),
        "SCRIPT08_UNDOCUMENTED_FAILURES": 0,
        "SCRIPT08_OUTPUT_STATUS": "PASS",
        "SCRIPT08_OUTPUT_MANIFEST_FROZEN": "YES",
        "inventory_scope": INVENTORY_SCOPE,
        "inventory_exclusions": INVENTORY_EXCLUSIONS,
        "inventory": _inventario(ctx.staging),
    })
    canon.update(portoes_pre)
    _gravar_manifesto(os.path.join(ctx.staging, "manifest.json"), canon)

    destino = os.path.join(RUNS_ROOT, ctx.run_id)
    _exigir(not os.path.exists(destino), "destino ja existe: %s" % destino)
    os.makedirs(RUNS_ROOT, exist_ok=True)
    shutil.move(ctx.staging, destino)
    _publicar(CURRENT_NNUNET, ctx.run_id, destino, os.path.join(destino, "manifest.json"),
              "PREPROCESS_COMPLETE_PASS", "YES", "%d/%d" % (passados, len(ctx.alvos)))

    log.info("-" * 68)
    for k in ("imagesTr", "labelsTr", "imagesTs",
              "IMAGES_TR_EXACT_SET_MATCH", "LABELS_TR_EXACT_SET_MATCH",
              "IMAGES_TS_EXACT_SET_MATCH", "EXCLUDED_DEVELOPMENT_IN_TRAIN",
              "EXCLUDED_INTERNAL_IN_TEST", "ADJUDICATED_MASKS_COPIED",
              "AUTHORIZED_EMPTY_MASKS_CREATED", "UNAUTHORIZED_EMPTY_MASKS",
              "COHORT_FOLDS_INPUT_HASH_MATCH", "SPLITS_SEMANTIC_EQUIVALENCE",
              "PREPROCESSING_INVOCATIONS", "DISTINCT_DATA_IDENTIFIERS",
              "PREPROCESSING_INVOCATION_SET_MATCH", "DATA_IDENTIFIER_EXACT_SET_MATCH",
              "PREPROCESSING_MAPPING_SHA256",
              "PREPROCESSING_TARGETS_PASSED", "SCRIPT08_OUTPUT_STATUS"):
        log.info("  %-40s = %s" % (k, canon[k]))
    log.info("  %-40s = %s" % ("SCRIPT08_OUTPUT_MANIFEST_BODY_SHA256",
                               canon["manifest_body_sha256"]))


if __name__ == "__main__":
    try:
        main()
    except FalhaMetodologica as e:
        log.error("FALHA METODOLOGICA: %s" % e)
        sys.exit(1)
