#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    09_train_nnunet.py

File type:
    Script

Purpose:
    Trains the fifty definitive nnU-Net models (ten configurations, five folds) through the shared training core.

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
    - registos_parciais.json
    - folds_ensaio.json

Outputs:
    - See docs/file_level_documentation.md.

Dependencies:
    - Python standard library only.

Execution:
    PULMONARY_BASE=<project_dir> python scripts/09_train_nnunet.py

Portability notes:
    All paths are resolved relative to the project base directory given by the PULMONARY_BASE environment variable. No machine-specific path is required.

Reproducibility notes:
    Environment pinned in requirements.txt and environment/. Seeds and frozen parameters are recorded in the run manifests under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) are not included and must be obtained from their original repositories.
"""

import os
import sys
import json
import time
import hashlib
import logging
import argparse
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pulmonary_base import BASE, sha256_file, escrever_ponteiro
import treino_core as tc

RUNS = os.path.join(BASE, "runs")
PONTEIRO = os.path.join(RUNS, "CURRENT_TRAINING.json")
RESULTADOS_PADRAO = os.path.join(BASE, "models")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("09")


def _exigir(cond, msg):
    if not cond:
        raise SystemExit("[FAIL] %s" % msg)


def _publicar_emissao(run_id, base_prov, cascatas, escritas, args):
    """Grava em disco o que a emissao de ordens verificou.

    Emitir uma ordem e um acto com consequencias: fixa o estagio anterior, o
    checkpoint de onde saiu cada bloco e a coorte. Sem registo, a unica prova de
    que o portao da cascata correu e a mensagem que passou no terminal.

    Escreve `runs/<run_id>/` com o `cascade_provenance`, a lista das ordens e o
    sha256 de cada uma — para que uma ordem no disco possa depois ser confrontada
    com o que foi emitido.
    """
    destino = os.path.join(RUNS, run_id)
    os.makedirs(destino, exist_ok=True)

    ordens = []
    for c in escritas:
        o = json.load(open(c))
        ordens.append({"ficheiro": os.path.basename(c),
                       "caminho": c,
                       "architecture": o["architecture"]["name"],
                       "fold": o["fold"],
                       "order_sha256": o.get("order_sha256"),
                       "ficheiro_sha256": sha256_file(c),
                       "treino_core_sha256": o.get("treino_core_sha256")})
    po = os.path.join(destino, "ordens.json")
    with open(po, "w") as f:
        json.dump(ordens, f, indent=1, sort_keys=True, default=str)

    m = dict(base_prov)
    m.update({"manifest_id": "emissao_%s" % run_id, "run_id": run_id,
              "script": "09_TRAIN_NNUNET", "modo": "EMITIR_ORDENS",
              "generated": time.strftime("%Y-%m-%dT%H:%M:%S BRT"),
              "producer_sha256": sha256_file(os.path.abspath(__file__)),
              "results_root": args.resultados,
              "ordens_dir": args.emitir_ordens,
              "n_ordens": len(ordens),
              "ordens_sha256": sha256_file(po),
              "cascade_provenance": cascatas,
              "SCRIPT09_EMISSION_STATUS": "PASS"})
    corpo = hashlib.sha256(json.dumps(
        {k: v for k, v in m.items() if k != "manifest_body_sha256"},
        indent=1, sort_keys=True, default=str).encode()).hexdigest()
    m["manifest_body_sha256"] = corpo
    pm = os.path.join(destino, "manifest.json")
    with open(pm, "w") as f:
        json.dump(m, f, indent=1, sort_keys=True, default=str)
    log.info("  emissao registada em %s", destino)
    return pm


def _alvos(args):
    """As (arquitetura, fold) a treinar, na ordem do registo."""
    nomes = args.arch or [a["name"] for a in tc.ARQUITETURAS]
    desconhecidas = [n for n in nomes if n not in tc.POR_NOME]
    _exigir(not desconhecidas, "arquitetura desconhecida: %s" % desconhecidas)
    folds = args.folds if args.folds is not None else list(range(5))
    return [(tc.POR_NOME[n], f) for a in tc.ARQUITETURAS
            for n in [a["name"]] if n in nomes for f in folds]


def executar_ordem(args):
    """
    EXECUTOR. Não resolve a cadeia: recebe uma ordem assinada, prova que os dados
    locais são os que ela descreve, treina, e grava o registo ao lado do fold.
    """
    _exigir(args.preprocessed, "--ordem exige --preprocessed <dir>")
    _exigir(args.resultados != RESULTADOS_PADRAO or os.path.isdir(args.resultados),
            "--ordem exige --resultados explicito")
    try:
        o, verif = tc.carregar_ordem(args.ordem, args.preprocessed)
    except tc.FalhaTreino as e:
        raise SystemExit("[FAIL] %s" % e)

    arq, fold = o["architecture"], int(o["fold"])
    run_id = "s09exec_%s_%d" % (time.strftime("%Y%m%dT%H%M%S"), os.getpid())
    destino = os.path.join(os.path.dirname(os.path.abspath(args.ordem)), run_id)
    os.makedirs(destino, exist_ok=True)
    log.info("Script 09 — EXECUTOR de ordem | run_id=%s", run_id)
    log.info("  ordem de %s, emitida por %s em %s",
             o.get("issuer_host"), o.get("issued_by_run"), o.get("issued_at"))
    log.info("  %d ficheiros de definicao conferidos contra a ordem",
             verif["json_verified"])
    if verif.get("warning"):
        log.warning("  %s", verif["warning"])

    _amb = tc.ambiente(args.python)
    try:
        _cv, _cs = tc.compile_para_placa(o.get("operational_overrides"), _amb.get("gpu"))
    except tc.FalhaTreino as e:
        raise SystemExit("[FAIL] %s" % e)
    os.environ["nnUNet_compile"] = str(_cv)
    log.info("  placa local: %s · compile=%s (%s)", _amb.get("gpu"), _cv, _cs)
    os.environ["PULMONARY_PREPROCESSED"] = args.preprocessed
    esperado = {"trainer_effective": o.get("expected_trainer_effective") or {},
                "operational_overrides": o.get("operational_overrides")}
    sobrepostos = dict(o.get("declared_overrides") or {})
    if args.smoke:
        sobrepostos["num_epochs"] = args.smoke
    log.info("── %s fold %d · compile=%s · %s ──", arq["name"], fold,
             os.environ["nnUNet_compile"], sobrepostos or "sem sobreposto")

    def registar(evento, detalhe):
        log.warning("  [%s] %s/fold_%d · %s", evento, arq["name"], fold, detalhe)

    r = tc.executar_fold(arq, fold, interpretador=args.python,
                         resultados=args.resultados, destino_run=destino,
                         run_id=run_id, sobrepostos=sobrepostos,
                         esperados=esperado, regra=o["collapse_rule"],
                         continuar=args.continuar, ao_registar=registar,
                         preprocessed=args.preprocessed)
    r["order"] = {"file": os.path.abspath(args.ordem),
                  "order_sha256": o.get("order_sha256"),
                  "issued_by_run": o.get("issued_by_run"),
                  "verification": verif}
    r["provenance"] = o.get("provenance")
    r["environment"] = _amb
    r["compile_resolution"] = {"value": _cv, "status": _cs, "gpu": _amb.get("gpu")}
    fr = os.path.join(destino, "registo_fold.json")
    with open(fr, "w") as f:
        json.dump(r, f, indent=1, sort_keys=True, default=str)
    log.info("  %s/fold_%d -> %s em %.2f h", arq["name"], fold, r["outcome"],
             (r.get("duration_s") or 0) / 3600.0)
    log.info("  registo em %s — devolver ao emissor", fr)
    return 0 if r["outcome"] == "COMPLETED" else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", nargs="+", metavar="NOME",
                    help="treina so estas arquiteturas (por omissao, as dez)")
    ap.add_argument("--folds", nargs="+", type=int, metavar="N",
                    help="treina so estes folds (por omissao, 0..4)")
    ap.add_argument("--resultados", default=RESULTADOS_PADRAO,
                    help="raiz do nnUNet_results")
    ap.add_argument("--python", default=sys.executable,
                    help="interpretador do venv com nnU-Net e os trainers")
    ap.add_argument("--continuar", action="store_true",
                    help="retoma folds interrompidos a partir do checkpoint")
    ap.add_argument("--supersede", action="store_true",
                    help="treina por cima de folds ja COMPLETED; invalida o que "
                         "deles dependa")
    ap.add_argument("--parar-na-primeira", action="store_true",
                    help="interrompe a serie no primeiro fold que nao complete")
    ap.add_argument("--plan-only", action="store_true",
                    help="corre todos os portoes e nao treina nada")
    ap.add_argument("--emitir-ordens", metavar="DIR",
                    help="EMISSOR: resolve a cadeia e escreve uma ordem por fold "
                         "em DIR, sem treinar. E o no que manda.")
    ap.add_argument("--ordem", metavar="FICHEIRO",
                    help="EXECUTOR: treina o fold que a ordem descreve. Nao "
                         "resolve ponteiros nenhuns — verifica a assinatura da "
                         "ordem e prova que os dados locais sao os que ela diz.")
    ap.add_argument("--preprocessed", metavar="DIR",
                    help="EXECUTOR: raiz do nnUNet_preprocessed local")
    ap.add_argument("--smoke", type=int, metavar="N",
                    help="ENSAIO: corre N epocas para exercitar o caminho de "
                         "execucao. NAO e um treino: `num_epochs` define o "
                         "escalonador de taxa, logo N epocas nao sao o inicio de "
                         "1000. Escreve em pasta descartavel e NAO publica.")
    args = ap.parse_args()

    if args.smoke:
        _exigir(args.resultados != RESULTADOS_PADRAO,
                "--smoke exige --resultados numa pasta descartavel, para o "
                "ensaio nao poder tocar nos modelos")
    if args.ordem:
        return executar_ordem(args)

    run_id = "%s_%s_%d" % ("s09smoke" if args.smoke else "s09",
                           time.strftime("%Y%m%dT%H%M%S"), os.getpid())
    destino = os.path.join(RUNS, run_id)
    _exigir(not os.path.exists(destino), "destino ja existe")
    log.info("Script 09 — treino | run_id=%s", run_id)
    log.info("  interpretador: %s", args.python)
    log.info("  resultados   : %s", args.resultados)

    try:
        cfg, pcfg = tc.carregar_config_congelada()
        regra, preg = tc.carregar_regra_colapso()
    except tc.FalhaTreino as e:
        raise SystemExit("[FAIL] %s" % e)
    from pulmonary_base import ler_ponteiro
    try:
        p08, _m08, _c08, _d08 = ler_ponteiro(
            os.path.join(RUNS, "CURRENT_NNUNET.json"), "08 (nnU-Net)")
        p07, _m07, _c07, _d07 = ler_ponteiro(
            os.path.join(RUNS, "CURRENT_BD1_PARTITION.json"), "07 (particao)")
    except SystemExit:
        raise
    log.info("  configuracao congelada: %s", pcfg.get("run_id"))
    log.info("  regra de colapso      : %s", regra.get("rule_id"))
    log.info("  dados do 08           : %s", p08.get("run_id"))

    alvos = _alvos(args)
    log.info("  %d folds a treinar em %d arquiteturas",
             len(alvos), len({a["name"] for a, _ in alvos}))

    _amb_local = tc.ambiente(args.python)
    _gpu_local = _amb_local.get("gpu")
    log.info("  placa que vai treinar: %s", _gpu_local)
    trainers, pendentes, sobrepostos_por_arq, compile_por_arq = {}, [], {}, {}
    for arq in {a["name"]: a for a, _ in alvos}.values():
        esperado = tc.esperado_para(cfg, arq["name"])
        ops = esperado.get("operational_overrides", {})

        try:
            _cv, _cs = tc.compile_para_placa(ops, _gpu_local)
            compile_por_arq[arq["name"]] = (_cv, _cs)
        except tc.FalhaTreino as e:
            pendentes.append("%s: %s" % (arq["name"], e))

        if arq["trainer"] not in trainers:
            try:
                trainers[arq["trainer"]] = tc.verificar_trainer_instalado(
                    args.python, arq["trainer"])
            except tc.FalhaTreino as e:
                raise SystemExit("[FAIL] %s" % e)
            log.info("  trainer %-30s OK", arq["trainer"])

        aplicar = {}
        se = ops.get("save_every", {})
        if se.get("runtime_override_needed"):
            aplicar["save_every"] = se["target"]
        if args.smoke:
            aplicar["num_epochs"] = args.smoke
        sobrepostos_por_arq[arq["name"]] = aplicar

    _exigir(not pendentes,
            "configuracao por decidir, treino recusado:\n    %s\n  Medir com o "
            "Script 08b --time --compile na maquina de destino e voltar a "
            "congelar com o 08c." % "\n    ".join(pendentes))

    _pre = args.preprocessed or tc.raiz_preprocessed()[0]
    cascatas = {}
    for arq, fold in alvos:
        if not arq.get("requires_lowres"):
            continue
        folds_arq = sorted({f for a, f in alvos if a["name"] == arq["name"]})
        if arq["name"] in cascatas:
            continue
        try:
            cascatas[arq["name"]] = tc.verificar_cascata(
                arq, args.resultados, folds_arq, _pre)
            log.info("  cascata %-14s estagio anterior verificado", arq["name"])
        except tc.FalhaTreino as e:
            raise SystemExit("[FAIL] %s" % e)

    ja_feitos = []
    for arq, fold in alvos:
        est = os.path.join(tc.pasta_do_fold(args.resultados, arq, fold),
                           "estado_fold.json")
        if os.path.isfile(est):
            e = json.load(open(est))
            if e.get("outcome") == "COMPLETED":
                ja_feitos.append("%s/fold_%d (run %s)"
                                 % (arq["name"], fold, e.get("run_id")))
    if ja_feitos and not args.supersede:
        _exigir(False, "ja existem folds COMPLETED:\n    %s\n  Nao se treina por "
                       "cima. Use --supersede sabendo que invalida o que deles "
                       "depender." % "\n    ".join(ja_feitos))

    base_prov = tc.proveniencia_base(args.python, run_id, {})
    log.info("  ambiente: %s · torch %s · cuda %s · %s",
             base_prov["environment"]["host"], base_prov["environment"]["torch"],
             base_prov["environment"]["cuda"], base_prov["environment"]["gpu"])

    if args.emitir_ordens:
        os.makedirs(args.emitir_ordens, exist_ok=True)
        escritas = []
        for arq, fold in alvos:
            o = tc.emitir_ordem(arq, fold, cfg, pcfg, regra, preg, p08, p07,
                                sobrepostos_por_arq[arq["name"]], run_id)
            c = os.path.join(args.emitir_ordens,
                             "ordem_%s_fold%d.json" % (arq["name"], fold))
            with open(c, "w") as f:
                json.dump(o, f, indent=1, sort_keys=True, default=str)
            escritas.append(c)
            log.info("    %-14s fold %d -> %s", arq["name"], fold,
                     os.path.basename(c))
        log.info("  %d ordens emitidas em %s", len(escritas), args.emitir_ordens)
        log.info("  o executor corre: 09_train_nnunet.py --ordem <ficheiro> "
                 "--preprocessed <dir> --resultados <dir>")
        _publicar_emissao(run_id, base_prov, cascatas, escritas, args)
        return 0

    if args.plan_only:
        log.info("  PLAN_STATUS = PASS · nenhum treino executado")
        for arq, fold in alvos:
            ops = tc.esperado_para(cfg, arq["name"])["operational_overrides"]
            log.info("    %-14s fold %d · compile=%s (%s) · save_every=%s · %s",
                     arq["name"], fold, compile_por_arq[arq["name"]][0],
                     compile_por_arq[arq["name"]][1], ops["save_every"]["target"],
                     sobrepostos_por_arq[arq["name"]] or "sem sobreposto")
        return 0

    os.makedirs(destino, exist_ok=False)
    os.makedirs(args.resultados, exist_ok=True)

    registos, contagem = [], {}
    for arq, fold in alvos:
        esperado = tc.esperado_para(cfg, arq["name"])
        ops = esperado["operational_overrides"]
        env_compile = str(compile_por_arq[arq["name"]][0])
        os.environ["nnUNet_compile"] = env_compile
        log.info("── %s fold %d · compile=%s · %s ──",
                 arq["name"], fold, env_compile,
                 sobrepostos_por_arq[arq["name"]] or "sem sobreposto")

        def registar(evento, detalhe, _a=arq, _f=fold):
            log.warning("  [%s] %s/fold_%d · %s", evento, _a["name"], _f, detalhe)

        try:
            r = tc.executar_fold(
                arq, fold, interpretador=args.python,
                resultados=args.resultados, destino_run=destino, run_id=run_id,
                sobrepostos=sobrepostos_por_arq[arq["name"]],
                esperados=esperado, regra=regra, continuar=args.continuar,
                ao_registar=registar)
        except tc.FalhaTreino as e:
            r = {"architecture": arq["name"], "fold": fold,
                 "outcome": "REFUSED", "reason": str(e)}
            log.error("  RECUSADO: %s", e)
        r["compile"] = ops["nnUNet_compile"]
        r["cascade_provenance"] = cascatas.get(arq["name"])
        registos.append(r)
        contagem[r["outcome"]] = contagem.get(r["outcome"], 0) + 1
        log.info("  %s/fold_%d -> %s em %.1f h", arq["name"], fold, r["outcome"],
                 (r.get("duration_s") or 0) / 3600.0)

        with open(os.path.join(destino, "registos_parciais.json"), "w") as f:
            json.dump(registos, f, indent=1, sort_keys=True, default=str)

        if r["outcome"] != "COMPLETED" and args.parar_na_primeira:
            log.error("  a interromper a serie (--parar-na-primeira)")
            break

    if args.smoke:
        log.warning("  ENSAIO (--smoke %d): nada publicado, nenhum ponteiro "
                    "escrito. Os %d registos ficam em %s",
                    args.smoke, len(registos), destino)
        with open(os.path.join(destino, "folds_ensaio.json"), "w") as f:
            json.dump(registos, f, indent=1, sort_keys=True, default=str)
        for r in registos:
            log.info("    %-14s fold %s -> %-20s debug.json=%s monitor=%s",
                     r.get("architecture"), r.get("fold"), r.get("outcome"),
                     (r.get("debug_json_check") or {}).get("status"),
                     (r.get("monitor") or {}).get("last_epoch"))
        return 0

    divergentes = [r for r in registos
                   if (r.get("debug_json_check") or {}).get("status") == "MISMATCH"]
    portoes = {
        "FOLDS_REQUESTED": len(alvos),
        "FOLDS_RUN": len(registos),
        "OUTCOMES": contagem,
        "DEBUG_JSON_MISMATCHES": len(divergentes),
        "COLLAPSE_RULE_ID": regra.get("rule_id"),
        "FROZEN_CONFIG_RUN": pcfg.get("run_id"),
        "SCRIPT09_OUTPUT_STATUS":
            "PASS" if (contagem.get("COMPLETED", 0) == len(alvos)
                       and not divergentes) else "PASS_WITH_FLAG",
    }
    for k, v in portoes.items():
        log.info("  %-28s %s", k, v)

    fr = os.path.join(destino, "folds.json")
    with open(fr, "w") as f:
        json.dump(registos, f, indent=1, sort_keys=True, default=str)

    m = dict(base_prov)
    m.update({"manifest_id": "training_%s" % run_id, "run_id": run_id,
              "script": "09_TRAIN_NNUNET", "script_version": "2026-08-11.train.v1",
              "generated": time.strftime("%Y-%m-%dT%H:%M:%S BRT"),
              "producer_sha256": sha256_file(os.path.abspath(__file__)),
              "results_root": args.resultados,
              "trainers_verified": trainers,
              "cascade_provenance": cascatas,
              "gates": portoes, "folds_sha256": sha256_file(fr),
              "SCRIPT09_OUTPUT_STATUS": portoes["SCRIPT09_OUTPUT_STATUS"]})
    corpo = hashlib.sha256(json.dumps(
        {k: v for k, v in m.items() if k != "manifest_body_sha256"},
        indent=1, sort_keys=True, default=str).encode()).hexdigest()
    m["manifest_body_sha256"] = corpo
    pm = os.path.join(destino, "manifest.json")
    with open(pm, "w") as f:
        json.dump(m, f, indent=1, sort_keys=True, default=str)

    escrever_ponteiro(PONTEIRO, pm, run_id=run_id, script="09",
                      script_version=m["script_version"],
                      results_root=args.resultados,
                      folds_json=os.path.relpath(fr, BASE),
                      folds_sha256=m["folds_sha256"],
                      outcomes=contagem,
                      SCRIPT09_OUTPUT_STATUS=portoes["SCRIPT09_OUTPUT_STATUS"])
    log.info("  publicado em %s", destino)
    return 0 if portoes["SCRIPT09_OUTPUT_STATUS"] == "PASS" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
