#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    treino_core.py

File type:
    Module

Purpose:
    Shared training core: architecture registry, nnU-Net invocation, collapse monitor, provenance and per-fold state.

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
    - Python standard library only.

Execution:
    Imported by the pipeline scripts in scripts/

Portability notes:
    All paths are resolved relative to the project base directory given by the PULMONARY_BASE environment variable. No machine-specific path is required.

Reproducibility notes:
    Environment pinned in requirements.txt and environment/. Seeds and frozen parameters are recorded in the run manifests under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) are not included and must be obtained from their original repositories.
"""

import os
import re
import sys
import json
import time
import signal
import hashlib
import logging
import platform
import subprocess

from pulmonary_base import BASE, sha256_file, ler_ponteiro

RUNS = os.path.join(BASE, "runs")
PONTEIRO_REGRA = os.path.join(RUNS, "CURRENT_COLLAPSE_RULE.json")
PONTEIRO_CONFIG = os.path.join(RUNS, "CURRENT_TRAINING_CONFIG.json")
PONTEIRO_08 = os.path.join(RUNS, "CURRENT_NNUNET.json")
PONTEIRO_07 = os.path.join(RUNS, "CURRENT_BD1_PARTITION.json")

log = logging.getLogger(__name__)

ARQUITETURAS = [
    {"name": "2D",           "config": "2d",
     "plans_id": "nnUNetPlans",             "trainer": "nnUNetTrainer"},
    {"name": "2D_REM",       "config": "2d",
     "plans_id": "nnUNetResEncUNetMPlans",  "trainer": "nnUNetTrainer"},
    {"name": "2D_REL",       "config": "2d",
     "plans_id": "nnUNetResEncUNetLPlans",  "trainer": "nnUNetTrainerCheckEvery10"},
    {"name": "3D_F",         "config": "3d_fullres",
     "plans_id": "nnUNetPlans",             "trainer": "nnUNetTrainer"},
    {"name": "3D_FREM",      "config": "3d_fullres",
     "plans_id": "nnUNetResEncUNetMPlans",  "trainer": "nnUNetTrainer"},
    {"name": "F100_lowres",  "config": "3d_lowres",
     "plans_id": "nnUNetPlans",             "trainer": "nnUNetTrainer_Foreground100"},
    {"name": "F100_ResEncM", "config": "3d_lowres",
     "plans_id": "nnUNetResEncUNetMPlans",  "trainer": "nnUNetTrainer_Foreground100"},
    {"name": "3D_CF",        "config": "3d_cascade_fullres",
     "plans_id": "nnUNetPlans",             "trainer": "nnUNetTrainer_Foreground100",
     "requires_lowres": "F100_lowres"},
    {"name": "3D_CFREM",     "config": "3d_cascade_fullres",
     "plans_id": "nnUNetResEncUNetMPlans",  "trainer": "nnUNetTrainer_Foreground100",
     "requires_lowres": "F100_ResEncM"},
    {"name": "F100_fullres", "config": "3d_fullres",
     "plans_id": "nnUNetPlans",             "trainer": "nnUNetTrainer_Foreground100"},
]

POR_NOME = {a["name"]: a for a in ARQUITETURAS}
DATASET = "Dataset001_PulmonaryNodules"
DATASET_ID = "001"


def pasta_do_fold(resultados, arq, fold):
    """O caminho que o nnU-Net usa, montado da mesma forma que ele o monta."""
    return os.path.join(resultados, DATASET,
                        "%s__%s__%s" % (arq["trainer"], arq["plans_id"], arq["config"]),
                        "fold_%d" % fold)


class FalhaTreino(Exception):
    """Erro que impede o fold de correr ou de ser aceite."""


def _exigir(cond, msg):
    if not cond:
        raise FalhaTreino(msg)


def carregar_regra_colapso():
    """A regra do Script 08a. Vem do ponteiro; não se escreve aqui."""
    p, m, caminho, _ = ler_ponteiro(PONTEIRO_REGRA, "08a (regra de colapso)")
    rel = p.get("collapse_rule")
    alvo = rel if os.path.isabs(rel or "") else os.path.join(BASE, rel or "")
    _exigir(rel and os.path.isfile(alvo), "regra de colapso ausente: %s" % alvo)
    _exigir(sha256_file(alvo) == p.get("collapse_rule_sha256"),
            "regra de colapso nao confere com o hash do ponteiro")
    return json.load(open(alvo)), p


def carregar_config_congelada():
    """A configuração do Script 08c: valores esperados de cada parâmetro."""
    p, m, caminho, _ = ler_ponteiro(PONTEIRO_CONFIG, "08c (configuracao de treino)")
    rel = p.get("training_config")
    alvo = rel if os.path.isabs(rel or "") else os.path.join(BASE, rel or "")
    _exigir(rel and os.path.isfile(alvo), "configuracao congelada ausente: %s" % alvo)
    _exigir(sha256_file(alvo) == p.get("training_config_sha256"),
            "configuracao congelada nao confere com o hash do ponteiro")
    return json.load(open(alvo)), p


def raiz_preprocessed():
    """
    A pasta `nnUNet_preprocessed` da execucao vigente do Script 08, resolvida
    pelo PONTEIRO — nunca pelo ambiente herdado. O nnU-Net le esta variavel no
    momento do IMPORT, para variaveis de modulo: se o subprocesso arrancar sem
    ela, morre em `TypeError: expected str ... not NoneType` antes de sequer
    construir o trainer, e nao ha treino nenhum.
    """
    p, m, _c, _d = ler_ponteiro(PONTEIRO_08, "08 (nnU-Net)")
    raiz = p.get("nnUNet_preprocessed")
    _exigir(raiz, "ponteiro do 08 nao declara nnUNet_preprocessed")
    if not os.path.isabs(raiz):
        raiz = os.path.join(BASE, raiz)

    local = os.environ.get("PULMONARY_PREPROCESSED")
    relocalizado = False
    if local:
        _exigir(os.path.isdir(os.path.join(local, DATASET)),
                "PULMONARY_PREPROCESSED=%s nao contem %s" % (local, DATASET))
        declarados = (m.get("preprocessed_json_sha256")
                      or (m.get("gates") or {}).get("preprocessed_json_sha256")
                      or {})
        _exigir(declarados,
                "o manifesto do 08 nao declara hashes dos JSON preprocessados; "
                "sem eles nao se pode provar que a copia e a mesma coisa")
        divergentes, ausentes = [], []
        for nome, esperado in sorted(declarados.items()):
            alvo = os.path.join(local, DATASET, nome)
            if not os.path.isfile(alvo):
                ausentes.append(nome)
            elif sha256_file(alvo) != esperado:
                divergentes.append(nome)
        _exigir(not ausentes and not divergentes,
                "a copia em %s NAO e a mesma do Script 08: ausentes=%s "
                "divergentes=%s" % (local, ausentes, divergentes))
        raiz, relocalizado = local, True

    _exigir(os.path.isdir(raiz), "nnUNet_preprocessed ausente: %s" % raiz)
    _exigir(os.path.isdir(os.path.join(raiz, DATASET)),
            "%s ausente em %s" % (DATASET, raiz))
    if relocalizado:
        log.info("  nnUNet_preprocessed relocalizado para %s "
                 "(JSON conferidos contra o manifesto do 08)", raiz)
    return raiz, p.get("nnUNet_raw")


def compile_para_placa(ops, nome_gpu):
    """
    O valor de `nnUNet_compile` para a placa que vai treinar.

    Medido em 13/08/2026: compilar atrasa o `ResEncM` 3d em 36% na 5070 e 22% na
    5060 Ti, e nao o atrasa na 5090. Nao e ruido — medicoes equivalentes na mesma
    placa concordam a 0,1%. Logo o parametro e do HARDWARE, nao do estudo, e
    congelar um valor unico imporia a escolha errada a uma das maquinas.
    """
    c = (ops or {}).get("nnUNet_compile") or {}
    if c.get("value") is not None:
        return int(c["value"]), c.get("status", "GLOBAL")
    porgpu = c.get("value_per_gpu") or {}
    if not porgpu:
        raise FalhaTreino(
            "nnUNet_compile sem valor e sem tabela por placa (status=%r). Medir "
            "com o Script 08b --time --compile nesta maquina." % c.get("status"))
    for g, v in porgpu.items():
        if nome_gpu and (g in nome_gpu or nome_gpu in g):
            return int(v), "PER_GPU_RESOLVED"
    raise FalhaTreino(
        "nnUNet_compile medido em %s, mas esta maquina tem %r. Correr o Script "
        "08b --time --compile aqui antes de treinar." % (sorted(porgpu), nome_gpu))


def esperado_para(config_congelada, nome_arq):
    """Os valores que o 08c congelou para esta arquitetura."""
    for a in config_congelada["architectures"]:
        if a["architecture"] == nome_arq:
            return a
    raise FalhaTreino("arquitetura %s ausente da configuracao congelada" % nome_arq)


RE_EPOCA = re.compile(r":\s*Epoch\s+(\d+)\s*$")
RE_DICE = re.compile(r":\s*Pseudo dice\s*\[(.*)\]\s*$")
RE_BEST = re.compile(r"New best EMA pseudo Dice:\s*([-\d.eE+naif]+)")
RE_ENVOLVIDO = re.compile(r"float\d*\(([^)]*)\)")
RE_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan|inf|-inf")
TOL_EMA = 1e-3
BURNIN_RETOMA = 100


def _numeros(interior):
    envolvidos = RE_ENVOLVIDO.findall(interior)
    brutos = envolvidos if envolvidos else RE_NUM.findall(interior)
    out = []
    for x in brutos:
        try:
            v = float(x)
        except ValueError:
            continue
        out.append(v)
    return out


class MonitorColapso:
    """
    Consome linhas do log e decide abortar. Só olha para o passado: o veredicto
    de cada época usa exclusivamente o que já aconteceu até ela.
    """

    def __init__(self, regra, ao_registar=None):
        self.regra = regra
        self.primeira_epoca = None
        self.epocas_vistas = 0
        self.ja_verificou_aprendizagem = False
        self.colapso = regra.get("collapse") or {}
        self.sem_aprender = regra.get("no_learning") or {}
        self.rule_id = regra.get("rule_id")
        self.ema = None
        self.ema_max = None
        self.epoca = None
        self.seguidas = 0
        self.armado = False
        self.desarmado_por = None
        self.desvio_max = 0.0
        self.n_verificacoes = 0
        self.disparo = None
        self._ao_registar = ao_registar or (lambda *_: None)

    def linha(self, texto):
        m = RE_EPOCA.search(texto)
        if m:
            self.epoca = int(m.group(1))
            return None
        m = RE_DICE.search(texto)
        if m and self.epoca is not None:
            vals = [v for v in _numeros(m.group(1)) if v == v]
            if not vals:
                return self._disparar("NON_FINITE", "pseudo dice nao finito")
            return self._epoca(sum(vals) / len(vals))
        m = RE_BEST.search(texto)
        if m and self.ema_max is not None:
            impresso = float(m.group(1))
            self.desvio_max = max(self.desvio_max, abs(self.ema_max - impresso))
            self.n_verificacoes += 1
            if self.desvio_max > TOL_EMA and not self.desarmado_por:
                self.desarmado_por = (
                    "EMA reconstruido diverge do impresso em %.4f (>%.4f)"
                    % (self.desvio_max, TOL_EMA))
                self._ao_registar("MONITOR_DISARMED", self.desarmado_por)
        return None

    def _epoca(self, dice):
        if self.primeira_epoca is None:
            self.primeira_epoca = self.epoca
        self.epocas_vistas += 1
        if self.ema is None:
            self.ema = dice
        else:
            self.ema = 0.9 * self.ema + 0.1 * dice
        self.ema_max = self.ema if self.ema_max is None else max(self.ema_max, self.ema)
        if self.desarmado_por:
            return None
        if self.primeira_epoca != 0 and self.epocas_vistas <= BURNIN_RETOMA:
            return None

        na = self.sem_aprender
        if (na and not self.ja_verificou_aprendizagem
                and self.epoca >= na.get("check_at_epoch", 10 ** 9)):
            self.ja_verificou_aprendizagem = True
            if self.ema_max < na["arm"]:
                return self._disparar(
                    "NO_LEARNING",
                    "EMA maximo %.4f abaixo de %.4f a epoca %d"
                    % (self.ema_max, na["arm"], self.epoca))

        c = self.colapso
        if not c:
            return None
        if self.epoca < c.get("min_epoch", 0) or self.ema_max < c["arm"]:
            self.seguidas = 0
            return None
        self.armado = True
        if self.ema <= c["frac_of_max"] * self.ema_max:
            self.seguidas += 1
        else:
            self.seguidas = 0
        if self.seguidas >= c["consecutive_epochs"]:
            return self._disparar(
                "COLLAPSE",
                "EMA %.4f abaixo de %.0f%% do maximo %.4f durante %d epocas"
                % (self.ema, 100 * c["frac_of_max"], self.ema_max, self.seguidas))
        return None

    def _disparar(self, qual, porque):
        self.disparo = {
            "event": "TRAINING_COLLAPSE", "which_rule": qual, "reason": porque,
            "epoch": self.epoca, "ema_at_abort": self.ema,
            "best_ema_before": self.ema_max, "rule_id": self.rule_id,
            "not_for": "seleccao de checkpoint; esta regra so aborta",
        }
        self._ao_registar("TRAINING_COLLAPSE", json.dumps(self.disparo, sort_keys=True))
        return self.disparo

    def resumo(self):
        return {"rule_id": self.rule_id, "first_epoch": self.primeira_epoca,
                "epochs_seen": self.epocas_vistas,
                "resume_burnin_applied": bool(self.primeira_epoca),
                "last_epoch": self.epoca,
                "ema_final": self.ema, "ema_max": self.ema_max,
                "armed": self.armado, "disarmed_reason": self.desarmado_por,
                "ema_reconstruction_max_dev": self.desvio_max,
                "ema_reconstruction_checks": self.n_verificacoes,
                "trigger": self.disparo}


RUNNER = '''"""
Runner gerado pelo treino_core. NAO EDITAR A MAO.

Reproduz `nnunetv2.run.run_training.run_training` e aplica, entre a construcao do
trainer e o arranque, os sobrepostos DECLARADOS na configuracao congelada. Existe
porque `run_training` nao expoe esses parametros, e mexer-lhes sem deixar rasto
foi o que produziu, em julho, cinco valores diferentes de `save_every` na mesma
configuracao sem que nada o registasse.

Este ficheiro e hasheado e guardado com a proveniencia do fold.
"""
import inspect
import json
import os
import sys

import torch
from nnunetv2.run.run_training import get_trainer_from_args, maybe_load_checkpoint
from torch.backends import cudnn

SOBREPOSTOS = json.loads(os.environ["PULMONARY_OVERRIDES"])


def main():
    dataset, config, fold, trainer, plans, continuar = sys.argv[1:7]
    # ADAPTACAO DE API. Ate ao 2.5.2 o `get_trainer_from_args` recebia
    # `use_compressed` e passava `unpack_dataset=not use_compressed` ao trainer.
    # No 2.8.1 esse parametro desapareceu — passa-lo da TypeError e o treino
    # morre antes da primeira epoca. Pergunta-se a assinatura em vez de ramificar
    # por numero de versao: o mesmo ficheiro serve as duas pilhas, e e obrigatorio
    # que sirva, porque o 17G corre as duas lado a lado.
    _kw = {"device": torch.device("cuda")}
    if "use_compressed" in inspect.signature(get_trainer_from_args).parameters:
        _kw["use_compressed"] = False
    t = get_trainer_from_args(dataset, config, int(fold), trainer, plans, **_kw)
    for chave, valor in sorted(SOBREPOSTOS.items()):
        if not hasattr(t, chave):
            raise SystemExit("[FAIL] sobreposto %r nao existe no trainer %s"
                             % (chave, type(t).__name__))
        anterior = getattr(t, chave)
        setattr(t, chave, valor)
        t.print_to_log_file("DECLARED_OVERRIDE %s: %r -> %r" % (chave, anterior, valor))
    maybe_load_checkpoint(t, continuar == "1", False, None)
    cudnn.deterministic = False
    cudnn.benchmark = True
    t.run_training()
    t.load_checkpoint(os.path.join(t.output_folder, "checkpoint_best.pth"))
    t.perform_actual_validation(False)


if __name__ == "__main__":
    main()
'''


def escrever_runner(destino):
    caminho = os.path.join(destino, "runner_treino.py")
    with open(caminho, "w") as f:
        f.write(RUNNER)
    return caminho, sha256_file(caminho)


def comando(interpretador, runner, arq, fold, continuar):
    """A linha de comando exacta, para correr e para registar."""
    return [interpretador, runner, DATASET_ID, arq["config"], str(fold),
            arq["trainer"], arq["plans_id"], "1" if continuar else "0"]


def verificar_trainer_instalado(interpretador, nome_trainer):
    """
    O trainer tem de EXISTIR e INSTANCIAR neste interpretador. Existir não chega:
    as tres classes do estudo importavam e nao instanciavam ate 11/08/2026.
    """
    codigo = (
        "import os, json, sys, nnunetv2\n"
        "from nnunetv2.utilities.find_class_by_name import recursive_find_python_class\n"
        "raiz = os.path.join(os.path.dirname(nnunetv2.__file__), 'training', 'nnUNetTrainer')\n"
        "c = recursive_find_python_class(raiz, %r, 'nnunetv2.training.nnUNetTrainer')\n"
        "if c is None: print(json.dumps({'ok': False, 'erro': 'nao encontrado'})); sys.exit(0)\n"
        "import inspect\n"
        "f = inspect.getsourcefile(c)\n"
        "p = list(inspect.signature(c.__init__).parameters)\n"
        "print(json.dumps({'ok': True, 'file': f, 'params': p,\n"
        "                  'star_args': any(x in p for x in ('args', 'kwargs'))}))\n"
        % nome_trainer)
    r = subprocess.run([interpretador, "-c", codigo], capture_output=True, text=True)
    _exigir(r.returncode == 0, "verificacao do trainer %s falhou: %s"
            % (nome_trainer, r.stderr.strip()[:300]))
    d = json.loads(r.stdout.strip().splitlines()[-1])
    _exigir(d.get("ok"), "trainer %s ausente em %s — ambiente por provisionar "
                         "(Script 00 --provision)" % (nome_trainer, interpretador))
    _exigir(not d.get("star_args"),
            "trainer %s declara *args/**kwargs: nao instancia em nnU-Net 2.5.2"
            % nome_trainer)
    return {"source_file": d.get("file"),
            "source_sha256": sha256_file(d["file"]) if d.get("file")
            and os.path.isfile(d["file"]) else None,
            "init_params": d.get("params")}


def verificar_cascata(arq, resultados, folds, preprocessed):
    """
    Uma cascata só arranca se o estágio anterior estiver completo e verificável.
    Devolve a proveniência do anterior, para o registo do fold.

    Três coisas que o nnU-Net faz e que esta verificação tem de espelhar, medidas
    na fonte instalada (`nnUNetTrainer.py`, 2.5.2) e não presumidas:

    1. As predições do estágio seguinte são escritas em
       `output_folder_base/predicted_next_stage/<config da cascata>` (linha 1321)
       — na pasta **pai** dos folds, não dentro de cada fold. E é de lá que a
       cascata as lê (linha 143), montando o caminho com o **seu próprio** nome
       de trainer. Por isso `requires_lowres` tem de apontar para o lowres que
       partilha trainer e planos com a cascata.
    2. Essa pasta é **partilhada pelos cinco folds**: cada fold do lowres só
       exporta os seus casos de validação. Só a união dos cinco cobre o universo.
    3. Logo, a cascata precisa dos **cinco** folds do lowres completos, mesmo que
       só se vá treinar um fold dela: o conjunto de treino de qualquer fold da
       cascata inclui casos que foram validação de outros folds do lowres.
    """
    nome_anterior = arq.get("requires_lowres")
    if not nome_anterior:
        return None
    anterior = POR_NOME.get(nome_anterior)
    _exigir(anterior, "estagio anterior %r nao esta no registo" % nome_anterior)

    splits = os.path.join(preprocessed, DATASET, "splits_final.json")
    _exigir(os.path.isfile(splits),
            "cascata %s recusada: falta %s — sem ele nao se sabe que casos a "
            "entrada tem de cobrir" % (arq["name"], splits))
    particoes = json.load(open(splits))
    folds_exigidos = list(range(len(particoes)))
    universo = sorted({c for p in particoes for c in p["train"]}
                      | {c for p in particoes for c in p["val"]})

    prov = {"previous_stage": nome_anterior,
            "previous_stage_folds_required": folds_exigidos,
            "folds_requested_for_cascade": sorted(folds),
            "folds": {}}
    for fold in folds_exigidos:
        pasta = pasta_do_fold(resultados, anterior, fold)
        estado = os.path.join(pasta, "estado_fold.json")
        _exigir(os.path.isfile(estado),
                "cascata %s recusada: fold %d do anterior nao tem estado_fold.json. "
                "Ausencia de registo nao e prova de sucesso." % (arq["name"], fold))
        e = json.load(open(estado))
        status = e.get("outcome")
        gatilho = e.get("monitor_trigger") or {}

        if status == "COMPLETED":
            fonte = "checkpoint_final.pth"
            ckpt = os.path.join(pasta, fonte)
            _exigir(os.path.isfile(ckpt),
                    "cascata %s recusada: fold %d declara COMPLETED mas falta %s. "
                    "Estado e artefacto tem de concordar." % (arq["name"], fold, ckpt))

        elif status == "ABORTED_BY_MONITOR":
            _exigir(gatilho,
                    "cascata %s recusada: fold %d diz ABORTED_BY_MONITOR sem "
                    "monitor_trigger. Um aborto sem decisao registada nao e um "
                    "resultado auditavel." % (arq["name"], fold))
            _exigir(e.get("resumable") is False,
                    "cascata %s recusada: fold %d abortado mas com resumable=%r. "
                    "Um corte do monitor e definitivo; resumable verdadeiro indica "
                    "registo incoerente." % (arq["name"], fold, e.get("resumable")))
            fonte = "checkpoint_latest.pth"
            ckpt = os.path.join(pasta, fonte)
            if not os.path.isfile(ckpt):
                alt = os.path.join(pasta, "checkpoint_abort.pth")
                _exigir(os.path.isfile(alt),
                        "cascata %s recusada: fold %d abortado e sem checkpoint "
                        "terminal preservado (nem %s nem %s). Sem ele nao ha "
                        "resultado deste fold que se possa materializar."
                        % (arq["name"], fold, ckpt, alt))
                fonte, ckpt = "checkpoint_abort.pth", alt

        else:
            _exigir(False,
                    "cascata %s recusada: fold %d do anterior terminou como %r. "
                    "So COMPLETED e ABORTED_BY_MONITOR sao estados terminais "
                    "reconhecidos." % (arq["name"], fold, status))

        sha = sha256_file(ckpt)
        prov["folds"][str(fold)] = {
            "previous_stage_checkpoint": ckpt,
            "previous_stage_checkpoint_sha256": sha,
            "previous_stage_run_id": e.get("run_id"),
            "previous_stage_outcome": status,
            "terminal_checkpoint_source": fonte,
            "terminal_checkpoint_sha256": sha,
            "terminal_checkpoint_epoch": gatilho.get("epoch") if gatilho else None,
            "training_status": status,
            "monitor_decision": gatilho.get("which_rule") if gatilho else None,
            "monitor_threshold": gatilho.get("arm") if gatilho else None,
            "monitor_epoch": gatilho.get("epoch") if gatilho else None,
        }

    n_abort = sum(1 for v in prov["folds"].values()
                  if v["training_status"] == "ABORTED_BY_MONITOR")
    prov["previous_stage_functional_state"] = (
        "FUNCTIONAL" if n_abort == 0 else
        "COMPLETE_DEGENERATE" if n_abort == len(folds_exigidos) else
        "PARTIALLY_DEGENERATE")
    prov["previous_stage_aborted_folds"] = sorted(
        int(k) for k, v in prov["folds"].items()
        if v["training_status"] == "ABORTED_BY_MONITOR")

    base = os.path.dirname(pasta_do_fold(resultados, anterior, folds_exigidos[0]))
    pnext = os.path.join(base, "predicted_next_stage", arq["config"])
    _exigir(os.path.isdir(pnext),
            "cascata %s recusada: falta %s" % (arq["name"], pnext))
    EXT_PNEXT = (".b2nd", ".npz")
    ficheiros = [f for f in os.listdir(pnext) if f.endswith(EXT_PNEXT)]
    _exigir(ficheiros,
            "cascata %s recusada: %s existe mas nao contem ficheiros %s. "
            "Se a pasta tem outra extensao, e a versao do nnU-Net que mudou: "
            "confirmar antes de assumir que o estagio anterior nao correu."
            % (arq["name"], pnext, "/".join(EXT_PNEXT)))
    casos = sorted({os.path.splitext(f)[0] for f in ficheiros})
    em_falta = sorted(set(universo) - set(casos))
    _exigir(not em_falta,
            "cascata %s recusada: a entrada nao cobre %d dos %d casos (ex.: %s). "
            "Cada fold do lowres so exporta a sua validacao; faltar um caso "
            "significa que um fold nao chegou ao fim."
            % (arq["name"], len(em_falta), len(universo), ", ".join(em_falta[:5])))
    prov["predicted_next_stage_dir"] = pnext
    prov["predicted_next_stage_ext"] = sorted({os.path.splitext(f)[1] for f in ficheiros})
    prov["predicted_next_stage_n_files"] = len(ficheiros)
    prov["predicted_next_stage_n_cases"] = len(casos)
    prov["predicted_next_stage_n_expected"] = len(universo)
    prov["predicted_next_stage_extra_cases"] = sorted(set(casos) - set(universo))
    prov["predicted_next_stage_listing_sha256"] = hashlib.sha256(
        json.dumps(casos, sort_keys=True).encode()).hexdigest()
    return prov


def ambiente(interpretador):
    codigo = ("import json, sys, torch, importlib.metadata as md\n"
              "print(json.dumps({'python': sys.version.split()[0],\n"
              " 'interpreter': sys.executable,\n"
              " 'torch': torch.__version__, 'cuda': torch.version.cuda,\n"
              " 'cudnn': torch.backends.cudnn.version(),\n"
              " 'nnunetv2': md.version('nnunetv2'),\n"
              " 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,\n"
              " 'gpu_total_gb': round(torch.cuda.get_device_properties(0).total_memory/2**30, 2)\n"
              "   if torch.cuda.is_available() else None}))")
    r = subprocess.run([interpretador, "-c", codigo], capture_output=True, text=True)
    _exigir(r.returncode == 0, "nao consegui interrogar o ambiente: %s"
            % r.stderr.strip()[:300])
    d = json.loads(r.stdout.strip().splitlines()[-1])
    d["host"] = platform.node()
    d["platform"] = platform.platform()
    return d


def escrever_estado(pasta, **campos):
    """
    Estado do fold, para o auto-resume. `outcome` diz se terminou POR DECISÃO —
    e nesse caso `resumable` é falso: relançar desfaria a decisão em silêncio.
    """
    caminho = os.path.join(pasta, "estado_fold.json")
    campos.setdefault("updated_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    tmp = caminho + ".tmp"
    with open(tmp, "w") as f:
        json.dump(campos, f, indent=1, sort_keys=True)
    os.replace(tmp, caminho)
    return caminho


def verificar_debug_json(pasta, esperados, sobrepostos, continuar=False):
    """
    Confere o `debug.json` que o nnU-Net escreve contra os valores congelados.
    É o único caminho que apanha uma alteração feita DEPOIS da construção — que
    é exactamente o que aconteceu em julho e que nada denunciou.

    `current_epoch` numa RETOMA
        A configuração congelada descreve um arranque de raiz, logo declara
        `current_epoch: 0`. Numa retoma o nnU-Net escreve a época do checkpoint,
        e comparar contra 0 reprovava o fold por estar a fazer exactamente o que
        lhe foi pedido. Em 2026-08-14 isto matou o fold 0 da `F100_lowres` à
        época 700, dois minutos depois de retomar.

        Com `continuar=True` o campo sai da comparação e passa a ser REGISTADO
        como `resume_epoch`. Não se ignora em silêncio: um valor 0 numa retoma
        seria sinal de que o checkpoint não foi lido, e por isso reprova na
        mesma. O que se deixa de exigir é a igualdade a zero, não a coerência.
    """
    caminho = os.path.join(pasta, "debug.json")
    if not os.path.isfile(caminho):
        return {"status": "ABSENT", "path": caminho}
    d = json.load(open(caminho))
    alvo = dict(esperados.get("trainer_effective", {}))
    alvo.update(sobrepostos)
    isento, retomada = {}, None
    if continuar and "current_epoch" in alvo:
        alvo.pop("current_epoch")
        try:
            retomada = int(float(d.get("current_epoch", 0)))
        except (TypeError, ValueError):
            retomada = None
        isento["current_epoch"] = {
            "motivo": "RETOMA: a configuracao congelada descreve arranque de raiz",
            "observado": d.get("current_epoch"),
        }
    divergencias = {}
    for k, v in sorted(alvo.items()):
        if k not in d:
            continue
        obtido = d[k]
        try:
            igual = (float(obtido) == float(v))
        except (TypeError, ValueError):
            igual = (str(obtido) == str(v))
        if not igual:
            divergencias[k] = {"esperado": v, "observado": obtido}
    if continuar and not retomada:
        divergencias["current_epoch"] = {
            "esperado": "> 0 numa retoma", "observado": d.get("current_epoch"),
            "significado": "o checkpoint nao foi lido; isto seria recomecar do zero"}
    r = {"status": "MISMATCH" if divergencias else "MATCH",
         "path": caminho, "divergences": divergencias,
         "checked": sorted(set(alvo) & set(d))}
    if isento:
        r["exempt_on_resume"] = isento
        r["resume_epoch"] = retomada
    return r


def _ultimo_log(pasta):
    """O log que o nnU-Net acabou de abrir para esta corrida."""
    if not os.path.isdir(pasta):
        return None
    cs = [os.path.join(pasta, f) for f in os.listdir(pasta)
          if f.startswith("training_log_") and f.endswith(".txt")]
    return max(cs, key=os.path.getmtime) if cs else None


def executar_fold(arq, fold, *, interpretador, resultados, destino_run, run_id,
                  sobrepostos, esperados, regra, continuar=False,
                  intervalo=5.0, ao_registar=None, preprocessed=None):
    """
    Corre UM fold e devolve o registo completo de proveniência.

    O treino corre em subprocesso; este processo acompanha o log e aplica o
    monitor. Se a regra disparar, o treino é terminado — primeiro com SIGTERM,
    para o nnU-Net poder fechar o que tem aberto, e só depois com SIGKILL.
    """
    ao_registar = ao_registar or (lambda *_: None)
    pasta = pasta_do_fold(resultados, arq, fold)
    os.makedirs(pasta, exist_ok=True)

    runner, runner_sha = escrever_runner(destino_run)
    cmd = comando(interpretador, runner, arq, fold, continuar)
    env = dict(os.environ)
    env["PULMONARY_OVERRIDES"] = json.dumps(sobrepostos, sort_keys=True)
    if preprocessed:
        _exigir(os.path.isdir(os.path.join(preprocessed, DATASET)),
                "%s ausente em %s" % (DATASET, preprocessed))
        pre, cru = preprocessed, None
    else:
        pre, cru = raiz_preprocessed()
    env["nnUNet_preprocessed"] = pre
    env["nnUNet_results"] = resultados
    env["nnUNet_raw"] = cru if cru else os.path.join(resultados, "_raw_nao_usado")
    os.makedirs(env["nnUNet_raw"], exist_ok=True)

    escrever_estado(pasta, run_id=run_id, architecture=arq["name"], fold=fold,
                    outcome="RUNNING", resumable=True,
                    started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    command=cmd, pid=None)

    inicio = time.time()
    monitor = MonitorColapso(regra, ao_registar=ao_registar)
    saida_bruta = os.path.join(destino_run, "%s_fold%d.stdout" % (arq["name"], fold))
    bruto = open(saida_bruta, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=bruto,
                            stderr=subprocess.STDOUT, text=True,
                            start_new_session=True)
    escrever_estado(pasta, run_id=run_id, architecture=arq["name"], fold=fold,
                    outcome="RUNNING", resumable=True,
                    started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    command=cmd, pid=proc.pid)

    caminho_log, posicao = None, 0
    abortado = None
    try:
        while True:
            if proc.poll() is not None and caminho_log is None:
                break
            if caminho_log is None:
                caminho_log = _ultimo_log(pasta)
                if caminho_log and os.path.getmtime(caminho_log) < inicio - 5:
                    caminho_log = None
            if caminho_log:
                with open(caminho_log, "r", errors="replace") as fh:
                    fh.seek(posicao)
                    for linha in fh:
                        veredicto = monitor.linha(linha.rstrip("\n"))
                        if veredicto and not abortado:
                            abortado = veredicto
                    posicao = fh.tell()
            if abortado:
                ao_registar("ABORT", "a terminar o treino de %s fold %d a epoca %s"
                            % (arq["name"], fold, abortado.get("epoch")))
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    proc.wait(timeout=120)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                break
            if proc.poll() is not None:
                if caminho_log:
                    with open(caminho_log, "r", errors="replace") as fh:
                        fh.seek(posicao)
                        for linha in fh:
                            monitor.linha(linha.rstrip("\n"))
                        posicao = fh.tell()
                break
            time.sleep(intervalo)
    finally:
        bruto.close()

    codigo = proc.wait()
    duracao = time.time() - inicio

    if abortado:
        desfecho, retomavel = "ABORTED_BY_MONITOR", False
    elif codigo == 0:
        desfecho, retomavel = "COMPLETED", False
    else:
        desfecho, retomavel = "FAILED", True

    ckpts = {}
    for nome in ("checkpoint_final.pth", "checkpoint_best.pth", "checkpoint_latest.pth"):
        c = os.path.join(pasta, nome)
        if os.path.isfile(c):
            ckpts[nome] = {"sha256": sha256_file(c),
                           "bytes": os.path.getsize(c)}

    conferencia = verificar_debug_json(pasta, esperados, sobrepostos, continuar)
    registo = {
        "run_id": run_id, "architecture": arq["name"], "fold": fold,
        "configuration": arq["config"], "plans_id": arq["plans_id"],
        "trainer": arq["trainer"],
        "command": cmd, "runner_sha256": runner_sha,
        "declared_overrides": sobrepostos,
        "exit_code": codigo, "outcome": desfecho,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(inicio)),
        "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "duration_s": round(duracao, 1),
        "training_log": caminho_log,
        "stdout_capture": saida_bruta,
        "monitor": monitor.resumo(),
        "checkpoints": ckpts,
        "debug_json_check": conferencia,
        "output_folder": pasta,
    }
    escrever_estado(pasta, run_id=run_id, architecture=arq["name"], fold=fold,
                    outcome=desfecho, resumable=retomavel,
                    exit_code=codigo, duration_s=registo["duration_s"],
                    monitor_trigger=monitor.disparo,
                    debug_json_status=conferencia["status"],
                    reason=("terminou por decisao do monitor de colapso; "
                            "relancar desfaria a decisao"
                            if desfecho == "ABORTED_BY_MONITOR" else None))
    return registo


def proveniencia_base(interpretador, run_id, sobrepostos):
    """
    O que e comum a todos os folds desta corrida: coorte, particao, planos,
    ambiente e a origem de cada valor. Vai inteiro para o manifesto.
    """
    p08, m08, c08, _ = ler_ponteiro(PONTEIRO_08, "08 (nnU-Net)")
    p07, m07, c07, _ = ler_ponteiro(PONTEIRO_07, "07 (particao)")
    regra, preg = carregar_regra_colapso()
    cfg, pcfg = carregar_config_congelada()

    raiz = p08.get("nnUNet_preprocessed")
    if raiz and not os.path.isabs(raiz):
        raiz = os.path.join(BASE, raiz)
    dir_ds = os.path.join(raiz, DATASET) if raiz else None
    hashes = {}
    if dir_ds and os.path.isdir(dir_ds):
        for f in sorted(os.listdir(dir_ds)):
            if f.endswith(".json"):
                hashes[f] = sha256_file(os.path.join(dir_ds, f))
    return {
        "run_id": run_id,
        "cohort_source": {"script08_run": p08.get("run_id"),
                          "script08_manifest_body_sha256":
                              p08.get("manifest_body_sha256")},
        "partition_source": {"script07_run": p07.get("run_id"),
                             "split_csv_sha256": p07.get("split_csv_sha256"),
                             "folds_json_sha256": p07.get("folds_json_sha256")},
        "frozen_config": {"script08c_run": pcfg.get("run_id"),
                          "training_config_sha256":
                              pcfg.get("training_config_sha256")},
        "collapse_rule": {"script08a_run": preg.get("run_id"),
                          "rule_id": regra.get("rule_id"),
                          "collapse_rule_sha256": preg.get("collapse_rule_sha256")},
        "preprocessed_json_sha256": hashes,
        "declared_overrides": sobrepostos,
        "environment": ambiente(interpretador),
        "treino_core_sha256": sha256_file(os.path.abspath(__file__)),
        "seed": 27,
    }


ESQUEMA_ORDEM = "pulmonary/ordem-treino/v1"


def _assinar(corpo):
    limpo = {k: v for k, v in corpo.items() if k != "order_sha256"}
    return hashlib.sha256(
        json.dumps(limpo, indent=1, sort_keys=True, default=str).encode()).hexdigest()


def emitir_ordem(arq, fold, cfg, pcfg, regra, preg, p08, p07, sobrepostos,
                 emissor_run_id):
    """
    Uma ordem por fold, auto-suficiente. Quem a executa não precisa da cadeia.
    """
    esperado = esperado_para(cfg, arq["name"])
    ordem = {
        "schema": ESQUEMA_ORDEM,
        "issued_by_run": emissor_run_id,
        "issued_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "issuer_host": platform.node(),
        "architecture": arq,
        "fold": fold,
        "dataset": DATASET,
        "dataset_id": DATASET_ID,
        "declared_overrides": sobrepostos,
        "operational_overrides": esperado.get("operational_overrides"),
        "expected_trainer_effective": esperado.get("trainer_effective"),
        "collapse_rule": regra,
        "preprocessed_json_sha256": (
            (json.load(open(p08["manifest"])) or {}).get("preprocessed_json_sha256")
            if os.path.isfile(p08.get("manifest", "")) else None),
        "provenance": {
            "script08_run": p08.get("run_id"),
            "script07_run": p07.get("run_id"),
            "script08a_run": preg.get("run_id"),
            "script08c_run": pcfg.get("run_id"),
            "collapse_rule_sha256": preg.get("collapse_rule_sha256"),
            "training_config_sha256": pcfg.get("training_config_sha256"),
            "split_csv_sha256": p07.get("split_csv_sha256"),
            "folds_json_sha256": p07.get("folds_json_sha256"),
        },
        "treino_core_sha256": sha256_file(os.path.abspath(__file__)),
    }
    _exigir(ordem["preprocessed_json_sha256"],
            "o manifesto do 08 nao declara `preprocessed_json_sha256`; sem isso "
            "a ordem nao consegue provar que o executor tem os dados certos. "
            "Reexecutar o Script 08 com a versao actual.")
    ordem["order_sha256"] = _assinar(ordem)
    return ordem


def carregar_ordem(caminho, raiz_preprocessed_local):
    """
    Lê uma ordem, verifica a assinatura, e PROVA que os dados locais são os que
    ela descreve. Devolve (ordem, relatorio_da_verificacao).
    """
    _exigir(os.path.isfile(caminho), "ordem inexistente: %s" % caminho)
    o = json.load(open(caminho))
    _exigir(o.get("schema") == ESQUEMA_ORDEM,
            "ordem de esquema desconhecido: %r" % o.get("schema"))
    _exigir(o.get("order_sha256") == _assinar(o),
            "a ordem foi alterada depois de emitida: assinatura nao confere")

    ds = os.path.join(raiz_preprocessed_local, o["dataset"])
    _exigir(os.path.isdir(ds), "%s ausente em %s" % (o["dataset"],
                                                     raiz_preprocessed_local))
    ausentes, divergentes = [], []
    for nome, esperado in sorted((o.get("preprocessed_json_sha256") or {}).items()):
        alvo = os.path.join(ds, nome)
        if not os.path.isfile(alvo):
            ausentes.append(nome)
        elif sha256_file(alvo) != esperado:
            divergentes.append(nome)
    _exigir(not ausentes and not divergentes,
            "os dados em %s NAO sao os que a ordem descreve: ausentes=%s "
            "divergentes=%s. Nao se treina sobre dados por confirmar."
            % (raiz_preprocessed_local, ausentes, divergentes))

    meu = sha256_file(os.path.abspath(__file__))
    aviso = None
    if meu != o.get("treino_core_sha256"):
        aviso = ("treino_core diferente do emissor (%s vs %s): o executor pode "
                 "aplicar regras diferentes das que a ordem pressupoe"
                 % (meu[:12], str(o.get("treino_core_sha256"))[:12]))
    return o, {"json_verified": len(o.get("preprocessed_json_sha256") or {}),
               "preprocessed_root": raiz_preprocessed_local,
               "treino_core_match": meu == o.get("treino_core_sha256"),
               "warning": aviso}
