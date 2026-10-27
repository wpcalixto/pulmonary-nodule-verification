#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    08c_freeze_training_config.py

File type:
    Script

Purpose:
    Freezes, by introspection, the effective training configuration of the ten architectures into the manifest consumed by stage 09.

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

Outputs:
    - See docs/file_level_documentation.md.

Dependencies:
    - nnunetv2
    - torch

Execution:
    PULMONARY_BASE=<project_dir> python scripts/08c_freeze_training_config.py

Portability notes:
    All paths are resolved relative to the project base directory given by the PULMONARY_BASE environment variable. No machine-specific path is required.

Reproducibility notes:
    Environment pinned in requirements.txt and environment/. Seeds and frozen parameters are recorded in the run manifests under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) are not included and must be obtained from their original repositories.
"""

import os
import ast
import csv
import sys
import json
import time
import inspect
import hashlib
import logging
import argparse
import tempfile
import textwrap
import importlib
import traceback

BASE = os.environ["PULMONARY_BASE"]
RUNS = os.path.join(BASE, "runs")
PONTEIRO = os.path.join(RUNS, "CURRENT_TRAINING_CONFIG.json")
PONTEIRO_08 = os.path.join(RUNS, "CURRENT_NNUNET.json")
PONTEIRO_07 = os.path.join(RUNS, "CURRENT_BD1_PARTITION.json")
PONTEIRO_08B = os.path.join(RUNS, "CURRENT_GPU_FIT.json")

SAVE_EVERY_ALVO = 10

ATRIBUTOS = ("initial_lr", "weight_decay", "num_epochs", "num_iterations_per_epoch",
             "num_val_iterations_per_epoch", "oversample_foreground_percent",
             "enable_deep_supervision", "save_every", "current_epoch",
             "probabilistic_oversampling")

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


def _ponteiro(caminho, etiqueta):
    _exigir(os.path.isfile(caminho), "sem ponteiro de %s em %s" % (etiqueta, caminho))
    p = json.load(open(caminho))
    m = p.get("manifest")
    _exigir(m and os.path.isfile(m), "manifesto de %s inacessivel: %s" % (etiqueta, m))
    h = sha(m)
    declarados = [p[k] for k in ("manifest_file_sha256", "manifest_sha256") if p.get(k)]
    _exigir(declarados, "ponteiro de %s nao declara hash do manifesto" % etiqueta)
    _exigir(h in declarados,
            "manifesto de %s nao confere com o ponteiro" % etiqueta)
    return p


CAMPOS_PLANO_GLOBAIS = ("original_median_spacing_after_transp",
                        "original_median_shape_after_transp",
                        "transpose_forward", "transpose_backward",
                        "image_reader_writer", "plans_name",
                        "experiment_planner_used")
CAMPOS_CONFIG = ("spacing", "patch_size", "batch_size", "data_identifier",
                 "normalization_schemes", "use_mask_for_norm",
                 "resampling_fn_data", "resampling_fn_seg", "batch_dice",
                 "inherits_from", "next_stage", "previous_stage")
CAMPOS_ARQ = ("n_stages", "features_per_stage", "kernel_sizes", "strides",
              "n_blocks_per_stage", "n_conv_per_stage_decoder")


def parametros_do_plano(caminho_plano, configuracao):
    """Tudo o que o planeador decidiu, para esta configuração."""
    cru = json.load(open(caminho_plano))
    fora = {"plans_file": os.path.basename(caminho_plano),
            "plans_sha256": sha(caminho_plano), "valores": {}}
    for k in CAMPOS_PLANO_GLOBAIS:
        if k in cru:
            fora["valores"][k] = cru[k]
    cfg = cru.get("configurations", {}).get(configuracao, {})
    for k in CAMPOS_CONFIG:
        if k in cfg:
            fora["valores"]["%s.%s" % (configuracao, k)] = cfg[k]
    arq = cfg.get("architecture", {}).get("arch_kwargs", {})
    for k in CAMPOS_ARQ:
        if k in arq:
            fora["valores"]["%s.arch.%s" % (configuracao, k)] = arq[k]
    if "network_class_name" in cfg.get("architecture", {}):
        fora["valores"]["%s.arch.class" % configuracao] = \
            cfg["architecture"]["network_class_name"]
    fp = cru.get("foreground_intensity_properties_per_channel", {})
    for canal, props in sorted(fp.items()):
        for k in ("mean", "std", "median", "percentile_00_5", "percentile_99_5"):
            if k in props:
                fora["valores"]["intensity.ch%s.%s" % (canal, k)] = props[k]
    return fora


def config_manager_efectivo(caminho_plano, configuracao):
    """Valores DEPOIS de resolvida a herança entre configurações."""
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    pm = PlansManager(caminho_plano)
    cm = pm.get_configuration(configuracao)
    fora = {}
    for k in ("spacing", "patch_size", "batch_size", "data_identifier",
              "normalization_schemes", "use_mask_for_norm", "batch_dice"):
        v = getattr(cm, k, None)
        if v is not None:
            fora[k] = list(v) if isinstance(v, (list, tuple)) else v
    fora["network_arch_class_name"] = getattr(cm, "network_arch_class_name", None)
    return pm, cm, fora


def _literal(no):
    try:
        return ast.literal_eval(no), True
    except (ValueError, SyntaxError):
        return ast.dump(no)[:120], False


def declarados_na_classe(cls):
    """
    Varre o `__init__` desta classe (só desta, não das bases) à procura de
    `self.<atributo> = <literal>`. Devolve {atributo: (valor, e_literal)}.
    """
    try:
        fonte = inspect.getsource(cls.__init__)
    except (OSError, TypeError):
        return {}
    try:
        arvore = ast.parse(textwrap.dedent(fonte))
    except SyntaxError:
        return {}
    fora = {}
    for no in ast.walk(arvore):
        if not isinstance(no, ast.Assign):
            continue
        for alvo in no.targets:
            if (isinstance(alvo, ast.Attribute) and isinstance(alvo.value, ast.Name)
                    and alvo.value.id == "self" and alvo.attr in ATRIBUTOS):
                fora[alvo.attr] = _literal(no.value)
    return fora


def cadeia_declarada(cls):
    """
    Percorre a MRO da subclasse para a base. O primeiro que declarar cada
    atributo na ordem de execução real — que é base primeiro, subclasse depois,
    porque `super().__init__()` corre antes — é quem fica a valer.
    """
    resolvido, proveniencia = {}, {}
    for c in reversed(cls.__mro__):
        for k, (v, lit) in declarados_na_classe(c).items():
            resolvido[k] = v
            proveniencia[k] = {"declared_in": c.__name__, "is_literal": lit}
    return resolvido, proveniencia


def carregar_trainer(nome):
    """
    Resolve o trainer pelo mesmo mecanismo que o nnU-Net usa
    (`recursive_find_python_class`), para que o que se audita seja o que vai
    correr — e não um homónimo que estivesse noutro sítio do `sys.path`.
    """
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
    import nnunetv2
    raiz = os.path.join(os.path.dirname(nnunetv2.__file__), "training", "nnUNetTrainer")
    cls = recursive_find_python_class(raiz, nome, "nnunetv2.training.nnUNetTrainer")
    _exigir(cls is not None,
            "trainer %s nao encontrado no nnunetv2 deste interpretador (%s). "
            "O ambiente nao esta provisionado: correr o Script 00 --provision."
            % (nome, sys.executable))
    return cls


def efectivos_por_instanciacao(cls, caminho_plano, configuracao, dir_dataset, fold):
    """
    Instancia o trainer a sério e lê os atributos do objecto. É o único caminho
    que apanha o que a cadeia de herança realmente produz.
    """
    plano = json.load(open(caminho_plano))
    dj = json.load(open(os.path.join(dir_dataset, "dataset.json")))
    import torch

    def _pede_continue_training(k):
        for base in inspect.getmro(k):
            ini = base.__dict__.get("__init__")
            if ini is None:
                continue
            try:
                if "continue_training" in inspect.getsource(ini):
                    return True
            except (OSError, TypeError):
                pass
        return False

    if _pede_continue_training(cls):
        plano.setdefault("continue_training", False)

    obj = cls(plans=plano, configuration=configuracao, fold=fold,
              dataset_json=dj, device=torch.device("cpu"))
    fora = {}
    for k in ATRIBUTOS:
        if hasattr(obj, k):
            v = getattr(obj, k)
            fora[k] = v if isinstance(v, (int, float, str, bool, type(None))) \
                else str(v)
    try:
        fonte = inspect.getsource(cls.configure_optimizers)
        fora["_optimizer_source"] = " ".join(fonte.split())[:400]
    except (OSError, TypeError):
        pass
    return fora


def preparar_caminhos_nnunet(dir_preprocessed):
    """
    O nnU-Net lê `nnUNet_preprocessed`, `nnUNet_results` e `nnUNet_raw` no momento
    do IMPORT, para variáveis de módulo. Defini-las depois não tem efeito nenhum —
    é preciso fazê-lo antes de qualquer import de `nnunetv2`, e ainda assim
    reescrevê-las no módulo, caso algo já o tenha importado.

    `nnUNet_results` aponta para um temporário: este script não treina e não
    escreve modelo nenhum. Só precisa que os caminhos existam para o trainer se
    conseguir construir.
    """
    tmp = tempfile.mkdtemp(prefix="s08c_results_")
    os.environ["nnUNet_preprocessed"] = dir_preprocessed
    os.environ["nnUNet_results"] = tmp
    os.environ.setdefault("nnUNet_raw", tmp)
    import nnunetv2.paths as _p

    for nome, valor in (("nnUNet_preprocessed", dir_preprocessed),
                        ("nnUNet_results", tmp),
                        ("nnUNet_raw", os.environ["nnUNet_raw"])):
        if isinstance(getattr(_p, nome, None), str):
            setattr(_p, nome, valor)
    return tmp


def decisao_compile(nome_plano, configuracao):
    """
    Liga ou desliga `torch.compile` POR CONFIGURACAO, a partir do que o Script
    08b mediu — nunca por regra escrita a mao.

    O nnU-Net liga a compilacao por omissao em CUDA. A medicao nas duas placas do
    cluster mostrou que isso acelera as configuracoes 2d e ATRASA as 3d, com o
    mesmo sinal em ambas. Onde nao houver medicao nas duas placas o valor fica
    PENDING_MEASUREMENT: extrapolar de uma configuracao para outra seria inventar,
    e o ganho ja decresce com o tamanho do modelo (-21% no 2d simples contra
    -14% no ResEncM 2d na mesma placa).

    NAO e neutro. Nao muda o desenho experimental, mas muda kernels e portanto
    pode mudar trajectorias numericas — por isso o valor, as versoes de PyTorch e
    CUDA, a GPU e a origem da decisao vao para a proveniencia de cada fold.
    """
    p = _ponteiro(PONTEIRO_08B, "08b (sondagem de GPU)")
    cmp_ = (p.get("compile_comparison") or {}).get(
        "%s/%s" % (nome_plano, configuracao))
    if not cmp_:
        return {"value": None, "status": "PENDING_MEASUREMENT",
                "evidence": None, "gpus_measured": 0,
                "note": ("sem medicao comparativa; correr o Script 08b "
                         "--time --compile na maquina de destino antes de "
                         "treinar esta configuracao")}
    porgpu = {k: v for k, v in cmp_.items() if not k.startswith("_")}
    veredicto = cmp_.get("_verdict")
    if veredicto == "COMPILE_FASTER":
        valor, estado = 1, "MEASURED_FASTER"
    elif veredicto == "COMPILE_SLOWER":
        valor, estado = 0, "MEASURED_SLOWER"
    else:
        por_placa = {g: (0 if d["delta_pct"] > 0 else 1) for g, d in porgpu.items()}
        return {"value": None, "status": "PER_GPU",
                "value_per_gpu": por_placa, "evidence": porgpu,
                "gpus_measured": cmp_.get("_gpus_measured", 0),
                "source_run": p.get("run_id"),
                "note": ("resolvido por placa: o parametro e operacional e "
                         "depende do hardware, nao do desenho experimental")}
    return {"value": valor, "status": estado, "evidence": porgpu,
            "gpus_measured": cmp_.get("_gpus_measured", 0),
            "source_run": p.get("run_id")}


def sobrepostos_operacionais(arq, efectivo_da_classe, nome_plano):
    """
    Sobrepostos OPERACIONAIS: valor alvo, o que a classe ja declara, e se e
    preciso mexer em tempo de execucao. Quando a classe ja declara o alvo — como
    o `nnUNetTrainerCheckEvery10` para o `save_every` — nao ha sobreposto a
    aplicar, e isso fica DITO em vez de aplicado por cima.
    """
    declarado = efectivo_da_classe.get("save_every")
    fora = {"save_every": {
        "target": SAVE_EVERY_ALVO, "class_declares": declarado,
        "runtime_override_needed": declarado != SAVE_EVERY_ALVO,
        "nature": "OPERATIONAL",
        "rationale": ("granularidade de retoma; nao altera o modelo treinado. "
                      "Ver Caderno de 2026-08-11.")}}
    c = decisao_compile(nome_plano, arq["config"])
    c.update({"nature": "OPERATIONAL_NUMERIC",
              "applied_via": "variavel de ambiente nnUNet_compile",
              "rationale": ("derivado da medicao do Script 08b nas placas de "
                            "destino; o nnU-Net liga por omissao em CUDA")})
    fora["nnUNet_compile"] = c
    return fora


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0,
                    help="fold usado apenas para instanciar o trainer; nenhum "
                         "parametro auditado depende dele")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--supersede", action="store_true")
    args = ap.parse_args()

    run_id = "s08c_%s_%d" % (time.strftime("%Y%m%dT%H%M%S"), os.getpid())
    dest = os.path.join(RUNS, run_id)
    _exigir(not os.path.exists(dest), "destino ja existe")
    if os.path.isfile(PONTEIRO) and not args.supersede and not args.plan_only:
        _p = json.load(open(PONTEIRO))
        _exigir(False, "ja existe configuracao congelada: %s. Nao se recongela "
                       "por cima; use --supersede sabendo que invalida a "
                       "comparabilidade com o que ja treinou." % _p.get("run_id"))

    log.info("Script 08c — congelamento da configuracao de treino | run_id=%s", run_id)
    log.info("  interpretador: %s", sys.executable)

    p08 = _ponteiro(PONTEIRO_08, "Script 08")
    p07 = _ponteiro(PONTEIRO_07, "Script 07")
    raiz = p08.get("nnUNet_preprocessed")
    _exigir(raiz, "ponteiro do 08 nao declara nnUNet_preprocessed")
    if not os.path.isabs(raiz):
        raiz = os.path.join(BASE, raiz)
    datasets = [d for d in sorted(os.listdir(raiz))
                if os.path.isdir(os.path.join(raiz, d))]
    _exigir(len(datasets) == 1, "esperava um dataset, encontrei %s" % datasets)
    dir_dataset = os.path.join(raiz, datasets[0])
    tmp_results = preparar_caminhos_nnunet(raiz)

    import nnunetv2
    import importlib.metadata as md
    versao_nnunet = md.version("nnunetv2")
    import torch
    log.info("  nnunetv2 %s · torch %s", versao_nnunet, torch.__version__)
    log.info("  planos do 08 run %s · particao do 07 run %s · results temporario %s",
             p08.get("run_id"), p07.get("run_id"), tmp_results)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import treino_core
    registo = treino_core.ARQUITETURAS
    origem_registo = os.path.abspath(treino_core.__file__)
    _exigir(registo, "registo de arquiteturas vazio em treino_core")
    log.info("  %d arquiteturas no registo do treino_core", len(registo))

    linhas, entradas, divergencias, trainers_vistos = [], [], [], {}
    for a in registo:
        nome, cfg = a["name"], a["config"]
        plano = os.path.join(dir_dataset, "%s.json" % a["plans_id"])
        _exigir(os.path.isfile(plano), "plano ausente: %s" % plano)
        nome_trainer = a.get("trainer", "nnUNetTrainer")

        cls = carregar_trainer(nome_trainer)
        if nome_trainer not in trainers_vistos:
            f = inspect.getsourcefile(cls)
            trainers_vistos[nome_trainer] = {
                "class": cls.__name__, "module": cls.__module__,
                "source_file": f, "source_sha256": sha(f) if f else None,
                "mro": [c.__name__ for c in cls.__mro__
                        if c.__name__ not in ("object",)]}

        decl, prov = cadeia_declarada(cls)
        efec = efectivos_por_instanciacao(cls, plano, cfg, dir_dataset, args.fold)
        planeador = parametros_do_plano(plano, cfg)
        _, _, cm_efec = config_manager_efectivo(plano, cfg)

        operacionais = sobrepostos_operacionais(a, efec, a["plans_id"])
        entrada = {"operational_overrides": operacionais,
                   "architecture": nome, "configuration": cfg,
                   "plans_id": a["plans_id"], "trainer": nome_trainer,
                   "requires_lowres": bool(a.get("requires_lowres")),
                   "planner": planeador, "configuration_effective": cm_efec,
                   "trainer_declared": decl, "trainer_declared_in": prov,
                   "trainer_effective": {k: v for k, v in efec.items()
                                         if not k.startswith("_")},
                   "optimizer_source": efec.get("_optimizer_source")}
        entradas.append(entrada)

        for k in sorted(set(decl) | set(efec)):
            if k.startswith("_"):
                continue
            d, e = decl.get(k, "<nao declarado>"), efec.get(k, "<inexistente>")
            if k not in decl:
                estado = "NOT_DECLARED_IN_INIT"
            elif d == e:
                estado = "OK"
            else:
                estado = "DECLARED_EFFECTIVE_MISMATCH"
            bate = (estado != "DECLARED_EFFECTIVE_MISMATCH")
            origem = ("study" if prov.get(k, {}).get("declared_in", "")
                      not in ("nnUNetTrainer", "") else "trainer")
            if k not in decl:
                origem = "trainer"
            if not bate:
                divergencias.append("%s/%s: %s declarado=%r efectivo=%r"
                                    % (nome, nome_trainer, k, d, e))
            linhas.append({"architecture": nome, "configuration": cfg,
                           "plans_id": a["plans_id"], "trainer": nome_trainer,
                           "parameter": k, "origin": origem,
                           "declared_value": d, "declared_in":
                               prov.get(k, {}).get("declared_in", ""),
                           "effective_value": e, "status": estado})
        for k, v in sorted(planeador["valores"].items()):
            linhas.append({"architecture": nome, "configuration": cfg,
                           "plans_id": a["plans_id"], "trainer": nome_trainer,
                           "parameter": k, "origin": "planner",
                           "declared_value": json.dumps(v), "declared_in":
                               planeador["plans_file"],
                           "effective_value": json.dumps(v), "status": "OK"})
        _c = operacionais["nnUNet_compile"]
        log.info("  %-14s %-20s lr=%s epocas=%s oversample=%-4s save_every=%s->%s "
                 "compile=%s (%s)",
                 nome, cfg, efec.get("initial_lr"), efec.get("num_epochs"),
                 efec.get("oversample_foreground_percent"),
                 efec.get("save_every"), SAVE_EVERY_ALVO,
                 "?" if _c["value"] is None else _c["value"], _c["status"])
        for _k, _v in sorted(operacionais.items()):
            linhas.append({"architecture": nome, "configuration": cfg,
                           "plans_id": a["plans_id"], "trainer": nome_trainer,
                           "parameter": _k, "origin": "study_operational",
                           "declared_value": json.dumps(_v.get("class_declares")),
                           "declared_in": "08b/medicao" if _k == "nnUNet_compile"
                           else nome_trainer,
                           "effective_value": json.dumps(
                               _v.get("target", _v.get("value"))),
                           "status": _v.get("status", "TARGET_SET")})

    for d in divergencias:
        log.info("  DIVERGENCIA  %s", d)

    portoes = {
        "ARCHITECTURES": len(registo),
        "PARAMETERS_RECORDED": len(linhas),
        "TRAINERS_RESOLVED": sorted(trainers_vistos),
        "DECLARED_EFFECTIVE_MISMATCHES": len(divergencias),
        "SAVE_EVERY_TARGET": SAVE_EVERY_ALVO,
        "COMPILE_PENDING": sorted(
            e["architecture"] for e in entradas
            if e["operational_overrides"]["nnUNet_compile"]["value"] is None
            and not e["operational_overrides"]["nnUNet_compile"].get("value_per_gpu")),
        "COMPILE_PER_GPU": sorted(
            e["architecture"] for e in entradas
            if e["operational_overrides"]["nnUNet_compile"].get("value_per_gpu")),
        "NNUNET_VERSION": versao_nnunet,
        "TORCH_VERSION": torch.__version__,
        "INTERPRETER": sys.executable,
        "CONFIG_FROZEN": "YES",
        "SCRIPT08C_OUTPUT_STATUS": "PASS" if not divergencias else "PASS_WITH_FLAG",
    }
    for k, v in portoes.items():
        log.info("  %-32s %s", k, v)

    if args.plan_only:
        log.info("  PLAN_STATUS = PASS · nada escrito")
        return

    os.makedirs(dest, exist_ok=False)
    fc = os.path.join(dest, "training_parameters.csv")
    with open(fc, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["architecture", "configuration",
                                          "plans_id", "trainer", "parameter",
                                          "origin", "declared_value",
                                          "declared_in", "effective_value",
                                          "status"])
        w.writeheader()
        for r in linhas:
            w.writerow(r)

    cfgj = os.path.join(dest, "training_config.json")
    json.dump({"architectures": entradas, "trainers": trainers_vistos,
               "declared_effective_mismatches": divergencias,
               "environment": {"nnunetv2": versao_nnunet,
                               "torch": torch.__version__,
                               "interpreter": sys.executable,
                               "nnunetv2_path": os.path.dirname(nnunetv2.__file__)},
               "inputs": {"script08_run": p08.get("run_id"),
                          "script07_run": p07.get("run_id"),
                          "dataset_dir": os.path.relpath(dir_dataset, BASE)},
               "attributes_audited": list(ATRIBUTOS),
               "note_runtime_override": (
                   "declarado vs efectivo compara CODIGO com OBJECTO CONSTRUIDO. "
                   "Uma alteracao feita depois da construcao, durante a corrida, "
                   "nao e visivel daqui: o Script 09 tem de comparar o debug.json "
                   "de cada fold contra estes valores e recusar o que nao bater.")},
              open(cfgj, "w"), indent=1, sort_keys=True)

    m = {"manifest_id": "training_config_%s" % run_id, "run_id": run_id,
         "script": "08C_FREEZE_TRAINING_CONFIG",
         "script_version": "2026-08-11.freeze.v1",
         "generated": time.strftime("%Y-%m-%dT%H:%M:%S BRT"),
         "producer_sha256": sha(os.path.abspath(__file__)),
         "registry_source": "treino_core.py:ARQUITETURAS",
         "registry_source_sha256": sha(origem_registo),
         "gates": portoes,
         "training_config_sha256": sha(cfgj),
         "training_parameters_csv_sha256": sha(fc),
         "script08_pointer_sha256": sha(PONTEIRO_08),
         "script07_pointer_sha256": sha(PONTEIRO_07),
         "SCRIPT08C_OUTPUT_STATUS": portoes["SCRIPT08C_OUTPUT_STATUS"]}
    corpo = hashlib.sha256(json.dumps(m, indent=1, sort_keys=True).encode()).hexdigest()
    m["manifest_sha256"] = corpo
    pm = os.path.join(dest, "manifest.json")
    json.dump(m, open(pm, "w"), indent=1, sort_keys=True)
    json.dump({"run_id": run_id, "script": "08c",
               "script_version": m["script_version"], "manifest": pm,
               "manifest_body_sha256": corpo, "manifest_sha256": sha(pm),
               "training_config": os.path.relpath(cfgj, BASE),
               "training_config_sha256": m["training_config_sha256"],
               "training_parameters_csv": os.path.relpath(fc, BASE),
               "trainers": trainers_vistos,
               "environment": {"nnunetv2": versao_nnunet,
                               "torch": torch.__version__},
               "SCRIPT08C_OUTPUT_STATUS": portoes["SCRIPT08C_OUTPUT_STATUS"]},
              open(PONTEIRO, "w"), indent=1, sort_keys=True)
    log.info("  publicado em %s", dest)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
