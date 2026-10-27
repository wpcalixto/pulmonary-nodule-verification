#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    08b_probe_gpu_fit.py

File type:
    Script

Purpose:
    Probes GPU memory to verify that each planned configuration fits the available GPU with the planned batch and patch sizes.

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
    - probe.json

Outputs:
    - See docs/file_level_documentation.md.

Dependencies:
    - nnunetv2
    - torch

Execution:
    PULMONARY_BASE=<project_dir> python scripts/08b_probe_gpu_fit.py

Portability notes:
    All paths are resolved relative to the project base directory given by the PULMONARY_BASE environment variable. No machine-specific path is required.

Reproducibility notes:
    Environment pinned in requirements.txt and environment/. Seeds and frozen parameters are recorded in the run manifests under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) are not included and must be obtained from their original repositories.
"""

import os
import csv
import sys
import json
import time
import socket
import hashlib
import logging
import argparse
import platform
import collections
import traceback

BASE = os.environ["PULMONARY_BASE"]
RUNS = os.path.join(BASE, "runs")
PONTEIRO = os.path.join(RUNS, "CURRENT_GPU_FIT.json")
PONTEIRO_08 = os.path.join(RUNS, "CURRENT_NNUNET.json")

ITERACOES = 3
ITER_TREINO_POR_EPOCA = 250
ITER_VALID_POR_EPOCA = 50
AQUECIMENTO = 3
CRONOMETRADAS = 10
SGD_MOMENTUM = 0.99
SGD_WEIGHT_DECAY = 3e-5
SGD_LR = 1e-2

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


def _preprocessed_do_08():
    """
    A pasta `nnUNet_preprocessed` da execução corrente do Script 08, resolvida
    pelo ponteiro. Nunca por caminho escrito à mão.
    """
    _exigir(os.path.isfile(PONTEIRO_08),
            "sem ponteiro do Script 08 em %s" % PONTEIRO_08)
    p = json.load(open(PONTEIRO_08))
    m = p.get("manifest")
    _exigir(m and os.path.isfile(m), "manifesto do 08 inacessivel: %s" % m)
    h = sha(m)
    declarados = {k: p.get(k) for k in ("manifest_file_sha256", "manifest_sha256")
                  if p.get(k)}
    _exigir(declarados, "ponteiro do 08 nao declara hash do manifesto")
    _exigir(h in declarados.values(),
            "manifesto do 08 nao confere: ficheiro %s, ponteiro declara %s"
            % (h[:16], {k: v[:16] for k, v in declarados.items()}))
    raiz = p.get("nnUNet_preprocessed")
    _exigir(raiz, "ponteiro do 08 nao declara nnUNet_preprocessed")
    if not os.path.isabs(raiz):
        raiz = os.path.join(BASE, raiz)
    _exigir(os.path.isdir(raiz), "nnUNet_preprocessed ausente em %s" % raiz)
    datasets = [d for d in sorted(os.listdir(raiz))
                if os.path.isdir(os.path.join(raiz, d))]
    _exigir(len(datasets) == 1,
            "esperava um dataset em nnUNet_preprocessed, encontrei %s" % datasets)
    return os.path.join(raiz, datasets[0]), p.get("run_id")


def _configuracoes(dir_dataset):
    """
    Todos os pares (plano, configuração) que o Script 08 publicou. Derivado do
    que está em disco — nenhuma lista escrita à mão.
    """
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    ficheiros = sorted(f for f in os.listdir(dir_dataset) if f.endswith("Plans.json"))
    _exigir(ficheiros, "nenhum ficheiro de plano em %s" % dir_dataset)
    dj = json.load(open(os.path.join(dir_dataset, "dataset.json")))
    saida = []
    for f in ficheiros:
        caminho = os.path.join(dir_dataset, f)
        pm = PlansManager(caminho)
        cru = json.load(open(caminho))
        for nome in sorted(cru.get("configurations", {})):
            try:
                cm = pm.get_configuration(nome)
                if not getattr(cm, "patch_size", None):
                    continue
            except Exception:
                continue
            saida.append({"plans_file": f, "plans_name": pm.plans_name,
                          "configuration": nome, "pm": pm, "cm": cm,
                          "dataset_json": dj,
                          "plans_sha256": sha(caminho)})
    return saida


def _cronometrar(fn, dispositivo, n):
    import torch
    torch.cuda.synchronize(dispositivo)
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize(dispositivo)
    return (time.perf_counter() - t) / n


def sondar(item, dispositivo, cronometrar=False, compilar=False):
    """Uma configuração numa placa. Devolve o registo, com OOM como resultado."""
    import torch
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from nnunetv2.utilities.label_handling.label_handling import \
        determine_num_input_channels

    cm, pm, dj = item["cm"], item["pm"], item["dataset_json"]
    patch = list(cm.patch_size)
    lote = int(cm.batch_size)
    n_ent = determine_num_input_channels(pm, cm, dj)
    rotulos = pm.get_label_manager(dj)
    n_sai = rotulos.num_segmentation_heads

    reg = {"plans_name": item["plans_name"], "configuration": item["configuration"],
           "batch_size": lote, "patch_size": "x".join(str(v) for v in patch),
           "input_channels": n_ent, "output_channels": n_sai,
           "peak_alloc_gb": "", "peak_reserved_gb": "", "status": "",
           "train_iter_s": "", "val_iter_s": "", "epoch_s": "", "fold_1000ep_h": "",
           "compiled": int(bool(compilar)),
           "plans_sha256": item["plans_sha256"], "detalhe": ""}

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dispositivo)
    rede = otim = None
    try:
        arq = cm.network_arch_class_name
        rede = get_network_from_plans(
            arq, cm.network_arch_init_kwargs,
            cm.network_arch_init_kwargs_req_import,
            n_ent, n_sai, allow_init=True, deep_supervision=True).to(dispositivo)
        if compilar:
            rede = torch.compile(rede)
        otim = torch.optim.SGD(rede.parameters(), SGD_LR, weight_decay=SGD_WEIGHT_DECAY,
                               momentum=SGD_MOMENTUM, nesterov=True)
        escala = torch.amp.GradScaler("cuda")
        rede.train()
        x = torch.randn([lote, n_ent] + patch, device=dispositivo)
        for _ in range(ITERACOES):
            otim.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=True):
                saidas = rede(x)
                if not isinstance(saidas, (list, tuple)):
                    saidas = [saidas]
                perda = 0
                for i, o in enumerate(saidas):
                    alvo = torch.randint(0, n_sai, [o.shape[0]] + list(o.shape[2:]),
                                         device=dispositivo)
                    perda = perda + (0.5 ** i) * torch.nn.functional.cross_entropy(o, alvo)
            escala.scale(perda).backward()
            escala.step(otim)
            escala.update()
        reg["status"] = "FITS"

        if cronometrar:
            def _passo_treino():
                otim.zero_grad(set_to_none=True)
                with torch.autocast("cuda", enabled=True):
                    outs = rede(x)
                    outs = outs if isinstance(outs, (list, tuple)) else [outs]
                    perda = 0
                    for i, o in enumerate(outs):
                        alvo = torch.randint(0, n_sai, [o.shape[0]] + list(o.shape[2:]),
                                             device=dispositivo)
                        perda = perda + (0.5 ** i) * torch.nn.functional.cross_entropy(o, alvo)
                escala.scale(perda).backward()
                escala.step(otim)
                escala.update()

            def _passo_valid():
                with torch.no_grad(), torch.autocast("cuda", enabled=True):
                    rede(x)

            _cronometrar(_passo_treino, dispositivo, AQUECIMENTO)
            t_tr = _cronometrar(_passo_treino, dispositivo, CRONOMETRADAS)
            _cronometrar(_passo_valid, dispositivo, AQUECIMENTO)
            t_va = _cronometrar(_passo_valid, dispositivo, CRONOMETRADAS)
            ep = ITER_TREINO_POR_EPOCA * t_tr + ITER_VALID_POR_EPOCA * t_va
            reg["train_iter_s"] = round(t_tr, 4)
            reg["val_iter_s"] = round(t_va, 4)
            reg["epoch_s"] = round(ep, 1)
            reg["fold_1000ep_h"] = round(ep * 1000 / 3600, 1)
    except torch.cuda.OutOfMemoryError as e:
        reg["status"] = "OOM"
        reg["detalhe"] = str(e).split("\n")[0][:200]
    except Exception as e:
        reg["status"] = "ERROR"
        reg["detalhe"] = "%s: %s" % (type(e).__name__, str(e).split("\n")[0][:180])
    finally:
        reg["peak_alloc_gb"] = round(torch.cuda.max_memory_allocated(dispositivo) / 2**30, 2)
        reg["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved(dispositivo) / 2**30, 2)
        del rede, otim
        torch.cuda.empty_cache()
    return reg


COLUNAS = ["host", "gpu_name", "gpu_total_gb", "plans_name", "configuration",
           "batch_size", "patch_size", "input_channels", "output_channels",
           "peak_alloc_gb", "peak_reserved_gb", "status", "train_iter_s",
           "val_iter_s", "epoch_s", "fold_1000ep_h", "compiled", "plans_sha256",
           "detalhe"]


def _escrever(caminho, linhas):
    with open(caminho, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUNAS, extrasaction="ignore")
        w.writeheader()
        for r in linhas:
            w.writerow(r)


def modo_merge(args):
    """Junta os CSV das várias placas e publica o ponteiro que o 09 vai ler."""
    linhas = []
    for c in args.merge:
        _exigir(os.path.isfile(c), "csv inexistente: %s" % c)
        with open(c) as f:
            linhas.extend(list(csv.DictReader(f)))
    _exigir(linhas, "nenhuma linha nos csv indicados")
    GEOM = ("batch_size", "patch_size", "input_channels", "output_channels")
    por_config = collections.defaultdict(lambda: {"sha": set(), "geom": set()})
    for r in linhas:
        k = (r["plans_name"], r["configuration"])
        por_config[k]["sha"].add(r.get("plans_sha256", ""))
        por_config[k]["geom"].add(tuple(r.get(c, "") for c in GEOM))
    divergentes = sorted("%s/%s" % k for k, v in por_config.items()
                         if len(v["geom"]) > 1)
    _exigir(not divergentes,
            "as placas sondaram GEOMETRIAS diferentes em: %s "
            "(batch/patch/canais nao coincidem — a comparacao seria invalida)"
            % divergentes)
    so_hash = sorted("%s/%s" % k for k, v in por_config.items()
                     if len(v["sha"]) > 1)
    if so_hash:
        log.warning("  planos com hash diferente mas geometria igual: %s", so_hash)
        log.warning("  aceite: o que muda nao afecta VRAM nem velocidade. Registado.")
    for r in linhas:
        k = "%s/%s" % (r["plans_name"], r["configuration"])
        r["plans_sha256_divergentes"] = "sim" if k in so_hash else "nao"
    run_id = "s08bmerge_%s_%d" % (time.strftime("%Y%m%dT%H%M%S"), os.getpid())
    dest = os.path.join(RUNS, run_id)
    os.makedirs(dest, exist_ok=False)
    fc = os.path.join(dest, "gpu_fit.csv")
    _escrever(fc, linhas)

    placas = sorted({r["gpu_name"] for r in linhas})
    onde = {}
    for r in linhas:
        chave = "%s/%s" % (r["plans_name"], r["configuration"])
        onde.setdefault(chave, {"fits_on": [], "oom_on": [], "error_on": []})
        alvo = {"FITS": "fits_on", "OOM": "oom_on"}.get(r["status"], "error_on")
        if r["gpu_name"] not in onde[chave][alvo]:
            onde[chave][alvo].append(r["gpu_name"])

    def _f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None
    tempos = {}
    for r in linhas:
        if r["status"] != "FITS" or not _f(r.get("epoch_s")):
            continue
        k = (r["plans_name"], r["configuration"], r["gpu_name"])
        tempos.setdefault(k, {})[int(r.get("compiled") or 0)] = _f(r["epoch_s"])
    compile_cmp = {}
    for (pl, cfg, gpu), v in sorted(tempos.items()):
        if 0 not in v or 1 not in v:
            continue
        chave = "%s/%s" % (pl, cfg)
        compile_cmp.setdefault(chave, {})[gpu] = {
            "epoch_s_raw": v[0], "epoch_s_compiled": v[1],
            "delta_pct": round(100.0 * (v[1] - v[0]) / v[0], 1),
            "compiled_faster": v[1] < v[0]}
    for chave, porgpu in compile_cmp.items():
        votos = [d["compiled_faster"] for d in porgpu.values()]
        compile_cmp[chave]["_verdict"] = (
            "COMPILE_FASTER" if all(votos) else
            "COMPILE_SLOWER" if not any(votos) else "INCONSISTENT")
        compile_cmp[chave]["_gpus_measured"] = len(votos)
    for chave in sorted(compile_cmp):
        v = compile_cmp[chave]
        log.info("  %-46s compile: %-15s (%d placas)", chave, v["_verdict"],
                 v["_gpus_measured"])
    nao_medidas = sorted(k for k in onde if k not in compile_cmp)
    sem_casa = sorted(k for k, v in onde.items() if not v["fits_on"])
    for k in sorted(onde):
        log.info("  %-46s cabe em: %s", k,
                 ", ".join(onde[k]["fits_on"]) or "NENHUMA PLACA")

    portoes = {"GPUS_PROBED": len(placas), "CONFIGURATIONS": len(onde),
               "CONFIGURATIONS_WITH_NO_GPU": len(sem_casa),
               "NO_GPU_LIST": sem_casa,
               "COMPILE_COMPARED": len(compile_cmp),
               "COMPILE_NOT_MEASURED": nao_medidas,
               "SCRIPT08B_OUTPUT_STATUS":
                   "PASS" if not sem_casa else "PASS_WITH_FLAG"}
    m = {"manifest_id": "gpu_fit_%s" % run_id, "run_id": run_id,
         "script": "08B_PROBE_GPU_FIT", "script_version": "2026-08-11.probe.v1",
         "generated": time.strftime("%Y-%m-%dT%H:%M:%S BRT"),
         "producer_sha256": sha(os.path.abspath(__file__)),
         "gpus": placas, "placement": onde, "compile_comparison": compile_cmp,
         "gates": portoes,
         "source_csv": [os.path.basename(c) for c in args.merge],
         "gpu_fit_csv_sha256": sha(fc),
         "SCRIPT08B_OUTPUT_STATUS": portoes["SCRIPT08B_OUTPUT_STATUS"]}
    corpo = hashlib.sha256(json.dumps(m, indent=1, sort_keys=True).encode()).hexdigest()
    m["manifest_sha256"] = corpo
    pm = os.path.join(dest, "manifest.json")
    json.dump(m, open(pm, "w"), indent=1, sort_keys=True)
    json.dump({"run_id": run_id, "script": "08b", "manifest": pm,
               "manifest_sha256": corpo, "manifest_file_sha256": sha(pm),
               "gpu_fit_csv": os.path.relpath(fc, BASE),
               "gpu_fit_csv_sha256": m["gpu_fit_csv_sha256"],
               "gpus": placas, "placement": onde,
               "compile_comparison": compile_cmp,
               "SCRIPT08B_OUTPUT_STATUS": portoes["SCRIPT08B_OUTPUT_STATUS"]},
              open(PONTEIRO, "w"), indent=1, sort_keys=True)
    for k, v in portoes.items():
        log.info("  %-30s %s", k, v)
    log.info("  publicado em %s", dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", nargs="+", metavar="CSV",
                    help="junta csv de varias placas e publica o ponteiro")
    ap.add_argument("--only", nargs="+", metavar="PLANS/CONFIG",
                    help="sonda so estes pares, para ensaio")
    ap.add_argument("--compile", action="store_true",
                    help="aplica torch.compile como o nnUNetTrainer faz; e o "
                         "caminho REAL do treino em CUDA na 2.5.2")
    ap.add_argument("--time", action="store_true",
                    help="cronometra o tempo de iteracao e deriva o tempo de "
                         "epoca e de fold desta placa")
    ap.add_argument("--plans-dir", metavar="DIR",
                    help="pasta com dataset.json e *Plans.json, para sondar uma "
                         "placa que ainda nao recebeu os dados preprocessados. O "
                         "hash de cada plano fica no CSV e o modo --merge exige "
                         "que todas as placas tenham sondado o mesmo plano.")
    args = ap.parse_args()

    if args.merge:
        return modo_merge(args)

    import torch
    _exigir(torch.cuda.is_available(),
            "sem CUDA nesta maquina: a sondagem nao se faz em CPU nem em MPS, "
            "porque o que se mede e memoria de GPU")
    dispositivo = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(dispositivo)
    total_gb = round(props.total_memory / 2**30, 2)
    host = socket.gethostname()

    run_id = "s08b_%s_%d" % (time.strftime("%Y%m%dT%H%M%S"), os.getpid())
    log.info("Script 08b — sondagem de VRAM | run_id=%s", run_id)
    log.info("  %s · %s · %.2f GB · torch %s · cuda %s",
             host, props.name, total_gb, torch.__version__, torch.version.cuda)

    if args.plans_dir:
        dir_dataset, run08 = args.plans_dir, "VIA_--plans-dir"
        _exigir(os.path.isfile(os.path.join(dir_dataset, "dataset.json")),
                "sem dataset.json em %s" % dir_dataset)
        log.info("  planos passados a mao: %s", dir_dataset)
    else:
        dir_dataset, run08 = _preprocessed_do_08()
        log.info("  planos do Script 08 run %s", run08)
    itens = _configuracoes(dir_dataset)
    if args.only:
        alvo = set(args.only)
        itens = [i for i in itens
                 if "%s/%s" % (i["plans_name"], i["configuration"]) in alvo]
        _exigir(itens, "nenhuma configuracao corresponde a %s" % sorted(alvo))
    log.info("  %d configuracoes a sondar", len(itens))

    linhas = []
    for i in itens:
        reg = sondar(i, dispositivo, cronometrar=args.time, compilar=args.compile)
        reg.update({"host": host, "gpu_name": props.name, "gpu_total_gb": total_gb})
        linhas.append(reg)
        log.info("  %-26s %-20s %s pico %5.2f GB  %-5s  epoca %7s s  fold %6s h  %s",
                 reg["plans_name"], reg["configuration"],
                 "compilado" if args.compile else "cru      ",
                 reg["peak_alloc_gb"], reg["status"], reg["epoch_s"] or "-",
                 reg["fold_1000ep_h"] or "-", reg["detalhe"][:40])

    dest = os.path.join(RUNS, run_id)
    os.makedirs(dest, exist_ok=False)
    fc = os.path.join(dest, "gpu_fit_%s%s.csv" % (host, "_compiled" if args.compile else ""))
    _escrever(fc, linhas)
    cabem = sum(1 for r in linhas if r["status"] == "FITS")
    json.dump({"run_id": run_id, "host": host, "gpu_name": props.name,
               "gpu_total_gb": total_gb, "torch": torch.__version__,
               "cuda": torch.version.cuda, "python": platform.python_version(),
               "nnunet_preprocessed_run": run08,
               "producer_sha256": sha(os.path.abspath(__file__)),
               "generated": time.strftime("%Y-%m-%dT%H:%M:%S BRT"),
               "configurations": len(linhas), "fits": cabem,
               "csv_sha256": sha(fc)},
              open(os.path.join(dest, "probe.json"), "w"), indent=1, sort_keys=True)
    log.info("  %d de %d configuracoes cabem nesta placa", cabem, len(linhas))
    log.info("  escrito em %s", fc)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
