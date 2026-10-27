#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    08a_calibrate_collapse_monitor.py

File type:
    Script

Purpose:
    Calibrates the training-collapse monitor rule against archived training logs (no GPU, no image access).

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
    PULMONARY_BASE=<project_dir> python scripts/08a_calibrate_collapse_monitor.py

Portability notes:
    All paths are resolved relative to the project base directory given by the PULMONARY_BASE environment variable. No machine-specific path is required.

Reproducibility notes:
    Environment pinned in requirements.txt and environment/. Seeds and frozen parameters are recorded in the run manifests under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) are not included and must be obtained from their original repositories.
"""

import os
import re
import csv
import json
import time
import glob
import hashlib
import logging
import argparse
import collections

BASE = os.environ["PULMONARY_BASE"]
RUNS = os.path.join(BASE, "runs")
PONTEIRO = os.path.join(RUNS, "CURRENT_COLLAPSE_RULE.json")

LOGS_ROOT_PADRAO = os.path.join(
    BASE, "evidence", "quarantine_custodia_2026-08-06", "metadados",
    "script06_legacy_descendants", "20260731T163920", "payload", "models",
    "Dataset001_PulmonaryNodules")

GRELHA_ARM = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
GRELHA_FRAC = (0.02, 0.05, 0.10, 0.20, 0.30, 0.50)
GRELHA_K = (5, 10, 20, 30, 50, 100, 200, 300, 400, 500)
GRELHA_MIN_EPOCH = (20, 50, 100, 150, 200)
GRELHA_E_ARM = (20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500)

TOL_EMA = 1e-4
BURNIN_CEGO = 100

RE_EPOCA = re.compile(r":\s*Epoch\s+(\d+)\s*$")
RE_LR = re.compile(r":\s*Current learning rate:\s*([-\d.eE+naif]+)\s*$")
RE_TRAIN = re.compile(r":\s*train_loss\s+([-\d.eE+naif]+)\s*$")
RE_VAL = re.compile(r":\s*val_loss\s+([-\d.eE+naif]+)\s*$")
RE_DICE = re.compile(r":\s*Pseudo dice\s*\[(.*)\]\s*$")
RE_BEST = re.compile(r"New best EMA pseudo Dice:\s*([-\d.eE+naif]+)")
RE_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan|inf|-inf")
RE_ENVOLVIDO = re.compile(r"float\d*\(([^)]*)\)")
RE_NOME = re.compile(r"training_log_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)\.txt$")

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


def _f(txt):
    """Float tolerante: devolve None onde o log escreveu nan/inf/lixo."""
    try:
        v = float(txt)
    except (TypeError, ValueError):
        return None
    return v if v == v and abs(v) != float("inf") else None


def _bruto(txt):
    """Como _f, mas distingue nan/inf de ilegível — o 09 aborta nesses."""
    try:
        v = float(txt)
    except (TypeError, ValueError):
        return None, False
    nao_finito = (v != v) or abs(v) == float("inf")
    return (None if nao_finito else v), nao_finito


def instante_do_nome(caminho):
    """Ordena os logs de um fold. O nome traz a data de arranque do processo."""
    m = RE_NOME.search(os.path.basename(caminho))
    _exigir(m, "nome de log fora do padrao: %s" % caminho)
    a, mes, d, h, mi, s = (int(x) for x in m.groups())
    return (a, mes, d, h, mi, s)


def _valores_do_dice(interior):
    """
    Números de dentro de `Pseudo dice [...]`. Se vierem embrulhados em
    `np.float32(...)`, lê-se só o embrulhado — o nome do tipo traz dígitos.
    """
    envolvidos = RE_ENVOLVIDO.findall(interior)
    brutos = envolvidos if envolvidos else RE_NUM.findall(interior)
    return [_f(x) for x in brutos]


def ler_log(caminho):
    """Um log -> lista de épocas por ordem de aparição."""
    epocas, actual = [], None
    with open(caminho, "r", errors="replace") as fh:
        for linha in fh:
            linha = linha.rstrip("\n")
            m = RE_EPOCA.search(linha)
            if m:
                actual = {"epoch": int(m.group(1)), "lr": None,
                          "train_loss": None, "val_loss": None,
                          "dice": None, "best_ema_impresso": None,
                          "nao_finito": False}
                epocas.append(actual)
                continue
            if actual is None:
                continue
            m = RE_LR.search(linha)
            if m:
                actual["lr"] = _f(m.group(1))
                continue
            achou = False
            for chave, rx in (("train_loss", RE_TRAIN), ("val_loss", RE_VAL)):
                m = rx.search(linha)
                if m:
                    v, nf = _bruto(m.group(1))
                    actual[chave] = v
                    actual["nao_finito"] = actual["nao_finito"] or nf
                    achou = True
                    break
            if achou:
                continue
            m = RE_DICE.search(linha)
            if m:
                vals = _valores_do_dice(m.group(1))
                validos = [v for v in vals if v is not None]
                if len(validos) < len(vals):
                    actual["nao_finito"] = True
                actual["dice"] = (sum(validos) / len(validos)) if validos else None
                continue
            m = RE_BEST.search(linha)
            if m:
                actual["best_ema_impresso"] = _f(m.group(1))
    nao_finitos = sum(1 for e in epocas if e["nao_finito"])
    return [e for e in epocas if e["dice"] is not None], nao_finitos


def juntar_fold(caminhos):
    """
    Funde as retomas de um fold. Por ordem cronológica, cada log descarta o que
    já estava registado a partir da PRIMEIRA época que ele traz — retoma em N
    preserva 0..N−1, recomeço do zero substitui tudo.
    """
    fundido, usados = {}, []
    for c in sorted(caminhos, key=instante_do_nome):
        eps, _ = ler_log(c)
        if not eps:
            continue
        inicio = eps[0]["epoch"]
        for k in [k for k in fundido if k >= inicio]:
            del fundido[k]
        eps[0]["seg_start"] = True
        for e in eps:
            fundido[e["epoch"]] = e
        usados.append(os.path.basename(c))
    serie = [fundido[k] for k in sorted(fundido)]
    return serie, sum(1 for e in serie if e["nao_finito"]), usados


def reconstruir_ema(serie):
    """
    EMA pela definição do nnU-Net, com detecção de quebra de linhagem, e
    verificação contra os valores impressos.

    Devolve (serie_da_linhagem_final, desvio_maximo, n_comparacoes, quebras).
    O desvio é medido SÓ na linhagem devolvida — o que ficou para trás pertence a
    outra tentativa e não se julga por valores que não são dele.
    """
    def _passar(sub, i0=0):
        """
        EMA sobre TODO `sub`; máximo corrente e verificação só a partir de `i0`.
        Separar as duas coisas é o que dá sentido ao aquecimento: as épocas de
        aquecimento continuam a alimentar a recorrência, apenas não são julgadas
        nem entram no máximo.

        A verificação é contra o EMA CORRENTE, não contra o máximo: o nnU-Net
        imprime «New best» no momento em que o melhor passa a ser o corrente, de
        modo que o valor impresso é o EMA daquela época.
        """
        ema, mx, desvio, n_cmp = None, None, 0.0, 0
        for i, e in enumerate(sub):
            impresso = e["best_ema_impresso"]
            if i == 0:
                ema = impresso if impresso is not None else e["dice"]
            else:
                ema = 0.9 * ema + 0.1 * e["dice"]
            e["ema"] = ema
            if i < i0:
                continue
            mx = ema if mx is None else max(mx, ema)
            e["ema_max"] = mx
            if impresso is not None and i > 0:
                desvio = max(desvio, abs(ema - impresso))
                n_cmp += 1
        return desvio, n_cmp

    _passar(serie)
    corte, quebras = 0, []
    for i in range(1, len(serie)):
        e, ant = serie[i], serie[i - 1]
        salto = e["epoch"] != ant["epoch"] + 1
        impresso = e["best_ema_impresso"]
        discorda = (e.get("seg_start") and impresso is not None
                    and abs(e["ema_max"] - impresso) > TOL_EMA)
        if salto or discorda:
            quebras.append({"epoch": e["epoch"],
                            "motivo": "EPOCH_JUMP" if salto else "EMA_MISMATCH",
                            "seedable": impresso is not None})
            corte = i
    if serie and serie[0]["epoch"] != 0:
        quebras.insert(0, {"epoch": serie[0]["epoch"], "motivo": "NO_EPOCH_ZERO",
                           "seedable": serie[0]["best_ema_impresso"] is not None})
    linhagem = serie[corte:]
    cego = bool(linhagem) and linhagem[0]["epoch"] != 0 \
        and linhagem[0]["best_ema_impresso"] is None
    i0 = BURNIN_CEGO if cego else 0
    if cego:
        quebras.append({"epoch": linhagem[0]["epoch"],
                        "motivo": "BLIND_SEED_BURNIN", "seedable": False})
    desvio, n_cmp = _passar(linhagem, i0)
    return linhagem[i0:], desvio, n_cmp, quebras


def limiar_por_maior_intervalo(valores):
    """
    O limiar que separa desfecho degenerado de saudável NÃO é escolhido: é o
    ponto médio do MAIOR intervalo vazio na distribuição dos EMA finais. Se a
    distribuição for bimodal, é ali que ela se parte; se não for, o intervalo
    sai pequeno e o portão abaixo reprova.
    """
    v = sorted(valores)
    _exigir(len(v) >= 2, "folds insuficientes para medir o limiar")
    maior, corte, esq, dir_ = -1.0, None, None, None
    for a, b in zip(v, v[1:]):
        if b - a > maior:
            maior, corte, esq, dir_ = b - a, (a + b) / 2.0, a, b
    return corte, maior, esq, dir_


def classificar(serie, corte):
    ema_final = serie[-1]["ema"]
    ema_max = max(e["ema"] for e in serie)
    if ema_final >= corte:
        return "HEALTHY", ema_final, ema_max
    return ("NEVER_LEARNED" if ema_max < corte else "COLLAPSED"), ema_final, ema_max


def disparo_colapso(serie, arm, frac, k, min_epoch):
    """Época do disparo, ou None. Só olha para o passado de cada época."""
    seguidas = 0
    for e in serie:
        if e["epoch"] < min_epoch or e["ema_max"] < arm:
            seguidas = 0
            continue
        seguidas = seguidas + 1 if e["ema"] <= frac * e["ema_max"] else 0
        if seguidas >= k:
            return e["epoch"]
    return None


def disparo_sem_aprendizagem(serie, arm, e_arm):
    """Dispara à época `e_arm` se o fold ainda não deu sinal de aprender."""
    vistas = [e for e in serie if e["epoch"] <= e_arm]
    if not vistas or vistas[-1]["epoch"] < e_arm:
        return None
    return e_arm if vistas[-1]["ema_max"] < arm else None


def evidencia_colapso(serie, arm, frac, min_epoch):
    """Maior corrida de épocas que satisfaz a condição de queda, sem exigir k."""
    run = melhor = 0
    for e in serie:
        if e["epoch"] < min_epoch or e["ema_max"] < arm:
            run = 0
            continue
        run = run + 1 if e["ema"] <= frac * e["ema_max"] else 0
        melhor = max(melhor, run)
    return melhor


def evidencia_sem_aprendizagem(serie, arm):
    """Época em que o fold atinge `arm`. None se nunca atinge."""
    return next((e["epoch"] for e in serie if e["ema"] >= arm), None)


def margem(folds, tipo, **kw):
    """
    Distância entre o parâmetro de paciência e o fold SAUDÁVEL que mais perto
    chegou de disparar. Uma regra com margem 1 está calibrada em cima de um caso
    único e mata o próximo treino que se lhe pareça.
    """
    sadios = [d["serie"] for d in folds.values() if d["verdade"] == "HEALTHY"]
    if not sadios:
        return None
    if tipo == "COLLAPSE":
        pior = max(evidencia_colapso(s, kw["arm"], kw["frac"], kw["min_epoch"])
                   for s in sadios)
        return kw["k"] - pior
    atinge = [evidencia_sem_aprendizagem(s, kw["arm"]) for s in sadios]
    if any(a is None for a in atinge):
        return None
    return kw["e_arm"] - max(atinge)


def avaliar(folds, disparar):
    """Sensibilidade, especificidade e épocas poupadas de um detector."""
    fp, tp, alvo, poupadas = [], [], 0, []
    for nome, d in folds.items():
        ep = disparar(d["serie"])
        if d["verdade"] == "HEALTHY":
            if ep is not None:
                fp.append((nome, ep))
        else:
            alvo += 1
            if ep is not None:
                tp.append((nome, ep))
                poupadas.append(d["serie"][-1]["epoch"] - ep)
    poupadas.sort()
    mediana = poupadas[len(poupadas) // 2] if poupadas else 0
    return {"falsos_positivos": len(fp), "detectados": len(tp), "alvo": alvo,
            "sensibilidade": (len(tp) / alvo) if alvo else 0.0,
            "epocas_poupadas_mediana": mediana,
            "exemplos_fp": fp[:5]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-root", default=LOGS_ROOT_PADRAO)
    ap.add_argument("--plan-only", action="store_true",
                    help="mede e relata sem escrever nada")
    ap.add_argument("--supersede", action="store_true",
                    help="substitui deliberadamente uma regra ja publicada")
    args = ap.parse_args()

    run_id = "s08a_%s_%d" % (time.strftime("%Y%m%dT%H%M%S"), os.getpid())
    dest = os.path.join(RUNS, run_id)
    _exigir(not os.path.exists(dest), "destino ja existe")

    if os.path.isfile(PONTEIRO) and not args.supersede and not args.plan_only:
        _p = json.load(open(PONTEIRO))
        _exigir(False, "ja existe regra publicada: %s. Nao se recalibra por "
                       "cima. Use --supersede sabendo que invalida a "
                       "comparabilidade dos treinos ja feitos." % _p.get("run_id"))

    log.info("Script 08a — calibracao do monitor de colapso | run_id=%s", run_id)
    _exigir(os.path.isdir(args.logs_root), "arquivo de logs inexistente: %s"
            % args.logs_root)

    padrao = os.path.join(args.logs_root, "*", "fold_*", "training_log_*.txt")
    ficheiros = sorted(glob.glob(padrao))
    _exigir(ficheiros, "nenhum log em %s" % padrao)
    por_fold = collections.defaultdict(list)
    for c in ficheiros:
        fold_dir = os.path.dirname(c)
        combo = os.path.basename(os.path.dirname(fold_dir))
        por_fold[(combo, os.path.basename(fold_dir))].append(c)
    log.info("  %d ficheiros de log · %d folds", len(ficheiros), len(por_fold))

    folds, desvio_max, n_cmp_total, nao_finitos_total, sem_serie = {}, 0.0, 0, 0, []
    for chave, caminhos in sorted(por_fold.items()):
        serie, nf, usados = juntar_fold(caminhos)
        nao_finitos_total += nf
        if not serie:
            sem_serie.append("%s/%s" % chave)
            continue
        serie, dev, n, quebras = reconstruir_ema(serie)
        desvio_max, n_cmp_total = max(desvio_max, dev), n_cmp_total + n
        partes = chave[0].split("__")
        folds["%s/%s" % chave] = {
            "serie": serie, "n_logs": len(caminhos), "logs_usados": usados,
            "trainer": partes[0],
            "plans": partes[1] if len(partes) > 1 else "",
            "config": partes[2] if len(partes) > 2 else "",
            "fold": chave[1].replace("fold_", ""),
            "nao_finitos": sum(1 for e in serie if e["nao_finito"]),
            "quebras": quebras,
            "historia_completa": serie[0]["epoch"] == 0,
        }
    _exigir(folds, "nenhum fold com serie de epocas utilizavel")
    completos = {k: v for k, v in folds.items() if v["historia_completa"]}
    log.info("  %d folds com serie · %d sem epocas · %d epocas ao todo",
             len(folds), len(sem_serie), sum(len(d["serie"]) for d in folds.values()))
    log.info("  %d com historia completa desde a epoca 0 · %d apenas desfecho",
             len(completos), len(folds) - len(completos))
    for nome, d in sorted(folds.items()):
        if not d["historia_completa"]:
            log.info("    OUTCOME_ONLY  %s  arranca na epoca %d  (%s)",
                     nome, d["serie"][0]["epoch"],
                     ", ".join("%s@%d" % (q["motivo"], q["epoch"]) for q in d["quebras"]))
    _exigir(completos, "nenhum fold com historia completa: nada para calibrar")

    log.info("  EMA reconstruido vs impresso: desvio maximo %.3e em %d comparacoes",
             desvio_max, n_cmp_total)
    _exigir(n_cmp_total > 0, "nenhum 'New best EMA' impresso: reconstrucao nao verificavel")
    _exigir(desvio_max <= TOL_EMA,
            "reconstrucao do EMA nao confere (desvio %.3e > %.3e): a calibracao "
            "assentaria num EMA que nao e o do nnU-Net" % (desvio_max, TOL_EMA))

    finais = [d["serie"][-1]["ema"] for d in folds.values()]
    corte, intervalo, esq, dir_ = limiar_por_maior_intervalo(finais)
    log.info("  maior intervalo vazio nos EMA finais: [%.4f , %.4f] -> corte %.4f",
             esq, dir_, corte)
    for nome, d in folds.items():
        d["verdade"], d["ema_final"], d["ema_max"] = classificar(d["serie"], corte)
    contagem = collections.Counter(d["verdade"] for d in folds.values())
    cont_cal = collections.Counter(d["verdade"] for d in completos.values())
    for k in ("HEALTHY", "COLLAPSED", "NEVER_LEARNED"):
        log.info("    %-14s %d  (dos quais %d calibram)", k,
                 contagem.get(k, 0), cont_cal.get(k, 0))
    _exigir(cont_cal.get("HEALTHY", 0) > 0 and
            cont_cal.get("COLLAPSED", 0) + cont_cal.get("NEVER_LEARNED", 0) > 0,
            "os folds de historia completa nao tem as duas classes de desfecho: "
            "nada a calibrar")

    grelha = []
    for arm in GRELHA_ARM:
        for frac in GRELHA_FRAC:
            for k in GRELHA_K:
                for me in GRELHA_MIN_EPOCH:
                    r = avaliar(completos, lambda s, a=arm, f=frac, kk=k, m=me:
                                disparo_colapso(s, a, f, kk, m))
                    r.update({"regra": "COLLAPSE", "arm": arm, "frac": frac,
                              "k": k, "min_epoch": me, "e_arm": "",
                              "margem": margem(completos, "COLLAPSE", arm=arm,
                                               frac=frac, k=k, min_epoch=me)})
                    grelha.append(r)
    for arm in GRELHA_ARM:
        for ea in GRELHA_E_ARM:
            r = avaliar(completos, lambda s, a=arm, e=ea: disparo_sem_aprendizagem(s, a, e))
            r.update({"regra": "NO_LEARNING", "arm": arm, "frac": "", "k": "",
                      "min_epoch": "", "e_arm": ea,
                      "margem": margem(completos, "NO_LEARNING", arm=arm, e_arm=ea)})
            grelha.append(r)

    def escolher(regra):
        """
        Especificidade de 100% é requisito de admissão. Entre as admitidas:
        primeiro sensibilidade; depois MARGEM — de duas regras que detectam o
        mesmo, fica a que passa mais longe do fold saudável mais próximo; só
        então épocas poupadas.
        """
        cand = [g for g in grelha if g["regra"] == regra
                and g["falsos_positivos"] == 0 and g["margem"] is not None]
        if not cand:
            return None
        return max(cand, key=lambda g: (g["sensibilidade"], g["margem"],
                                        g["epocas_poupadas_mediana"]))

    sel_col, sel_nl = escolher("COLLAPSE"), escolher("NO_LEARNING")
    for etiqueta, s in (("COLLAPSE", sel_col), ("NO_LEARNING", sel_nl)):
        if s is None:
            log.info("  %-12s SEM REGRA com especificidade 100%%", etiqueta)
        else:
            log.info("  %-12s arm=%s frac=%s k=%s min_epoch=%s e_arm=%s -> "
                     "%d/%d degenerados, 0 falsos positivos, margem %s epocas, "
                     "mediana %d epocas poupadas",
                     etiqueta, s["arm"], s["frac"], s["k"], s["min_epoch"],
                     s["e_arm"], s["detectados"], s["alvo"], s["margem"],
                     s["epocas_poupadas_mediana"])

    def conjunta(serie):
        a = disparo_colapso(serie, sel_col["arm"], sel_col["frac"], sel_col["k"],
                            sel_col["min_epoch"]) if sel_col else None
        b = disparo_sem_aprendizagem(serie, sel_nl["arm"], sel_nl["e_arm"]) if sel_nl else None
        return min([x for x in (a, b) if x is not None], default=None)

    res_conj = avaliar(completos, conjunta)
    log.info("  CONJUNTA     %d/%d degenerados, %d falsos positivos, mediana %d epocas poupadas",
             res_conj["detectados"], res_conj["alvo"], res_conj["falsos_positivos"],
             res_conj["epocas_poupadas_mediana"])

    portoes = {
        "LOG_FILES_READ": len(ficheiros),
        "FOLDS_WITH_SERIES": len(folds),
        "FOLDS_CALIBRATING": len(completos),
        "FOLDS_OUTCOME_ONLY": len(folds) - len(completos),
        "FOLDS_WITHOUT_SERIES": len(sem_serie),
        "EPOCHS_TOTAL": sum(len(d["serie"]) for d in folds.values()),
        "EMA_RECONSTRUCTION_MAX_DEV": desvio_max,
        "EMA_RECONSTRUCTION_COMPARISONS": n_cmp_total,
        "OUTCOME_THRESHOLD": corte,
        "OUTCOME_THRESHOLD_GAP": intervalo,
        "OUTCOME_THRESHOLD_METHOD": ("ponto medio do maior intervalo vazio na "
                                     "distribuicao dos EMA finais; nao escolhido a mao"),
        "OUTCOME_HEALTHY": contagem.get("HEALTHY", 0),
        "OUTCOME_COLLAPSED": contagem.get("COLLAPSED", 0),
        "OUTCOME_NEVER_LEARNED": contagem.get("NEVER_LEARNED", 0),
        "NONFINITE_EPOCHS_IN_ARCHIVE": nao_finitos_total,
        "SELECTED_SENSITIVITY": res_conj["sensibilidade"],
        "SELECTED_FALSE_POSITIVES": res_conj["falsos_positivos"],
        "SELECTED_EPOCHS_SAVED_MEDIAN": res_conj["epocas_poupadas_mediana"],
        "RULE_FROZEN": "YES",
        "SCRIPT08A_OUTPUT_STATUS": "PASS",
    }
    _exigir(sel_col is not None or sel_nl is not None,
            "nenhuma regra atinge especificidade de 100%: nao ha o que congelar")
    _exigir(res_conj["falsos_positivos"] == 0,
            "a regra conjunta dispara em fold saudavel: nao se publica")

    if args.plan_only:
        log.info("  PLAN_STATUS = PASS · nada escrito")
        return

    os.makedirs(dest, exist_ok=False)

    fs = os.path.join(dest, "fold_series.csv")
    with open(fs, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold_key", "epoch", "lr", "train_loss", "val_loss",
                    "pseudo_dice", "ema", "ema_max", "nonfinite"])
        for nome, d in sorted(folds.items()):
            for e in d["serie"]:
                w.writerow([nome, e["epoch"], e["lr"], e["train_loss"],
                            e["val_loss"], "%.6f" % e["dice"], "%.6f" % e["ema"],
                            "%.6f" % e["ema_max"], int(e["nao_finito"])])

    fr = os.path.join(dest, "folds_summary.csv")
    with open(fr, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold_key", "trainer", "plans", "config", "fold", "n_logs",
                    "n_epochs", "last_epoch", "ema_final", "ema_max",
                    "epoch_of_ema_max", "outcome", "trigger_epoch",
                    "epochs_saved", "nonfinite_epochs"])
        for nome, d in sorted(folds.items()):
            ep = conjunta(d["serie"])
            pico = max(d["serie"], key=lambda e: e["ema"])["epoch"]
            w.writerow([nome, d["trainer"], d["plans"], d["config"], d["fold"],
                        d["n_logs"], len(d["serie"]), d["serie"][-1]["epoch"],
                        "%.6f" % d["ema_final"], "%.6f" % d["ema_max"], pico,
                        d["verdade"], "" if ep is None else ep,
                        "" if ep is None else d["serie"][-1]["epoch"] - ep,
                        d["nao_finitos"]])

    fg = os.path.join(dest, "grid.csv")
    with open(fg, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["regra", "arm", "frac", "k", "min_epoch", "e_arm",
                    "falsos_positivos", "detectados", "alvo", "sensibilidade",
                    "margem_epocas", "epocas_poupadas_mediana"])
        for g in grelha:
            w.writerow([g["regra"], g["arm"], g["frac"], g["k"], g["min_epoch"],
                        g["e_arm"], g["falsos_positivos"], g["detectados"],
                        g["alvo"], "%.4f" % g["sensibilidade"],
                        "" if g["margem"] is None else g["margem"],
                        g["epocas_poupadas_mediana"]])

    regra = {
        "rule_id": "collapse_monitor_%s" % run_id,
        "calibrated_from": os.path.relpath(args.logs_root, BASE),
        "calibrated_on_folds": len(completos),
        "outcome_threshold": corte,
        "nonfinite": {"action": "ABORT_IMMEDIATELY",
                      "applies_to": ["train_loss", "val_loss", "pseudo_dice"],
                      "calibrated": False,
                      "note": ("NaN/Inf nao se calibra: aborta na epoca em que "
                               "aparece. Ocorrencias no arquivo: %d"
                               % nao_finitos_total)},
        "collapse": None if sel_col is None else {
            "arm": sel_col["arm"], "frac_of_max": sel_col["frac"],
            "consecutive_epochs": sel_col["k"], "min_epoch": sel_col["min_epoch"],
            "margin_epochs": sel_col["margem"],
            "definition": ("apos min_epoch e depois de ema_max >= arm, abortar "
                           "quando ema <= frac_of_max * ema_max durante "
                           "consecutive_epochs epocas seguidas")},
        "no_learning": None if sel_nl is None else {
            "arm": sel_nl["arm"], "check_at_epoch": sel_nl["e_arm"],
            "margin_epochs": sel_nl["margem"],
            "definition": ("abortar a epoca check_at_epoch se ema_max < arm")},
        "on_trigger": {
            "emit": "TRAINING_COLLAPSE",
            "fields": ["epoch", "best_ema_before", "ema_at_abort", "checkpoint",
                       "rule_id", "which_rule"],
            "not_for": ("seleccao de melhor epoca; a regra so aborta, nunca "
                        "escolhe checkpoint")},
        "measured": {"sensitivity": res_conj["sensibilidade"],
                     "false_positives": res_conj["falsos_positivos"],
                     "detected": res_conj["detectados"], "target": res_conj["alvo"],
                     "epochs_saved_median": res_conj["epocas_poupadas_mediana"]},
        "caveat": ("calibrado em logs da coorte contaminada de julho de 2026 "
                   "(113/821 alvos vazios). Separa colapso de saudavel; nao "
                   "prediz valores de EMA da coorte corrigida."),
    }
    fj = os.path.join(dest, "collapse_rule.json")
    json.dump(regra, open(fj, "w"), indent=1, sort_keys=True)

    m = {"manifest_id": "collapse_rule_%s" % run_id, "run_id": run_id,
         "script": "08A_CALIBRATE_COLLAPSE_MONITOR",
         "script_version": "2026-08-11.calib.v1",
         "generated": time.strftime("%Y-%m-%dT%H:%M:%S BRT"),
         "producer_sha256": sha(os.path.abspath(__file__)),
         "logs_root": os.path.relpath(args.logs_root, BASE),
         "log_files": [os.path.relpath(c, BASE) for c in ficheiros],
         "folds_without_series": sem_serie,
         "gates": portoes,
         "SCRIPT08A_OUTPUT_STATUS": "PASS",
         "collapse_rule_sha256": sha(fj),
         "fold_series_sha256": sha(fs),
         "folds_summary_sha256": sha(fr),
         "grid_sha256": sha(fg)}
    corpo = hashlib.sha256(json.dumps(m, indent=1, sort_keys=True).encode()).hexdigest()
    m["manifest_sha256"] = corpo
    pm = os.path.join(dest, "manifest.json")
    json.dump(m, open(pm, "w"), indent=1, sort_keys=True)

    p = {"run_id": run_id, "script": "08a", "script_version": m["script_version"],
         "manifest": pm, "manifest_sha256": corpo, "manifest_file_sha256": sha(pm),
         "collapse_rule": os.path.relpath(fj, BASE),
         "collapse_rule_sha256": m["collapse_rule_sha256"],
         "folds_summary": os.path.relpath(fr, BASE),
         "folds_summary_sha256": m["folds_summary_sha256"],
         "calibrated_on_folds": len(completos),
         "sensitivity": res_conj["sensibilidade"],
         "false_positives": res_conj["falsos_positivos"],
         "SCRIPT08A_OUTPUT_STATUS": "PASS"}
    json.dump(p, open(PONTEIRO, "w"), indent=1, sort_keys=True)

    for k, v in portoes.items():
        log.info("  %-32s %s", k, v)
    log.info("  publicado em %s", dest)


if __name__ == "__main__":
    main()
