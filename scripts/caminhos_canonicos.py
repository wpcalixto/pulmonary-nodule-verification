#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    caminhos_canonicos.py

File type:
    Module

Purpose:
    Shared module that resolves the canonical directory layout relative to the project base directory.

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
import json
import hashlib

from pulmonary_base import BASE

RUNS_DIR          = os.path.join(BASE, "runs")
CURRENT_CROPS     = os.path.join(RUNS_DIR, "CURRENT.json")
CURRENT_ANNOT     = os.path.join(RUNS_DIR, "CURRENT_LIDC_ANNOTATIONS.json")

BANCOS = ("lidc_idri", "lndb", "rider")


class FalhaEntrada(RuntimeError):
    """FAIL_INPUT: entrada ausente ou incoerente. Nunca vira saída vazia."""


def _exigir(cond, msg):
    if not cond:
        raise FalhaEntrada(msg)


def sha256(caminho, bloco=1 << 20):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for p in iter(lambda: f.read(bloco), b""):
            h.update(p)
    return h.hexdigest()


def _reancorar(abs_registado, rotulo, run_id_esperado):
    """Resolve um caminho absoluto registado noutra máquina.

    Mesma regra do Script 08 (`_reancorar`) e do Script 10, para que todos os
    consumidores do Script 06 resolvam o mesmo caminho da mesma maneira.
    """
    if os.path.isfile(abs_registado):
        return abs_registado, "ABSOLUTE_PATH_AS_REGISTERED"
    partes = os.path.normpath(abs_registado).split(os.sep)
    n = partes.count(run_id_esperado)
    _exigir(n == 1,
            "%s: parte relativa AMBIGUA — run_id %r aparece %d vezes em %s"
            % (rotulo, run_id_esperado, n, abs_registado))
    i = partes.index(run_id_esperado)
    _exigir(i >= 1 and partes[i - 1] == "runs",
            "%s: run_id nao esta sob 'runs/' em %s" % (rotulo, abs_registado))
    resolvido = os.path.join(BASE, os.path.join(*partes[i - 1:]))
    dentro = (os.path.commonpath([os.path.realpath(os.path.dirname(resolvido)),
                                  os.path.realpath(BASE)]) == os.path.realpath(BASE))
    _exigir(dentro, "%s: caminho resolvido ESCAPA de PULMONARY_BASE: %s"
            % (rotulo, resolvido))
    _exigir(os.path.isfile(resolvido),
            "%s: manifesto ausente no registado (%s) e no reancorado (%s)"
            % (rotulo, abs_registado, resolvido))
    return resolvido, "REANCHORED_TO_PULMONARY_BASE"


def _carregar(ponteiro, script, chave_manifesto, rotulo):
    _exigir(os.path.isfile(ponteiro),
            "%s ausente: %s. O Script %s ainda nao publicou."
            % (rotulo, ponteiro, script))
    p = json.load(open(ponteiro))
    _exigir(p.get("script") in (script, "%s_LIDC_ANNOTATIONS" % script,
                                "04_BUILD_LIDC_ANNOTATIONS"),
            "%s: ponteiro.script=%r inesperado" % (rotulo, p.get("script")))
    reg = p.get(chave_manifesto) or p.get("manifest")
    _exigir(reg, "%s: ponteiro sem caminho de manifesto" % rotulo)
    if not os.path.isabs(reg):
        reg = os.path.join(BASE, reg)
    run_id = p.get("run_id") or ""
    if os.path.isfile(reg):
        man, modo = reg, "ABSOLUTE_PATH_AS_REGISTERED"
    else:
        man, modo = _reancorar(reg, rotulo, run_id)
    esperado = p.get("manifest_sha256") or p.get("manifest_file_sha256")
    if esperado:
        h_ficheiro = sha256(man)
        try:
            from pulmonary_base import hash_corpo_manifesto
            h_corpo = hash_corpo_manifesto(json.load(open(man)))
        except Exception:
            h_corpo = None
        _exigir(esperado in (h_ficheiro, h_corpo),
                "%s: manifesto com hash diferente do ponteiro\n"
                "  ponteiro: %s\n  ficheiro: %s\n  corpo   : %s"
                % (rotulo, esperado, h_ficheiro, h_corpo))
    return p, man, modo


_CACHE = {}


def raiz_recortes():
    """Diretório com `lidc_idri/`, `lndb/`, `rider/` e `lung_masks/`.

    A raiz vem do caminho REANCORADO, não do que o ponteiro registou: noutra
    máquina o registado não existe, e os recortes estão ao lado do manifesto
    efetivamente resolvido. Mesma regra do Script 08, linha ~470.
    """
    if "recortes" in _CACHE:
        return _CACHE["recortes"]
    p, man, modo = _carregar(CURRENT_CROPS, "06", "manifest", "recortes do 06")
    m = json.load(open(man))
    g = m.get("gates", m)
    _exigir(g.get("SCRIPT06_REFERENCE_GUIDED_CROP") == "NO",
            "Script 06 com recorte guiado por rotulo — vazamento de referencia")
    raiz = os.path.dirname(man)
    for b in BANCOS:
        _exigir(os.path.isdir(os.path.join(raiz, b)),
                "banco %r ausente nos recortes do 06: %s" % (b, raiz))
    _CACHE["recortes"] = raiz
    return raiz


def ct_de(raiz, banco, pid):
    """`{raiz}/{banco}/images/{pid}/ct.nii.gz` — a CT recortada.

    Repare no nível `images/`: a hierarquia do Script 06 é
    `{banco}/images/{caso}/ct.nii.gz` e `{banco}/labels/{caso}/*.nii.gz`. No
    RIDER o caso é `{pid}_{1,2}` — o exame, não o doente.
    """
    _exigir(banco in BANCOS, "banco desconhecido: %r" % banco)
    p = os.path.join(raiz, banco, "images", pid, "ct.nii.gz")
    _exigir(os.path.isfile(p), "CT ausente para %s/%s: %s" % (banco, pid, p))
    return p


def label_de(raiz, banco, pid, nome="consensus.nii.gz"):
    """`{raiz}/{banco}/labels/{pid}/consensus.nii.gz` — a referência RECORTADA.

    **É esta e não a do Script 04.** A distinção não é de gosto: as predições
    são feitas sobre a CT recortada, e o consenso do Script 04 está no volume
    inteiro. Verificado em 09/09/2026 no LIDC-IDRI-0001 — CT recortada e
    consenso recortado dão ambos `size=(391, 302, 113)`; o do Script 04 tem a
    dimensão original. Comparar predição com referência de geometria diferente
    dá IoU zero em todo o lado e parece um modelo que não detecta nada.

    O Script 04 continua a ser a fonte da coorte e da regra de consenso; o que
    se lê aqui é o mesmo objecto transportado para o espaço do recorte.
    """
    _exigir(banco in BANCOS, "banco desconhecido: %r" % banco)
    p = os.path.join(raiz, banco, "labels", pid, nome)
    _exigir(os.path.isfile(p),
            "referencia recortada ausente para %s/%s: %s" % (banco, pid, p))
    return p


def tem_label(raiz, banco, pid, nome="consensus.nii.gz"):
    """Existe referência recortada? Sem interromper — para distinguir «caso sem
    referência» de «ficheiro em falta», que o protocolo manda separar."""
    return os.path.isfile(os.path.join(raiz, banco, "labels", pid, nome))


def mascara_pulmonar_de(raiz, banco, pid, recortada=True):
    """`{raiz}/lung_masks/{banco}/{pid}/lung_mask[_crop].nii.gz`.

    O Script 18 lia de `BASE/lung_masks/rider`, que não existe. A hierarquia
    real é a que o 06 produziu, sob a raiz do seu run, e cada caso tem duas
    máscaras: `lung_mask.nii.gz` no volume original e `lung_mask_crop.nii.gz`
    no espaço do recorte. Por omissão devolve-se a **recortada**, porque é a
    que casa com a CT que os scripts 10+ consomem.
    """
    _exigir(banco in BANCOS, "banco desconhecido: %r" % banco)
    d = os.path.join(raiz, "lung_masks", banco, pid)
    _exigir(os.path.isdir(d),
            "mascara pulmonar ausente para %s/%s: %s" % (banco, pid, d))
    preferida = "lung_mask_crop.nii.gz" if recortada else "lung_mask.nii.gz"
    p = os.path.join(d, preferida)
    _exigir(os.path.isfile(p), "sem %s em %s" % (preferida, d))
    return p


def raiz_consensos():
    """Diretório `annotations/consensus` publicado pelo Script 04."""
    if "consensos" in _CACHE:
        return _CACHE["consensos"]
    p, man, modo = _carregar(CURRENT_ANNOT, "04", "annotations_manifest",
                             "anotacoes do 04")
    rel = p.get("consensus_dir", os.path.join("annotations", "consensus"))
    raiz = rel if os.path.isabs(rel) else os.path.join(BASE, rel)
    if not os.path.isdir(raiz):
        alt = os.path.dirname(raiz)
        _exigir(os.path.isdir(alt),
                "raiz dos consensos ausente: %s (nem %s)" % (raiz, alt))
        raiz = alt
    _CACHE["consensos"] = raiz
    return raiz


def consenso_de(banco, pid, raiz=None):
    """`{raiz}/{banco}/{pid}/consensus.nii.gz` — a referência canónica.

    Para o LNDb é o que o protocolo já impõe ao Script 22:
    `annotations/consensus/lndb/{scan_id}/consensus.nii.gz`.
    """
    _exigir(banco in ("lidc", "lndb"), "banco de consenso desconhecido: %r" % banco)
    raiz = raiz or raiz_consensos()
    p = os.path.join(raiz, banco, pid, "consensus.nii.gz")
    _exigir(os.path.isfile(p),
            "consenso ausente para %s/%s: %s" % (banco, pid, p))
    return p


def tem_consenso(banco, pid, raiz=None):
    """Existe referência para este caso? Sem interromper.

    Serve para distinguir «caso sem referência» de «ficheiro em falta» — a
    distinção que o protocolo exige e que os ramos antigos apagavam. Quem chama
    tem de registar a diferença, não de a converter em negativo.
    """
    raiz = raiz or raiz_consensos()
    return os.path.isfile(os.path.join(raiz, banco, pid, "consensus.nii.gz"))
