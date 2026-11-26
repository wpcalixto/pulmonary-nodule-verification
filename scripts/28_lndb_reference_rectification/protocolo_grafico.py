#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    protocolo_grafico.py

File type:
    Module

Purpose:
    Shared plotting helpers (number formatting, figure registration) driven by the visual-identity YAML file.
    Pipeline position: Stage 28 (LNDb N1 reference rectification and re-evaluation).

Repository:
    pulmonary_nodule_verification

Version:
    v1.1.0

Date:
    2026-11-26

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
    Calixto, W. P., Camargo, T. F. O., Pacheco, V. M. G., Martins, M. S. R., & Coimbra, A. P. (2026). Reproducibility package for multiscale morphology-informed post-detection verification of deep learning pulmonary nodule candidates in computed tomography: code, trained nnU-Net models, cohort splits and analysis data (v1.1.0) [Data set and software]. Zenodo. https://doi.org/10.5281/zenodo.23674672

Related publication:
    Multiscale morphology-informed post-detection verification of deep learning pulmonary nodule candidates in computed tomography: false-positive reduction, feature repeatability and external validation (manuscript; DOI not yet available)

Inputs:
    - .registro.json
    - identidade_visual_figuras.yaml

Outputs:
    - Tables written to the stage output directory (see docs/post_v1_stages.md).

Dependencies:
    - matplotlib
    - yaml

Execution:
    cd <project_dir> && python scripts/28_lndb_reference_rectification/protocolo_grafico.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import json
import os

import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter

def virgula(x, casas=2):
    s = f"{x:.{casas}f}"
    return s

def formatador_virgula(casas=2):
    return FuncFormatter(lambda x, _pos: virgula(x, casas))

def marcar(linha, configuracao):
    """Identifica a série desenhada com o nome da configuração."""
    for l in (linha if isinstance(linha, (list, tuple)) else [linha]):
        l.set_gid(configuracao)

def _hex(c):
    try:
        return mcolors.to_hex(c).upper()
    except ValueError:
        return str(c)

def registrar(fig, caminho_pdf, papel):
    """Grava `<pdf>.registro.json` com o que a figura contém de facto."""
    series = {}
    for l in fig.findobj(Line2D):
        g = l.get_gid()
        if not g:
            continue
        m = l.get_marker()
        series.setdefault(g, []).append({
            "marker": m if isinstance(m, str) else [list(map(float, v)) for v in m],
            "edgecolor": _hex(l.get_markeredgecolor()),
            "facecolor": _hex(l.get_markerfacecolor()) if l.get_markerfacecolor() != "none" else "none",
        })
    textos = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    titulos = [ax.get_title() for ax in fig.axes if ax.get_title()]
    sup = fig._suptitle.get_text() if getattr(fig, "_suptitle", None) else ""
    reg = {
        "pdf": os.path.basename(caminho_pdf),
        "papel": papel,
        "largura_in": float(fig.get_figwidth()),
        "altura_in": float(fig.get_figheight()),
        "tamanhos_fonte_pt": sorted({float(t.get_fontsize()) for t in textos}),
        "n_textos": len(textos),
        "titulos_internos": titulos + ([sup] if sup else []),
        "series_por_configuracao": series,
    }
    with open(caminho_pdf + ".registro.json", "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=1, ensure_ascii=False)
    return reg

_IDENT_TAB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "config", "identidade_visual_figuras.yaml")

def _tabelas():
    import yaml
    return yaml.safe_load(open(_IDENT_TAB, encoding="utf-8")).get("tables", {})

def dec_tab(s):
    """Separador decimal das TABELAS (secao `tables` do YAML), independente das figuras."""
    return s.replace(".", _tabelas().get("decimal_separator", ","))

def dec_tab_math(s):
    """Idem, em modo matematico: virgula como {,} para nao criar espaco."""
    return s.replace(".", "{,}" if _tabelas().get("decimal_separator", ",") == "," else ".")

def nome_cfg(c):
    """Nome de configuracao para figuras e tabelas: '_' -> '-' (3D_FREM -> 3D-FREM)."""
    return c.replace("_", "-")

def rotulo_resposta_tabela(ident, f):
    t = _tabelas(); d = ident["morphological_features"]["features"][f]
    base = t["feature_labels_pt"][f] if t.get("language", "pt") == "pt" else d["label"]
    return base + (f" [{d['unit']}]" if d["unit"] else "")

def _corpo_tabular(tex):
    i = tex.index("\\begin{tabular}") + len("\\begin{tabular}")
    d = 0
    for j in range(i, len(tex)):
        d += {"{": 1, "}": -1}.get(tex[j], 0)
        if d == 0:
            break
    return j + 1, tex.index("\\end{tabular}", j)

def preenche_molde(gerado, nome_molde):
    """Molde editorial congelado (texto atual do manuscrito) preenchido com as linhas
    calculadas pelo gerador. So as linhas com '&' vem do gerador; as marcas \tnote{x}
    ausentes no molde sao retiradas. Falha se a estrutura das linhas divergir."""
    import re
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moldes", nome_molde)
    molde = open(base, encoding="utf-8").read()
    am, bm = _corpo_tabular(molde); ag, bg = _corpo_tabular(gerado)
    sm, sg = molde[am:bm].split("\\\\"), gerado[ag:bg].split("\\\\")
    if len(sm) != len(sg):
        raise SystemExit(f"STOP: {nome_molde}: {len(sm)} linhas no molde, {len(sg)} geradas")
    out = []
    for x, y in zip(sm, sg):
        if "&" not in x:
            out.append(x); continue
        if "&" not in y:
            raise SystemExit(f"STOP: {nome_molde}: estrutura de linhas divergente")
        lx, ly = x.split("\n"), y.split("\n")
        kx = next(k for k, l in enumerate(lx) if "&" in l)
        ky = next(k for k, l in enumerate(ly) if "&" in l)
        nucleo = "\n".join(ly[ky:])
        for t in set(re.findall(r"\\tnote\{[a-z]\}", nucleo)):
            if t not in x:
                nucleo = nucleo.replace(t, "")
        out.append("\n".join(lx[:kx] + [nucleo]))
    return molde[:am] + "\\\\".join(out) + molde[bm:]

def grava_legenda(grupos, caminho_pdf, papel, gap_in=0.16, pad_in=0.06, **kw):
    """Grava um PDF so com a legenda (um ou mais grupos empilhados), sem eixos.
    O tamanho e medido com a mesma fonte LaTeX dos paineis, para que, incluido na
    mesma escala dos paineis, o texto tenha o mesmo tamanho efetivo."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 8))
    legs = [fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0, 1), frameon=False,
                       borderaxespad=0.0, borderpad=0.0, **kw) for h in grupos]
    fig.draw_without_rendering()
    tam = [(lg.get_window_extent().width / fig.dpi, lg.get_window_extent().height / fig.dpi) for lg in legs]
    W = max(w for w, _ in tam) + 2 * pad_in
    H = sum(h for _, h in tam) + gap_in * (len(tam) - 1) + 2 * pad_in
    fig.set_size_inches(W, H)
    y = H - pad_in
    for lg, (w, h) in zip(legs, tam):
        lg.set_bbox_to_anchor((pad_in / W, y / H), transform=fig.transFigure)
        y -= h + gap_in
    fig.savefig(caminho_pdf, format="pdf")
    reg = registrar(fig, caminho_pdf, papel)
    plt.close(fig)
    return reg
