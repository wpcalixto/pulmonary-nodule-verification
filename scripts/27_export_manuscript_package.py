#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    27_export_manuscript_package.py

File type:
    Script

Purpose:
    Assembles the manuscript results package (main and supplementary results, LaTeX tables, captions and figures).

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
    PULMONARY_BASE=<project_dir> python scripts/27_export_manuscript_package.py

Portability notes:
    All paths are resolved relative to the project base directory given by the PULMONARY_BASE environment variable. No machine-specific path is required.

Reproducibility notes:
    Environment pinned in requirements.txt and environment/. Seeds and frozen parameters are recorded in the run manifests under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) are not included and must be obtained from their original repositories.
"""

import os
import csv
import json
import shutil
import logging
from datetime import datetime
from pathlib import Path

from pulmonary_base import BASE
ANALYSIS_DIR = os.path.join(BASE, "analysis")
FIGDATA_DIR  = os.path.join(BASE, "figdata")
FIGURES_DIR  = os.path.join(BASE, "figures")
TABLES_DIR   = os.path.join(BASE, "tables")
REPORTS_DIR  = os.path.join(BASE, "reports")
LOGS_DIR     = os.path.join(BASE, "logs")
CONFIG_DIR   = os.path.join(BASE, "config")
SCRIPTS_DIR  = os.path.join(os.path.dirname(BASE), "Scripts", "novos")
for d in [REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_versions():
    ver_path = os.path.join(LOGS_DIR, "package_versions.txt")
    if not os.path.isfile(ver_path):
        return "package_versions.txt não encontrado."
    with open(ver_path) as f:
        return f.read()


def load_json(path, default=None):
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return default or {}


def csv_to_md_table(path, max_rows=30):
    if not os.path.isfile(path):
        return f"*Arquivo não encontrado: {os.path.basename(path)}*\n"
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            rows.append(row)
    if not rows:
        return "*Sem dados.*\n"
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines) + "\n"


def list_figures():
    if not os.path.isdir(FIGURES_DIR):
        return "*Pasta figures/ não encontrada.*\n"
    figs = sorted(Path(FIGURES_DIR).glob("*.pdf")) + sorted(Path(FIGURES_DIR).glob("*.png"))
    if not figs:
        return "*Nenhuma figura encontrada.*\n"
    return "\n".join(f"- `{f.name}`" for f in figs) + "\n"


def list_latex_tables():
    if not os.path.isdir(TABLES_DIR):
        return "*Pasta tables/ não encontrada.*\n"
    tex_files = sorted(Path(TABLES_DIR).glob("*.tex"))
    if not tex_files:
        return "*Nenhuma tabela LaTeX encontrada.*\n"
    return "\n".join(f"- `{f.name}`" for f in tex_files) + "\n"


def list_figure_captions():
    if not os.path.isdir(FIGURES_DIR):
        return "*Pasta figures/ não encontrada.*\n"
    figs = sorted(Path(FIGURES_DIR).glob("*.pdf")) + sorted(Path(FIGURES_DIR).glob("*.png"))
    if not figs:
        return "*Nenhuma figura encontrada.*\n"
    lines = []
    for f in figs:
        stem = f.stem
        lines.append(f"**{f.name}** — [inserir legenda para {stem}]")
    return "\n\n".join(lines) + "\n"


def write_manuscript_results():
    out = os.path.join(REPORTS_DIR, "manuscript_results.md")
    with open(out, "w") as f:
        f.write(f"# Resultados do Manuscrito\n\n")
        f.write(f"Gerado automaticamente em {datetime.now().isoformat()}\n\n")

        f.write("## BD1 — Detecção e Segmentação\n\n")
        f.write(csv_to_md_table(os.path.join(ANALYSIS_DIR, "bd1_detection_metrics.csv")))

        f.write("\n## BD1 — Segmentação (Dice, HD3D)\n\n")
        f.write(csv_to_md_table(os.path.join(ANALYSIS_DIR, "bd1_segmentation_metrics.csv")))

        f.write("\n## BD1 — Estratificação por Tamanho\n\n")
        f.write(csv_to_md_table(os.path.join(ANALYSIS_DIR, "bd1_size_stratified_metrics.csv")))

        f.write("\n## RIDER — Repetibilidade (BA, ICC, SCCR)\n\n")
        f.write(csv_to_md_table(os.path.join(FIGDATA_DIR, "rider_icc.csv")))
        f.write("\n")
        f.write(csv_to_md_table(os.path.join(FIGDATA_DIR, "rider_sccr.csv")))

        f.write("\n## LNDb — Avaliação Externa\n\n")
        f.write(csv_to_md_table(os.path.join(FIGDATA_DIR, "lndb_external_metrics.csv")))

        f.write("\n## Filtros Morfológicos — BD1\n\n")
        f.write(csv_to_md_table(os.path.join(FIGDATA_DIR, "filter_bd1_auc.csv")))

        f.write("\n## Tabelas LaTeX (tables/)\n\n")
        f.write(list_latex_tables())

    log.info(f"Resultados: {out}")


def write_supplementary_methods():
    versions = load_versions()
    thresh = load_json(os.path.join(CONFIG_DIR, "filter_verifier_threshold.json"))
    out = os.path.join(REPORTS_DIR, "supplementary_methods.md")
    with open(out, "w") as f:
        f.write("# Métodos Suplementares\n\n")
        f.write("## Versões de Software\n\n")
        f.write("```\n" + versions + "\n```\n\n")
        f.write("## Parâmetros Congelados\n\n")
        f.write("| Parâmetro | Valor |\n|---|---|\n")
        f.write("| τ LIDC-IDRI | 0.60 |\n")
        f.write("| τ LNDb | 0.50 |\n")
        f.write("| Semente (bootstrap, splits) | 27 |\n")
        f.write("| N bootstrap | 2000 |\n")
        f.write("| Casos de desenvolvimento BD1 | 822 |\n")
        f.write("| Casos de avaliação interna BD1 | 158 |\n")
        f.write("| Pacientes RIDER | 31 |\n")
        f.write("| CTs LNDb | 236 |\n\n")
        if thresh.get("trained"):
            f.write("## Verificador Morfológico\n\n")
            f.write(f"- Modelo: {thresh.get('model_label', '?')}\n")
            f.write(f"- Features: {', '.join(thresh.get('features', []))}\n")
            f.write(f"- Limiar congelado: {thresh.get('threshold', '?')}\n")
            f.write(f"- AUROC (CV): {thresh.get('auroc_cv', '?')}\n\n")
        f.write("## Escalas LoG\n\n")
        f.write("σ ∈ {0.8, 1.2, 1.8, 2.7, 4.0, 6.0, 8.0} mm\n\n")
        f.write("## Intervalo HU\n\n")
        f.write("[-1000, 400] HU\n\n")
    log.info(f"Métodos suplementares: {out}")


def write_supplementary_results():
    out = os.path.join(REPORTS_DIR, "supplementary_results.md")
    with open(out, "w") as f:
        f.write("# Resultados Suplementares\n\n")
        f.write(f"Gerado em {datetime.now().isoformat()}\n\n")

        f.write("## Interobservador\n\n")
        f.write(csv_to_md_table(os.path.join(ANALYSIS_DIR, "interobserver_summary.csv")))

        f.write("\n## RIDER — Bland-Altman\n\n")
        f.write(csv_to_md_table(os.path.join(ANALYSIS_DIR, "rider_bland_altman.csv")))

        f.write("\n## RIDER — Filtros\n\n")
        f.write(csv_to_md_table(os.path.join(FIGDATA_DIR, "rider_filter_repeatability.csv")))

        f.write("\n## LNDb — Filtros\n\n")
        f.write(csv_to_md_table(os.path.join(ANALYSIS_DIR, "lndb_filter_group_comparisons.csv")))

        f.write("\n## Figuras geradas\n\n")
        f.write(list_figures())

        f.write("\n## Legendas das Figuras\n\n")
        f.write(list_figure_captions())

    log.info(f"Resultados suplementares: {out}")


def write_computational_reproducibility():
    audit_report = os.path.join(ANALYSIS_DIR, "final_audit_report.md")
    manifest_path = os.path.join(ANALYSIS_DIR, "reproducibility_manifest.json")
    manifest = load_json(manifest_path)
    out = os.path.join(REPORTS_DIR, "computational_reproducibility.md")
    with open(out, "w") as f:
        f.write("# Reprodutibilidade Computacional\n\n")
        f.write(f"Gerado em {datetime.now().isoformat()}\n\n")
        f.write("## Declaração de disponibilidade\n\n")
        f.write("Os scripts de análise estão disponíveis em:\n")
        f.write(f"`{SCRIPTS_DIR}`\n\n")
        f.write("Os dados originais estão disponíveis nos repositórios:\n")
        f.write("- LIDC-IDRI: https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX\n")
        f.write("- RIDER Lung CT: https://doi.org/10.7937/K9/TCIA.2015.U1X8A5NR\n")
        f.write("- LNDb: https://zenodo.org/record/3944464\n\n")
        f.write("## Resultado da auditoria\n\n")
        n_ok   = sum(1 for v in manifest.get("checks", {}).values() if v["status"] == "OK")
        n_fail = sum(1 for v in manifest.get("checks", {}).values() if v["status"] != "OK")
        f.write(f"Verificações OK: {n_ok} | Falhas: {n_fail}\n\n")
        if os.path.isfile(audit_report):
            with open(audit_report) as ar:
                f.write(ar.read())
    log.info(f"Reprodutibilidade: {out}")


if __name__ == "__main__":
    write_manuscript_results()
    write_supplementary_methods()
    write_supplementary_results()
    write_computational_reproducibility()
    log.info("Pacote do manuscrito gerado em reports/")
