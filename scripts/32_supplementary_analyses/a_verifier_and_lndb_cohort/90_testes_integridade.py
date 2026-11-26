#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    90_testes_integridade.py

File type:
    Script

Purpose:
    Integrity tests of the inputs and outputs of the analyses V01, V02, V03 and V09.
    Pipeline position: Stage 32 (supplementary analyses).

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
    - MANIFESTO_SAIDAS.json
    - V03_sensibilidade_coorte_expandida.csv
    - inventario_entradas.json

Outputs:
    - V01_controlos.csv
    - V02_chaves_particoes_validacao.json
    - V02_controlo_c14.csv
    - V09_validacao.json

Dependencies:
    - Python standard library only

Execution:
    cd <project_dir> && python scripts/32_supplementary_analyses/a_verifier_and_lndb_cohort/90_testes_integridade.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import json, glob, os, sys
from comum import *
inv = json.load(open(out("manifests", "inventario_entradas.json")))["artefactos"]
T = []
for a in inv:
    if a["sha256"]: T.append(dict(teste=f"entrada inalterada: {a['chave']}", ok=sha(a["caminho"]) == a["sha256"]))
T.append(dict(teste="sem ficheiros de falha", ok=not glob.glob(out("outputs", "audits", "*FALHA*"))))
import csv
T.append(dict(teste="V01: todos os controlos OK", ok=all(r["ok"] == "True" for r in csv.DictReader(open(out("outputs", "audits", "V01_controlos.csv"))))))
v2 = json.load(open(out("outputs", "audits", "V02_chaves_particoes_validacao.json")))
T.append(dict(teste="V02: porta aprovada", ok=not v2["porta"]["falhas"]))
T.append(dict(teste="V02: c14 reproduzido (|dif| < 1e-12)", ok=all(float(r["dif_abs"]) < 1e-12 for r in csv.DictReader(open(out("outputs", "audits", "V02_controlo_c14.csv"))))))
v9 = json.load(open(out("outputs", "audits", "V09_validacao.json"))); T.append(dict(teste="V09: modelo congelado reproduz D2 e D4 coerente", ok=not v9["falhas"]))
T.append(dict(teste="V03: coorte expandida não gerada (não estimável)", ok=not os.path.exists(out("outputs", "tables", "V03_sensibilidade_coorte_expandida.csv"))))
files = sorted(p for p in glob.glob(OUT + "/**/*", recursive=True) if os.path.isfile(p) and "MANIFESTO_SAIDAS" not in p and "__pycache__" not in p)
dump(dict(ambiente=env(), testes=T, ficheiros={os.path.relpath(p, OUT): sha(p) for p in files}), "manifests", "MANIFESTO_SAIDAS.json")
for t in T: print(("OK   " if t["ok"] else "FALHA"), t["teste"])
sys.exit(0 if all(t["ok"] for t in T) else 1)
