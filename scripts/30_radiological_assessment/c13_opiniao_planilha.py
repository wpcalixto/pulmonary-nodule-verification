#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    c13_opiniao_planilha.py

File type:
    Script

Purpose:
    Reads the radiologist's interpretation of each record from the primary spreadsheet.
    Pipeline position: Stage 30 (radiological assessment linkage).

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
    - Outputs of the preceding pipeline stages (see docs/post_v1_stages.md).

Outputs:
    - c08_ligacao_congelada.csv
    - c10_opiniao_radiologista.csv
    - c13_opiniao_planilha.csv

Dependencies:
    - pandas

Execution:
    cd <project_dir> && python scripts/30_radiological_assessment/c13_opiniao_planilha.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import pandas as pd
O = "resultados/radiological_reanalysis"
NN = "não nódulo"
T = {25: (NN, "vaso", "pequenas imagens roxas no pulmão direito… Parecem vasos pulmonares"),
     26: (NN, "vaso", "região justapleural direita. Não existe… pode mimetizar vaso"),
     27: (NN, "via aérea", "diminutas esparsas… preenchimento de vias aéreas distais"),
     28: (NN, "cicatriz/atelectasia", "cicatriz pulmonar… estrias atelectásicas"),
     29: (NN, "cicatriz/atelectasia", "justapleural à direita. Atelectasia"),
     30: (NN, "vaso ou via aérea", "pode mimetizar vaso ou preenchimento de via aérea"),
     31: (NN, "vaso ou via aérea", "podem mimetizar vasos pulmonares ou preenchimento de via aérea distal"),
     32: (NN, "cicatriz/atelectasia", "cicatriz pulmonar (estria atelectásica)… não se trata de nódulo"),
     33: (NN, "cicatriz/atelectasia", "opacidades adjacentes à área de cicatriz"),
     34: ("nódulo real fora da referência", "", "Acho que este nódulo é real"),
     35: ("linfonodo", "", "provavelmente correspondendo a um linfonodo"),
     36: (NN, "cicatriz/atelectasia", "área de atelectasia, região onde o pulmão expande menos"),
     37: (NN, "vaso", "Ínfimo micronódulo… Vaso pulmonar"),
     38: (NN, "vaso ou via aérea", "preenchimento de via aérea distal; também… vaso pulmonar"),
     39: (NN, "vaso", "Esse nódulo não existe. Vaso pulmonar"),
     40: ("nódulo real fora da referência", "", "Este nódulo existe"),
     41: ("nódulo real fora da referência", "", "Nódulo semissólido… Eu concordei com o modelo"),
     42: (NN, "vaso", "Corresponde a um vaso pulmonar"),
     43: ("linfonodo", "", "Provavelmente corresponde a um diminuto linfonodo"),
     44: ("nódulo real fora da referência", "", "eu concordei com o modelo e não com o radiologista"),
     45: (NN, "vaso", "trata-se de um vaso pulmonar"),
     46: ("linfonodo", "", "aspecto nodular, mas é linfonodo (morfologia triangular)"),
     47: (NN, "vaso ou via aérea", "Podem mimetizar vasos pulmonares ou preenchimento de via aérea distal"),
     48: (NN, "vaso", "Vaso pulmonar"), 49: (NN, "vaso", "Vaso pulmonar"),
     50: ("nódulo", "", "imagem nodular… justapleural, que o radiologista marcou"),
     51: ("nódulo", "", "nódulo espiculado no segmento basal do LIE"),
     52: ("nódulo", "", "micronódulo no ápice esquerdo"), 53: ("nódulo", "", "Outro micronódulo superior esquerdo"),
     54: ("nódulo", "", "Nodulo pulmonar no ápice pulmonar esquerdo"),
     55: ("nódulo", "", "nódulo de paredes regulares… justadiafragmática direita"),
     56: ("linfonodo", "", "Provavelmente correspondendo a um linfonodo"),
     57: ("nódulo", "", "Opacidade nodular no lobo médio, junto a uma atelectasia"),
     58: (NN, "vaso", "opacidade… composta por vasos pulmonares. O modelo classificou como nódulo")}
L = pd.read_csv(f"{O}/c08_ligacao_congelada.csv", dtype={"exame": str})
L["opiniao"] = [T[n][0] if n in T else "nódulo" for n in L.n]
L["tipo_nao_nodulo"] = [T[n][1] if n in T else "" for n in L.n]
L["trecho_planilha"] = [T[n][2] if n in T else "nota de delineamento (" + L.descricao[i].split("nota ")[-1] + ")" for i, n in enumerate(L.n)]
old = pd.read_csv(f"{O}/c10_opiniao_radiologista.csv", dtype={"exame": str})
L = L.merge(old.rename(columns={"opiniao_radiologista": "S20_atual"}), on=["exame", "registo", "classe"], how="left", validate="1:1")
def m(s): return "nódulo" if s.startswith("Nódulo") else (NN if s.startswith("Não nódulo") else "inconclusivo/não especificado")
def m2(o): return {"nódulo real fora da referência": "nódulo", "linfonodo": "linfonodo"}.get(o, o)
L["difere_S20"] = [m(a) != m2(b) for a, b in zip(L.S20_atual, L.opiniao)]
L[["n", "exame", "registo", "classe", "opiniao", "tipo_nao_nodulo", "trecho_planilha", "S20_atual", "difere_S20"]].to_csv(f"{O}/c13_opiniao_planilha.csv", index=False)
print(L[L.difere_S20][["n", "exame", "registo", "classe", "S20_atual", "opiniao"]].to_string(index=False))
print(L.groupby(["classe", "opiniao"]).size().to_string())
