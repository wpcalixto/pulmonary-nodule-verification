#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    c09_congela_ligacao.py

File type:
    Script

Purpose:
    Freezes the confirmed record-to-marker link with its SHA-256.
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
    -   c08_ligacao_congelada.csv

    - c08_ligacao_congelada.SHA256

Outputs:
    - c07_confirmacao_cega_RESPONDIDA_20261109.csv
    - c08_ligacao_congelada.csv

Dependencies:
    - pandas

Execution:
    cd <project_dir> && python scripts/30_radiological_assessment/c09_congela_ligacao.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import re, hashlib, datetime
import pandas as pd
O = "resultados/radiological_reanalysis"
src = f"{O}/c07_confirmacao_cega_RESPONDIDA_20261109.csv"
d = pd.read_csv(src, dtype={"exame": str})
assert len(d) == 58 and d.confirmacao.notna().all()
fin, alv = [], []
for r in d.itertuples():
    c = r.confirmacao
    if r.n == 33:
        f, a = "agrupado ao registo 32", ""
    elif c.startswith("confirmada como múltipla"):
        f, a = "múltipla", "/".join("m" + x for x in re.findall(r"m(\d+)", c.split(";")[0]))
    elif c.startswith("confirmada: ref"):
        f, a = "confirmada", "ref " + re.match(r"confirmada: ref (\d+)", c).group(1)
    elif c.startswith("ambígua mantida"):
        f, a = "ambígua", "/".join("m" + x for x in re.findall(r"m(\d+)", c))
    elif c.startswith("ok"):
        f = r.ligacao
        a = "" if r.alvo == "--" else (r.alvo.replace("marcador ", "").replace("/", "/m").join(["m", ""]) if r.alvo.startswith("marcador") else r.alvo)
    else:
        raise ValueError(c)
    fin.append(f); alv.append(a)
d["ligacao_final"], d["alvos_finais"] = fin, alv
d["ligado"] = d.ligacao_final.isin(["única", "provável", "confirmada", "múltipla"])
out = f"{O}/c08_ligacao_congelada.csv"
d.to_csv(out, index=False)
h = hashlib.sha256(open(out, "rb").read()).hexdigest()
hs = hashlib.sha256(open(src, "rb").read()).hexdigest()
agora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z").strip()
open(f"{O}/c08_ligacao_congelada.SHA256", "w").write(f"{h}  c08_ligacao_congelada.csv\n{hs}  c07_confirmacao_cega_RESPONDIDA_20261109.csv\n# congelado em {agora} (antes de qualquer junção com desfechos)\n")
print(h[:16], agora)
print(d.ligacao_final.value_counts().to_string()); print("ligados:", d.ligado.sum(), "(+ registo 33 agrupado)")
print(d[["n", "exame", "registo", "classe", "ligacao_final", "alvos_finais"]].to_string(index=False))
