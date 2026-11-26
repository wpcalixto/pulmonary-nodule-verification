#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    V04_reconcilia_722_686.py

File type:
    Script

Purpose:
    Reconciles the N1 lesion counts of the reference scan with those of the 224-image cohort.
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
    - F_respostas.csv
    - LNDb-*.npz
    - exames_236.csv
    - input_manifest.csv

Outputs:
    - Tables written to the stage output directory (see docs/post_v1_stages.md).

Dependencies:
    - numpy
    - pandas
    - scipy

Execution:
    cd <project_dir> && python scripts/32_supplementary_analyses/d_interobserver_matching_and_partition/V04_reconcilia_722_686.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import sys, glob, json, csv, hashlib, ast
import numpy as np, scipy.ndimage as nd, pandas as pd
O, OUT = sys.argv[1], sys.argv[2]; R = O.rsplit("/", 1)[0]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
src = open(O + "/scripts/F_respostas.py").read(); assert sha(O + "/scripts/F_respostas.py").startswith("0c309e1b")
EX13 = ast.literal_eval([n for n in ast.parse(src).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "EX13"][0].value)
S22 = R + "/pulmonar_pos_oof_prep/script22_final/s22_20260924T084718_826234/input_manifest.csv"
EX211 = sorted(r["scan_id"] for r in csv.DictReader(open(S22)))
CAN = set(EX211) | set(EX13)
S6 = nd.generate_binary_structure(3, 1)
todas = []
for f in sorted(glob.glob(O + "/referencias/LNDb-*.npz")):
    e = f.split("/")[-1][:-4]; Z = np.load(f); n1 = Z["N1"]
    for cid in range(1, int(n1.max()) + 1):
        m = n1 == cid
        if not m.any(): continue
        todas.append(dict(exame=e, id=cid, FindingID=int(Z["N1_findings"][cid - 1]), comp6=int(nd.label(m, structure=S6)[1]), canonica=e in CAN))
T = pd.DataFrame(todas)
FR = pd.read_csv(O + "/saidas/F_respostas.csv", low_memory=False); FN1 = FR[FR.tipo == "N1"]
kT = set(zip(T.exame, T.FindingID)); kF = set(zip(FN1.exame, FN1.FindingID.astype(int)))
kC = set(zip(T[T.canonica].exame, T[T.canonica].FindingID))
fora = T[~T.canonica]
man = O + "/referencias/exames_236.csv"
M = pd.read_csv(man); cab = dict(lesoes_N1_236=int(M.lesoes_N1.sum()), lesoes_N1_canonicas=int(M[M.lndb_id.isin(CAN)].lesoes_N1.sum()), exames=len(M))
res = dict(npz=int(len(glob.glob(O + "/referencias/LNDb-*.npz"))), lesoes_varridas=len(T), exames_canonicos=len(CAN), EX211=len(EX211), EX13=len(EX13),
    lesoes_canonicas=int(T.canonica.sum()), linhas_N1_F_respostas=len(FN1), chaves_canonicas_iguais_F_respostas=kC == kF,
    exames_fora_da_coorte=sorted(fora.exame.unique().tolist()), n_exames_fora=int(fora.exame.nunique()), lesoes_fora=len(fora),
    lesoes_211=int(T.exame.isin(EX211).sum()),
    LNDb0031_les4_canonica=bool(((T.exame == "LNDb-0031") & (T.id == 4) & T.canonica).any()), LNDb0031_em_EX211="LNDb-0031" in EX211,
    multicomponente_canonica=T[T.canonica & (T.comp6 > 1)][["exame", "id", "FindingID", "comp6"]].to_dict("records"),
    multicomponente_todas=T[T.comp6 > 1][["exame", "id", "FindingID", "comp6"]].to_dict("records"),
    manifesto_exames_236=cab, sha=dict(F_respostas=sha(O + "/saidas/F_respostas.csv"), input_manifest_s22=sha(S22), exames_236=sha(man)))
json.dump(res, open(OUT, "w"), indent=1); print(json.dumps(res, indent=1))
