#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    V01_reproduz_particao.py

File type:
    Script

Purpose:
    Reproduces the D1/D2 partition from the recorded seed and compares it with the archived partition.
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
    - Outputs of the preceding pipeline stages (see docs/post_v1_stages.md).

Outputs:
    - Tables written to the stage output directory (see docs/post_v1_stages.md).

Dependencies:
    - sklearn

Execution:
    cd <project_dir> && python scripts/32_supplementary_analyses/d_interobserver_matching_and_partition/V01_reproduz_particao.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import sys, csv, json, zipfile, hashlib, xml.etree.ElementTree as ET
from sklearn.model_selection import KFold, train_test_split
SPLIT, FOLDS, XLSX = sys.argv[1:4]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
assert sha(SPLIT).startswith("cb212bdfc9623bc0") and sha(FOLDS).startswith("708639690a7bd594") and sha(XLSX).startswith("9860039cbd9af7a6")
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
def ler_planilha(p):
    z = zipfile.ZipFile(p); ss = [t.text or "" for t in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(NS + "t")]; out = {}
    for r in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(NS + "row"):
        v = []
        for c in r:
            e = c.find(NS + "v"); x = e.text if e is not None else ""
            if c.get("t") == "s" and x: x = ss[int(x)]
            v.append(x)
        if v and str(v[0]).startswith("LIDC-IDRI"): out[v[0]] = {"total": v[1], "ge3mm": v[2]}
    return out
can = {r["patient_id"]: r["split"] for r in csv.DictReader(open(SPLIT))}
universo = sorted(can); cont = ler_planilha(XLSX)
estrato = {p: "%s_%s" % (cont[p]["total"], cont[p]["ge3mm"]) for p in universo}
freq = {}
for e in estrato.values(): freq[e] = freq.get(e, 0) + 1
unicos = sorted(p for p in universo if freq[estrato[p]] == 1); resto = sorted(p for p in universo if freq[estrato[p]] > 1)
tr, va = train_test_split([[p] for p in resto], test_size=0.2, stratify=[estrato[p] for p in resto], random_state=27)
dev = sorted({x[0] for x in tr} | set(unicos)); interno = sorted({x[0] for x in va})
ok_split = set(dev) == {p for p, s in can.items() if s == "development"} and set(interno) == {p for p, s in can.items() if s == "internal"}
kf = KFold(n_splits=5, shuffle=True, random_state=27)
rep = {"fold_%d" % i: sorted(dev[j] for j in iva) for i, (itr, iva) in enumerate(kf.split(dev))}
fj = json.load(open(FOLDS)); txt = json.dumps(fj)[:0]
def val(f):
    v = fj.get("folds", fj).get(f, {}); return sorted(v.get("val", v.get("validation", [])))
ok_folds = all(rep[f] == val(f) for f in rep)
print(json.dumps(dict(universo=len(universo), estratos=len(freq), unicos=len(unicos), D1=len(dev), D2=len(interno),
    particao_reproduzida=ok_split, dobras_reproduzidas=ok_folds, tamanhos_dobras=[len(v) for v in rep.values()],
    chaves_folds=list(fj.keys())[:8])))
