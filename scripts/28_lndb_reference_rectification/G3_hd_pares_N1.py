#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    G3_hd_pares_N1.py

File type:
    Script

Purpose:
    Surface Hausdorff distance of the N1 true-positive pairs, with a Dice parity gate.
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
    - Outputs of the preceding pipeline stages (see docs/post_v1_stages.md).

Outputs:
    - E1_objetos_tau010.csv
    - G3_hd_pares_N1.csv

Dependencies:
    - numpy
    - pandas

Execution:
    cd <project_dir> && python scripts/28_lndb_reference_rectification/G3_hd_pares_N1.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import json, sys, numpy as np, pandas as pd
exec(open(__file__.replace("G3_hd_pares_N1.py", "E1_associacao_geometria.py")).read().split("def avaliar(")[0])
OBJ = pd.read_csv(O + "/saidas/E1_objetos_tau010.csv", dtype={"coorte": str}, low_memory=False); OBJ = OBJ[(OBJ.coorte == "211") & (OBJ.referencia == "N1") & (OBJ.fonte == "ref") & (OBJ.status == "TP")]
def um(e):
    oi, size, spx = geo(e); sp_zyx = (spx[2], spx[1], spx[0]); Z0 = np.load(f"{O}/referencias/{e}.npz"); out = []
    for a in ARCH:
        q = OBJ[(OBJ.exame == e) & (OBJ.configuracao == a)]
        if q.empty: continue
        p = PRED211.format(a=a, e=e); pr = sitk.ReadImage(p); lab_c, _ = nd.label(sitk.GetArrayFromImage(pr) > 0, structure=STRUCT_6)
        z0, y0, x0 = encaixe(pr, oi, size); Zs, Ys, Xs = lab_c.shape; pl = np.zeros(Z0["N1"].shape, np.int32); pl[z0:z0 + Zs, y0:y0 + Ys, x0:x0 + Xs] = lab_c
        for r in q.itertuples():
            Pm, Gm = pl == int(r.par), Z0["N1"] == int(r.id); sl = G["recorte"](Pm, Gm); mm = G["metricas_par"](Pm[sl], Gm[sl], sp_zyx)
            out.append(dict(exame=e, configuracao=a, id=int(r.id), dice=mm["dice"], dice_E1=r.dice, hd100=mm["hd100_sup"]))
    return out
if __name__ == "__main__":
    with Pool(7) as pool: R_ = pool.map(um, sorted(OBJ.exame.unique()))
    df = pd.DataFrame([x for r in R_ for x in r]); df.to_csv(O + "/saidas/G3_hd_pares_N1.csv", index=False)
    d = float((df.dice - df.dice_E1).abs().max()); print(len(df), d)
    if d > 1e-12: sys.exit("PORTA G3 FALHOU")
