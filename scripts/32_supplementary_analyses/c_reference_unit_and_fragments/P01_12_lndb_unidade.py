#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    P01_12_lndb_unidade.py

File type:
    Script

Purpose:
    Diagnosis of the reference unit in LNDb (N1).
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
    - P01_12_lndb_contagens.csv
    - P01_12_lndb_lesoes.csv
    - P01_12_lndb_portas.csv

Dependencies:
    - SimpleITK
    - numpy
    - pandas
    - scipy

Execution:
    cd <project_dir> && python scripts/32_supplementary_analyses/c_reference_unit_and_fragments/P01_12_lndb_unidade.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import sys, importlib.util, numpy as np, pandas as pd, scipy.ndimage as nd
from multiprocessing import Pool
E1P = "resultados/retificacao_LNDb_20261111/scripts/E1_associacao_geometria.py"
sp = importlib.util.spec_from_file_location("E1", E1P); E1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(E1)
OUT = sys.argv[1]; S6 = E1.STRUCT_6
def um(e):
    Z = np.load(f"{E1.O}/referencias/{e}.npz"); L = Z["N1"].astype(np.int32); fin = Z["N1_findings"]; n = int(L.max())
    les, comp = [], np.zeros(L.shape, np.int32); k = 0
    for i in range(1, n + 1):
        m = L == i; lab, nc = nd.label(m, structure=S6); vs = np.bincount(lab.ravel())[1:] if nc else []
        les.append(dict(exame=e, lesao=i, FindingID=int(fin[i - 1]), vox=int(m.sum()), n_componentes=nc, vox_maior=int(max(vs)) if nc else 0))
        for c in range(1, nc + 1): k += 1; comp[lab == c] = k
    porta = dict(exame=e, n_lesoes=n, n_findings=len(fin), rotulos_contiguos=sorted(set(np.unique(L)) - {0}) == list(range(1, n + 1)), n_comp=k)
    C = []
    if k != n:
        oi, size, spx = E1.geo(e); spz = (spx[2], spx[1], spx[0])
        for a in E1.ARCH:
            import os, SimpleITK as sitk
            p = E1.PRED211.format(a=a, e=e)
            if not os.path.exists(p): p = E1.PRED211.format(a=[x for x in E1.ARCH if x != a][0], e=e)
            assert E1.sha(p) == E1.man[e][f"prediction_{a}_sha256"]
            pr = sitk.ReadImage(p); lc, npd = nd.label(sitk.GetArrayFromImage(pr) > 0, structure=S6); z0, y0, x0 = E1.encaixe(pr, oi, size)
            pl = np.zeros(L.shape, np.int32); pl[z0:z0 + lc.shape[0], y0:y0 + lc.shape[1], x0:x0 + lc.shape[2]] = lc
            for rot, G, ng in (("lesao", L, n), ("componente", comp, k)):
                c, _ = E1.avaliar(G, ng, pl, npd, Z["ignorar_N1"], a, e, rot, e, "211", spz, False); C += c
    print(e, n, k, flush=True); return les, porta, C
if __name__ == "__main__":
    with Pool(7) as pool: R = pool.map(um, E1.EX211, chunksize=1)
    pd.DataFrame([x for a, _, _ in R for x in a]).to_csv(OUT + "/P01_12_lndb_lesoes.csv", index=False)
    pd.DataFrame([b for _, b, _ in R]).to_csv(OUT + "/P01_12_lndb_portas.csv", index=False)
    C = [x for _, _, c in R for x in c]
    if C: pd.DataFrame(C).to_csv(OUT + "/P01_12_lndb_contagens.csv", index=False)
    print("FIM exames", len(R), "fragmentados", sum(1 for _, b, _ in R if b["n_comp"] != b["n_lesoes"]))
