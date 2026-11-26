#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    V03_lesoes_multicomponente.py

File type:
    Script

Purpose:
    Checks LNDb N1 lesions with more than one connected component and their response vectors.
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

Outputs:
    - Tables written to the stage output directory (see docs/post_v1_stages.md).

Dependencies:
    - numpy
    - pandas
    - scipy

Execution:
    cd <project_dir> && python scripts/32_supplementary_analyses/d_interobserver_matching_and_partition/V03_lesoes_multicomponente.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import sys, glob, json, hashlib
import numpy as np, scipy.ndimage as nd, pandas as pd
O = sys.argv[1]; OUT = sys.argv[2]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
S6 = nd.generate_binary_structure(3, 1); S26 = nd.generate_binary_structure(3, 3)
FR = pd.read_csv(O + "/saidas/F_respostas.csv", low_memory=False)
res = []; n_les = 0
for f in sorted(glob.glob(O + "/referencias/LNDb-*.npz")):
    e = f.split("/")[-1][:-4]; Z = np.load(f); n1 = Z["N1"]
    for cid in range(1, int(n1.max()) + 1):
        m = n1 == cid
        if not m.any(): continue
        n_les += 1
        lab6, k6 = nd.label(m, structure=S6)
        if k6 > 1:
            _, k26 = nd.label(m, structure=S26)
            r = FR[(FR.tipo == "N1") & (FR.exame == e) & (FR.id == cid)]
            res.append(dict(exame=e, id=cid, FindingID=int(Z["N1_findings"][cid - 1]), componentes_6=int(k6), componentes_26=int(k26),
                            voxels_mascara=int(m.sum()), voxels_por_componente=[int((lab6 == i).sum()) for i in range(1, k6 + 1)],
                            linhas_respostas=len(r), voxels_linha=r.voxels.tolist(), voxels_no_recorte=r.voxels_no_recorte.tolist(),
                            blob_max=r.blob_max.tolist(), diam_axial_mm=r.diam_axial_mm.tolist(), sha_npz=sha(f)))
out = dict(lesoes_N1_varridas=n_les, linhas_N1_F_respostas=int((FR.tipo == "N1").sum()), multicomponente=res,
           sha=dict(F_respostas=sha(O + "/saidas/F_respostas.csv"), F_respostas_py=sha(O + "/scripts/F_respostas.py"),
                    E2_estatisticas_py=sha(O + "/scripts/E2_estatisticas.py"),
                    s13=sha(O + "/sandbox/scripts/13_prepare_filter_objects.py"), s14=sha(O + "/sandbox/scripts/14_compute_filter_maps_gpu.py"),
                    s15=sha(O + "/sandbox/scripts/15_extract_filter_features.py")))
json.dump(out, open(OUT, "w"), indent=1); print("FIM")
