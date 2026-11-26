#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    c04_juncao.py

File type:
    Script

Purpose:
    Joins the frozen marker correspondence with the verifier decisions.
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
    - _cc.nii.gz

Outputs:
    - c02_esferas.csv
    - c02_referencias.csv
    - c03_decisoes.csv
    - c04_esferas_verificador.csv
    - c04_referencias_verificador.csv

Dependencies:
    - SimpleITK
    - numpy
    - pandas
    - scipy

Execution:
    cd <project_dir> && python scripts/30_radiological_assessment/c04_juncao.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import csv, hashlib
import numpy as np, pandas as pd, SimpleITK as sitk
from scipy import ndimage as nd

P = "."
R = P + "/resultados"
O = R + "/radiological_reanalysis"
OOF = R + "/Pulmonary_Project/Script10_canonical_oof/blocks"
INT = R + "/Pulmonary_Project/Script10_canonical/blocks"
EX = {"0061": 3, "0080": 0, "0097": None, "0235": 1, "0620": 2, "0879": 2, "0922": 2, "0925": 2, "1008": None}
S6 = nd.generate_binary_structure(3, 1)

for linha in open(f"{O}/c02_SHA256SUMS"):
    h, n = linha.split()
    assert hashlib.sha256(open(f"{O}/{n}", "rb").read()).hexdigest() == h, f"c02 alterado: {n}"
for e, f in EX.items():
    for a in ("3D_F", "3D_FREM"):
        d = f"{OOF}/lidc_oof__{a}__fold_{f}" if f is not None else f"{INT}/lidc_internal__{a}"
        cc = sitk.GetArrayFromImage(sitk.ReadImage(f"{d}/LIDC-IDRI-{e}_cc.nii.gz"))
        lab, _ = nd.label(sitk.GetArrayFromImage(sitk.ReadImage(f"{d}/LIDC-IDRI-{e}.nii.gz")) > 0, structure=S6)
        assert (cc == lab).all(), f"rotulos diferentes {e} {a}"
print("rotulos _cc == nd.label: OK")

dec = pd.read_csv(f"{O}/c03_decisoes.csv", dtype={"patient_id": str})
dec["exame"] = dec.patient_id.str[-4:]
D = {(r.exame, r.architecture, int(r.component_id)): (r.group, r.p_A, r.aceite) for r in dec.itertuples()}

esf = pd.read_csv(f"{O}/c02_esferas.csv", dtype={"exame": str})
for a in ("3D_F", "3D_FREM"):
    g = [D.get((r.exame, a, int(r[f"{a}_obj"]))) if r[f"{a}_ate5mm"] else None for _, r in esf.iterrows()]
    esf[f"{a}_grupo"] = [x[0] if x else "" for x in g]
    esf[f"{a}_pA"] = [round(x[1], 3) if x else np.nan for x in g]
    esf[f"{a}_decisao"] = [("aceito" if x[2] else "rejeitado") if x else "" for x in g]
esf.to_csv(f"{O}/c04_esferas_verificador.csv", index=False)

ref = pd.read_csv(f"{O}/c02_referencias.csv", dtype={"exame": str, "3D_F_pred": str, "3D_FREM_pred": str})
for a in ("3D_F", "3D_FREM"):
    g = [D.get((r.exame, a, int(float(r[f"{a}_pred"])))) if r[f"{a}_estado"] == "TP" else None for _, r in ref.iterrows()]
    ref[f"{a}_decisao"] = [("aceito" if x[2] else "rejeitado") if x else "" for x in g]
    ref[f"{a}_pA"] = [round(x[1], 3) if x else np.nan for x in g]
ref.to_csv(f"{O}/c04_referencias_verificador.csv", index=False)

u = esf[esf.usar == "sim"]
print(f"\nESFERAS utilizaveis: {len(u)} em {u.exame.nunique()} exames (0061 excluido; 0620 sem ficheiro)")
for a in ("3D_F", "3D_FREM"):
    cat = np.where(u[f"{a}_ate5mm"] & u[f"{a}_estado"].eq("FP"), "sobre FP atual",
          np.where(u.ref_ate5mm, "sobre referencia (" + u.ref_estado_3D_F.astype(str) + ")",
          np.where(u[f"{a}_ate5mm"], "sobre TP atual", "sem objeto atual a <=5 mm")))
    t = pd.crosstab(cat, u[f"{a}_decisao"].replace("", "-"), margins=True)
    print(f"\n{a}:\n{t}")
print("\nREFERENCIAS (34) por estado 3D-F e decisao:")
print(pd.crosstab(ref["3D_F_estado"], ref["3D_F_decisao"].replace("", "-"), margins=True))
