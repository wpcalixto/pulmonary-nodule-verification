#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    c03_verificador.py

File type:
    Script

Purpose:
    Frozen Model A decisions for the 3D-F and 3D-FREM objects of the assessed scans.
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
    - bd1_lung_mask_authority.csv
    - filter_verifier.pkl
    - filter_verifier_threshold.json
    - internal_verifier_object_decisions.csv
    - lung_mask_crop.nii.gz

Outputs:
    - c03_decisoes.csv
    - c03_validacao.json

Dependencies:
    - SimpleITK
    - numpy
    - pandas
    - scipy

Execution:
    cd <project_dir> && python scripts/30_radiological_assessment/c03_verificador.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import json, os, pickle, hashlib, sys
import numpy as np, pandas as pd, SimpleITK as sitk
from scipy import ndimage as nd

P = "."
R = P + "/resultados"
PP = R + "/pulmonar_pos_oof_prep"
RAW = PP + "/script14_final/s14b_20260922T203413_344007/features/filter_features_raw.parquet"
MODEL = PP + "/script17_final/s17_20260923T160316_563312/models/filter_verifier.pkl"
THR = PP + "/script17_final/s17_20260923T160316_563312/config/filter_verifier_threshold.json"
S17A = PP + "/script17a_final/s17a_20260924T225525_1009246/internal_verifier_object_decisions.csv"
AUTH = PP + "/script15_final/s15_20260922T223850_371159/figdata/bd1_lung_mask_authority.csv"
LUNG = P + "/arquivo/versoes_mais_novas_que_a_oficial/Pulmonar_v281_2026-09-01/08_DADOS_INTERMEDIOS/s06_20260810T183307_96979/lung_masks/lidc_idri"
OOF = R + "/Pulmonary_Project/Script10_canonical_oof/blocks"
INT = R + "/Pulmonary_Project/Script10_canonical/blocks"
OUT = R + "/radiological_reanalysis"
EX = {"0061": 3, "0080": 0, "0097": None, "0235": 1, "0620": 2, "0879": 2, "0922": 2, "0925": 2, "1008": None}
S6 = nd.generate_binary_structure(3, 1)

def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

assert sha(RAW).startswith("a48c2164"), "fonte s14b diferente"
assert sha(MODEL).startswith("a5f727af"), "modelo A diferente"
thr = json.load(open(THR))["model_A"]
FEAT, theta = thr["features"], thr["threshold"]
_pk = pickle.load(open(MODEL, "rb"))
assert _pk["features"] == FEAT and abs(_pk["threshold"] - theta) < 1e-15 and _pk["model_label"] == "A"
model = _pk["model"]
auth = pd.read_csv(AUTH).set_index("patient_id")["sha256"]

def pleura_edt(mask_path):
    img = sitk.ReadImage(mask_path)
    arr = sitk.GetArrayFromImage(img)
    assert np.isin(arr, (0, 1)).all()
    lung = arr.astype(bool)
    surface = lung & ~nd.binary_erosion(lung, structure=S6, border_value=0)
    sp = img.GetSpacing(); sp_zyx = (float(sp[2]), float(sp[1]), float(sp[0]))
    return nd.distance_transform_edt(~surface, sampling=sp_zyx), lung, sp_zyx

def pleural_features(comp, dist, sp_zyx):
    idx = np.argwhere(comp)
    d_comp = dist[comp]
    pmin = float(d_comp.min())
    centroid = idx.mean(axis=0)
    dd = ((idx - centroid) * np.asarray(sp_zyx)) ** 2
    rep = idx[int(np.argmin(dd.sum(axis=1)))]
    return round(pmin, 4), round(float(dist[tuple(rep)]), 4)

raw = pd.read_parquet(RAW)
pids = [f"LIDC-IDRI-{e}" for e in EX]
sel = raw[raw.patient_id.isin(pids) & raw.architecture.isin(["3D_F", "3D_FREM"])
          & raw.group.isin(["TP_pred", "FP_pred"])].copy()
print("linhas selecionadas:", len(sel), sel.groupby(["subset", "architecture"]).size().to_dict())
rows = []
for pid, g in sel.groupby("patient_id"):
    e = pid[-4:]
    lm = f"{LUNG}/{pid}/lung_mask_crop.nii.gz"
    assert sha(lm) == auth[pid], f"mascara pulmonar diferente da autoridade: {pid}"
    dist, lung, sp = pleura_edt(lm)
    for arch, ga in g.groupby("architecture"):
        f = EX[e]
        pp = f"{OOF}/lidc_oof__{arch}__fold_{f}/{pid}.nii.gz" if f is not None else f"{INT}/lidc_internal__{arch}/{pid}.nii.gz"
        lab, _ = nd.label(sitk.GetArrayFromImage(sitk.ReadImage(pp)) > 0, structure=S6)
        assert lab.shape == lung.shape
        for _, r in ga.iterrows():
            pmin, pcen = pleural_features(lab == int(r.component_id), dist, sp)
            d = r.to_dict(); d.update(pleura_min_dist_mm=pmin, pleura_centroid_dist_mm=pcen)
            rows.append(d)
df = pd.DataFrame(rows)
df["p_A"] = model.predict_proba(df[FEAT])[:, 1]
df["aceite"] = df.p_A >= theta

v = pd.read_csv(S17A)
v = v[v.patient_id.isin(pids) & v.architecture.isin(["3D_F", "3D_FREM"])]
m = df[df.subset == "internal"].merge(v, left_on=["patient_id", "architecture", "component_id"],
                                      right_on=["patient_id", "architecture", "component_id"], how="outer", indicator=True)
val = {"n_internal_calc": int((df.subset == "internal").sum()), "n_17A": len(v),
       "so_um_lado": int((m._merge != "both").sum()),
       "max_dif_prob": float((m.p_A - m.verifier_probability).abs().max()),
       "decisoes_diferentes": int(((m.p_A >= theta) != m.decision.eq("ACCEPT")).sum())}
print("VALIDACAO D2:", val)
ok = val["so_um_lado"] == 0 and val["max_dif_prob"] < 1e-6 and val["decisoes_diferentes"] == 0
json.dump({**val, "aprovado": ok, "theta": theta}, open(f"{OUT}/c03_validacao.json", "w"), indent=1)
if not ok:
    sys.exit("FALHA na validacao: nada escrito para D1")
cols = ["subset", "patient_id", "architecture", "group", "component_id", "diam_mm", "pleura_min_dist_mm", "p_A", "aceite"]
df[cols].sort_values(cols[:5]).to_csv(f"{OUT}/c03_decisoes.csv", index=False)
print(df.groupby(["subset", "architecture", "group", "aceite"]).size())
