#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    c12_geometria_tp.py

File type:
    Script

Purpose:
    Dice and HD95 of the true-positive pairs linked to radiological records.
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
    - bd1_object_matches.csv
    - consensus.nii.gz
    - geometria_por_objeto.csv

Outputs:
    - c10_objetos.csv
    - c12_geometria_tp.csv

Dependencies:
    - SimpleITK
    - numpy
    - pandas
    - scipy

Execution:
    cd <project_dir> && python scripts/30_radiological_assessment/c12_geometria_tp.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import importlib.util, sys
import numpy as np, pandas as pd, SimpleITK as sitk
from scipy import ndimage as nd
P = "."; R = P + "/resultados"; O = R + "/radiological_reanalysis"
spec = importlib.util.spec_from_file_location("s02", R + "/analises_complementares_v8/00_scripts/s02_geometria_pares.py")
s02 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s02)
CONS = P + "/arquivo/Ultima_Versao/Resultados/evidencia_nvme/do_5070/annotations/consensus/lidc"
OOF = R + "/Pulmonary_Project/Script10_canonical_oof/blocks"; INT = R + "/Pulmonary_Project/Script10_canonical/blocks"
EX = {"0061": 3, "0080": 0, "0097": None, "0235": 1, "0620": 2, "0879": 2, "0922": 2, "0925": 2, "1008": None}
M = pd.read_csv(R + "/pulmonar_pos_oof_prep/script11_final/s11B_20260921T164539_27883/analysis/bd1_object_matches.csv")
M = M[(M.model == "3D_F") & (M.status == "TP")].set_index(["patient_id", "gt_component_id", "pred_component_id"])
G2 = pd.read_csv(R + "/analises_complementares_v8/frenteB/geometria_por_objeto.csv")
G2 = G2[(G2.arch == "3D_F") & (G2.status == "TP")]
G2 = G2.assign(gt_id=G2.gt_id.astype(float).astype(int), pred_id=G2.pred_id.astype(float).astype(int)).set_index(["id", "gt_id", "pred_id"])
X = pd.read_csv(f"{O}/c10_objetos.csv", dtype={"exame": str})
X = X[(X.configuracao == "3D_F") & (X.classe == "TP") & (X.estado == "TP")]
S6 = nd.generate_binary_structure(3, 1); rows = []; cache = {}
for r in X.itertuples():
    e, pid = r.exame, f"LIDC-IDRI-{r.exame}"; k = int(r.alvo.split()[1]); p = int(r.objeto)
    if e not in cache:
        d = f"{OOF}/lidc_oof__3D_F__fold_{EX[e]}" if EX[e] is not None else f"{INT}/lidc_internal__3D_F"
        img = sitk.ReadImage(f"{d}/{pid}.nii.gz"); pl, _ = nd.label(sitk.GetArrayFromImage(img) > 0, structure=S6)
        c = sitk.Resample(sitk.ReadImage(f"{CONS}/{pid}/consensus.nii.gz"), img, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
        gl, _ = nd.label(sitk.GetArrayFromImage(c) > 0, structure=S6); sp = img.GetSpacing(); cache[e] = (pl, gl, (sp[2], sp[1], sp[0]))
    pl, gl, sp = cache[e]; Gm, Pm = gl == k, pl == p; sl = s02.recorte(Gm, Pm); m = s02.metricas_par(Pm[sl], Gm[sl], sp)
    reg = M.loc[(pid, k, p)]
    v = dict(dif_iou=abs(m["jaccard"] - reg.iou), dif_dice=abs(m["dice"] - reg.dice_matched), dif_hd=abs(m["hd_s11_reproduzido"] - reg.hausdorff_mm))
    if r.conjunto == "D2":
        g = G2.loc[(pid, k, p)]; v.update(dif_dice_art=abs(m["dice"] - g.dice), dif_hd95_art=abs(m["hd95"] - g.hd95))
    rows.append(dict(n=r.n, exame=e, conjunto=r.conjunto, registo=r.registo, ref=k, pred=p, diam_ref_mm=reg.diam_mm_gt,
                     dice=round(m["dice"], 3), iou=round(m["jaccard"], 3), hd95_mm=round(m["hd95"], 2), **{a: b for a, b in v.items()}))
D = pd.DataFrame(rows)
tol = dict(dif_iou=6e-5, dif_dice=6e-5, dif_hd=6e-3, dif_dice_art=1e-9, dif_hd95_art=1e-9)
fal = {c: float(D[c].max()) for c in tol if c in D and D[c].max() > tol[c]}
print("max diferencas:", {c: float(D[c].max()) for c in tol if c in D});
if fal: sys.exit(f"FALHA validacao: {fal}")
D.to_csv(f"{O}/c12_geometria_tp.csv", index=False)
print(D[["n", "exame", "registo", "diam_ref_mm", "dice", "iou", "hd95_mm"]].to_string(index=False))
