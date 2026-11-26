#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    S4_curvas_operacionais.py

File type:
    Script

Purpose:
    Descriptive comparison of the operating curves of selective and full verification.
    Pipeline position: Stage 29 (selective verification).

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
    - E1_contagens_por_exame.csv
    - internal_verifier_object_decisions.csv
    - lndb_verifier_object_scores.csv

Outputs:
    - E1b_predicoes_por_tau.csv
    - S4_curvas_operacionais.csv

Dependencies:
    - numpy
    - pandas

Execution:
    cd <project_dir> && python scripts/29_selective_verification/S4_curvas_operacionais.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import json, numpy as np, pandas as pd
exec(open(__file__.replace("S4_curvas_operacionais.py", "S2_avaliacao_seletiva.py")).read().split("RES, BOOT, FRONT = [], [], []")[0])
GT = [0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]; L = []
def curva(rot, df, ex, nref):
    n = len(ex); R_ = sum(int(nref.get(e, 0)) for e in ex); tp_, fp_ = (df.status == "TP").to_numpy(), (df.status == "FP").to_numpy()
    pts = sorted(((fp_ & (df.p >= g).to_numpy()).sum() / n, (tp_ & (df.p >= g).to_numpy()).sum() / R_) for g in sorted(set([round(x, 2) for x in np.arange(0.05, 0.951, 0.05)] + [TH])))
    xs, ys = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    for t in GT:
        a = ((df.q >= t) | (df.p >= TH)).to_numpy(); fpe = (fp_ & a).sum() / n; s = (tp_ & a).sum() / R_
        sb = float(np.interp(fpe, xs, ys)) if xs.min() <= fpe <= xs.max() else np.nan
        L.append(dict(conjunto=rot, t=t, aceites_diretos=int((df.q >= t).sum()), FP_exame_C=fpe, S_C=s, S_B_mesma_carga=sb, diferenca=s - sb if np.isfinite(sb) else np.nan))
for a in ARCH:
    curva(f"D1 OOF {a}", d1[d1.configuracao == a], ex1, nref1.get(a, pd.Series(dtype=int)).to_dict())
s17 = pd.read_csv(R + "/pulmonar_pos_oof_prep/script17a_final/s17a_20260924T225525_1009246/internal_verifier_object_decisions.csv")
d2 = s17[s17.architecture.isin(ARCH)].merge(Q[Q.conjunto == "D2"].rename(columns={"exame": "patient_id", "configuracao": "architecture", "id": "component_id"}), on=["patient_id", "architecture", "component_id"])
d2 = d2.rename(columns={"patient_id": "exame", "architecture": "configuracao", "verifier_probability": "p", "q_media": "q", "status_original": "status"})
mi = M[M.subset == "internal"]; ex2 = sorted(mi.patient_id.unique())
for a in ARCH: curva(f"D2 {a}", d2[d2.configuracao == a], ex2, mi[(mi.model == a) & mi.status.isin(["TP", "FN"])].groupby("patient_id").size().to_dict())
pb = pd.read_csv(O + "/saidas/E1b_predicoes_por_tau.csv", dtype={"coorte": str}); pb = pb[(pb.coorte == "211") & (pb.referencia == "N1") & np.isclose(pb.tau, 0.10) & (pb.status != "IGN")]
s23 = pd.read_csv(R + "/pulmonar_pos_oof_prep/script23_final/s23_20260924T153750_909396/lndb_verifier_object_scores.csv").rename(columns={"scan_id": "exame", "architecture": "configuracao", "object_id": "id", "p_tp": "p"})
d4 = pb.merge(s23[["exame", "configuracao", "id", "p"]], on=["exame", "configuracao", "id"]).merge(Q[Q.conjunto == "D4"].rename(columns={"q_media": "q"})[["exame", "configuracao", "id", "q"]], on=["exame", "configuracao", "id"])
CE = pd.read_csv(RL + "/saidas/E1_contagens_por_exame.csv", dtype={"coorte": str}); CE = CE[(CE.coorte == "211") & (CE.referencia == "N1") & np.isclose(CE.tau, 0.10)]
for a in ARCH:
    c = CE[CE.configuracao == a]; curva(f"D4 N1 211 {a}", d4[d4.configuracao == a], sorted(c.exame), c.set_index("exame").n_ref.to_dict())
pd.DataFrame(L).to_csv(O + "/saidas/S4_curvas_operacionais.csv", index=False); print(pd.DataFrame(L).round(4).to_string())
