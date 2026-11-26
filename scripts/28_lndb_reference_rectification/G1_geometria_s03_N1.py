#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    G1_geometria_s03_N1.py

File type:
    Script

Purpose:
    Segmentation geometry (Dice, Jaccard, surface distances) of the N1 true-positive pairs, by size stratum and verifier decision, with the stage-32b geometry functions.
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
    - B1_tabelaC_geometria_conjunto_configuracao_estrato.csv
    - B2_tabelaD_geometria_tamanho_verificador.csv
    - B3_efeitos_verificador_aceites_rejeitados.csv
    - B5_comparacao_3DFREM_3DF_pareada.csv
    - _N1.csv
    - geometria_por_objeto.csv
    - gt_axial_diameters.csv
    - internal_verifier_object_decisions.csv
    - lndb_verifier_object_scores.csv

Outputs:
    - B0_geometria_por_objeto_N1.csv
    - E1_objetos_tau010.csv
    - F_respostas.csv
    - G1_porta.json

Dependencies:
    - numpy
    - pandas

Execution:
    cd <project_dir> && python scripts/28_lndb_reference_rectification/G1_geometria_s03_N1.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import sys, types, importlib.util, hashlib, json, math, os
import numpy as np, pandas as pd
R = "resultados"; AC = R + "/analises_complementares_v8"; O = R + "/retificacao_LNDb_20261111"
S03 = AC + "/00_scripts/s03_geometria_analise.py"; assert hashlib.sha256(open(S03, "rb").read()).hexdigest().startswith("9104a71d")
sp = importlib.util.spec_from_file_location("s03", S03); S = importlib.util.module_from_spec(sp); sp.loader.exec_module(S)
PP = R + "/pulmonar_pos_oof_prep"
geo = pd.read_csv(AC + "/frenteB/geometria_por_objeto.csv")
d2 = geo[geo.conjunto == "D2"].copy()
dd = pd.read_csv(PP + "/script11_final/s11B_20260921T164539_27883/gt_axial_diameters.csv"); dd = dd[dd.subset == "internal"].rename(columns={"patient_id": "id", "gt_component_id": "gt_id"})
ref = d2.gt_id.notna(); phi = d2.loc[ref].merge(dd[["id", "gt_id", "d_axial_bbox_mm"]], on=["id", "gt_id"], how="left"); assert phi.d_axial_bbox_mm.notna().all()
d2["phi_ref"] = np.nan; d2.loc[ref, "phi_ref"] = phi.d_axial_bbox_mm.values
v2 = pd.read_csv(PP + "/script17a_final/s17a_20260924T225525_1009246/internal_verifier_object_decisions.csv").rename(columns={"patient_id": "id", "architecture": "arch", "component_id": "pred_id"})
v2["aceite"] = v2.decision.map({"ACCEPT": True, "REJECT": False})
d2 = d2.merge(v2[["id", "arch", "pred_id", "verifier_probability", "aceite"]], on=["id", "arch", "pred_id"], how="left")
ob = pd.read_csv(O + "/saidas/E1_objetos_tau010.csv", dtype={"coorte": str}, low_memory=False); ob = ob[(ob.coorte == "211") & (ob.referencia == "N1")]
fr = pd.read_csv(O + "/saidas/F_respostas.csv", low_memory=False); fr = fr[fr.tipo == "N1"]
s23 = pd.read_csv(PP + "/script23_final/s23_20260924T153750_909396/lndb_verifier_object_scores.csv").rename(columns={"scan_id": "id", "architecture": "arch", "object_id": "pred_id", "p_tp": "verifier_probability", "retained": "aceite"})
L = []
for _, r in ob[ob.fonte == "ref"].iterrows():
    Z = np.load(f"{O}/referencias/{r.exame}.npz"); fid = int(Z["N1_findings"][int(r.id) - 1]); dia = float(fr[(fr.exame == r.exame) & (fr.FindingID == fid)].diam_axial_mm.iloc[0])
    L.append(dict(conjunto="D4", id=r.exame, arch=r.configuracao, status=r.status, gt_id=int(r.id), pred_id=(int(r.par) if r.status == "TP" else np.nan), diam_mm_ref=dia, phi_ref=dia,
                  dice=r.dice if r.status == "TP" else np.nan, jaccard=r.iou if r.status == "TP" else np.nan, hd95=r.hd95 if r.status == "TP" else np.nan, dsm=r.dsm if r.status == "TP" else np.nan))
for _, r in ob[(ob.fonte == "pred") & (ob.status != "TP")].iterrows():
    L.append(dict(conjunto="D4", id=r.exame, arch=r.configuracao, status=r.status, gt_id=np.nan, pred_id=int(r.id)))
d4 = pd.DataFrame(L).merge(s23[["id", "arch", "pred_id", "verifier_probability", "aceite"]], on=["id", "arch", "pred_id"], how="left")
assert d4[d4.pred_id.notna()].aceite.notna().all() and len(d4[d4.pred_id.notna()]) == len(s23[s23.source == "pred"])
df = pd.concat([d2, d4], ignore_index=True)
df["estrato_det"] = [S.estrato_det(x) if not (isinstance(x, float) and math.isnan(x)) else None for x in df.phi_ref]
df["estrato_com"] = [S.estrato_com(x) if not (isinstance(x, float) and math.isnan(x)) else None for x in df.phi_ref]
df["aceite"] = df.aceite.astype(object)
OUT = O + "/saidas/geometria_N1"; os.makedirs(OUT, exist_ok=True)
df.to_csv(OUT + "/B0_geometria_por_objeto_N1.csv", index=False)
est = S.tabela_estratos(df); ver, ef, ter = S.tabela_verificador(df); cp = S.comparacao_primarias(df)
for n, t in [("B1", est), ("B2", ver), ("B3", ef), ("B4", ter), ("B5", cp)]: t.to_csv(f"{OUT}/{n}_N1.csv", index=False)
orig = {n: pd.read_csv(f"{AC}/tabelas/{f}") for n, f in [("B1", "B1_tabelaC_geometria_conjunto_configuracao_estrato.csv"), ("B2", "B2_tabelaD_geometria_tamanho_verificador.csv"), ("B3", "B3_efeitos_verificador_aceites_rejeitados.csv"), ("B5", "B5_comparacao_3DFREM_3DF_pareada.csv")]}
falhas = {}
for n, t in [("B1", est), ("B2", ver), ("B3", ef), ("B5", cp)]:
    a = orig[n][orig[n].conjunto == "D2"].reset_index(drop=True); b = t[t.conjunto == "D2"].reset_index(drop=True)[a.columns]
    num = a.select_dtypes("number").columns
    falhas[n] = int(len(a) != len(b)) + int(((a[num] - b[num]).abs() > 1e-9).sum().sum()) + int((a.drop(columns=num).astype(str) != b.drop(columns=num).astype(str)).sum().sum())
porta = dict(falhas_D2=falhas, aprovada=all(v == 0 for v in falhas.values()))
json.dump(porta, open(OUT + "/G1_porta.json", "w"), indent=1); print(json.dumps(porta))
if not porta["aprovada"]: sys.exit("PORTA G1 FALHOU")
