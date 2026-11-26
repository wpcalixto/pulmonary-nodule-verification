#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    30_v03_coorte_lndb.py

File type:
    Script

Purpose:
    Documentary and quantitative audit of the LNDb cohort selection (236 to 211 scans).
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
    - *.csv
    - V03_FALHA.txt
    - V03_estado_referencias.md
    - inference_manifest_s10_20260918T114645_2953858.csv
    - inference_manifest_s10_20260918T194618_3177777.csv

Outputs:
    - V03_comparacao_metadados.csv
    - V03_exames_auditoria_236.csv
    - V03_exames_excluidos_auditoria.csv
    - V03_fluxo_coorte.csv
    - V03_pares_duplicatas.csv

Dependencies:
    - numpy
    - pandas

Execution:
    cd <project_dir> && python scripts/32_supplementary_analyses/a_verifier_and_lndb_cohort/30_v03_coorte_lndb.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import sys, json, os, glob, ast
import numpy as np, pandas as pd
from comum import *
man = json.load(open(ART["lndb_annotations_manifest"])); rows = man["rows"]
rows = rows if isinstance(rows, list) else ast.literal_eval(rows)
M = pd.DataFrame(rows); assert len(M) == 236 and M.lndb_id.is_unique
rec = pd.read_csv(ART["lndb_reconciliacao"]).rename(columns={"case_id": "lndb_id"})
alt = json.load(open(ART["lndb_alternate_label_set"]))
s22 = set(pd.read_csv(ART["s22_input_manifest_D4"]).scan_id); inm = set(pd.read_csv(ART["lndb_input_manifest"]).case_id)
fail = []
D = M.merge(rec[["lndb_id", "final_status", "exclusion_reason"]], on="lndb_id", how="left", validate="1:1")
dup_out = set(alt["swapped_in"])
def grupo(r):
    if r.lndb_id in s22: return "incluido_211"
    if r.lndb_id in dup_out: return "duplicata_nao_representante_12"
    if r.analytic_status == "EXCLUDED_REFERENCE_READER_DISCORDANCE": return "referencia_vazia_com_achados_13"
    return "OUTRO"
D["grupo"] = D.apply(grupo, axis=1)
cnt = D.grupo.value_counts().to_dict()
if cnt != {"incluido_211": 211, "referencia_vazia_com_achados_13": 13, "duplicata_nao_representante_12": 12}: fail.append(f"grupos {cnt}")
if s22 != inm: fail.append("211 de s22 != LNDB_INPUT_MANIFEST")
if set(D[D.grupo == "referencia_vazia_com_achados_13"].lndb_id) != set(rec[rec.exclusion_reason == "consensus_empty"].lndb_id): fail.append("13 != reconciliação")
if set(D[D.grupo == "duplicata_nao_representante_12"].lndb_id) != set(rec[rec.exclusion_reason == "sem razao documentada"].lndb_id): fail.append("12 != reconciliação")
ip = man["intrapair"]; ip = ip if isinstance(ip, list) else ast.literal_eval(ip)
P = pd.DataFrame(ip)
P["representante_no_211"] = P.representative.isin(s22); P["excluido_fora_211"] = ~P.excluded.isin(s22)
P["excluido_e_dos_13"] = P.excluded.isin(D[D.grupo == "referencia_vazia_com_achados_13"].lndb_id)
P["representante_e_dos_13"] = P.representative.isin(D[D.grupo == "referencia_vazia_com_achados_13"].lndb_id)
P["n_vox_excluido"] = P.excluded.map(D.set_index("lndb_id").n_vox_consensus)
if not (len(P) == 12 and P.excluido_fora_211.all() and (P.representante_no_211 ^ P.representante_e_dos_13).all() and not P.excluido_e_dos_13.any()): fail.append("pares de duplicatas")
def hdr(i):
    h = {}
    for line in open(f"{LNDB_RAW}/{i}.mhd"):
        if "=" in line: k, v = line.split("=", 1); h[k.strip()] = v.strip()
    sp = [float(x) for x in h["ElementSpacing"].split()]; dm = [int(x) for x in h["DimSize"].split()]
    return dict(esp_x_mm=sp[0], esp_y_mm=sp[1], esp_z_mm=sp[2], dim_x=dm[0], dim_y=dm[1], n_cortes=dm[2], extensao_z_mm=sp[2] * dm[2])
H = pd.DataFrame([dict(lndb_id=i, **hdr(i)) for i in D.lndb_id]); D = D.merge(H, on="lndb_id", validate="1:1")
tn = pd.read_csv(ART["lndb_trainNodules"]); tg = pd.read_csv(ART["lndb_trainNodules_gt"])
tn["lndb_id"] = tn.LNDbID.map(lambda x: f"LNDb-{int(x):04d}"); tg["lndb_id"] = tg.LNDbID.map(lambda x: f"LNDb-{int(x):04d}")
deq = lambda v: 2 * (3 * np.asarray(v, float) / (4 * np.pi)) ** (1 / 3)
tg["d_eq_mm"] = deq(tg.Volume)
a = tn.groupby("lndb_id").agg(achados_leitores=("FindingID", "size"), achados_leitores_nodulo=("Nodule", "sum"), leitores_com_achados=("RadID", "nunique"))
b = tg.groupby("lndb_id").agg(achados_consenso=("FindingID", "size"), achados_consenso_nodulo=("Nodule", "sum"))
c = tg[tg.Nodule == 1].groupby("lndb_id").agg(nodulos_consenso_d_eq_max_mm=("d_eq_mm", "max"),
        nodulos_consenso_ge3mm=("d_eq_mm", lambda s: int((s >= 3).sum())), nodulos_consenso_lt3mm=("d_eq_mm", lambda s: int((s < 3).sum())))
D = D.merge(a, on="lndb_id", how="left").merge(b, on="lndb_id", how="left").merge(c, on="lndb_id", how="left")
for k in ["achados_leitores", "achados_leitores_nodulo", "leitores_com_achados", "achados_consenso", "achados_consenso_nodulo", "nodulos_consenso_ge3mm", "nodulos_consenso_lt3mm"]:
    D[k] = D[k].fillna(0).astype(int)
IM = {a: R + "/Pulmonary_Project/analysis/" + f for a, f in [("3D_F", "inference_manifest_s10_20260918T114645_2953858.csv"), ("3D_FREM", "inference_manifest_s10_20260918T194618_3177777.csv")]}
for a, f in IM.items():
    q = pd.read_csv(f)
    if not (len(q) == 1 and q.architecture.iloc[0] == a and int(q.n_expected.iloc[0]) == 211 and int(q.n_obtained.iloc[0]) == 211 and q.status.iloc[0] == "ok"): fail.append(f"manifesto de inferência {a}")
D["inferencia_canonica_3D_F_3D_FREM"] = D.lndb_id.isin(s22)
radiologist = " ".join(open(f, errors="ignore").read() for f in glob.glob(R + "/radiological_reanalysis/*.csv"))
D["na_avaliacao_radiologica"] = [i in radiologist for i in D.lndb_id]
D["referencia_valida"] = D.grupo == "incluido_211"
D["estado_inferencia"] = np.where(D.grupo == "incluido_211", "ANALISADO (coorte primária)",
                          np.where(D.grupo == "referencia_vazia_com_achados_13", "NÃO ESTIMÁVEL: referência oficial vazia com achados de leitor; sem adjudicação",
                                   "FORA: imagem idêntica à do representante incluído"))
if fail: open(out("outputs", "audits", "V03_FALHA.txt"), "w").write("\n".join(fail)); sys.exit("FALHA: " + "; ".join(fail))
cols = ["lndb_id", "grupo", "estado_inferencia", "analytic_status", "official_reference_status", "adjudication", "duplicate_group_id", "n_readers_documented", "n_readers_with_nodules",
        "n_reader_findings", "n_reader_findings_marked_nodule", "n_vox_consensus", "achados_leitores", "achados_leitores_nodulo", "leitores_com_achados", "achados_consenso",
        "achados_consenso_nodulo", "nodulos_consenso_ge3mm", "nodulos_consenso_lt3mm", "nodulos_consenso_d_eq_max_mm", "esp_x_mm", "esp_z_mm", "dim_x", "n_cortes", "extensao_z_mm",
        "geometry_status", "structural_validity_status", "inferencia_canonica_3D_F_3D_FREM", "na_avaliacao_radiologica", "referencia_valida"]
D[cols].sort_values(["grupo", "lndb_id"]).to_csv(out("outputs", "tables", "V03_exames_auditoria_236.csv"), index=False)
D[D.grupo != "incluido_211"][cols].sort_values(["grupo", "lndb_id"]).to_csv(out("outputs", "tables", "V03_exames_excluidos_auditoria.csv"), index=False)
P.to_csv(out("outputs", "tables", "V03_pares_duplicatas.csv"), index=False)
F = pd.DataFrame([dict(etapa="exames disponíveis no LNDb", n=236), dict(etapa="membros não representantes de pares de imagens idênticas", n=-12),
                  dict(etapa="referência oficial vazia com achados de leitor (não avaliáveis)", n=-13), dict(etapa="coorte analítica", n=211), dict(etapa="imagens únicas (236 - 12 duplicatas)", n=224),
                  dict(etapa="pares de imagens idênticas cujo representante é um dos 13 (nenhum membro analisado)", n=int(P.representante_e_dos_13.sum())),
                  dict(etapa="conjunto alternativo de pares trocados (só anotações; registo)", n=int(alt["alternate_label_set_n"]))])
F.to_csv(out("outputs", "tables", "V03_fluxo_coorte.csv"), index=False)
V = ["n_readers_documented", "n_readers_with_nodules", "achados_leitores", "achados_leitores_nodulo", "achados_consenso", "achados_consenso_nodulo", "nodulos_consenso_ge3mm",
     "nodulos_consenso_lt3mm", "nodulos_consenso_d_eq_max_mm", "esp_x_mm", "esp_z_mm", "n_cortes", "extensao_z_mm"]
C = []
for g, q in D.groupby("grupo"):
    for v in V:
        x = q[v].dropna().astype(float)
        C.append(dict(grupo=g, variavel=v, n_exames=len(q), n_com_valor=len(x), media=x.mean() if len(x) else np.nan, mediana=x.median() if len(x) else np.nan,
                      q1=x.quantile(.25) if len(x) else np.nan, q3=x.quantile(.75) if len(x) else np.nan, min=x.min() if len(x) else np.nan, max=x.max() if len(x) else np.nan,
                      n_zero=int((x == 0).sum())))
    for v, lab in [("consensus_empty", "referência oficial vazia"), ("inferencia_canonica_3D_F_3D_FREM", "na inferência canónica 3D-F/3D-FREM"), ("na_avaliacao_radiologica", "na avaliação radiológica")]:
        C.append(dict(grupo=g, variavel=lab, n_exames=len(q), n_com_valor=len(q), n_verdadeiro=int(q[v].astype(bool).sum())))
pd.DataFrame(C).to_csv(out("outputs", "tables", "V03_comparacao_metadados.csv"), index=False)
x13 = D[D.grupo == "referencia_vazia_com_achados_13"]
E = ["# V03 — estado das referências dos 13 exames", "",
     f"- Regra de adjudicação registada: «{man['ADJUDICATION_RULE']}»",
     f"- Esferas por leitor: `PER_READER_IS_APPROXIMATION = {man['PER_READER_IS_APPROXIMATION']}` (modelo `{man['PER_READER_MODEL']}`) — aproximações, não referência válida.",
     f"- Máscaras corrigidas ou adjudicação radiológica independente para os 13 exames: **não localizadas**. A avaliação radiológica do projeto inclui {int(x13.na_avaliacao_radiologica.sum())} dos 13.",
     f"- Inferência canónica 3D-F/3D-FREM nos 13: {int(x13.inferencia_canonica_3D_F_3D_FREM.sum())} (os manifestos de inferência esperam e obtêm 211 exames por configuração); não há candidatos, e sem referência não haveria TP/FP/FN.",
     "- Estado da análise diagnóstica da coorte expandida (224 exames únicos): **NÃO ESTIMÁVEL**. Dependência externa: adjudicação radiológica das 13 referências, não realizada e não simulada.",
     "- `V03_sensibilidade_coorte_expandida.csv` **não foi gerado**."]
open(out("outputs", "audits", "V03_estado_referencias.md"), "w").write("\n".join(E) + "\n")
print(cnt); print(D[D.grupo == "referencia_vazia_com_achados_13"][["lndb_id", "n_readers_documented", "n_readers_with_nodules", "achados_leitores", "achados_leitores_nodulo", "achados_consenso", "achados_consenso_nodulo", "nodulos_consenso_ge3mm", "nodulos_consenso_lt3mm", "nodulos_consenso_d_eq_max_mm", "esp_z_mm", "na_avaliacao_radiologica"]].to_string())
print(pd.DataFrame(C).query("variavel in ['esp_z_mm','esp_x_mm','n_cortes','achados_consenso_nodulo','nodulos_consenso_ge3mm','n_readers_documented','referência oficial vazia','achados_leitores_nodulo']").round(3).to_string())
print(P[["member_a", "member_b", "representative", "excluded", "deciding_criterion", "representante_e_dos_13", "n_vox_excluido"]].to_string())
