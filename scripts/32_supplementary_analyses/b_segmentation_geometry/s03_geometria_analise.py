#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    s03_geometria_analise.py

File type:
    Script

Purpose:
    Geometry summaries by configuration, dataset, size stratum and verifier decision, with bootstrap intervals.
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
    - 02_PLANO_ANALITICO_CONGELADO.md
    - bd1_object_matches.csv
    - filter_verifier_threshold.json
    - gt_axial_diameters.csv
    - internal_verifier_object_decisions.csv
    - lndb_verifier_object_scores.csv

Outputs:
    - B0_geometria_por_objeto_com_verificador.csv
    - B1_tabelaC_geometria_conjunto_configuracao_estrato.csv
    - B2_tabelaD_geometria_tamanho_verificador.csv
    - B3_efeitos_verificador_aceites_rejeitados.csv
    - B4_rejeicao_por_tercil_dice.csv
    - B5_comparacao_3DFREM_3DF_pareada.csv
    - GATE_s02.json
    - GATE_s03.json
    - geometria_por_objeto.csv

Dependencies:
    - numpy
    - pandas
    - scipy

Execution:
    cd <project_dir> && python scripts/32_supplementary_analyses/b_segmentation_geometry/s03_geometria_analise.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import hashlib
import json
import math
import os
import sys
import datetime

import numpy as np
import pandas as pd
from scipy import stats

ROOT = "."
AC = os.path.join(ROOT, "analises_complementares_v8")
PLANO_SHA = "d519c9e76a6d8710254a5ed8d028ac5b935ded869067c5977e3a8565e6c1a5f1"
PP = os.path.join(ROOT, "pulmonar_pos_oof_prep")
GEOM = os.path.join(AC, "frenteB", "geometria_por_objeto.csv")
GATE_S02 = os.path.join(AC, "frenteB", "GATE_s02.json")
S11B_DIAM = os.path.join(PP, "script11_final", "s11B_20260921T164539_27883", "gt_axial_diameters.csv")
S11B_MATCH = os.path.join(PP, "script11_final", "s11B_20260921T164539_27883", "analysis", "bd1_object_matches.csv")
S17A = os.path.join(PP, "script17a_final", "s17a_20260924T225525_1009246", "internal_verifier_object_decisions.csv")
S23 = os.path.join(PP, "script23_final", "s23_20260924T153750_909396", "lndb_verifier_object_scores.csv")
S17_THR = os.path.join(PP, "script17_final", "s17_20260923T160316_563312", "config", "filter_verifier_threshold.json")
TAB = os.path.join(AC, "tabelas")

N_BOOT, SEED = 2000, 27
MIN_PARES_IC, MIN_PAC_IC = 5, 2
PRIMARIAS = ["3D_F", "3D_FREM"]
NOME = {"2D": "2D", "2D_REM": "2D-REM", "2D_REL": "2D-REL", "3D_F": "3D-F", "3D_FREM": "3D-FREM",
        "3D_CF": "3D-CF", "3D_CFREM": "3D-CFREM", "F100_lowres": "F100-lowres",
        "F100_ResEncM": "F100-ResEncM", "F100_fullres": "F100-fullres"}
METR = ["dice", "jaccard", "hd95", "dsm"]

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()

def falha(msg):
    sys.exit(f"FAIL: {msg}")

def estrato_det(d):
    if d < 3: return "<3"
    if d < 6: return "3-6"
    if d < 15: return "6-15"
    return ">=15"

def estrato_com(d):
    if d < 6: return "<6"
    if d < 15: return "6-15"
    return ">=15"

def resumo(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {"n": 0, "media": None, "dp": None, "mediana": None, "p25": None, "p75": None}
    return {"n": len(v), "media": float(v.mean()), "dp": float(v.std(ddof=1)) if len(v) > 1 else None,
            "mediana": float(np.median(v)), "p25": float(np.percentile(v, 25)),
            "p75": float(np.percentile(v, 75))}

def boot_pacientes(df, fn, cols=("dice", "hd95", "aceite", "dice_F", "dice_R", "hd_F", "hd_R"),
                   n_boot=N_BOOT, seed=SEED):
    """Bootstrap não paramétrico com o paciente como unidade (Secção 2.9 do v8): cada réplica
    sorteia, com reposição, tantos pacientes quantos existem e inclui todos os objetos de cada
    paciente sorteado. fn(dict de arrays) -> float; réplicas com nan são descartadas."""
    uniq, inv = np.unique(df["id"].to_numpy(), return_inverse=True)
    if len(uniq) < MIN_PAC_IC:
        return None, None, 0
    arr = {c: df[c].to_numpy() for c in cols if c in df.columns}
    idx = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        s = rng.integers(0, len(uniq), size=len(uniq))
        ii = np.concatenate([idx[k] for k in s])
        v = fn({c: a[ii] for c, a in arr.items()})
        if v is not None and not math.isnan(v):
            vals.append(v)
    if not vals:
        return None, None, 0
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(vals)

def media(col):
    return lambda d: float(np.nanmean(d[col])) if np.isfinite(d[col].astype(float)).any() else math.nan

def mediana(col):
    return lambda d: float(np.nanmedian(d[col])) if np.isfinite(d[col].astype(float)).any() else math.nan

def holm(p):
    p = np.asarray(p, float)
    ordem = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    acum = 0.0
    for k, i in enumerate(ordem):
        acum = max(acum, min(1.0, (m - k) * p[i]))
        adj[i] = acum
    return adj

def carregar():
    g = json.load(open(GATE_S02))
    if g["estado"] != "PASS" or g["plano_sha256"] != PLANO_SHA:
        falha("GATE_s02 não PASS ou plano diferente")
    if g["saida"]["geometria_por_objeto.csv"] != sha256(GEOM):
        falha("geometria_por_objeto.csv diferente do registado no GATE_s02")
    df = pd.read_csv(GEOM)
    thr = json.load(open(S17_THR))["model_A"]["threshold"]

    dd = pd.read_csv(S11B_DIAM)
    dd = dd[dd.subset == "internal"].rename(columns={"patient_id": "id", "gt_component_id": "gt_id"})
    m11 = pd.read_csv(S11B_MATCH)
    m11 = m11[(m11.subset == "internal") & m11.gt_component_id.notna()]
    cat11 = m11.groupby(["patient_id", "gt_component_id"]).size_category.first()
    d2 = df.conjunto == "D2"
    ref = df.gt_id.notna()
    phi = df.loc[d2 & ref].merge(dd[["id", "gt_id", "d_axial_bbox_mm"]], on=["id", "gt_id"], how="left")
    if phi.d_axial_bbox_mm.isna().any():
        falha("objeto de referência de D2 sem diâmetro no S11B")
    if (np.abs(phi.d_axial_bbox_mm - phi.diam_mm_ref) > 0.0051).any():
        falha("diâmetro S11B (gt_axial_diameters) diverge de bd1_object_matches")
    df["phi_ref"] = np.nan
    df.loc[d2 & ref, "phi_ref"] = phi.d_axial_bbox_mm.values
    df.loc[~d2 & ref, "phi_ref"] = df.loc[~d2 & ref, "diam_mm_ref"]
    chk = df.loc[d2 & ref].copy()
    chk["cat11"] = [cat11.get((i, g)) for i, g in zip(chk.id, chk.gt_id)]
    mapa = {"small": "3-6", "medium": "6-15", "large": ">=15"}
    for _, r in chk.iterrows():
        e = estrato_det(r.phi_ref)
        if isinstance(r.cat11, str) and mapa[r.cat11] != e:
            falha(f"estrato divergente {r.id} g{r.gt_id}: {e} vs {r.cat11}")
        if not isinstance(r.cat11, str) and e != "<3":
            falha(f"objeto sem categoria no S11B com φ≥3: {r.id} g{r.gt_id}")
    df["estrato_det"] = [estrato_det(x) if not math.isnan(x) else None for x in df.phi_ref]
    df["estrato_com"] = [estrato_com(x) if not math.isnan(x) else None for x in df.phi_ref]

    v2 = pd.read_csv(S17A).rename(columns={"patient_id": "id", "architecture": "arch", "component_id": "pred_id"})
    v2 = v2[["id", "arch", "pred_id", "status_original", "verifier_probability", "decision"]]
    v2["aceite"] = v2.decision.map({"ACCEPT": True, "REJECT": False})
    v2["conjunto"] = "D2"
    v4 = pd.read_csv(S23)
    v4 = v4[v4.source == "pred"].rename(columns={"scan_id": "id", "architecture": "arch", "object_id": "pred_id",
                                                 "p_tp": "verifier_probability", "retained": "aceite"})
    v4["status_original"] = v4.group.map({"TP_pred": "TP", "FP_pred": "FP"})
    v4["conjunto"] = "D4"
    v = pd.concat([v2, v4[["id", "arch", "pred_id", "status_original", "verifier_probability", "aceite", "conjunto"]]])
    v["aceite"] = v.aceite.astype(bool)
    if ((v.verifier_probability >= thr) != v.aceite).any():
        falha("decisão do verificador incoerente com θ congelado")
    pred = df[df.pred_id.notna()].merge(v, on=["conjunto", "id", "arch", "pred_id"], how="left")
    if pred.aceite.isna().any():
        falha(f"{int(pred.aceite.isna().sum())} objetos preditos sem decisão do verificador")
    if (pred.status != pred.status_original).any():
        falha("estado original do verificador diverge da associação")
    nv = v.groupby("conjunto").size().to_dict()
    np_ = pred.groupby("conjunto").size().to_dict()
    if nv != np_:
        falha(f"cobertura verificador {nv} != objetos preditos {np_}")
    df = df.merge(pred[["conjunto", "id", "arch", "pred_id", "verifier_probability", "aceite"]],
                  on=["conjunto", "id", "arch", "pred_id"], how="left")
    return df, thr

def tabela_estratos(df):
    """Etapa 11 + Tabela C: por conjunto × configuração × esquema × estrato."""
    linhas = []
    refs = df[df.gt_id.notna()]
    for (cj, a), g in refs.groupby(["conjunto", "arch"]):
        esquemas = [("global", None)] + [("comum", "estrato_com")] + ([("detalhado", "estrato_det")] if cj == "D2" else [])
        for esq, col in esquemas:
            partes = [("todos", g)] if col is None else list(g.groupby(col))
            for est, h in partes:
                tp = h[h.status == "TP"]
                r = {"conjunto": cj, "configuracao": NOME[a], "arch": a, "primaria": a in PRIMARIAS,
                     "esquema": esq, "estrato": est, "n_pacientes_ref": h.id.nunique(),
                     "n_ref": len(h), "n_tp": len(tp), "n_fn": int((h.status == "FN").sum()),
                     "cobertura": len(tp) / len(h) if len(h) else None,
                     "n_pacientes_pares": tp.id.nunique()}
                for m in METR:
                    s = resumo(tp[m])
                    for k, val in s.items():
                        r[f"{m}_{k}"] = val
                if len(tp) >= MIN_PARES_IC and tp.id.nunique() >= MIN_PAC_IC:
                    lo, hi, nb = boot_pacientes(tp, media("dice"))
                    r.update(dice_media_ic_lo=lo, dice_media_ic_hi=hi, dice_media_ic_nrep=nb)
                    lo, hi, nb = boot_pacientes(tp, mediana("hd95"))
                    r.update(hd95_mediana_ic_lo=lo, hd95_mediana_ic_hi=hi, hd95_mediana_ic_nrep=nb)
                r["classe"] = ("pre-especificada" if a in PRIMARIAS else "exploratoria")
                linhas.append(r)
    return pd.DataFrame(linhas)

def tabela_verificador(df):
    """Etapa 12 + Tabela D: TP aceites × rejeitados."""
    tp = df[(df.status == "TP")].copy()
    if tp.aceite.isna().any():
        falha("TP sem decisão do verificador")
    tp["aceite"] = tp.aceite.astype(bool)
    linhas, efeitos, tercis = [], [], []
    for (cj, a), g in tp.groupby(["conjunto", "arch"]):
        for esq, col in [("global", None), ("comum", "estrato_com")] + ([("detalhado", "estrato_det")] if cj == "D2" else []):
            partes = [("todos", g)] if col is None else list(g.groupby(col))
            for est, h in partes:
                for grupo, hh in [("TP_todos", h), ("TP_aceite", h[h.aceite]), ("TP_rejeitado", h[~h.aceite])]:
                    r = {"conjunto": cj, "configuracao": NOME[a], "arch": a, "esquema": esq, "estrato": est,
                         "grupo": grupo, "n": len(hh), "n_pacientes": hh.id.nunique()}
                    for m in METR:
                        for k, val in resumo(hh[m]).items():
                            r[f"{m}_{k}"] = val
                    if len(hh) >= MIN_PARES_IC and hh.id.nunique() >= MIN_PAC_IC:
                        lo, hi, _ = boot_pacientes(hh, media("dice")); r.update(dice_media_ic_lo=lo, dice_media_ic_hi=hi)
                        lo, hi, _ = boot_pacientes(hh, mediana("hd95")); r.update(hd95_mediana_ic_lo=lo, hd95_mediana_ic_hi=hi)
                    linhas.append(r)
        ac, rj = g[g.aceite], g[~g.aceite]
        e = {"conjunto": cj, "configuracao": NOME[a], "arch": a, "n_aceite": len(ac), "n_rejeitado": len(rj),
             "pac_aceite": ac.id.nunique(), "pac_rejeitado": rj.id.nunique(),
             "taxa_rejeicao_tp": len(rj) / len(g) if len(g) else None,
             "dice_media_todos_tp": float(g.dice.mean()), "dice_media_aceites": float(ac.dice.mean()) if len(ac) else None}
        if len(ac) and len(rj):
            e["delta_dice_media"] = float(rj.dice.mean() - ac.dice.mean())
            e["delta_hd95_mediana"] = float(rj.hd95.median() - ac.hd95.median())

            def dd(d):
                ac_ = d["aceite"].astype(bool)
                if ac_.all() or not ac_.any():
                    return math.nan
                return float(d["dice"][~ac_].mean() - d["dice"][ac_].mean())

            def dh(d):
                ac_ = d["aceite"].astype(bool)
                if ac_.all() or not ac_.any():
                    return math.nan
                return float(np.nanmedian(d["hd95"][~ac_]) - np.nanmedian(d["hd95"][ac_]))
            if len(ac) >= MIN_PARES_IC and len(rj) >= MIN_PARES_IC:
                e["delta_dice_ic_lo"], e["delta_dice_ic_hi"], e["delta_dice_nrep"] = boot_pacientes(g, dd)
                e["delta_hd95_ic_lo"], e["delta_hd95_ic_hi"], e["delta_hd95_nrep"] = boot_pacientes(g, dh)
            e["mw_p_dice"] = float(stats.mannwhitneyu(rj.dice, ac.dice, alternative="two-sided").pvalue)
            e["mw_p_hd95"] = float(stats.mannwhitneyu(rj.hd95.dropna(), ac.hd95.dropna(), alternative="two-sided").pvalue)
        efeitos.append(e)
        q1, q2 = np.quantile(g.dice, [1 / 3, 2 / 3])
        t = np.where(g.dice <= q1, "T1_inferior", np.where(g.dice <= q2, "T2_medio", "T3_superior"))
        for tt in ["T1_inferior", "T2_medio", "T3_superior"]:
            h = g[t == tt]
            tercis.append({"conjunto": cj, "configuracao": NOME[a], "tercil": tt, "corte_q1": q1, "corte_q2": q2,
                           "n": len(h), "n_rejeitado": int((~h.aceite).sum()),
                           "prop_rejeicao": float((~h.aceite).mean()) if len(h) else None,
                           "dice_min": float(h.dice.min()) if len(h) else None,
                           "dice_max": float(h.dice.max()) if len(h) else None})
    ef = pd.DataFrame(efeitos)
    for cj in ef.conjunto.unique():
        for col in ["mw_p_dice", "mw_p_hd95"]:
            msk = (ef.conjunto == cj) & ef[col].notna()
            ef.loc[msk, col + "_holm"] = holm(ef.loc[msk, col].values)
            ef.loc[msk, col.replace("mw_p_", "familia_holm_")] = f"{cj}|{col.replace('mw_p_', '')}|{int(msk.sum())}"
    return pd.DataFrame(linhas), ef, pd.DataFrame(tercis)

def comparacao_primarias(df):
    """Secção 3.7: 3D-FREM − 3D-F, pareado por objeto de referência detectado pelas duas."""
    out = []
    for cj in ["D2", "D4"]:
        x = df[(df.conjunto == cj) & df.arch.isin(PRIMARIAS) & df.gt_id.notna()]
        w = x.pivot_table(index=["id", "gt_id"], columns="arch", values=["dice", "hd95"], aggfunc="first")
        st = x.pivot_table(index=["id", "gt_id"], columns="arch", values="status", aggfunc="first")
        ambos = st[(st["3D_F"] == "TP") & (st["3D_FREM"] == "TP")].index
        so_f = int(((st["3D_F"] == "TP") & (st["3D_FREM"] != "TP")).sum())
        so_r = int(((st["3D_F"] != "TP") & (st["3D_FREM"] == "TP")).sum())
        p = pd.DataFrame({"id": [i for i, _ in ambos],
                          "dice_F": w.loc[ambos, ("dice", "3D_F")].values,
                          "dice_R": w.loc[ambos, ("dice", "3D_FREM")].values,
                          "hd_F": w.loc[ambos, ("hd95", "3D_F")].values,
                          "hd_R": w.loc[ambos, ("hd95", "3D_FREM")].values})

        def dmed(d): return float(np.mean(d["dice_R"]) - np.mean(d["dice_F"]))
        def dhd(d): return float(np.median(d["hd_R"]) - np.median(d["hd_F"]))
        lo1, hi1, _ = boot_pacientes(p, dmed)
        lo2, hi2, _ = boot_pacientes(p, dhd)
        out.append({"conjunto": cj, "n_ref": len(st), "n_pares_ambas": len(p), "n_pacientes": p.id.nunique(),
                    "so_3D_F": so_f, "so_3D_FREM": so_r,
                    "dice_media_3D_F": float(p.dice_F.mean()), "dice_media_3D_FREM": float(p.dice_R.mean()),
                    "delta_dice_media": dmed({c: p[c].to_numpy() for c in p.columns}), "delta_dice_ic_lo": lo1, "delta_dice_ic_hi": hi1,
                    "hd95_mediana_3D_F": float(p.hd_F.median()), "hd95_mediana_3D_FREM": float(p.hd_R.median()),
                    "delta_hd95_mediana": dhd({c: p[c].to_numpy() for c in p.columns}), "delta_hd95_ic_lo": lo2, "delta_hd95_ic_hi": hi2,
                    "classe": "pre-especificada; estimativa sem teste de superioridade"})
    return pd.DataFrame(out)

def main():
    if sha256(os.path.join(AC, "02_PLANO_ANALITICO_CONGELADO.md")) != PLANO_SHA:
        falha("plano congelado mudou")
    inicio = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    df, thr = carregar()
    os.makedirs(TAB, exist_ok=True)
    df.to_csv(os.path.join(TAB, "B0_geometria_por_objeto_com_verificador.csv"), index=False)
    est = tabela_estratos(df)
    est.to_csv(os.path.join(TAB, "B1_tabelaC_geometria_conjunto_configuracao_estrato.csv"), index=False)
    ver, ef, ter = tabela_verificador(df)
    ver.to_csv(os.path.join(TAB, "B2_tabelaD_geometria_tamanho_verificador.csv"), index=False)
    ef.to_csv(os.path.join(TAB, "B3_efeitos_verificador_aceites_rejeitados.csv"), index=False)
    ter.to_csv(os.path.join(TAB, "B4_rejeicao_por_tercil_dice.csv"), index=False)
    cp = comparacao_primarias(df)
    cp.to_csv(os.path.join(TAB, "B5_comparacao_3DFREM_3DF_pareada.csv"), index=False)
    saidas = {f: sha256(os.path.join(TAB, f)) for f in sorted(os.listdir(TAB)) if f.startswith("B")}
    gate = {"script": os.path.basename(__file__), "script_sha256": sha256(__file__), "plano_sha256": PLANO_SHA,
            "theta": thr, "inicio": inicio,
            "fim": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "entradas_sha256": {os.path.basename(p): sha256(p) for p in (GEOM, S11B_DIAM, S11B_MATCH, S17A, S23, S17_THR)},
            "saidas_sha256": saidas, "estado": "PASS",
            "n_boot": N_BOOT, "semente": SEED}
    json.dump(gate, open(os.path.join(AC, "frenteB", "GATE_s03.json"), "w"), indent=2, ensure_ascii=False)
    print(json.dumps({k: gate[k] for k in ("estado", "inicio", "fim")}))

if __name__ == "__main__":
    main()
