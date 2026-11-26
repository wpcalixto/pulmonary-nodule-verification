#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    10_v01_iou_verificador.py

File type:
    Script

Purpose:
    Sensitivity of the detector-plus-verifier procedure to the IoU matching threshold.
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
    - V01_FALHA.txt
    - V01_validacao_baseline.md

Outputs:
    - V01_contagens_por_exame.csv
    - V01_controlos.csv
    - V01_diferencas_bootstrap.csv
    - V01_iou_pre_pos_por_configuracao.csv

Dependencies:
    - numpy
    - pandas

Execution:
    cd <project_dir> && python scripts/32_supplementary_analyses/a_verifier_and_lndb_cohort/10_v01_iou_verificador.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import sys, json
import numpy as np, pandas as pd
from comum import *
TAUS = [0.10, 0.25, 0.50]
TH = json.load(open(ART["s17_threshold"]))["model_A"]["threshold"]
fail = []
m = pd.read_csv(ART["s11_matches_D1_D2"]); m = m[m.subset == "internal"]
dec = pd.read_csv(ART["s17a_decisions_D2"])
met = pd.read_csv(ART["s17a_metrics_D2"])
pats = sorted(set(pd.read_csv(ART["s11_matches_D1_D2"]).query("subset=='internal'").patient_id))
if len(pats) != 158: sys.exit("D2: pacientes != 158")
rows_obj = []
for arch in sorted(m.model.unique()):
    mm = m[m.model == arch]; dd = dec[dec.architecture == arch]
    tp = mm[mm.status == "TP"].copy()
    tp["pred_component_id"] = tp.pred_component_id.astype(int)
    acc = dd.set_index(["patient_id", "component_id"]).decision.to_dict()
    tp["acc"] = [acc.get((p, c)) for p, c in zip(tp.patient_id, tp.pred_component_id)]
    if tp.acc.isna().any(): fail.append(f"D2 {arch}: TP sem decisão"); continue
    if int((dd.status_original == "TP").sum()) != int(tp.tp_010.sum()): fail.append(f"D2 {arch}: TP decisões != tp_010")
    for p in pats:
        t = tp[tp.patient_id == p]; d = dd[dd.patient_id == p]; q = mm[mm.patient_id == p]
        r = dict(dataset="D2", arch=arch, exam=p, n_ref=int((q.status != "FP").sum()), n_pred=len(d), n_acc=int((d.decision == "ACCEPT").sum()))
        for tau, col in zip(TAUS, ["tp_010", "tp_025", "tp_050"]):
            e = t[t[col] == 1]; r[f"tp_pre_{tau}"] = len(e); r[f"tp_pos_{tau}"] = int((e.acc == "ACCEPT").sum())
        rows_obj.append(r)
lm = pd.read_csv(ART["s22_matches_D4"]); sc = pd.read_csv(ART["s23_scores_D4"])
exams = sorted(pd.read_csv(ART["s22_input_manifest_D4"]).scan_id.astype(str))
if len(exams) != 211: sys.exit("D4: exames != 211")
for arch in sorted(lm.architecture.unique()):
    q0 = lm[lm.architecture == arch]; s = sc[sc.architecture == arch]
    ret = s.set_index(["scan_id", "object_id"]).retained.to_dict()
    tp = q0[q0.status == "TP"].copy(); tp["pred_component_id"] = tp.pred_component_id.astype(int)
    tp["acc"] = [ret.get((e, c)) for e, c in zip(tp.scan_id, tp.pred_component_id)]
    if tp.acc.isna().any(): fail.append(f"D4 {arch}: TP sem decisão")
    if int((s.group == "TP_pred").sum()) != len(tp): fail.append(f"D4 {arch}: TP_pred != TP associação")
    for e in exams:
        t = tp[tp.scan_id == e]; q = q0[q0.scan_id == e]; d = s[s.scan_id == e]
        r = dict(dataset="D4", arch=arch, exam=e, n_ref=int((q.status != "FP").sum()), n_pred=len(d), n_acc=int(d.retained.sum()))
        if r["n_pred"] != int((q.status != "FN").sum()): fail.append(f"D4 {arch} {e}: N_pred escores != associação")
        for tau in TAUS:
            el = t[t.iou >= tau]; r[f"tp_pre_{tau}"] = len(el); r[f"tp_pos_{tau}"] = int(el.acc.astype(bool).sum())
        rows_obj.append(r)
ex = pd.DataFrame(rows_obj)
def metr(tp, npred, nref, nex):
    fp, fn = npred - tp, nref - tp
    return dict(TP=tp, FP=fp, FN=fn, S=tp / nref if nref else np.nan, P=tp / npred if npred else np.nan,
                F1=2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan, FP_exame=fp / nex)
res = []
for (ds, arch), q in ex.groupby(["dataset", "arch"]):
    nref, npred, nacc, nex = int(q.n_ref.sum()), int(q.n_pred.sum()), int(q.n_acc.sum()), len(q)
    for tau in TAUS:
        a = metr(int(q[f"tp_pre_{tau}"].sum()), npred, nref, nex); b = metr(int(q[f"tp_pos_{tau}"].sum()), nacc, nref, nex)
        r = dict(dataset=ds, arch=arch, tau=tau, n_exames=nex, n_ref=nref, n_pred=npred, n_aceitos=nacc)
        for k, v in a.items(): r[k + "_pre"] = v
        for k, v in b.items(): r[k + "_pos"] = v
        for k in ["S", "P", "F1", "FP_exame"]: r["d_" + k] = r[k + "_pos"] - r[k + "_pre"]
        r["retencao_TP"] = r["TP_pos"] / r["TP_pre"] if r["TP_pre"] else np.nan
        r["reducao_FP"] = 1 - r["FP_pos"] / r["FP_pre"] if r["FP_pre"] else np.nan
        res.append(r)
res = pd.DataFrame(res)
ctl = []
def chk(nome, obs, esp):
    ok = obs == esp; ctl.append(dict(controlo=nome, observado=str(obs), esperado=str(esp), ok=bool(ok)))
    if not ok: fail.append(nome)
g = lambda ds, a, t: res[(res.dataset == ds) & (res.arch == a) & (np.isclose(res.tau, t))].iloc[0]
for a, pre, pos in [("3D_F", (290, 534, 560), (253, 210, 597)), ("3D_FREM", (304, 495, 546), (262, 182, 588))]:
    r = g("D4", a, 0.10); chk(f"D4 {a} tau0,10 antes", (r.TP_pre, r.FP_pre, r.FN_pre), pre); chk(f"D4 {a} tau0,10 depois", (r.TP_pos, r.FP_pos, r.FN_pos), pos)
for a, t25, t50 in [("3D_F", (243, 581, 607), (96, 728, 754)), ("3D_FREM", (252, 547, 598), (100, 699, 750))]:
    r = g("D4", a, 0.25); chk(f"D4 {a} tau0,25 antes (Tabela S15)", (r.TP_pre, r.FP_pre, r.FN_pre), t25)
    r = g("D4", a, 0.50); chk(f"D4 {a} tau0,50 antes (Tabela S15)", (r.TP_pre, r.FP_pre, r.FN_pre), t50)
for _, q in met.iterrows():
    r = g("D2", q.architecture, 0.10); s = "pre" if q.detector == "original" else "pos"
    chk(f"D2 {q.architecture} tau0,10 {q.detector} (s17a)", (r["TP_" + s], r["FP_" + s], r["FN_" + s]), (int(q.tp), int(q.fp), int(q.fn)))
for _, r in res.iterrows():
    chk(f"{r.dataset} {r.arch} tau{r.tau} TP+FN=Nref", int(r.TP_pre + r.FN_pre), int(r.n_ref))
    chk(f"{r.dataset} {r.arch} tau{r.tau} TP+FP=Npred", int(r.TP_pre + r.FP_pre), int(r.n_pred))
    if r.TP_pos > r.TP_pre or r.FP_pos > r.FP_pre: fail.append(f"{r.dataset} {r.arch} {r.tau}: contagem pós > pré")
for ds, n in [("D2", 412), ("D4", 850)]:
    for v in res[res.dataset == ds].n_ref.unique(): chk(f"{ds} N_ref", int(v), n)
pd.DataFrame(ctl).to_csv(out("outputs", "audits", "V01_controlos.csv"), index=False)
if fail: open(out("outputs", "audits", "V01_FALHA.txt"), "w").write("\n".join(fail)); sys.exit("CONTROLOS FALHARAM: " + "; ".join(fail[:10]))
res.to_csv(out("outputs", "tables", "V01_iou_pre_pos_por_configuracao.csv"), index=False)
ex.to_csv(out("outputs", "figdata", "V01_contagens_por_exame.csv"), index=False)
bt = []
for (ds, arch), q in ex.groupby(["dataset", "arch"], sort=True):
    q = q.reset_index(drop=True); rng = np.random.default_rng(SEED_E); n = len(q)
    idx = [rng.integers(0, n, size=n) for _ in range(NBOOT)]
    for tau in TAUS:
        D = {k: [] for k in ["S", "P", "F1", "FP_exame", "retencao_TP", "reducao_FP"]}; nan_p = 0
        for ii in idx:
            s = q.iloc[ii]; R, NP, NA = int(s.n_ref.sum()), int(s.n_pred.sum()), int(s.n_acc.sum())
            a = metr(int(s[f"tp_pre_{tau}"].sum()), NP, R, n); b = metr(int(s[f"tp_pos_{tau}"].sum()), NA, R, n)
            for k in ["S", "P", "F1", "FP_exame"]: D[k].append(b[k] - a[k])
            D["retencao_TP"].append(b["TP"] / a["TP"] if a["TP"] else np.nan); D["reducao_FP"].append(1 - b["FP"] / a["FP"] if a["FP"] else np.nan)
            nan_p += int(np.isnan(a["P"]) or np.isnan(b["P"]))
        ob = g(ds, arch, tau)
        for k, v in D.items():
            v = np.asarray(v, float); obs = ob["d_" + k] if k in ["S", "P", "F1", "FP_exame"] else ob[k]
            fin = np.isfinite(v)
            bt.append(dict(dataset=ds, arch=arch, tau=tau, medida=("delta_" + k) if k in ["S", "P", "F1", "FP_exame"] else k, observado=obs,
                           ic_lo=np.quantile(v[fin], .025) if fin.any() else np.nan, ic_hi=np.quantile(v[fin], .975) if fin.any() else np.nan,
                           replicas_validas=int(fin.sum()), replicas=NBOOT, semente=SEED_E, unidade="paciente" if ds == "D2" else "exame"))
pd.DataFrame(bt).to_csv(out("outputs", "tables", "V01_diferencas_bootstrap.csv"), index=False)
L = ["# V01 — validação do baseline", "", f"theta = {TH!r} (precisão integral, `filter_verifier_threshold.json`).", "",
     "| Controlo | Observado | Esperado | OK |", "|---|---|---|---|"] + [f"| {c['controlo']} | {c['observado']} | {c['esperado']} | {'sim' if c['ok'] else 'NÃO'} |" for c in ctl if "TP+" not in c["controlo"] and "N_ref" not in c["controlo"]]
L += ["", f"Identidades TP+FN=N_ref e TP+FP=N_pred antes da filtragem: {sum(c['ok'] for c in ctl if 'TP+' in c['controlo'])} de {sum('TP+' in c['controlo'] for c in ctl)} conferem.",
      f"N_ref: D2 = 412 e D4 = 850 em todas as configurações; exames: D2 = 158, D4 = 211.", f"Ambiente: {env()}"]
open(out("outputs", "audits", "V01_validacao_baseline.md"), "w").write("\n".join(L) + "\n")
print("V01 OK", len(res), "linhas;", len(ctl), "controlos")
