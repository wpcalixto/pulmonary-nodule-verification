#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    c14_modeloC_ablacoes.py

File type:
    Script

Purpose:
    Fits Models B and C and the Model A ablations in the same five patient-grouped development folds, and applies Model C to the internal and external candidates.
    Pipeline position: Stage 31 (Model B, Model C and ablation in the development folds).

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
    - A1_quadro_candidatos_por_objeto.csv
    - B04_A_C_OPERACIONAL_CORRIGIDO.csv
    - bd1_filter_features.csv
    - filter_verifier_cv.csv
    - filter_verifier_threshold.json
    - geometria_por_objeto.csv
    - input_manifest.csv
    - manifesto_imagens_D2_D4.csv

Outputs:
    - c14_controlo_operacao.json
    - c14_ic_pareados_A_C_D2_D4.csv
    - c14_limiares_C.json
    - c14_modelos_D1.csv
    - c14_operacao_D2_D4.csv
    - c14_por_dobra.csv
    - c14_por_exame_D2_D4.csv
    - c14_porta_dobras.json

Dependencies:
    - numpy
    - pandas
    - sklearn

Execution:
    cd <project_dir> && python scripts/31_model_c_and_ablation/c14_modeloC_ablacoes.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import json, hashlib, sys, platform
import numpy as np, pandas as pd, sklearn
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
P = "."
S17 = P + "/resultados/pulmonar_pos_oof_prep/script17_final/s17_20260923T160316_563312"
FEATCSV = P + "/artigos_relatorios/Artigos_14_10_2026/RESULTADOS_ADICIONAIS_CMIG/AUDITORIA_v4/fontes/bd1_filter_features.csv"
V4 = P + "/artigos_relatorios/Artigos_14_10_2026/RESULTADOS_ADICIONAIS_CMIG/AUDITORIA_v4"
A1 = P + "/resultados/analises_complementares_v8/frenteA/A1_quadro_candidatos_por_objeto.csv"
GEO = P + "/resultados/analises_complementares_v8/frenteB/geometria_por_objeto.csv"
EXAM = P + "/resultados/analises_complementares_v8/01_inventario/manifesto_imagens_D2_D4.csv"
S22 = P + "/resultados/pulmonar_pos_oof_prep/script22_final/s22_20260924T084718_826234/input_manifest.csv"
O = P + "/resultados/modelo_c_ablacoes_dobras"
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
H = {FEATCSV: "87fd83ce96096d241a3113e23c7529fd052dff8eeb8f724acbe735f0d60e0f8e",
     A1: "4211e396fc12eedc32baa056ad88cd148ca37a0125ebeb563b21c78f843e3c32",
     GEO: "71abca0af2f18d7afd59dd77f7d57208900089eafdbc7a4b55e3e4bfad9b9b41",
     EXAM: "275cd0ace33a0a745d4321622c001c09a87725aa001c634f42b4a2685a7ff047",
     S22: "55ac16b54aa5111ce1043a998dd642ca626dd7d335627796c539a518399c8afe"}
for p, h in H.items(): assert sha(p) == h, p
thr = json.load(open(S17 + "/config/filter_verifier_threshold.json"))["model_A"]
FEAT, THETA = thr["features"], thr["threshold"]
cv = pd.read_csv(S17 + "/analysis/filter_verifier_cv.csv"); cv = cv[cv.fold != "pooled_oof"].assign(fold=lambda d: d.fold.astype(int))
df = pd.read_csv(FEATCSV, low_memory=False)
df = df[(df.subset == "oof") & df.group.isin(["TP_pred", "FP_pred"])].reset_index(drop=True)
assert len(df) == 37987 and df.patient_id.nunique() == 718
y = (df.group == "TP_pred").astype(int).values
df["size"] = np.log1p(df.diam_mm.astype(float))
FAM = {"LoG": ["log_max","log_mean","log_p95","log_scale_mm"], "blobness": ["blob_max","blob_mean","blob_p95","blob_scale_mm"],
       "vesselness": ["vessel_max","vessel_mean","vessel_p95","vessel_scale_mm"], "contrast": ["contrast_hu"],
       "geometria_pleural": ["centroid_peak_dist_mm","pleura_min_dist_mm","pleura_centroid_dist_mm"]}
assert sorted(sum(FAM.values(), [])) == sorted(FEAT)
MODELS = {"A": FEAT, "B": FEAT + ["size"], "C": ["size"]}
for f, cols in FAM.items(): MODELS["sem_" + f] = [c for c in FEAT if c not in cols]
splits = list(GroupKFold(n_splits=5).split(df, y, df.patient_id))
def pipe(): return Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=27))])
def youden(yy, s):
    fpr, tpr, t = roc_curve(yy, s); j = tpr - fpr; m = np.nanmax(j); return float(np.min(t[np.isclose(j, m)])), float(m)
oof, fold = {}, np.full(len(df), -1)
perfold = []
for name, cols in MODELS.items():
    s = np.full(len(df), np.nan)
    for k, (tr, va) in enumerate(splits):
        m = pipe().fit(df.loc[tr, cols], y[tr]); s[va] = m.predict_proba(df.loc[va, cols])[:, 1]; fold[va] = k
        perfold.append(dict(model=name, fold=k, patients=df.loc[va, "patient_id"].nunique(), TP=int(y[va].sum()), FP=int((1 - y[va]).sum()),
                            auroc=roc_auc_score(y[va], s[va]), pm=average_precision_score(y[va], s[va])))
    oof[name] = s
pf = pd.DataFrame(perfold)
gate = {}
for mdl in ["A", "B"]:
    ref = cv[cv.model == mdl].set_index("fold"); q = pf[pf.model == mdl].set_index("fold")
    gate[mdl] = dict(max_dif_auroc=float((q.auroc - ref.auroc_fold).abs().max()), max_dif_pm=float((q.pm - ref.average_precision_fold).abs().max()))
tA, _ = youden(y, oof["A"]); gate["theta_reproduzido"] = tA; gate["theta_congelado"] = THETA; gate["dif_theta"] = abs(tA - THETA)
gate["aprovado"] = gate["A"]["max_dif_auroc"] < 1e-9 and gate["A"]["max_dif_pm"] < 1e-9 and gate["dif_theta"] < 1e-9
gate["B_igual"] = gate["B"]["max_dif_auroc"] < 1e-9 and gate["B"]["max_dif_pm"] < 1e-9
gate["ambiente"] = dict(python=platform.python_version(), sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__)
import os; os.makedirs(O, exist_ok=True)
json.dump(gate, open(O + "/c14_porta_dobras.json", "w"), indent=1)
pf.to_csv(O + "/c14_por_dobra.csv", index=False)
if not gate["aprovado"]: sys.exit("PORTA FALHOU: dobras do Modelo A diferentes do desenvolvimento; nada mais é gravado")
tCy, jC = youden(y, oof["C"])
target = int(((y == 1) & (oof["A"] >= THETA)).sum()); vals = []
for t in np.unique(oof["C"]):
    z = oof["C"] >= t; vals.append((abs(int((z & (y == 1)).sum()) - target), float(t), int((z & (y == 1)).sum())))
vals.sort(key=lambda q: (q[0], q[1])); tCm = vals[0][1]
g = df.patient_id.astype(str).values; u = np.array(sorted(set(g))); ix = {p: np.flatnonzero(g == p) for p in u}
rng = np.random.default_rng(27); names = list(MODELS); B = {n: [] for n in names}
for _ in range(2000):
    q = np.concatenate([ix[p] for p in rng.choice(u, size=len(u), replace=True)])
    for n in names: B[n].append((roc_auc_score(y[q], oof[n][q]), average_precision_score(y[q], oof[n][q])))
B = {n: np.asarray(v) for n, v in B.items()}
rows = []
for n in names:
    a, p = roc_auc_score(y, oof[n]), average_precision_score(y, oof[n]); t, j = youden(y, oof[n]); z = oof[n] >= t
    tp, fp = int((z & (y == 1)).sum()), int((z & (y == 0)).sum()); fn, tn = int((~z & (y == 1)).sum()), int((~z & (y == 0)).sum())
    d = B[n] - B["A"]
    rows.append(dict(model=n, features=";".join(MODELS[n]), n=len(y), patients=len(u), auroc_oof=a, pm_oof=p,
        auroc_ci_lo=np.quantile(B[n][:, 0], .025), auroc_ci_hi=np.quantile(B[n][:, 0], .975), pm_ci_lo=np.quantile(B[n][:, 1], .025), pm_ci_hi=np.quantile(B[n][:, 1], .975),
        delta_auroc_vs_A=a - roc_auc_score(y, oof["A"]), delta_auroc_ci_lo=np.quantile(d[:, 0], .025), delta_auroc_ci_hi=np.quantile(d[:, 0], .975),
        delta_pm_vs_A=p - average_precision_score(y, oof["A"]), delta_pm_ci_lo=np.quantile(d[:, 1], .025), delta_pm_ci_hi=np.quantile(d[:, 1], .975),
        threshold_youden=t, youden_J=j, tp=tp, fp=fp, fn=fn, tn=tn, sensitivity=tp/(tp+fn), specificity=tn/(tn+fp), precision=tp/(tp+fp), f1=2*tp/(2*tp+fp+fn),
        bootstrap_replicas=2000, bootstrap_seed=27))
pd.DataFrame(rows).to_csv(O + "/c14_modelos_D1.csv", index=False)
json.dump(dict(theta_A=THETA, threshold_C_youden=tCy, threshold_C_matched_TP=tCm, D1_TP_total=int(y.sum()), D1_A_TP_accepted=target,
               D1_C_youden_TP_accepted=int(((oof["C"] >= tCy) & (y == 1)).sum()), D1_C_matched_TP_accepted=vals[0][2],
               regra_matched="min |TP_C - TP_A| em D1 OOF; empate -> menor limiar"), open(O + "/c14_limiares_C.json", "w"), indent=1)
cfin = pipe().fit(df[["size"]].values, y)
a1 = pd.read_csv(A1); a1 = a1[a1.status.isin(["TP", "FP"])].copy(); a1["id"] = a1.id.astype(str); a1["y"] = (a1.status == "TP").astype(int)
a1["score_C"] = cfin.predict_proba(np.log1p(a1.phi_pred_mm.astype(float)).values.reshape(-1, 1))[:, 1]
geo = pd.read_csv(GEO); refc = geo[geo.status.isin(["TP", "FN"])].groupby(["conjunto", "arch", "id"]).size().rename("n_ref").reset_index(); refc["id"] = refc.id.astype(str)
phys = set(pd.read_csv(EXAM).query("conjunto=='D2' and tipo=='ct'").id.astype(str)); anal = set(pd.read_csv(S22).scan_id.astype(str))
def exams(tY, tM):
    a = a1.copy(); a["A"] = a.escore.astype(float) >= THETA; a["CY"] = a.score_C >= tY; a["CM"] = a.score_C >= tM; r = []
    for (ds, ar), h in a.groupby(["conjunto", "arch"], sort=True):
        rq = refc[(refc.conjunto == ds) & (refc.arch == ar)].set_index("id").n_ref.to_dict()
        for e in sorted(phys if ds == "D2" else anal):
            q = h[h.id == e]; row = dict(dataset=ds, arch=ar, exam_id=e, n_ref=int(rq.get(e, 0)), cand_tp=int((q.y == 1).sum()), cand_fp=int((q.y == 0).sum()))
            for m in ["A", "CY", "CM"]: row[m + "_tp"] = int(((q.y == 1) & q[m]).sum()); row[m + "_fp"] = int(((q.y == 0) & q[m]).sum())
            r.append(row)
    return pd.DataFrame(r)
def operacao(ex, tY, tM):
    r = []
    for (ds, ar), q in ex.groupby(["dataset", "arch"]):
        nref, nex = int(q.n_ref.sum()), len(q); assert nref == (412 if ds == "D2" else 850)
        for m, lab, t in [("A", "Modelo A", THETA), ("CY", "Modelo C (Youden)", tY), ("CM", "Modelo C (retenção-alvo D1)", tM)]:
            tp, fp = int(q[m + "_tp"].sum()), int(q[m + "_fp"].sum()); fn = nref - tp
            r.append(dict(dataset=ds, arch=ar, regra=lab, limiar=t, TP=tp, FP=fp, S=tp/nref, Pr=tp/(tp+fp), F1=2*tp/(2*tp+fp+fn), FP_exame=fp/nex, n_exames=nex, n_ref=nref))
    return pd.DataFrame(r)
old = json.load(open(O + "/c14_limiares_C.json")) and None
ctrl = operacao(exams(0.3148295615379871, 0.3441626766117495), 0.3148295615379871, 0.3441626766117495)
ref = pd.read_csv(V4 + "/B04_A_C_OPERACIONAL_CORRIGIDO.csv")
mp = {"A": "Modelo A", "C_youden": "Modelo C (Youden)", "C_matched_TP": "Modelo C (retenção-alvo D1)"}
ref = ref.assign(regra=ref.method.map(mp)).rename(columns={"conjunto": "dataset"})
mm = ctrl.merge(ref[["dataset", "arch", "regra", "candidate_TP_after", "candidate_FP_after"]], on=["dataset", "arch", "regra"])
ctrl_ok = bool(len(mm) == 12 and (mm.TP == mm.candidate_TP_after).all() and (mm.FP == mm.candidate_FP_after).all())
json.dump(dict(controlo_limiares_antigos_reproduz_S21=ctrl_ok), open(O + "/c14_controlo_operacao.json", "w"))
if not ctrl_ok: sys.exit("CONTROLO FALHOU: a lógica de D2/D4 não reproduz a S21 com os limiares antigos")
ex = exams(tCy, tCm); ex.to_csv(O + "/c14_por_exame_D2_D4.csv", index=False)
op = operacao(ex, tCy, tCm); op.to_csv(O + "/c14_operacao_D2_D4.csv", index=False)
recs = []
for (ds, ar), q in ex.groupby(["dataset", "arch"]):
    q = q.reset_index(drop=True); rng = np.random.default_rng(270014); reps = []
    for _ in range(2000):
        s = q.iloc[rng.integers(0, len(q), size=len(q))]; R = int(s.n_ref.sum()); v = {}
        for m in ["A", "CY", "CM"]:
            tp, fp = int(s[m + "_tp"].sum()), int(s[m + "_fp"].sum())
            v[m] = dict(sensitivity_global=tp/R, precision=tp/(tp+fp) if tp+fp else np.nan, F1=2*tp/(2*tp+fp+(R-tp)), FP_per_exam=fp/len(q))
        reps.append(v)
    for c in ["CY", "CM"]:
        for met in ["sensitivity_global", "precision", "F1", "FP_per_exam"]:
            a = np.array([x["A"][met] for x in reps]); cc = np.array([x[c][met] for x in reps]); d = a - cc
            ob = op[(op.dataset == ds) & (op.arch == ar)].set_index("regra"); col = dict(sensitivity_global="S", precision="Pr", F1="F1", FP_per_exam="FP_exame")[met]
            Ao = float(ob.loc["Modelo A", col]); Co = float(ob.loc["Modelo C (Youden)" if c == "CY" else "Modelo C (retenção-alvo D1)", col])
            recs.append(dict(dataset=ds, arch=ar, comparator="C_Youden" if c == "CY" else "C_matched_TP", metric=met, A_observed=Ao, C_observed=Co, delta_A_minus_C=Ao - Co,
                             delta_ci_lo=np.nanquantile(d, .025), delta_ci_hi=np.nanquantile(d, .975), replicas=2000, seed=270014))
pd.DataFrame(recs).to_csv(O + "/c14_ic_pareados_A_C_D2_D4.csv", index=False)
print("OK", json.dumps(gate["A"]), "B_igual", gate["B_igual"], "tCy", tCy, "tCm", tCm)
