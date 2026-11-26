#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    E2_estatisticas.py

File type:
    Script

Purpose:
    Recomputes the LNDb-dependent analyses with the N1 reference (211 scans) and the sensitivity cohorts (224 unique images, N2), with scan-level bootstrap.
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
    - bd1_filter_features.csv
    - c14_limiares_C.json
    - filter_verifier_threshold.json
    - lndb_filter_features.csv
    - lndb_verifier_object_scores.csv

Outputs:
    - E1_contagens_por_exame.csv
    - E1_objetos_tau010.csv
    - E2_M_ic.csv
    - E2_portas.json
    - F_escores_pred13.csv
    - F_respostas.csv

Dependencies:
    - numpy
    - pandas
    - scipy
    - sklearn

Execution:
    cd <project_dir> && python scripts/28_lndb_reference_rectification/E2_estatisticas.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import json, sys, warnings, ast, hashlib, pickle
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
P = "."; R = P + "/resultados"; O = R + "/retificacao_LNDb_20261111"
SEED, NB = 270014, 2000
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
S23F = P + "/codigo/scripts_e_relatorios/script14_run/23_validate_filters_lndb.py"
src = open(S23F).read(); ns = {"np": np, "mannwhitneyu": mannwhitneyu}
for n in ["auroc", "mw", "holm"]: exec(ast.get_source_segment(src, [x for x in ast.parse(src).body if isinstance(x, ast.FunctionDef) and x.name == n][0]), ns)
auroc, mw, holm = ns["auroc"], ns["mw"], ns["holm"]
F16 = ["log_max", "log_mean", "log_p95", "log_scale_mm", "blob_max", "blob_mean", "blob_p95", "blob_scale_mm", "vessel_max", "vessel_mean", "vessel_p95", "vessel_scale_mm",
       "contrast_hu", "centroid_peak_dist_mm", "pleura_min_dist_mm", "pleura_centroid_dist_mm"]
C = pd.read_csv(O + "/saidas/E1_contagens_por_exame.csv", dtype={"coorte": str}); OB = pd.read_csv(O + "/saidas/E1_objetos_tau010.csv", low_memory=False, dtype={"coorte": str})
FR = pd.read_csv(O + "/saidas/F_respostas.csv", low_memory=False)
s22b = pd.read_csv(R + "/pulmonar_pos_oof_prep/script22b_final/s22b_20260924T134153_884131/lndb_filter_features.csv")
s23 = pd.read_csv(R + "/pulmonar_pos_oof_prep/script23_final/s23_20260924T153750_909396/lndb_verifier_object_scores.csv")
p13 = pd.read_csv(O + "/saidas/F_escores_pred13.csv")
PRED = pd.concat([s22b[s22b.source == "pred"].drop_duplicates(["scan_id", "architecture", "object_id"]).merge(s23[["scan_id", "architecture", "object_id", "p_tp", "retained"]], on=["scan_id", "architecture", "object_id"])
                  .rename(columns={"scan_id": "exame", "architecture": "configuracao", "object_id": "id"})[["exame", "configuracao", "id", "diam_mm", "p_tp", "retained"] + F16],
                  p13[["exame", "configuracao", "id", "diam_mm", "p_tp", "retained"] + F16]])
TH = json.load(open(R + "/pulmonar_pos_oof_prep/script17_final/s17_20260923T160316_563312/config/filter_verifier_threshold.json"))["model_A"]["threshold"]
LC = json.load(open(R + "/modelo_c_ablacoes_dobras/c14_limiares_C.json"))
df = pd.read_csv(P + "/artigos_relatorios/Artigos_14_10_2026/RESULTADOS_ADICIONAIS_CMIG/AUDITORIA_v4/fontes/bd1_filter_features.csv", low_memory=False)
df = df[(df.subset == "oof") & df.group.isin(["TP_pred", "FP_pred"])].reset_index(drop=True); assert len(df) == 37987
cfin = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=27))]).fit(np.log1p(df.diam_mm.astype(float)).values.reshape(-1, 1), (df.group == "TP_pred").astype(int))
PRED["score_C"] = cfin.predict_proba(np.log1p(PRED.diam_mm.astype(float)).values.reshape(-1, 1))[:, 1]
def met(tp, fp, fn, nex):
    return (tp / (tp + fn) if tp + fn else np.nan, tp / (tp + fp) if tp + fp else np.nan, 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan, fp / nex)
COH = {"211": ["211"], "224": ["211", "13"]}
portas = {}
pa = OB[(OB.referencia == "antiga") & (OB.fonte == "pred")].merge(PRED, on=["exame", "configuracao", "id"])
portas["C_AUROC_antiga"] = {a: float(roc_auc_score((q.status == "TP").astype(int), q.score_C)) for a, q in pa.groupby("configuracao")}
portas["C_ok"] = abs(portas["C_AUROC_antiga"]["3D_F"] - 0.6678) < 5e-5 and abs(portas["C_AUROC_antiga"]["3D_FREM"] - 0.6940) < 5e-5
portas["A_AUROC_antiga"] = {a: float(roc_auc_score((q.status == "TP").astype(int), q.p_tp)) for a, q in pa.groupby("configuracao")}
portas["A_ok"] = abs(portas["A_AUROC_antiga"]["3D_F"] - 0.8269) < 5e-5 and abs(portas["A_AUROC_antiga"]["3D_FREM"] - 0.8263) < 5e-5
json.dump(portas, open(O + "/saidas/E2_portas.json", "w"), indent=1)
if not (portas["C_ok"] and portas["A_ok"]): sys.exit("PORTA E2 FALHOU " + json.dumps(portas))
out = {k: [] for k in "MTDCGVQ"}; boot = []
for coh, partes in COH.items():
    for ref in ["N1", "N2", "N1_alt", "N2_alt"]:
        for a in ["3D_F", "3D_FREM"]:
            c = C[C.coorte.isin(partes) & (C.referencia == ref) & (C.configuracao == a)]
            if c.empty: continue
            ex = sorted(c.exame.unique()); n = len(ex); rng = np.random.default_rng(SEED); idx = [rng.integers(0, n, n) for _ in range(NB)]
            for t, ct in c.groupby("tau"):
                s = ct[["TP", "FP", "FN", "TP_pos", "FP_pos", "FN_pos", "n_ref", "n_pred", "ignoradas"]].sum(); pre = met(s.TP, s.FP, s.FN, n); pos = met(s.TP_pos, s.FP_pos, s.FN_pos, n)
                out["M"].append(dict(coorte=coh, referencia=ref, configuracao=a, tau=t, exames=n, n_ref=int(s.n_ref), ignoradas=int(s.ignoradas), TP=int(s.TP), FP=int(s.FP), FN=int(s.FN),
                                     TP_pos=int(s.TP_pos), FP_pos=int(s.FP_pos), S=pre[0], P=pre[1], F1=pre[2], FP_exame=pre[3], S_pos=pos[0], P_pos=pos[1], F1_pos=pos[2], FP_exame_pos=pos[3],
                                     retencao_TP=s.TP_pos / s.TP if s.TP else np.nan))
                if ref in ("N1", "N2") and round(t, 2) in (0.10, 0.25, 0.50):
                    A = ct.set_index("exame").loc[ex][["TP", "FP", "FN", "TP_pos", "FP_pos", "FN_pos"]].to_numpy(); V = []
                    for ii in idx:
                        q = A[ii].sum(0); x = met(*q[:3], n); y = met(*q[3:], n); V.append(list(x) + list(y) + [y[k] - x[k] for k in range(4)] + [q[3] / q[0] if q[0] else np.nan])
                    V = np.array(V, float); obs = list(pre) + list(pos) + [pos[k] - pre[k] for k in range(4)] + [s.TP_pos / s.TP if s.TP else np.nan]
                    for k, nm in enumerate(["S", "P", "F1", "FP_exame", "S_pos", "P_pos", "F1_pos", "FP_exame_pos", "dS", "dP", "dF1", "dFP_exame", "retencao_TP"]):
                        x = V[:, k]; f = np.isfinite(x); boot.append(dict(coorte=coh, referencia=ref, configuracao=a, tau=t, medida=nm, observado=obs[k], ic_lo=np.quantile(x[f], .025), ic_hi=np.quantile(x[f], .975), exames=n))
            if ref not in ("N1", "N2"): continue
            ob = OB[OB.coorte.isin(partes) & (OB.referencia == ref) & (OB.configuracao == a)]
            les = []
            for e in ex:
                Z = np.load(f"{O}/referencias/{e}.npz"); fids = Z[f"{ref}_findings"]
                fr = FR[(FR.tipo == "N1") & (FR.exame == e)].set_index("FindingID")
                for i, fid in enumerate(fids, 1): les.append(dict(exame=e, id=i, FindingID=int(fid), diam=float(fr.loc[int(fid), "diam_axial_mm"])))
            les = pd.DataFrame(les)
            r_ = ob[ob.fonte == "ref"].merge(les, on=["exame", "id"], how="left"); assert r_.diam.notna().all()
            r_["aceite_b"] = r_.aceite.astype(str) == "True"
            for nome, lo, hi in [("<3", 0, 3), ("3-6", 3, 6), ("<6", 0, 6), ("6-15", 6, 15), (">=15", 15, 1e9)]:
                q = r_[(r_.diam >= lo) & (r_.diam < hi)]
                out["T"].append(dict(coorte=coh, referencia=ref, configuracao=a, estrato=nome, n_lesoes=len(q), TP=int((q.status == "TP").sum()),
                                     S=(q.status == "TP").mean() if len(q) else np.nan, TP_pos=int(((q.status == "TP") & q.aceite_b).sum()), S_pos=((q.status == "TP") & q.aceite_b).mean() if len(q) else np.nan))
            pr = ob[(ob.fonte == "pred") & ob.status.isin(["TP", "FP"])].merge(PRED, on=["exame", "configuracao", "id"], how="left"); assert pr.p_tp.notna().all()
            fl = FR[FR.tipo == "N1"].set_index(["exame", "FindingID"])
            r_["key"] = list(zip(r_.exame, r_.FindingID)); rf = r_.join(fl[F16], on=["exame", "FindingID"])
            for comp, pos_, neg_ in [("A_TP_pred_vs_FP_pred", pr[pr.status == "TP"], pr[pr.status == "FP"]), ("B_TP_ref_vs_FN_ref", rf[rf.status == "TP"], rf[rf.status == "FN"])]:
                for f in F16:
                    x, y = pos_[f].dropna().to_numpy(float), neg_[f].dropna().to_numpy(float); u, p = mw(x, y)
                    out["D"].append(dict(coorte=coh, referencia=ref, comparacao=comp, configuracao=a, resposta=f, n_pos=len(x), n_neg=len(y), auroc=auroc(x, y), p_raw=p))
            yb = (pr.status == "TP").astype(int).to_numpy()
            if 0 < yb.sum() < len(yb):
                out["C"].append(dict(coorte=coh, referencia=ref, configuracao=a, n=len(yb), auroc_A=roc_auc_score(yb, pr.p_tp), pm_A=average_precision_score(yb, pr.p_tp),
                                     auroc_C=roc_auc_score(yb, pr.score_C), pm_C=average_precision_score(yb, pr.score_C)))
            allp = ob[ob.fonte == "pred"].merge(PRED, on=["exame", "configuracao", "id"], how="left")
            nref = ob[ob.fonte == "ref"].groupby("exame").size().reindex(ex, fill_value=0)
            regras = {"A": allp.p_tp >= TH, "C_Youden": allp.score_C >= LC["threshold_C_youden"], "C_retencao": allp.score_C >= LC["threshold_C_matched_TP"]}
            tab = {}
            for rg, acc in regras.items():
                q = allp.assign(acc=acc.values)
                tab[rg] = pd.DataFrame({"TP": q[(q.status == "TP") & q.acc].groupby("exame").size(), "FP": q[(q.status == "FP") & q.acc].groupby("exame").size()}).reindex(ex, fill_value=0).fillna(0)
            obs = {rg: met(t.TP.sum(), t.FP.sum(), nref.sum() - t.TP.sum(), n) for rg, t in tab.items()}
            for rg in regras: out["C"].append(dict(coorte=coh, referencia=ref, configuracao=a, regra=rg, S=obs[rg][0], P=obs[rg][1], F1=obs[rg][2], FP_exame=obs[rg][3]))
            Ar = {rg: t[["TP", "FP"]].to_numpy() for rg, t in tab.items()}; NR = nref.to_numpy(); D_ = {k: [] for k in ["C_Youden", "C_retencao"]}
            rng = np.random.default_rng(SEED)
            for _ in range(NB):
                ii = rng.integers(0, n, n); R_ = NR[ii].sum()
                m = {rg: met(Ar[rg][ii, 0].sum(), Ar[rg][ii, 1].sum(), R_ - Ar[rg][ii, 0].sum(), n) for rg in regras}
                for k in D_: D_[k].append([m["A"][j] - m[k][j] for j in range(4)])
            for k, v in D_.items():
                v = np.array(v, float)
                for j, nm in enumerate(["S", "P", "F1", "FP_exame"]):
                    x = v[:, j]; f = np.isfinite(x); out["C"].append(dict(coorte=coh, referencia=ref, configuracao=a, regra=f"A-{k}", medida=nm, observado=obs["A"][j] - obs[k][j], ic_lo=np.quantile(x[f], .025), ic_hi=np.quantile(x[f], .975)))
            for g in [round(x, 2) for x in np.arange(0.05, 0.951, 0.05)] + [TH]:
                q = allp[allp.p_tp >= g]; tp = int((q.status == "TP").sum()); fp = int((q.status == "FP").sum()); m = met(tp, fp, int(nref.sum()) - tp, n)
                out["G"].append(dict(coorte=coh, referencia=ref, configuracao=a, limiar=g, TP=tp, FP=fp, S=m[0], P=m[1], F1=m[2], FP_exame=m[3]))
            for cl in ["TP", "FP"]:
                q = pr[pr.status == cl]; out["V"].append(dict(coorte=coh, referencia=ref, configuracao=a, classe=cl, n=len(q), frac_acima_theta=float((q.p_tp >= TH).mean()) if len(q) else np.nan,
                                                              mediana=float(q.p_tp.median()) if len(q) else np.nan))
            g_ = r_[r_.status == "TP"].copy(); g_["estrato"] = pd.cut(g_.diam, [0, 6, 15, 1e9], right=False, labels=["<6", "6-15", ">=15"])
            for (lab, q) in [("todos", g_)] + [(f"estrato {k}", v) for k, v in g_.groupby("estrato", observed=True)] + [("aceites", g_[g_.aceite_b]), ("rejeitados", g_[~g_.aceite_b])]:
                exq = sorted(q.exame.unique()); rng = np.random.default_rng(SEED); bm = []
                gq = {e: q[q.exame == e] for e in exq}
                for _ in range(NB):
                    s_ = pd.concat([gq[exq[i]] for i in rng.integers(0, len(exq), len(exq))]) if exq else q; bm.append((s_.dice.mean(), s_.hd95.mean()))
                bm = np.array(bm, float)
                out["Q"].append(dict(coorte=coh, referencia=ref, configuracao=a, grupo=lab, n=len(q), dice_media=q.dice.mean(), dice_ic_lo=np.nanquantile(bm[:, 0], .025), dice_ic_hi=np.nanquantile(bm[:, 0], .975),
                                     dice_mediana=q.dice.median(), hd95_media=q.hd95.mean(), hd95_ic_lo=np.nanquantile(bm[:, 1], .025), hd95_ic_hi=np.nanquantile(bm[:, 1], .975), hd95_mediana=q.hd95.median()))
Dd = pd.DataFrame(out["D"]); Dd["p_holm"] = np.nan
for k, q in Dd.groupby(["coorte", "referencia", "comparacao"]): Dd.loc[q.index, "p_holm"] = holm(q.p_raw.to_numpy())
Dd["abaixo_alpha"] = Dd.p_holm < 0.05; out["D"] = Dd
for k, v in out.items(): (v if isinstance(v, pd.DataFrame) else pd.DataFrame(v)).to_csv(O + f"/saidas/E2_{k}.csv", index=False)
pd.DataFrame(boot).to_csv(O + "/saidas/E2_M_ic.csv", index=False); print("E2 OK", json.dumps(portas))
