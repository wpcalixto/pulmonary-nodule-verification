#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    S2_avaliacao_seletiva.py

File type:
    Script

Purpose:
    Selective verification: direct acceptance of candidates above the confidence threshold and Model A for the remainder, compared with full verification.
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
    - E2_M.csv
    - F_escores_pred13.csv
    - V02_escores_oof_ABC.csv.gz
    - bd1_filter_features.csv
    - bd1_object_matches.csv
    - filter_verifier_threshold.json
    - internal_verifier_metrics.csv
    - internal_verifier_object_decisions.csv
    - lndb_annotations_manifest.json

Outputs:
    - E1b_predicoes_por_tau.csv
    - S1_confianca_candidatos.csv
    - S2_diferencas_C_B.csv
    - S2_fronteira_B.csv
    - S2_limiar_congelado.json
    - S2_metricas.csv
    - S2_portas.json

Dependencies:
    - numpy
    - pandas

Execution:
    cd <project_dir> && python scripts/29_selective_verification/S2_avaliacao_seletiva.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
R = "resultados"; P = "."; O = R + "/verificacao_seletiva_20261111"; RL = R + "/retificacao_LNDb_20261111"
TH = json.load(open(R + "/pulmonar_pos_oof_prep/script17_final/s17_20260923T160316_563312/config/filter_verifier_threshold.json"))["model_A"]["threshold"]
SEED, NB = 270014, 2000; ARCH = ["3D_F", "3D_FREM"]
Q = pd.read_csv(O + "/saidas/S1_confianca_candidatos.csv")
M = pd.read_csv(R + "/pulmonar_pos_oof_prep/script11_final/s11B_20260921T164539_27883/analysis/bd1_object_matches.csv")
port = {}
fe = pd.read_csv(P + "/artigos_relatorios/Artigos_14_10_2026/RESULTADOS_ADICIONAIS_CMIG/AUDITORIA_v4/fontes/bd1_filter_features.csv", low_memory=False)
fe = fe[(fe.subset == "oof") & fe.group.isin(["TP_pred", "FP_pred"]) & fe.architecture.isin(ARCH)][["patient_id", "architecture", "component_id", "group", "voxel_count"]]
sa = pd.read_csv(R + "/analises_complementares_CMIG/outputs/figdata/V02_escores_oof_ABC.csv.gz")[["patient_id", "architecture", "component_id", "score_A", "dobra"]]
d1 = fe.merge(sa, on=["patient_id", "architecture", "component_id"], how="left").merge(Q[Q.conjunto == "D1"].rename(columns={"exame": "patient_id", "configuracao": "architecture", "id": "component_id"}), on=["patient_id", "architecture", "component_id"], how="left")
port["D1_sem_q"] = int(d1.q_media.isna().sum()); port["D1_sem_scoreA"] = int(d1.score_A.isna().sum()); port["D1_voxels_diferentes"] = int((d1.voxels != d1.voxel_count).sum())
d1 = d1.rename(columns={"patient_id": "exame", "architecture": "configuracao", "score_A": "p", "q_media": "q"}); d1["status"] = np.where(d1.group == "TP_pred", "TP", "FP")
nref1 = M[(M.subset == "oof") & M.status.isin(["TP", "FN"])].groupby(["model", "patient_id"]).size()
ex1 = sorted(M[M.subset == "oof"].patient_id.unique()); assert len(ex1) == 718
def escolhe_t(c):
    c = c.sort_values("q", ascending=False).reset_index(drop=True); tp, fp = (c.status == "TP").to_numpy(), (c.status == "FP").to_numpy(); b = (c.p >= TH).to_numpy()
    FPB, TPB = int((fp & b).sum()), int((tp & b).sum())
    ctp_d, cfp_d = np.concatenate([[0], np.cumsum(tp)]), np.concatenate([[0], np.cumsum(fp)])
    ctp_b, cfp_b = np.concatenate([[0], np.cumsum(tp & b)]), np.concatenate([[0], np.cumsum(fp & b)])
    qv = c.q.to_numpy(); fronteira = [k for k in range(len(c) + 1) if k == 0 or k == len(c) or qv[k - 1] != qv[k]]
    best = (TPB, -1, np.inf)
    for k in fronteira:
        TPc = ctp_d[k] + (TPB - ctp_b[k]); FPc = cfp_d[k] + (FPB - cfp_b[k])
        t = qv[k - 1] if k > 0 else np.inf
        if FPc <= FPB and (TPc > best[0] or (TPc == best[0] and t > best[2])): best = (TPc, k, t)
    return best[2]
T = {}; cv = []
for a in ARCH:
    c = d1[d1.configuracao == a]; T[a] = float(escolhe_t(c))
    for k in range(5):
        tk = escolhe_t(c[c.dobra != k]); h = c[c.dobra == k]; cv.append(h.assign(t=tk))
json.dump(dict(theta=TH, t_rota=T, regra="max TP aceites s.a. FP(C) <= FP(B) em D1 OOF; empate -> maior t"), open(O + "/saidas/S2_limiar_congelado.json", "w"), indent=1)
d1cv = pd.concat(cv)
def aceita(df, t, arq):
    if arq == "A": return np.ones(len(df), bool)
    if arq == "B": return (df.p >= TH).to_numpy()
    return ((df.q >= t) | (df.p >= TH)).to_numpy()
def contas(df, ex, nref, arq, tcol=None, t=None):
    a = aceita(df, df[tcol].to_numpy() if tcol else t, arq) if arq == "C" else aceita(df, None, arq)
    g = pd.DataFrame({"exame": df.exame.to_numpy(), "TP": (df.status == "TP").to_numpy() & a, "FP": (df.status == "FP").to_numpy() & a,
                      "rota": ((df.q < (df[tcol].to_numpy() if tcol else t)).to_numpy() if arq == "C" else np.full(len(df), arq == "B")),
                      "direto": ((df.q >= (df[tcol].to_numpy() if tcol else t)).to_numpy() if arq == "C" else np.full(len(df), arq == "A")),
                      "TPdet": (df.status == "TP").to_numpy(), "FPdet": (df.status == "FP").to_numpy()})
    g["TP_dir"] = g.TP & g.direto; g["FP_dir"] = g.FP & g.direto; g["TP_rot"] = g.TPdet & g.rota; g["FP_rot"] = g.FPdet & g.rota
    s = g.groupby("exame")[["TP", "FP", "rota", "direto", "TPdet", "FPdet", "TP_dir", "FP_dir", "TP_rot", "FP_rot"]].sum().reindex(ex, fill_value=0)
    s["n_ref"] = [int(nref.get(e, 0)) for e in ex]; return s
def met(s, n):
    tp, fp, R_ = s.TP.sum(), s.FP.sum(), s.n_ref.sum(); fn = R_ - tp
    return dict(TP=int(tp), FP=int(fp), FN=int(fn), S=tp / R_ if R_ else np.nan, P=tp / (tp + fp) if tp + fp else np.nan, F1=2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan,
                FP_exame=fp / n, TP_detector=int(s.TPdet.sum()), FP_detector=int(s.FPdet.sum()), FN_detector=int(R_ - s.TPdet.sum()), FN_adicionais=int(s.TPdet.sum() - tp),
                retencao_TP=tp / s.TPdet.sum() if s.TPdet.sum() else np.nan, reducao_FP_abs=int(s.FPdet.sum() - fp), reducao_FP_rel=1 - fp / s.FPdet.sum() if s.FPdet.sum() else np.nan,
                encaminhados=int(s.rota.sum()), aceites_diretos=int(s.direto.sum()), TP_diretos=int(s.TP_dir.sum()), FP_diretos=int(s.FP_dir.sum()), TP_encaminhados=int(s.TP_rot.sum()), FP_encaminhados=int(s.FP_rot.sum()),
                candidatos=int(s.TPdet.sum() + s.FPdet.sum()), n_ref=int(R_), exames=n)
RES, BOOT, FRONT = [], [], []
def avalia(rot, df, ex, nref, tcol=None, t=None):
    n = len(ex); S = {k: contas(df, ex, nref, k, tcol, t) for k in "ABC"}
    for k, s in S.items(): RES.append(dict(conjunto=rot, arquitetura=k, t_rota=(t if t is not None else "CV"), **met(s, n)))
    rng = np.random.default_rng(SEED); A_ = {k: s[["TP", "FP", "n_ref", "TPdet"]].to_numpy() for k, s in S.items()}; V = []
    for _ in range(NB):
        ii = rng.integers(0, n, n); r = {}
        for k in "BC":
            tp, fp, R_, td = A_[k][ii].sum(0); fn = R_ - tp
            r[k] = (tp / R_ if R_ else np.nan, tp / (tp + fp) if tp + fp else np.nan, 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan, fp / n, tp, tp / td if td else np.nan)
        V.append([r["C"][j] - r["B"][j] for j in range(6)])
    V = np.array(V, float); mb, mc = met(S["B"], n), met(S["C"], n)
    for j, nm in enumerate(["S", "P", "F1", "FP_exame", "TP", "retencao_TP"]):
        x = V[:, j]; f = np.isfinite(x); BOOT.append(dict(conjunto=rot, medida=f"C-B {nm}", observado=mc[nm] - mb[nm], ic_lo=np.quantile(x[f], .025), ic_hi=np.quantile(x[f], .975)))
    pts = []
    for g in sorted(set([round(x, 2) for x in np.arange(0.05, 0.951, 0.05)] + [TH])):
        a = (df.p >= g).to_numpy(); tp = int(((df.status == "TP").to_numpy() & a).sum()); fp = int(((df.status == "FP").to_numpy() & a).sum())
        pts.append((fp / n, tp / sum(int(nref.get(e, 0)) for e in ex), g))
    pts.sort(); xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    sB = float(np.interp(mc["FP_exame"], xs, ys)) if xs.min() <= mc["FP_exame"] <= xs.max() else np.nan
    FRONT.append(dict(conjunto=rot, FP_exame_C=mc["FP_exame"], S_C=mc["S"], S_B_mesma_carga=sB, C_acima_fronteira_B=bool(mc["S"] > sB) if np.isfinite(sB) else None, pontos_B=json.dumps(pts)))
for a in ARCH:
    c = d1cv[d1cv.configuracao == a]; avalia(f"D1 OOF (CV aninhada) {a}", c, ex1, nref1.get(a, pd.Series(dtype=int)).to_dict(), tcol="t")
    c = d1[d1.configuracao == a]; avalia(f"D1 OOF (t final, otimista) {a}", c, ex1, nref1.get(a, pd.Series(dtype=int)).to_dict(), t=T[a])
s17 = pd.read_csv(R + "/pulmonar_pos_oof_prep/script17a_final/s17a_20260924T225525_1009246/internal_verifier_object_decisions.csv")
d2 = s17[s17.architecture.isin(ARCH)].merge(Q[Q.conjunto == "D2"].rename(columns={"exame": "patient_id", "configuracao": "architecture", "id": "component_id"}), on=["patient_id", "architecture", "component_id"], how="left")
port["D2_sem_q"] = int(d2.q_media.isna().sum())
d2 = d2.rename(columns={"patient_id": "exame", "architecture": "configuracao", "verifier_probability": "p", "q_media": "q"})
mi = M[M.subset == "internal"]; ex2 = sorted(mi.patient_id.unique()); assert len(ex2) == 158
for a in ARCH:
    nref = mi[(mi.model == a) & mi.status.isin(["TP", "FN"])].groupby("patient_id").size().to_dict()
    for tau, col in [(0.10, "tp_010"), (0.25, "tp_025"), (0.50, "tp_050")]:
        tps = set(zip(mi[(mi.model == a) & (mi[col] == 1)].patient_id, mi[(mi.model == a) & (mi[col] == 1)].pred_component_id.astype(int)))
        c = d2[d2.configuracao == a].copy(); c["status"] = ["TP" if (e, int(i)) in tps else "FP" for e, i in zip(c.exame, c.component_id)]
        avalia(f"D2 IoU {tau:.2f} {a}", c, ex2, nref, t=T[a])
pb = pd.read_csv(O + "/saidas/E1b_predicoes_por_tau.csv", dtype={"coorte": str})
s23 = pd.read_csv(R + "/pulmonar_pos_oof_prep/script23_final/s23_20260924T153750_909396/lndb_verifier_object_scores.csv").rename(columns={"scan_id": "exame", "architecture": "configuracao", "object_id": "id", "p_tp": "p"})
p13 = pd.read_csv(RL + "/saidas/F_escores_pred13.csv").rename(columns={"p_tp": "p"})
PS = pd.concat([s23[["exame", "configuracao", "id", "p"]], p13[["exame", "configuracao", "id", "p"]]])
q4 = Q[Q.conjunto == "D4"][["exame", "configuracao", "id", "q_media", "voxels"]].rename(columns={"q_media": "q"})
CE = pd.read_csv(RL + "/saidas/E1_contagens_por_exame.csv", dtype={"coorte": str})
d4 = pb.merge(PS, on=["exame", "configuracao", "id"], how="left")
ann = json.load(open(R + "/pulmonar_canonical_pre_oof/cohort/lndb_annotations_manifest.json")); ALTINV = {q_["excluded"]: q_["representative"] for q_ in ann["intrapair"]}
d4["exame_pred"] = [ALTINV.get(e, e) if r == "N1_alt" else e for e, r in zip(d4.exame, d4.referencia)]
d4 = d4.drop(columns=["p"]).merge(PS.rename(columns={"exame": "exame_pred"}), on=["exame_pred", "configuracao", "id"], how="left").merge(q4.rename(columns={"exame": "exame_pred"}), on=["exame_pred", "configuracao", "id"], how="left")
port["D4_sem_q"] = int(d4.q.isna().sum()); port["D4_sem_p"] = int(d4.p.isna().sum())
d4 = d4[d4.status != "IGN"]
for coh, partes in [("211", ["211"]), ("224", ["211", "13"])]:
    for ref in ["N1", "N2", "N1_alt"]:
        for a in ARCH:
            for tau in ([0.10, 0.25, 0.50] if (coh == "211" and ref == "N1") else [0.10]):
                ce = CE[CE.coorte.isin(partes) & (CE.referencia == ref) & (CE.configuracao == a) & np.isclose(CE.tau, tau)]
                ex = sorted(ce.exame.unique()); nref = ce.set_index("exame").n_ref.to_dict()
                c = d4[d4.coorte.isin(partes) & (d4.referencia == ref) & (d4.configuracao == a) & np.isclose(d4.tau, tau)]
                avalia(f"D4 {ref} {coh} IoU {tau:.2f} {a}", c, ex, nref, t=T[a])
e2 = pd.read_csv(RL + "/saidas/E2_M.csv", dtype={"coorte": str}); RR = pd.DataFrame(RES)
for a in ARCH:
    b = RR[(RR.conjunto == f"D4 N1 211 IoU 0.10 {a}") & (RR.arquitetura == "B")].iloc[0]; e = e2[(e2.coorte == "211") & (e2.referencia == "N1") & (e2.configuracao == a) & np.isclose(e2.tau, 0.10)].iloc[0]
    port[f"B_igual_Fase2_{a}"] = bool(b.TP == e.TP_pos and b.FP == e.FP_pos and b.TP_detector == e.TP and b.FP_detector == e.FP)
    b2 = RR[(RR.conjunto == f"D2 IoU 0.10 {a}") & (RR.arquitetura == "B")].iloc[0]
    s17m = pd.read_csv(R + "/pulmonar_pos_oof_prep/script17a_final/s17a_20260924T225525_1009246/internal_verifier_metrics.csv")
    q_ = s17m[(s17m.architecture == a) & (s17m.detector == "filtered")].iloc[0]; port[f"B_igual_s17a_{a}"] = bool(b2.TP == q_.tp and b2.FP == q_.fp)
port["aprovada"] = all(v == 0 for k, v in port.items() if k.startswith(("D1_", "D2_", "D4_"))) and all(v for k, v in port.items() if k.startswith("B_igual"))
json.dump(port, open(O + "/saidas/S2_portas.json", "w"), indent=1)
RR.to_csv(O + "/saidas/S2_metricas.csv", index=False); pd.DataFrame(BOOT).to_csv(O + "/saidas/S2_diferencas_C_B.csv", index=False); pd.DataFrame(FRONT).to_csv(O + "/saidas/S2_fronteira_B.csv", index=False)
print(json.dumps(port)); print(json.dumps(T))
if not port["aprovada"]: sys.exit("PORTAS S2 FALHARAM")
