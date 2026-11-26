#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    c10_desfechos.py

File type:
    Script

Purpose:
    Joins the frozen link with the detection outcomes and verifier decisions.
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
    - c08_ligacao_congelada.SHA256
    - c10_tabelas.md

Outputs:
    - c03_decisoes.csv
    - c04_esferas_verificador.csv
    - c04_referencias_verificador.csv
    - c08_ligacao_congelada.csv
    - c08_oof_objetos_radiologist.csv
    - c10_objetos.csv
    - c10_opiniao_radiologista.csv
    - c10_registos.csv

Dependencies:
    - numpy
    - pandas

Execution:
    cd <project_dir> && python scripts/30_radiological_assessment/c10_desfechos.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import hashlib, re
import numpy as np, pandas as pd
O = "resultados/radiological_reanalysis"
for l in open(f"{O}/c08_ligacao_congelada.SHA256"):
    if l.startswith("#"): continue
    h, n = l.split(); assert hashlib.sha256(open(f"{O}/{n}", "rb").read()).hexdigest() == h, n
for l in open(f"{O}/c02_SHA256SUMS"):
    h, n = l.split(); assert hashlib.sha256(open(f"{O}/{n}", "rb").read()).hexdigest() == h, n
L = pd.read_csv(f"{O}/c08_ligacao_congelada.csv", dtype={"exame": str})
op = pd.read_csv(f"{O}/c10_opiniao_radiologista.csv", dtype={"exame": str})
L = L.merge(op[["exame", "registo", "classe", "opiniao_radiologista"]], on=["exame", "registo", "classe"], how="left", validate="1:1")
assert L.opiniao_radiologista.notna().all()
esf = pd.read_csv(f"{O}/c04_esferas_verificador.csv", dtype={"exame": str}).set_index(["exame", "esfera"])
ref = pd.read_csv(f"{O}/c04_referencias_verificador.csv", dtype={"exame": str, "3D_F_pred": str, "3D_FREM_pred": str}).set_index(["exame", "ref_obj"])
dec = pd.read_csv(f"{O}/c03_decisoes.csv"); dec["exame"] = dec.patient_id.str[-4:]
INS = {(r.exame, r.architecture, int(r.component_id)): (r.p_A, bool(r.aceite)) for r in dec.itertuples()}
oo = pd.read_csv(f"{O}/c08_oof_objetos_radiologist.csv"); oo["exame"] = oo.patient_id.str[-4:]
OOF = {(r.exame, r.architecture, int(r.component_id)): (r.p_oof, bool(r.aceite_oof)) for r in oo.itertuples()}
D2 = {"0097", "1008"}
rows, conflito = [], []
for r in L[L.ligado | (L.ligacao_final == "ambígua")].itertuples():
    for t in r.alvos_finais.split("/"):
        tt = t.strip()
        for a in ("3D_F", "3D_FREM"):
            if tt.startswith("ref"):
                k = int(tt.split()[1]); rr = ref.loc[(r.exame, k)]; st = rr[f"{a}_estado"]
                obj = int(float(rr[f"{a}_pred"])) if st == "TP" else None; alvo = f"ref {k}"
            else:
                k = int(tt[1:]); e = esf.loc[(r.exame, k)]; alvo = f"m{k}"
                if e.ref_ate5mm:
                    rr = ref.loc[(r.exame, int(e.ref_obj))]; st = rr[f"{a}_estado"]
                    obj = int(float(rr[f"{a}_pred"])) if st == "TP" else None
                    if e[f"{a}_ate5mm"] and e[f"{a}_estado"] == "FP": conflito.append((r.n, alvo, a))
                elif e[f"{a}_ate5mm"]:
                    st, obj = e[f"{a}_estado"], int(e[f"{a}_obj"])
                else:
                    st, obj = "sem objeto", None
            conj = "D2" if r.exame in D2 else "D1"
            p, ac = (INS if conj == "D2" else OOF).get((r.exame, a, obj), (np.nan, None)) if obj is not None else (np.nan, None)
            pi, aci = INS.get((r.exame, a, obj), (np.nan, None)) if obj is not None else (np.nan, None)
            if obj is not None: assert ac is not None, (r.n, alvo, a)
            rows.append(dict(n=r.n, exame=r.exame, conjunto=conj, registo=r.registo, classe=r.classe,
                             opiniao=r.opiniao_radiologista, ligacao=r.ligacao_final, alvo=alvo, configuracao=a,
                             estado=st, objeto=obj, p=round(p, 3) if obj is not None else np.nan,
                             decisao=("aceito" if ac else "rejeitado") if obj is not None else "",
                             p_dentro_amostra=round(pi, 3) if (obj is not None and conj == "D1") else np.nan,
                             decisao_dentro_amostra=(("aceito" if aci else "rejeitado") if (obj is not None and conj == "D1") else "")))
X = pd.DataFrame(rows); X.to_csv(f"{O}/c10_objetos.csv", index=False)
print("conflitos marcador sobre referencia e sobre FP:", conflito)
X["desfecho"] = np.where(X.decisao != "", X.estado + " " + X.decisao, X.estado)
def op3(s):
    return "nódulo" if s.startswith("Nódulo") else ("não nódulo" if s.startswith("Não nódulo") else "inconclusivo")
X["grupo"] = X.classe + " / " + X.opiniao.map(op3)
lig = X[X.ligacao != "ambígua"]
out = []
out.append("# Resultados da transferência (ligação congelada 44b68dab…)\n")
def md(df):
    df = df.reset_index(); c = [str(x) for x in df.columns]
    return "| " + " | ".join(c) + " |\n|" + "---|" * len(c) + "\n" + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)) + "\n"
t8 = pd.crosstab(L.classe, L.ligacao_final, margins=True, margins_name="Total")
cj = L.exame.isin(D2).map({True: "D2", False: "D1"})
t8 = pd.concat([pd.crosstab(L.classe, cj, margins=True, margins_name="Total"), t8.drop(columns="Total")], axis=1)
out.append("## Tabela 8 nova: registos por classe e grau de ligação\n\n" + md(t8.reindex(["TP", "FP", "FN", "Total"])))
for a in ("3D_F", "3D_FREM"):
    for c in ("D1", "D2"):
        s = lig[(lig.configuracao == a) & (lig.conjunto == c)]
        out.append(f"## Objetos ligados × desfecho atual: {a}, {c} ({'escore fora da dobra' if c=='D1' else 'decisão congelada'})\n\n" + md(pd.crosstab(s.grupo, s.desfecho, margins=True, margins_name="Total")))
s = lig[lig.conjunto == "D1"].copy(); s["muda"] = s.decisao != s.decisao_dentro_amostra
out.append(f"## Complemento D1: decisões diferentes entre o escore fora da dobra e o verificador final (dentro da amostra): {int(s[s.decisao!=''].muda.sum())} de {int((s.decisao!='').sum())} objetos ligados\n\n" + md(s[s.muda & (s.decisao != '')][["n", "exame", "registo", "alvo", "configuracao", "estado", "p", "decisao", "p_dentro_amostra", "decisao_dentro_amostra"]].set_index("n")))
def resume(g, a):
    g = g[g.configuracao == a]
    if g.empty: return "--"
    v = g.desfecho.value_counts()
    return "; ".join(f"{k}" if n == 1 and len(g) == 1 else f"{k} ({n})" for k, n in v.items())
R = L[["n", "exame", "registo", "classe", "opiniao_radiologista", "ligacao_final", "alvos_finais"]].copy()
R["conjunto"] = R.exame.isin(D2).map({True: "D2", False: "D1"})
for a in ("3D_F", "3D_FREM"):
    R[a] = [resume(X[X.n == n], a) for n in R.n]
R.to_csv(f"{O}/c10_registos.csv", index=False)
out.append("## S20 nova: 58 registos\n\n" + md(R.set_index("n")))
open(f"{O}/c10_tabelas.md", "w").write("\n".join(out))
print(t8.reindex(["TP", "FP", "FN", "Total"]).to_string())
for a in ("3D_F",):
    for c in ("D1", "D2"):
        s = lig[(lig.configuracao == a) & (lig.conjunto == c)]
        print(f"\n{a} {c}\n", pd.crosstab(s.grupo, s.desfecho, margins=True).to_string())
