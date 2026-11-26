#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    N2_concordancia_leitores_LNDb.py

File type:
    Script

Purpose:
    Inter-reader identification agreement in LNDb through the official finding groupings.
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
    - Outputs of the preceding pipeline stages (see docs/post_v1_stages.md).

Outputs:
    - N2_pares.csv
    - N2_resumo.csv
    - trainCTs.csv
    - trainNodules.csv
    - trainNodules_gt.csv

Dependencies:
    - numpy
    - pandas

Execution:
    cd <project_dir> && python scripts/32_supplementary_analyses/d_interobserver_matching_and_partition/N2_concordancia_leitores_LNDb.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import sys, numpy as np, pandas as pd, itertools
RAW, LES, AUD, OUT = sys.argv[1:5]
ct = pd.read_csv(RAW + '/trainCTs.csv'); tn = pd.read_csv(RAW + '/trainNodules.csv'); gt = pd.read_csv(RAW + '/trainNodules_gt.csv', dtype=str)
les = pd.read_csv(LES); aud = pd.read_csv(AUD)
NOD = {(int(r.LNDbID), int(r.RadID), int(r.FindingID)): int(r.Nodule) for r in tn.itertuples()}
vox = {(r.lndb_id, int(r.FindingID)): [int(v) for v in str(r.voxels_por_leitor).split('|')] for r in les.itertuples()}
radn = {f"LNDb-{int(r.LNDbID):04d}": int(r.RadN) for r in ct.itertuples()}
marc = {}
for r in gt.itertuples():
    e = f"LNDb-{int(r.LNDbID):04d}"; f = int(r.FindingID)
    rads = [int(x) for x in str(r.RadID).split(',')]; rfs = [int(float(x)) for x in str(r.RadFindingID).split(',')]
    vv = vox[(e, f)]; assert len(vv) == len(rads)
    for rad, rf, nv in zip(rads, rfs, vv):
        if NOD[(int(r.LNDbID), rad, rf)] == 1:
            marc.setdefault((e, 'todos'), {}).setdefault(rad, set()).add(f)
            if nv > 1: marc.setdefault((e, 'segmentados'), {}).setdefault(rad, set()).add(f)
coortes = {'211': set(aud[aud.grupo == 'incluido_211'].lndb_id), '224': set(aud[aud.grupo.isin(['incluido_211', 'referencia_vazia_com_achados_13'])].lndb_id)}
assert len(coortes['211']) == 211 and len(coortes['224']) == 224
P = []
for c, ex in coortes.items():
    for e in sorted(ex):
        R = radn[e]
        if R < 2: continue
        for var in ['todos', 'segmentados']:
            m = marc.get((e, var), {})
            for a, b in itertools.combinations(range(1, R + 1), 2):
                A, B = m.get(a, set()), m.get(b, set()); u = len(A | B)
                if u == 0: continue
                P.append(dict(coorte=c, variante=var, exame=e, leitores=R, par=f"{a}-{b}", n_A=len(A), n_B=len(B), n_ambos=len(A & B), n_uniao=u, concordancia=len(A & B) / u))
P = pd.DataFrame(P); P.to_csv(OUT + '/N2_pares.csv', index=False)
S = []
for (c, var), g in P.groupby(['coorte', 'variante']):
    ex = g.exame.unique(); idx = {e: np.where(g.exame.values == e)[0] for e in ex}; v = g.concordancia.values
    rng = np.random.default_rng(270014); bs = []
    for _ in range(2000):
        d = rng.choice(ex, len(ex)); ii = np.concatenate([idx[e] for e in d]); bs.append(v[ii].mean())
    tot_amb, tot_un = g.n_ambos.sum(), g.n_uniao.sum()
    S.append(dict(coorte=c, variante=var, exames=len(ex), exames_2_leitores=int(g.drop_duplicates('exame').leitores.eq(2).sum()), exames_3_leitores=int(g.drop_duplicates('exame').leitores.eq(3).sum()),
                  pares=len(g), media=v.mean(), ic_lo=np.percentile(bs, 2.5), ic_hi=np.percentile(bs, 97.5), mediana=np.median(v),
                  q1=np.percentile(v, 25), q3=np.percentile(v, 75), frac_pares_zero=(v == 0).mean(), frac_pares_um=(v == 1).mean(), agregada=tot_amb / tot_un, n_ambos=int(tot_amb), n_uniao=int(tot_un)))
S = pd.DataFrame(S); S.to_csv(OUT + '/N2_resumo.csv', index=False); print(S.round(3).to_string())
