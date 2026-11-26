#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    s04_quadro_candidatos.py

File type:
    Script

Purpose:
    Table of all predicted objects of 3D-F and 3D-FREM with original class, verifier score and decision.
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
    - bd1_filter_features.csv
    - bd1_object_matches.csv
    - ct.nii.gz
    - filter_verifier_threshold.json
    - internal_verifier_object_decisions.csv
    - lndb_filter_features.csv
    - lndb_object_matches.csv
    - lndb_verifier_object_scores.csv

Outputs:
    - A1_quadro_candidatos_por_objeto.csv
    - A1_quadro_candidatos_por_rid.csv
    - GATE_s04.json

Dependencies:
    - SimpleITK
    - numpy
    - pandas
    - scipy

Execution:
    cd <project_dir> && python scripts/32_supplementary_analyses/b_segmentation_geometry/s04_quadro_candidatos.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import hashlib
import json
import os
import sys
import datetime
from multiprocessing import Pool

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage as nd

ROOT = "."
AC = os.path.join(ROOT, "analises_complementares_v8")
OUT = os.path.join(AC, "frenteA")
PLANO_SHA = "d519c9e76a6d8710254a5ed8d028ac5b935ded869067c5977e3a8565e6c1a5f1"
PP = os.path.join(ROOT, "pulmonar_pos_oof_prep")
S11B_MATCH = os.path.join(PP, "script11_final", "s11B_20260921T164539_27883", "analysis", "bd1_object_matches.csv")
S15_FEAT = os.path.join(PP, "script15_final", "s15_20260922T223850_371159", "figdata", "bd1_filter_features.csv")
S17A = os.path.join(PP, "script17a_final", "s17a_20260924T225525_1009246", "internal_verifier_object_decisions.csv")
S17_THR = os.path.join(PP, "script17_final", "s17_20260923T160316_563312", "config", "filter_verifier_threshold.json")
S22 = os.path.join(PP, "script22_final", "s22_20260924T084718_826234", "lndb_object_matches.csv")
S22B = os.path.join(PP, "script22b_final", "s22b_20260924T134153_884131", "lndb_filter_features.csv")
S23 = os.path.join(PP, "script23_final", "s23_20260924T153750_909396", "lndb_verifier_object_scores.csv")
BLK_INT = os.path.join(ROOT, "Pulmonary_Project", "Script10_canonical", "blocks")
BLK_LNDB = os.path.join(ROOT, "Pulmonary_Project", "Script10_canonical_lndb", "blocks")
S06 = os.path.join(ROOT, "Pulmonar_v281_2026-09-01", "08_DADOS_INTERMEDIOS", "s06_20260810T183307_96979")
ARCHS = ["3D_F", "3D_FREM"]
STRUCT_6 = nd.generate_binary_structure(3, 1)

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()

def falha(m):
    sys.exit(f"FAIL: {m}")

def pred_path(cj, ident, a):
    return (os.path.join(BLK_INT, f"lidc_internal__{a}", f"{ident}.nii.gz") if cj == "D2"
            else os.path.join(BLK_LNDB, f"lndb__{a}", f"{ident}.nii.gz"))

def ct_path(cj, ident):
    return os.path.join(S06, "lidc_idri" if cj == "D2" else "lndb", "images", ident, "ct.nii.gz")

def exame(t):
    cj, ident = t
    labs, imgs = {}, {}
    for a in ARCHS:
        img = sitk.ReadImage(pred_path(cj, ident, a))
        labs[a], _ = nd.label(sitk.GetArrayFromImage(img) > 0, structure=STRUCT_6)
        imgs[a] = img
    ct = sitk.ReadImage(ct_path(cj, ident))
    for a in ARCHS:
        if not (imgs[a].GetSize() == ct.GetSize()
                and np.allclose(imgs[a].GetOrigin(), ct.GetOrigin(), atol=1e-4)
                and np.allclose(imgs[a].GetSpacing(), ct.GetSpacing(), atol=1e-6)
                and np.allclose(imgs[a].GetDirection(), ct.GetDirection(), atol=1e-6)):
            return cj, ident, None, f"FAIL_GEOMETRY {cj} {ident} {a} vs TC"
    objs = []
    for a in ARCHS:
        lab = labs[a]
        n = int(lab.max())
        if n == 0:
            continue
        idx = nd.center_of_mass(np.ones_like(lab), lab, range(1, n + 1))
        cnt = np.bincount(lab.ravel(), minlength=n + 1)[1:]
        for k in range(1, n + 1):
            cz, cy, cx = idx[k - 1]
            px, py, pz = ct.TransformContinuousIndexToPhysicalPoint((float(cx), float(cy), float(cz)))
            objs.append({"conjunto": cj, "id": ident, "arch": a, "pred_id": k, "n_vox": int(cnt[k - 1]),
                         "centroide_idx_z": cz, "centroide_idx_y": cy, "centroide_idx_x": cx,
                         "centroide_mm_x": px, "centroide_mm_y": py, "centroide_mm_z": pz})
    a, b = labs["3D_F"], labs["3D_FREM"]
    m = (a > 0) & (b > 0)
    arestas = sorted(set(zip(a[m].tolist(), b[m].tolist())))
    return cj, ident, (objs, arestas), None

def main():
    if sha256(os.path.join(AC, "02_PLANO_ANALITICO_CONGELADO.md")) != PLANO_SHA:
        falha("plano mudou")
    inicio = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    thr = json.load(open(S17_THR))["model_A"]["threshold"]

    m11 = pd.read_csv(S11B_MATCH)
    m11 = m11[(m11.subset == "internal") & m11.model.isin(ARCHS) & m11.pred_component_id.notna()]
    v2 = pd.read_csv(S17A)
    v2 = v2[v2.architecture.isin(ARCHS)]
    f15 = pd.read_csv(S15_FEAT, usecols=lambda c: c in {"subset", "patient_id", "architecture", "group",
                                                         "component_id", "pleura_min_dist_mm"})
    f15 = f15[(f15.subset == "internal") & f15.architecture.isin(ARCHS) & f15.group.isin(["TP_pred", "FP_pred"])]
    d2 = v2.rename(columns={"patient_id": "id", "architecture": "arch", "component_id": "pred_id",
                            "verifier_probability": "escore", "diam_mm": "phi_pred_mm"})
    d2["aceite"] = d2.decision.eq("ACCEPT")
    d2 = d2.merge(m11.rename(columns={"patient_id": "id", "model": "arch", "pred_component_id": "pred_id"})
                  [["id", "arch", "pred_id", "gt_component_id", "status"]], on=["id", "arch", "pred_id"], how="left")
    d2 = d2.merge(f15.rename(columns={"patient_id": "id", "architecture": "arch", "component_id": "pred_id"})
                  [["id", "arch", "pred_id", "pleura_min_dist_mm"]], on=["id", "arch", "pred_id"], how="left")
    d2["conjunto"] = "D2"
    s22 = pd.read_csv(S22)
    s22 = s22[s22.pred_component_id.notna()]
    v4 = pd.read_csv(S23)
    v4 = v4[v4.source == "pred"]
    f22 = pd.read_csv(S22B, usecols=["source", "scan_id", "architecture", "group", "object_id", "pleura_min_dist_mm"])
    f22 = f22[f22.source == "pred"]
    d4 = v4.rename(columns={"scan_id": "id", "architecture": "arch", "object_id": "pred_id", "p_tp": "escore",
                            "retained": "aceite"})
    d4["status_original"] = d4.group.map({"TP_pred": "TP", "FP_pred": "FP"})
    d4 = d4.merge(s22.rename(columns={"scan_id": "id", "architecture": "arch", "pred_component_id": "pred_id",
                                      "diam_mm_pred": "phi_pred_mm"})
                  [["id", "arch", "pred_id", "gt_component_id", "status", "phi_pred_mm"]],
                  on=["id", "arch", "pred_id"], how="left")
    d4 = d4.merge(f22.rename(columns={"scan_id": "id", "architecture": "arch", "object_id": "pred_id"})
                  [["id", "arch", "pred_id", "pleura_min_dist_mm"]], on=["id", "arch", "pred_id"], how="left")
    d4["conjunto"] = "D4"
    cols = ["conjunto", "id", "arch", "pred_id", "status", "status_original", "gt_component_id", "aceite", "escore",
            "phi_pred_mm", "pleura_min_dist_mm"]
    q = pd.concat([d2[cols], d4[cols]], ignore_index=True)
    for c in ["status", "pleura_min_dist_mm", "phi_pred_mm", "escore"]:
        if q[c].isna().any():
            falha(f"{int(q[c].isna().sum())} objetos sem {c}")
    if (q.status != q.status_original).any():
        falha("estado do verificador diverge da associação")
    if ((q.escore >= thr) != q.aceite.astype(bool)).any():
        falha("decisão incoerente com θ")
    q["aceite"] = q.aceite.astype(bool)
    q["pred_id"] = q.pred_id.astype(int)

    exames = sorted(set(zip(q.conjunto, q.id)))
    objs, arestas, erros = [], [], []
    with Pool(8) as pool:
        for cj, ident, res, err in pool.imap_unordered(exame, exames):
            if err:
                erros.append(err)
                continue
            o, ar = res
            objs.extend(o)
            arestas.extend((cj, ident, f, r) for f, r in ar)
    if erros:
        falha("; ".join(erros[:5]))
    g = pd.DataFrame(objs)
    q = q.merge(g, on=["conjunto", "id", "arch", "pred_id"], how="outer", indicator=True)
    if (q._merge != "both").any():
        falha(f"objetos sem correspondência entre máscara e registos: {q._merge.value_counts().to_dict()}")
    q = q.drop(columns="_merge")

    pai = {}

    def acha(x):
        while pai.setdefault(x, x) != x:
            pai[x] = pai[pai[x]]
            x = pai[x]
        return x
    for cj, ident, f, r in arestas:
        pai[acha((cj, ident, "3D_F", f))] = acha((cj, ident, "3D_FREM", r))
    q["raiz"] = [acha((r.conjunto, r.id, r.arch, r.pred_id)) for r in q.itertuples()]
    q = q.sort_values(["conjunto", "id", "centroide_idx_z", "centroide_idx_y", "centroide_idx_x", "arch"])
    rid = {}
    for (cj, ident), h in q.groupby(["conjunto", "id"], sort=False):
        k = 0
        for rz in h.raiz:
            if rz not in rid:
                k += 1
                rid[rz] = f"{cj}-{ident}-R{k:03d}"
    q["rid"] = q.raiz.map(rid)
    q = q.drop(columns="raiz")
    q["celula_objeto"] = q.status + "-" + np.where(q.aceite, "aceite", "rejeitado")
    q["contacto_pleural"] = q.pleura_min_dist_mm.eq(0)
    q["referencia_anotada"] = q.gt_component_id.notna()
    q["theta"] = thr

    def celula(h):
        c = set(h.celula_objeto)
        return c.pop() if len(c) == 1 else "misto"

    def estr(phi):
        return "<6" if phi < 6 else ("6-15" if phi < 15 else ">=15")
    r = (q.groupby("rid")
          .apply(lambda h: pd.Series({
              "conjunto": h.conjunto.iat[0], "id": h.id.iat[0], "n_membros": len(h),
              "membros": ";".join(f"{a}:{p}" for a, p in zip(h.arch, h.pred_id)),
              "configuracoes": "+".join(sorted(set(h.arch))), "celula": celula(h),
              "phi_pred_max_mm": h.phi_pred_mm.max(), "estrato_tamanho": estr(h.phi_pred_mm.max()),
              "contacto_pleural_algum": bool(h.contacto_pleural.any()),
              "algum_tp": bool((h.status == "TP").any()),
              "centroide_mm_x": h.centroide_mm_x.mean(), "centroide_mm_y": h.centroide_mm_y.mean(),
              "centroide_mm_z": h.centroide_mm_z.mean(),
              "centroide_idx_z": h.centroide_idx_z.mean(), "centroide_idx_y": h.centroide_idx_y.mean(),
              "centroide_idx_x": h.centroide_idx_x.mean()}), include_groups=False)
          .reset_index())
    os.makedirs(OUT, exist_ok=True)
    p1 = os.path.join(OUT, "A1_quadro_candidatos_por_objeto.csv")
    p2 = os.path.join(OUT, "A1_quadro_candidatos_por_rid.csv")
    q.to_csv(p1, index=False)
    r.to_csv(p2, index=False)
    resumo = {
        "objetos_por_conjunto_config_celula": {f"{a}|{b}|{c}": int(n) for (a, b, c), n in
                                               q.groupby(["conjunto", "arch", "celula_objeto"]).size().items()},
        "rid_por_conjunto_celula": {f"{a}|{b}": int(n) for (a, b), n in r.groupby(["conjunto", "celula"]).size().items()},
        "rid_por_conjunto_celula_estrato": {f"{a}|{b}|{c}": int(n) for (a, b, c), n in
                                            r.groupby(["conjunto", "celula", "estrato_tamanho"]).size().items()},
        "rid_por_conjunto": r.groupby("conjunto").size().to_dict(),
        "rid_com_mais_de_2_membros": int((r.n_membros > 2).sum()),
    }
    gate = {"script": os.path.basename(__file__), "script_sha256": sha256(__file__), "plano_sha256": PLANO_SHA,
            "inicio": inicio, "fim": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "entradas_sha256": {os.path.basename(p): sha256(p) for p in
                                (S11B_MATCH, S15_FEAT, S17A, S17_THR, S22, S22B, S23)},
            "saidas_sha256": {os.path.basename(p): sha256(p) for p in (p1, p2)},
            "resumo": resumo, "estado": "PASS"}
    json.dump(gate, open(os.path.join(OUT, "GATE_s04.json"), "w"), indent=2, ensure_ascii=False)
    print(json.dumps(resumo, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
