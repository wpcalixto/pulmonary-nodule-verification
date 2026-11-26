#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    F_respostas.py

File type:
    Script

Purpose:
    Extracts the sixteen morphological responses, the largest axial diameter and the frozen Model A score of the LNDb predicted objects through the stage 13/14/15 modules.
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
    - ct.nii.gz
    - filter_verifier.pkl
    - input_manifest.csv
    - lndb_filter_features.csv
    - lndb_verifier_object_scores.csv
    - lung_mask_crop.nii.gz
    - {e}.nii.gz

Outputs:
    - F_escores_pred13.csv
    - F_porta_paridade.json
    - F_respostas.csv

Dependencies:
    - SimpleITK
    - numpy
    - pandas
    - scipy

Execution:
    cd <project_dir> && python scripts/28_lndb_reference_rectification/F_respostas.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import os, sys, json, hashlib, importlib.util, ast, csv, pickle
os.environ["PULMONARY_BASE"] = "resultados/retificacao_LNDb_20261111/sandbox"
import numpy as np, pandas as pd, scipy.ndimage as nd, SimpleITK as sitk
from multiprocessing import Pool
P = "."; R = P + "/resultados"; O = R + "/retificacao_LNDb_20261111"; SB = O + "/sandbox/scripts"
sys.path.insert(0, SB)
CAN = {"13_prepare_filter_objects.py": "18533b0b1d53c14d46ec08637b3629afdc9846f1dcb2d04630fb8cabcdc3762a",
       "14_compute_filter_maps_gpu.py": "64fe7a82c5cec0146371f7cc7b9c6927932a8816110890f4c293b82cb1e1ea99",
       "15_extract_filter_features.py": "d9adb7af9bf7c0b39c2020118abafd5e85ccf2befb595319e2e6a47267468497"}
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def load(n, f):
    assert sha(f"{SB}/{f}") == CAN[f], f; s = importlib.util.spec_from_file_location(n, f"{SB}/{f}"); m = importlib.util.module_from_spec(s); sys.modules[n] = m; s.loader.exec_module(m); return m
S13 = load("s13", "13_prepare_filter_objects.py"); S14 = load("s14", "14_compute_filter_maps_gpu.py"); S15 = load("s15", "15_extract_filter_features.py")
_src = open(f"{SB}/11_match_and_score_bd1.py").read(); assert sha(f"{SB}/11_match_and_score_bd1.py").startswith("438c50d3")
_ns = {"np": np}; exec(ast.get_source_segment(_src, [x for x in ast.parse(_src).body if isinstance(x, ast.FunctionDef) and x.name == "diam_mm"][0]), _ns); diam_mm = _ns["diam_mm"]
CROP = R + "/Pulmonar_v281_2026-09-01/08_DADOS_INTERMEDIOS/s06_20260810T183307_96979"; LM211 = O + "/s06_mascaras"; C13 = O + "/s06_13"
C_GLOBAL = 499.89288330078125; STRUCT_6 = nd.generate_binary_structure(3, 1)
F16 = ["log_max", "log_mean", "log_p95", "log_scale_mm", "blob_max", "blob_mean", "blob_p95", "blob_scale_mm", "vessel_max", "vessel_mean", "vessel_p95", "vessel_scale_mm",
       "contrast_hu", "centroid_peak_dist_mm", "pleura_min_dist_mm", "pleura_centroid_dist_mm"]
S22D = R + "/pulmonar_pos_oof_prep/script22_final/s22_20260924T084718_826234"
S22B = R + "/pulmonar_pos_oof_prep/script22b_final/s22b_20260924T134153_884131/lndb_filter_features.csv"
S23 = R + "/pulmonar_pos_oof_prep/script23_final/s23_20260924T153750_909396/lndb_verifier_object_scores.csv"
PKL = R + "/pulmonar_pos_oof_prep/script17_final/s17_20260923T160316_563312/models/filter_verifier.pkl"
PRED211 = R + "/Pulmonary_Project/Script10_canonical_lndb/blocks/lndb__{a}/{e}.nii.gz"; PRED13 = O + "/pred13/{a}/{e}.nii.gz"
EX211 = sorted(r["scan_id"] for r in csv.DictReader(open(S22D + "/input_manifest.csv")))
EX13 = ["LNDb-0005", "LNDb-0007", "LNDb-0020", "LNDb-0037", "LNDb-0043", "LNDb-0044", "LNDb-0049", "LNDb-0109", "LNDb-0123", "LNDb-0124", "LNDb-0126", "LNDb-0171", "LNDb-0179"]
PAR_PRED = ["LNDb-0001", "LNDb-0002", "LNDb-0003", "LNDb-0004", "LNDb-0010"]
def feats(ct_arr, lab, binall, sp_zyx, cid, dist, lung, rot):
    r = S13.object_record(lab, cid, sp_zyx, ct_arr.shape, rot)
    bbox = {k: r[k] for k in ("bbox_z0", "bbox_z1", "bbox_y0", "bbox_y1", "bbox_x0", "bbox_x1")}
    out, err = S14.process_roi(ct_arr, lab, binall, sp_zyx, cid, bbox, C_GLOBAL, "cpu", rot)
    if err is not None: raise SystemExit(f"ROI {rot}: {err}")
    f = dict(out[0]); f.update(S15.pleural_features(lab == cid, dist, lung, sp_zyx, rot)); return {k: f[k] for k in F16}
def um(e):
    if e in EX13: ctp, lmp = f"{C13}/lndb/images/{e}/ct.nii.gz", f"{C13}/lung_masks/lndb/{e}/lung_mask_crop.nii.gz"
    else:
        ctp = f"{CROP}/lndb/images/{e}/ct.nii.gz"; lmp = f"{CROP}/lung_masks/lndb/{e}/lung_mask_crop.nii.gz"
        if not os.path.exists(lmp): lmp = f"{LM211}/lung_masks/lndb/{e}/lung_mask_crop.nii.gz"
    ct_img = sitk.ReadImage(ctp, sitk.sitkFloat32); sp = ct_img.GetSpacing(); sp_zyx = (float(sp[2]), float(sp[1]), float(sp[0]))
    ct_arr = np.clip(sitk.GetArrayFromImage(ct_img), *S14.HU_CLIP).astype(np.float32)
    dist, lung, _ = S15.pleura_edt(lmp, ct_img, e); dist = dist.astype(np.float32)
    Z0 = np.load(f"{O}/referencias/{e}.npz")
    rd = sitk.ImageFileReader(); rd.SetFileName(f"{P}/dados/brutos/raw_5070_2026-08/lndb/data/{e}.mhd"); rd.ReadImageInformation()
    full = sitk.Image([1, 1, 1], sitk.sitkUInt8); full.SetOrigin(rd.GetOrigin()); full.SetSpacing(rd.GetSpacing()); full.SetDirection(rd.GetDirection())
    ci = np.array(full.TransformPhysicalPointToContinuousIndex(ct_img.GetOrigin())); assert np.abs(ci - np.round(ci)).max() <= 1e-3; o = np.round(ci).astype(int)
    Zs, Ys, Xs = ct_arr.shape; crop = lambda a: a[o[2]:o[2] + Zs, o[1]:o[1] + Ys, o[0]:o[0] + Xs]
    out = []
    n1 = Z0["N1"].astype(np.int32); nc = crop(n1); fids = Z0["N1_findings"]
    for cid in range(1, int(n1.max()) + 1):
        base = dict(tipo="N1", exame=e, id=cid, FindingID=int(fids[cid - 1]), diam_axial_mm=diam_mm((n1 == cid).astype(np.uint8), rd.GetSpacing()), voxels=int((n1 == cid).sum()), voxels_no_recorte=int((nc == cid).sum()))
        out.append({**base, **(feats(ct_arr, nc, nc > 0, sp_zyx, cid, dist, lung, f"N1/{e}/{cid}") if base["voxels_no_recorte"] else {})})
    if e in EX211:
        old, n_old = nd.label(Z0["ref_antiga"], structure=STRUCT_6); oc = crop(old).astype(np.int32)
        for cid in range(1, n_old + 1):
            if (oc == cid).any(): out.append(dict(tipo="antiga", exame=e, id=cid, **feats(ct_arr, oc, oc > 0, sp_zyx, cid, dist, lung, f"antiga/{e}/{cid}")))
    if e in EX13 or e in PAR_PRED:
        for a in ["3D_F", "3D_FREM"]:
            pp = (PRED13 if e in EX13 else PRED211).format(a=a, e=e)
            pr = sitk.ReadImage(pp); assert np.allclose(pr.GetOrigin(), ct_img.GetOrigin(), atol=1e-3) and pr.GetSize() == ct_img.GetSize()
            pl, npr = nd.label(sitk.GetArrayFromImage(pr) > 0, structure=STRUCT_6); pl = pl.astype(np.int32)
            for cid in range(1, npr + 1):
                out.append(dict(tipo="pred", exame=e, configuracao=a, id=cid, diam_mm=diam_mm((pl == cid).astype(np.uint8), (sp[0], sp[1], sp[2])), voxels=int((pl == cid).sum()),
                                **feats(ct_arr, pl, pl > 0, sp_zyx, cid, dist, lung, f"pred/{a}/{e}/{cid}")))
    print(e, flush=True); return out
if __name__ == "__main__":
    M = pickle.load(open(PKL, "rb")); thr = M["threshold"]; FEAT = M["features"]
    s22b = pd.read_csv(S22B); s23 = pd.read_csv(S23)
    m = s22b[s22b.source == "pred"].merge(s23, left_on=["scan_id", "architecture", "object_id"], right_on=["scan_id", "architecture", "object_id"], how="inner")
    m["p_rec"] = M["model"].predict_proba(m[FEAT])[:, 1]
    porta_c = dict(objetos=len(m), max_dif=float((m.p_rec - m.p_tp).abs().max()), decisoes_iguais=bool(((m.p_rec >= thr) == m.retained).all()))
    with Pool(7) as p: RES = p.map(um, EX211 + EX13, chunksize=1)
    df = pd.DataFrame([x for r in RES for x in r]); df.to_csv(O + "/saidas/F_respostas.csv", index=False)
    g = s22b[s22b.source == "gt"].drop_duplicates(["scan_id", "object_id"])
    a = df[df.tipo == "antiga"].merge(g, left_on=["exame", "id"], right_on=["scan_id", "object_id"], suffixes=("", "_22b"))
    pa = df[(df.tipo == "pred") & df.exame.isin(PAR_PRED)].merge(s22b[s22b.source == "pred"], left_on=["exame", "configuracao", "id"], right_on=["scan_id", "architecture", "object_id"], suffixes=("", "_22b"))
    npa = int(((s22b.source == "pred") & s22b.scan_id.isin(PAR_PRED)).sum())
    porta = dict(a_ref_antiga=dict(comparadas=len(a), esperadas=len(g), max_dif=max(float((a[k] - a[k + "_22b"]).abs().max()) for k in F16)),
                 b_pred=dict(comparadas=len(pa), esperadas=npa, max_dif=max(float((pa[k] - pa[k + "_22b"]).abs().max()) for k in F16) if len(pa) else None,
                             max_dif_diam=float((pa.diam_mm - pa.diam_mm_22b).abs().max()) if len(pa) else None),
                 c_modelo=porta_c)
    porta["aprovada"] = bool(porta["a_ref_antiga"]["comparadas"] == porta["a_ref_antiga"]["esperadas"] and porta["a_ref_antiga"]["max_dif"] <= 1e-9
                             and porta["b_pred"]["comparadas"] == npa and porta["b_pred"]["max_dif"] <= 1e-9 and porta_c["max_dif"] <= 1e-12 and porta_c["decisoes_iguais"])
    json.dump(porta, open(O + "/saidas/F_porta_paridade.json", "w"), indent=1, default=float); print(json.dumps(porta, default=float))
    if not porta["aprovada"]: sys.exit("PARIDADE FALHOU")
    q = df[(df.tipo == "pred") & df.exame.isin(EX13)].copy()
    q["p_tp"] = M["model"].predict_proba(q[FEAT])[:, 1] if len(q) else []; q["retained"] = q.p_tp >= thr
    q.to_csv(O + "/saidas/F_escores_pred13.csv", index=False)
