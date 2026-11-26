#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    P1_recorte_s06_replica.py

File type:
    Script

Purpose:
    Strict reproduction of the lung-mask generation and cropping of stage 06 for the 13 LNDb scans outside the analytical cohort, with byte-level parity gates against the recorded outputs.
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
    - lung_mask.nii.gz
    - lung_mask_crop.nii.gz

Outputs:
    - P1_porta.json

Dependencies:
    - SimpleITK
    - lungmask
    - numpy
    - torch

Execution:
    cd <project_dir> && python scripts/28_lndb_reference_rectification/P1_recorte_s06_replica.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import os, sys, json, hashlib, platform
import numpy as np, SimpleITK as sitk
IN, OUT = "tmp/s06rep/in", "tmp/s06rep/out"; MARGIN_MM = 10.0; INVERSE_TOL_MM = 1e-3
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
_INF = None
def _lungmask(ct):
    global _INF
    if _INF is None:
        from lungmask import LMInferer
        _INF = LMInferer(modelname="R231", force_cpu=False, tqdm_disable=True)
        assert getattr(_INF.device, "type", str(_INF.device)) == "mps" and _INF.modelname == "R231"
    seg = _INF.apply(ct)
    arr = sitk.GetArrayFromImage(seg) if isinstance(seg, sitk.Image) else np.asarray(seg)
    lung = (arr > 0).astype(np.uint8); img = sitk.GetImageFromArray(lung); img.CopyInformation(ct); return img, lung
def _caixa(lung_arr, spacing_xyz):
    coords = np.argwhere(lung_arr > 0); sp = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]); forma = lung_arr.shape; bbox = []
    for eixo in range(3):
        m_vox = int(np.ceil(MARGIN_MM / sp[eixo])); lo_p = int(coords[:, eixo].min()); hi_p = int(coords[:, eixo].max())
        bbox += [max(lo_p - m_vox, 0), min(hi_p + m_vox + 1, forma[eixo])]
    return tuple(bbox)
def _recortar(img, bbox):
    z0, z1, y0, y1, x0, x1 = bbox; arr = sitk.GetArrayFromImage(img)[z0:z1, y0:y1, x0:x1]; out = sitk.GetImageFromArray(arr)
    out.SetOrigin(img.TransformIndexToPhysicalPoint((int(x0), int(y0), int(z0)))); out.SetSpacing(img.GetSpacing()); out.SetDirection(img.GetDirection()); return out
def _inversa(ct, ct_crop, bbox):
    z0, _, y0, _, x0, _ = bbox
    for idx in ((0, 0, 0), (1, 2, 3)):
        d = float(np.max(np.abs(np.asarray(ct_crop.TransformIndexToPhysicalPoint(idx)) - np.asarray(ct.TransformIndexToPhysicalPoint((idx[0] + x0, idx[1] + y0, idx[2] + z0))))))
        assert d <= INVERSE_TOL_MM
def caso(e):
    ct = sitk.ReadImage(f"{IN}/{e}/ct.nii.gz"); arr = sitk.GetArrayFromImage(ct); assert np.isfinite(arr).all()
    lung_img, lung = _lungmask(ct); assert lung.sum() > 0
    bbox = _caixa(lung, ct.GetSpacing()); ct_c = _recortar(ct, bbox); _inversa(ct, ct_c, bbox); lm_c = _recortar(lung_img, bbox)
    os.makedirs(f"{OUT}/lndb/images/{e}", exist_ok=True); os.makedirs(f"{OUT}/lung_masks/lndb/{e}", exist_ok=True)
    sitk.WriteImage(ct_c, f"{OUT}/lndb/images/{e}/ct.nii.gz", useCompression=True)
    sitk.WriteImage(lung_img, f"{OUT}/lung_masks/lndb/{e}/lung_mask.nii.gz", useCompression=True)
    sitk.WriteImage(lm_c, f"{OUT}/lung_masks/lndb/{e}/lung_mask_crop.nii.gz", useCompression=True)
    return dict(exame=e, sha_ct_fonte=sha(f"{IN}/{e}/ct.nii.gz"), crop_bbox=list(bbox), sha_ct_crop=sha(f"{OUT}/lndb/images/{e}/ct.nii.gz"),
                sha_lung_mask_crop=sha(f"{OUT}/lung_masks/lndb/{e}/lung_mask_crop.nii.gz"), cropped_shape=list(sitk.GetArrayFromImage(ct_c).shape))
if __name__ == "__main__":
    import torch, lungmask, importlib.metadata as md
    env = dict(python=platform.python_version(), torch=torch.__version__, lungmask=md.version("lungmask"), simpleitk=sitk.Version_VersionString(), numpy=np.__version__)
    assert env == dict(python="3.9.6", torch="2.8.0", lungmask="0.2.20", simpleitk="2.5.5", numpy="2.0.2"), env
    ref = {"LNDb-0001": ("f917eb77da40cfc4c5d04aefa692d98ebc18840fcb9ab10404f1a1ba3a38fd76", "c59f42175d699c3d0c7e8002c07ba405fbd3e7b833eb4f9a81e416a41d4fb76a", "b214e76fe18cad40686a52819ce474d99d6abc2b19c21c12517c010216cc6d24"),
           "LNDb-0002": ("cc506a885fd012a39876c9c5afd8ee2cac62bc81c32b88cc71d6fb893742d6ae", "65426b6235e0081409b23b921d25242208578a8170a2aa6ff63cb23210f47f2e", "ae1da0c984004766627a0c41e8f6b69f9b287c09f62f192daf8696b474a49ebd"),
           "LNDb-0003": ("38f79b8ac917e14a5d5f0c96fffa89ed2699e251a11b64fc0b8c52b82867a67c", "91532833f6d45da2f52381f9be91dee0f5b49a1c22d16cadfc818a739245f798", "efb442f1ce6ce735a04dde5a69512814f656621c44e54a29b990e776d8807a5f")}
    res = {"ambiente": env, "porta": [], "casos13": []}
    for e, (sf, sc, sl) in ref.items():
        r = caso(e); r["ok"] = (r["sha_ct_fonte"] == sf, r["sha_ct_crop"] == sc, r["sha_lung_mask_crop"] == sl); res["porta"].append(r); print(e, r["ok"], flush=True)
    res["porta_aprovada"] = all(all(r["ok"]) for r in res["porta"])
    pesos = os.path.expanduser("~/.cache/torch/hub/checkpoints/unet_r231-d5d2fc3d.pth")
    res["pesos_sha256"] = sha(pesos) if os.path.exists(pesos) else "NAO_LOCALIZADO"
    res["pesos_ok"] = res["pesos_sha256"] == "d5d2fc3df432115933491f115b283e94d1f5c841c0e82f6e7bd2d65d3ccde69f"
    json.dump(res, open(f"{OUT}/P1_porta.json", "w"), indent=1)
    if not (res["porta_aprovada"] and res["pesos_ok"]): sys.exit("PORTA P1 FALHOU")
    for e in ["LNDb-0005", "LNDb-0007", "LNDb-0020", "LNDb-0037", "LNDb-0043", "LNDb-0044", "LNDb-0049", "LNDb-0109", "LNDb-0123", "LNDb-0124", "LNDb-0126", "LNDb-0171", "LNDb-0179"]:
        res["casos13"].append(caso(e)); print(e, flush=True)
    json.dump(res, open(f"{OUT}/P1_porta.json", "w"), indent=1); print("OK")
