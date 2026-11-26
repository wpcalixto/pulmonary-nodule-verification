#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    P01_11_unidade_referencia.py

File type:
    Script

Purpose:
    Sensitivity of the LIDC-IDRI reference unit: face-connected components versus reader groupings with merged fragments.
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
    - *.nii.gz
    - BLOCK_HASHES.csv
    - bd1_folds.json
    - bd1_split.csv
    - consensus.nii.gz
    - crop_transform.json
    - parcial.pkl

Outputs:
    - P01_11_contagens.csv
    - P01_11_portas.csv
    - P01_11_preditos.csv
    - P01_11_referencias.csv
    - consensus_thresholds.csv

Dependencies:
    - SimpleITK
    - numpy
    - scipy

Execution:
    cd <project_dir> && python scripts/32_supplementary_analyses/c_reference_unit_and_fragments/P01_11_unidade_referencia.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import os, sys, re, csv, json, glob, math, collections, numpy as np, scipy.ndimage as nd, SimpleITK as sitk
from scipy.optimize import linear_sum_assignment
from multiprocessing import Pool
P = "."; R = P + "/resultados"; ARQ = P + "/arquivo/Ultima_Versao/Resultados/evidencia_nvme/do_5070/annotations"
OUT = sys.argv[1]; TAUS = (0.10, 0.25, 0.50)
ARCH = ["2D", "2D_REL", "2D_REM", "3D_F", "3D_FREM", "3D_CF", "3D_CFREM", "F100_lowres", "F100_fullres", "F100_ResEncM"]
S6 = nd.generate_binary_structure(3, 1); S26 = nd.generate_binary_structure(3, 3)
split = {r["patient_id"]: r["split"] for r in csv.DictReader(open(R + "/pulmonar_canonical_pre_oof/splits/bd1_split.csv"))}
fj = json.load(open(R + "/pulmonar_canonical_pre_oof/splits/bd1_folds.json"))["folds"]
fold = {}
for i, f in enumerate(fj if isinstance(fj, list) else fj.values()):
    for p in (f.get("val", f.get("validation", [])) if isinstance(f, dict) else f): fold[p] = i
TH = collections.defaultdict(dict)
for r in csv.DictReader(open(R + "/Pulmonar_v281_2026-09-01/05_ANALISE/analysis/consensus_thresholds.csv")): TH[r["patient_id"]][int(r["cluster"])] = (int(r["threshold"]), int(r["n_vox"]))
BH = {}
for d in glob.glob(R + "/Pulmonary_Project/Script10_canonical_oof/blocks/*") + glob.glob(R + "/Pulmonary_Project/Script10_canonical/blocks/lidc_internal__*"):
    for h in csv.DictReader(open(d + "/BLOCK_HASHES.csv")): BH[d + "/" + h["file"]] = h["sha256"]
import hashlib
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
_IDX = None
def indice():
    global _IDX
    if _IDX is None:
        _IDX = {}
        for p in glob.glob(R + "/Pulmonary_Project/predictions/bd1_oof/*/fold_*/LIDC-*[0-9].nii.gz") + glob.glob(R + "/Pulmonary_Project/predictions/bd1_internal/*/LIDC-*[0-9].nii.gz") + [k for k in BH if os.path.exists(k)]:
            _IDX.setdefault(os.path.basename(p), []).append(p)
    return _IDX
def resolve(p):
    s = BH[p]
    if os.path.exists(p) and sha(p) == s: return p, "bloco"
    for q in indice().get(os.path.basename(p), []):
        if sha(q) == s: return q, "copia_sha_igual"
    raise FileNotFoundError(p)
def diam(m, sy, sx):
    best = 0.0; z = np.nonzero(m.any(axis=(1, 2)))[0]
    for k in z:
        ys, xs = np.nonzero(m[k]); best = max(best, (ys.max() - ys.min() + 1) * sy, (xs.max() - xs.min() + 1) * sx)
    return best
def associar(m, t):
    if m.size == 0: return set()
    w = np.where(m >= t, 1000.0 + m, 0.0); r, c = linear_sum_assignment(-w); return {(int(a), int(b)) for a, b in zip(r, c) if m[a, b] >= t}
def um(pid):
    try:
        return _um(pid)
    except Exception as e:
        return dict(pid=pid, linhas=[], refs=[], preds=[], portas={"ERRO": repr(e)[:300]})
def _um(pid):
    out = dict(pid=pid, linhas=[], refs=[], preds=[], portas={})
    ct = json.load(open(f"{R}/pulmonar_canonical_pre_oof/recrops/images/{pid}/crop_transform.json")); z0, z1, y0, y1, x0, x1 = ct["crop_bbox"]
    fc = f"{ARQ}/consensus/lidc/{pid}/consensus.nii.gz"; lab_rec = f"{R}/pulmonar_canonical_pre_oof/recrops/labels/{pid}/consensus.nii.gz"
    cons = None
    if os.path.exists(fc):
        cons = (sitk.GetArrayFromImage(sitk.ReadImage(fc)) > 0)[z0:z1, y0:y1, x0:x1]; out["portas"]["fonte_consenso"] = "arquivo_volume_inteiro"
    elif os.path.exists(lab_rec):
        cons = sitk.GetArrayFromImage(sitk.ReadImage(lab_rec)) > 0; out["portas"]["fonte_consenso"] = "recorte_canonico"
    if cons is not None:
        assert list(cons.shape) == list(ct["cropped_shape"]), pid
        if os.path.exists(lab_rec) and os.path.exists(fc): out["portas"]["recorte_igual_canonico"] = bool(np.array_equal(cons, sitk.GetArrayFromImage(sitk.ReadImage(lab_rec)) > 0))
    clus = {}; th = TH.get(pid, {})
    d = f"{ARQ}/per_reader/lidc/{pid}"
    if len(th) == 1 and not os.path.isdir(d):
        assert cons is not None, pid; clus[next(iter(th))] = cons.copy(); out["portas"]["agrupamento_unico_sem_mascaras"] = True
    elif th:
        por = collections.defaultdict(lambda: collections.defaultdict(lambda: None))
        for f in sorted(glob.glob(d + "/*.nii.gz")):
            k = int(re.search(r"nodule(\d+)", f).group(1)); s = re.search(r"(rec\w+_session\d+)", f).group(1)
            a = sitk.GetArrayFromImage(sitk.ReadImage(f)) > 0; por[k][s] = a if por[k][s] is None else (por[k][s] | a)
        nv_ok = True
        sem = []
        for k, (t, nv) in th.items():
            ms = list(por[k].values())
            if not ms: sem.append(k); nv_ok = False; continue
            acc = np.sum(ms, axis=0); c = acc >= t
            nv_ok &= int(c.sum()) == nv; clus[k] = c[z0:z1, y0:y1, x0:x1]
        out["portas"]["n_vox_agrupamentos"] = bool(nv_ok)
        if sem: out["portas"]["agrupamentos_sem_mascaras"] = ";".join(map(str, sem))
    if cons is None:
        cons = np.zeros(tuple(ct["cropped_shape"]), bool)
        for c in clus.values(): cons |= c
        out["portas"]["fonte_consenso"] = "reconstruido_dos_agrupamentos"
    uni = np.zeros(cons.shape, bool)
    for c in clus.values(): uni |= c
    out["portas"]["uniao_agrupamentos_igual_consenso"] = bool(np.array_equal(uni, cons))
    refs = {}
    la, na = nd.label(cons, structure=S6); refs["A"] = [la == i for i in range(1, na + 1)]
    lb, nb = nd.label(cons, structure=S26); refs["B"] = [lb == i for i in range(1, nb + 1)]
    refs["C"] = [clus[k] for k in sorted(clus) if clus[k].any()]
    for i, g in enumerate(refs["A"]):
        ov = {k: int((g & c).sum()) for k, c in clus.items()}; k = max(ov, key=ov.get) if ov else None
        out["refs"].append(dict(cen="A_mapa", id=i + 1, agrupamento=k, n_agrupamentos_tocados=sum(1 for v in ov.values() if v > 0), vox=int(g.sum())))
    sub = "oof" if split[pid] == "development" else "internal"
    for arch in ARCH:
        p = (f"{R}/Pulmonary_Project/Script10_canonical_oof/blocks/lidc_oof__{arch}__fold_{fold[pid]}/{pid}.nii.gz" if sub == "oof" else f"{R}/Pulmonary_Project/Script10_canonical/blocks/lidc_internal__{arch}/{pid}.nii.gz")
        p, fonte = resolve(p); out["portas"].setdefault("fontes", collections.Counter())[fonte] += 1
        img = sitk.ReadImage(p); arr = sitk.GetArrayFromImage(img) > 0; assert arr.shape == cons.shape, (pid, arch, arr.shape)
        sx, sy, _ = img.GetSpacing(); pl, npd = nd.label(arr, structure=S6)
        cls = {}
        for cen in ("A", "B", "C"):
            G = refs[cen]; m = np.zeros((len(G), npd))
            if npd and G:
                for i, g in enumerate(G):
                    hit = pl[g]; hit = hit[hit > 0]
                    if hit.size:
                        sp = np.bincount(pl.ravel(), minlength=npd + 1)
                        for j, inter in zip(*np.unique(hit, return_counts=True)): m[i, j - 1] = inter / (g.sum() + sp[j] - inter)
            for t in TAUS:
                pr = associar(m, t); pm = {c for _, c in pr}; gm = {r for r, _ in pr}
                out["linhas"].append(dict(pid=pid, subset=sub, arch=arch, cen=cen, tau=t, n_ref=len(G), n_pred=npd, TP=len(pr), FP=npd - len(pr), FN=len(G) - len(pr)))
                if t == 0.10:
                    cls[cen] = ["TP" if j in pm else "FP" for j in range(npd)]
                    for i, g in enumerate(G): out["refs"].append(dict(cen=cen, arch=arch, subset=sub, id=i + 1, diam=diam(g, sy, sx) if arch == "2D" or True else None, detectado=i in gm))
        for j in range(npd): out["preds"].append(dict(arch=arch, subset=sub, id=j + 1, A=cls["A"][j], B=cls["B"][j], C=cls["C"][j]))
    return out
if __name__ == "__main__":
    import pickle
    pids = sorted(split); feitos = set(); PK = OUT + "/parcial.pkl"
    res = []
    if os.path.exists(PK):
        with open(PK, "rb") as f:
            while True:
                try: o = pickle.load(f); res.append(o); feitos.add(o["pid"])
                except EOFError: break
    falta = [p for p in pids if p not in feitos]
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 6) as pool, open(PK, "ab") as fk:
        for i, o in enumerate(pool.imap_unordered(um, falta, chunksize=2)):
            res.append(o); pickle.dump(o, fk); fk.flush()
            if i % 50 == 0: print(i, o["pid"], o["portas"], flush=True)
    L, Rf, Pr, Po = [], [], [], []
    for o in res:
        L += o["linhas"]; Rf += [dict(pid=o["pid"], **r) for r in o["refs"]]; Pr += [dict(pid=o["pid"], **r) for r in o["preds"]]; Po.append(dict(pid=o["pid"], **{k: (dict(v) if isinstance(v, collections.Counter) else v) for k, v in o["portas"].items()}))
    for nome, X in (("P01_11_contagens.csv", L), ("P01_11_referencias.csv", Rf), ("P01_11_preditos.csv", Pr), ("P01_11_portas.csv", Po)):
        keys = []
        for x in X:
            for k in x:
                if k not in keys: keys.append(k)
        with open(OUT + "/" + nome, "w", newline="") as f: w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(X)
    print("FIM", len(res), "erros", sum(1 for o in res if "ERRO" in o["portas"]))
