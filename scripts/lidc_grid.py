#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    lidc_grid.py

File type:
    Module

Purpose:
    Shared utilities for the canonical LIDC-IDRI image grid.

Repository:
    pulmonary_nodule_verification

Version:
    v1.0.0

Date:
    2026-10-26

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
    Calixto, W. P., Camargo, T. F. O., Pacheco, V. M. G., Martins, M. S. R., & Coimbra, A. P. (2026). Multiscale morphology-informed post-detection verification of pulmonary nodule candidates in CT: code, trained nnU-Net models, cohort splits and analysis data (v1.0.0) [Data set and software]. Zenodo. https://doi.org/10.5281/zenodo.23674673

Related publication:
    Multiscale morphology-informed post-detection verification of pulmonary nodule candidates in computed tomography (manuscript; DOI not yet available)

Inputs:
    - Outputs of the preceding pipeline stage (see docs/usage.md).

Outputs:
    - See docs/file_level_documentation.md.

Dependencies:
    - SimpleITK
    - numpy

Execution:
    Imported by the pipeline scripts in scripts/

Portability notes:
    All paths are resolved relative to the project base directory given by the PULMONARY_BASE environment variable. No machine-specific path is required.

Reproducibility notes:
    Environment pinned in requirements.txt and environment/. Seeds and frozen parameters are recorded in the run manifests under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) are not included and must be obtained from their original repositories.
"""

import os

import numpy as np
import SimpleITK as sitk

from pulmonary_base import BASE

NIFTI_LIDC_DIR = os.path.join(BASE, "data", "interim", "nifti", "lidc_idri")
GRID_TOL_MM  = 1e-3
GRID_TOL_DIR = 1e-6
SLICE_MATCH_FRACTION = 0.5


class FalhaGrade(Exception):
    """Grade incoerente. Nunca absorvida: bloqueia a publicação do caso."""

def carregar_grade_ct(patient_id):
    """
    Devolve a grade CANÓNICA a partir do `ct.nii.gz` do Script 03.

    A versão anterior construía a sua própria referência com
    `sitk.ImageSeriesReader`, que nesta série emite
    «Non uniform sampling or missing slices detected» e devolve 448 fatias com
    espaçamento 0,623602 e origem no extremo superior — porque o GDCM não
    desduplica posições repetidas, entrega os ficheiros por ordem descendente, e
    o espaçamento é DERIVADO de extensão/(n−1) com n errado.

    Aqui nada é derivado: `size`, `spacing`, `origin` e `direction` são copiados
    da CT. O Script 03 já resolveu a duplicação.
    """
    ct_p = os.path.join(NIFTI_LIDC_DIR, patient_id, "ct.nii.gz")
    if not os.path.isfile(ct_p):
        raise FalhaGrade("%s: ct.nii.gz canónica ausente em %s" % (patient_id, ct_p))
    r = sitk.ImageFileReader()
    r.SetFileName(ct_p)
    r.ReadImageInformation()
    size = tuple(r.GetSize())
    grade = {"size_xyz": size,
             "shape_zyx": (size[2], size[1], size[0]),
             "spacing": tuple(r.GetSpacing()),
             "origin": tuple(r.GetOrigin()),
             "direction": tuple(r.GetDirection()),
             "ct_path": ct_p}
    D = np.asarray(grade["direction"], float).reshape(3, 3)
    if not np.allclose(D.T @ D, np.eye(3), atol=1e-4):
        raise FalhaGrade("%s: direção da CT não ortonormal" % patient_id)
    grade["normal"] = D[:, 2] / np.linalg.norm(D[:, 2])
    return grade


def mask_to_sitk(arr, grade):
    """Aplica a grade da CT explicitamente. Nada é derivado nem inferido."""
    img = sitk.GetImageFromArray(arr.astype(np.uint8))
    img.SetSpacing(grade["spacing"])
    img.SetOrigin(grade["origin"])
    img.SetDirection(grade["direction"])
    if tuple(img.GetSize()) != tuple(grade["size_xyz"]):
        raise FalhaGrade("máscara %s != grade da CT %s"
                         % (img.GetSize(), grade["size_xyz"]))
    return img


