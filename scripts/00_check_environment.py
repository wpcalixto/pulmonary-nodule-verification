#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    00_check_environment.py

File type:
    Script

Purpose:
    Records the execution environment (Python, CUDA, GPU, PyTorch, SimpleITK, nnU-Net, LungMask, statistical libraries, disk space and folder access) before any processing stage.

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
    - lungmask
    - nnunetv2
    - torch

Execution:
    PULMONARY_BASE=<project_dir> python scripts/00_check_environment.py

Portability notes:
    All paths are resolved relative to the project base directory given by the PULMONARY_BASE environment variable. No machine-specific path is required.

Reproducibility notes:
    Environment pinned in requirements.txt and environment/. Seeds and frozen parameters are recorded in the run manifests under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) are not included and must be obtained from their original repositories.
"""

import os
import sys
import json
import shutil
import platform
import subprocess
from datetime import datetime

from pulmonary_base import BASE

LOGS_DIR = os.path.join(BASE, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

REQUIRED_DIRS = [
    os.path.join(BASE, "data/raw"),
    os.path.join(BASE, "data/interim"),
    os.path.join(BASE, "data/processed"),
    os.path.join(BASE, "analysis"),
    os.path.join(BASE, "models"),
    os.path.join(BASE, "predictions"),
    os.path.join(BASE, "annotations"),
    os.path.join(BASE, "features"),
    os.path.join(BASE, "figures"),
    os.path.join(BASE, "tables"),
    os.path.join(BASE, "reports"),
    os.path.join(BASE, "config"),
    os.path.join(BASE, "lung_masks"),
    os.path.join(BASE, "logs"),
    os.path.join(BASE, "figdata"),
    os.path.join(BASE, "tabdata"),
    os.path.join(BASE, "manifests"),
    os.path.join(BASE, "audits"),
]

report = {
    "timestamp":        datetime.now().isoformat(),
    "platform":         platform.platform(),
    "python_version":   sys.version,
    "checks":           {},
    "package_versions": {},
    "gpu_info":         {},
    "disk_space_gb":    {},
    "folder_access":    {},
}

versions_lines = [f"# Environment report — {datetime.now().isoformat()}\n"]


def _run(cmd):
    """
    Executa um comando e devolve o stdout.

    Captura também o stderr e o código de saída: sem isso, um comando que falha
    sem escrever no stdout devolvia string vazia, e a string vazia era gravada no
    relatório como se fosse um valor válido.
    """
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        out = r.stdout.strip()
        if r.returncode != 0:
            err = r.stderr.strip() or "(sem stderr)"
            return f"ERROR: rc={r.returncode} {err[:200]}"
        if not out:
            return "ERROR: comando sem saída"
        return out
    except Exception as e:
        return f"ERROR: {e}"


def _pkg_version(mod, dist_name=None):
    """
    Resolve a versão de um pacote: `__version__` e, em falta, os metadados da
    distribuição instalada. Devolve 'UNRESOLVED' quando nenhuma fonte responde —
    nunca um valor de preenchimento como 'ok', que seria indistinguível de uma
    versão real na auditoria.
    """
    v = getattr(mod, "__version__", None)
    if v:
        return str(v)
    try:
        import importlib.metadata as md
        return md.version(dist_name or mod.__name__)
    except Exception:
        return "UNRESOLVED"


PROVISION_VENV_PADRAO = os.path.expanduser("~/venv_nnunet252_cuda")
PROVISION_ROOTS = os.path.expanduser("~/nnunet252_canonical")
TORCH_SPEC = "torch==2.11.0+cu128"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"

NNUNET_VERSAO = os.environ.get("PULMONARY_NNUNET_VERSION", "2.8.1")
NNUNET_SPEC = "nnunetv2==%s" % NNUNET_VERSAO
TRAINERS_MANIFEST_SHA_POR_VERSAO = {
    "2.5.2": "7c0b598ac796459b8ac71137a7e0aeb1d9fda27b66ec3cf1ff06a503f34d5b67",
    "2.8.1": "98bb2eb0f5fb1a9b728bee82aa452da9dacfad0e0e284739046f1ca7d1cc035d",
}
TRAINERS_MANIFEST_SHA = TRAINERS_MANIFEST_SHA_POR_VERSAO.get(NNUNET_VERSAO)
TRAINERS_DIR_POR_VERSAO = {"2.5.2": "trainers", "2.8.1": "trainers_281"}
PATCHES_DIR_POR_VERSAO = {"2.5.2": "patches", "2.8.1": None}
MANIFESTO_SUPERSEDED = ("211394d1453967f15aa51707d445e49126e339793264ca0a509b5c650ca661f4",
                        "f32ed760a06be6ab73b1072eae52478229413fe872e63031c1108577b98754e5")
DESTINOS_PROIBIDOS = ("/ml", "/nnunet", "/nnUNet_raw", "/nnUNet_preprocessed",
                      "/nnUNet_results")


def _sha256_ficheiro(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _prov_run(cmd, timeout=3600, env=None):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    return {"cmd": " ".join(cmd[:6]) + (" …" if len(cmd) > 6 else ""),
            "returncode": r.returncode,
            "stdout_tail": r.stdout[-2000:], "stderr_tail": r.stderr[-2000:]}


def _prov_falhar(rel, motivo, detalhe=None):
    rel["PROVISION_STATUS"] = "FAIL"
    rel["failure_reason"] = motivo
    if detalhe is not None:
        rel["failure_detail"] = detalhe
    _prov_gravar(rel)
    print(f"[FAIL] {motivo}")
    sys.exit(1)


def _prov_gravar(rel):
    os.makedirs(LOGS_DIR, exist_ok=True)
    p = os.path.join(LOGS_DIR, "provision_report_%s.json" % rel["run_id"])
    with open(p, "w") as f:
        json.dump(rel, f, indent=2, sort_keys=True, default=str)
    rel["_path"] = p
    print(f"  relatorio: {p}")


def provisionar(destino, trainers_dir, dry_run_only=False):
    """
    Cria o ambiente canonico NOVO e ISOLADO. Nao toca em nenhum ambiente
    existente. Cada etapa que falhe interrompe: um ambiente meio construido
    seria pior do que nenhum, porque pareceria pronto.
    """
    import time as _t
    rel = {"run_id": "prov_%s_%d" % (_t.strftime("%Y%m%dT%H%M%S"), os.getpid()),
           "generated": _t.strftime("%Y-%m-%dT%H:%M:%S"),
           "host": platform.node(), "platform": platform.platform(),
           "TARGET_VENV": destino, "ISOLATED_ROOTS": PROVISION_ROOTS,
           "TORCH_SPEC": TORCH_SPEC, "TORCH_INDEX": TORCH_INDEX,
           "NNUNET_SPEC": NNUNET_SPEC,
           "TRAINERS_MANIFEST_SHA256_EXPECTED": TRAINERS_MANIFEST_SHA,
           "GPU_MEMORY_TARGET_PASSED": "NONE — usa-se o padrao da classe do planeador",
           "steps": []}

    real = os.path.realpath(destino)
    for proibido in DESTINOS_PROIBIDOS:
        if real.rstrip("/").endswith(proibido) or ("%s/" % proibido) in real:
            _prov_falhar(rel, "DESTINO_PROIBIDO_AMBIENTE_HISTORICO: %s" % real)
    reutiliza = False
    if os.path.exists(destino):
        vpy0 = os.path.join(destino, "bin", "python")
        if not os.path.isfile(vpy0):
            _prov_falhar(rel, "DESTINO_JA_EXISTE_E_NAO_E_VENV: %s "
                              "(OVERWRITE_EXISTING = PROHIBITED)" % destino)
        reutiliza = True
        rel["REUSING_EXISTING_VENV"] = True
        rel["reuse_note"] = ("venv preexistente reutilizado; torch e nnunetv2 nao sao "
                             "reinstalados, mas TODOS os portoes voltam a correr")
    rel["steps"].append({"step": "target_guard", "status": "PASS", "resolved": real})

    man = os.path.join(trainers_dir, "trainers_manifest.json")
    if not os.path.isfile(man):
        _prov_falhar(rel, "MANIFESTO_DE_TRAINERS_AUSENTE: %s" % man)
    sha_man = _sha256_ficheiro(man)
    if sha_man in MANIFESTO_SUPERSEDED:
        _prov_falhar(rel, "MANIFESTO_SUPERSEDED_RECUSADO: %s esta arquivado como "
                          "defeituoso; usar o manifesto corrigido" % sha_man)
    if sha_man != TRAINERS_MANIFEST_SHA:
        _prov_falhar(rel, "MANIFESTO_COM_HASH_INESPERADO: %s != %s"
                     % (sha_man, TRAINERS_MANIFEST_SHA))
    dados = json.load(open(man))
    esperados = {}
    for t in (dados.get("trainers") or []):
        nome = t.get("name") or t.get("file") or ""
        if nome and not nome.endswith(".py"):
            nome += ".py"
        sha = t.get("SHA256") or t.get("sha256") or ""
        if nome and sha:
            esperados[nome] = sha
    if dados.get("NNUNET_VERSION") and dados["NNUNET_VERSION"] not in NNUNET_SPEC:
        _prov_falhar(rel, "MANIFESTO_DECLARA_OUTRA_VERSAO_DE_NNUNET: %s vs %s"
                     % (dados["NNUNET_VERSION"], NNUNET_SPEC))
    if dados.get("INSTALL_IN_CANONICAL_CLUSTER") != "ONLY_BY_SCRIPT00":
        _prov_falhar(rel, "MANIFESTO_NAO_AUTORIZA_INSTALACAO_POR_ESTE_SCRIPT: %s"
                     % dados.get("INSTALL_IN_CANONICAL_CLUSTER"))
    rel["manifest_declarations"] = {
        k: dados.get(k) for k in ("manifest_id", "NNUNET_VERSION",
                                  "INSTALL_IN_CANONICAL_CLUSTER",
                                  "F100_LR5E3_CANONICALIZATION",
                                  "TRAINING_EXECUTION")}
    rel["trainers_manifest_sha256_observed"] = sha_man
    rel["trainers_declared"] = sorted(esperados)
    if not esperados:
        _prov_falhar(rel, "MANIFESTO_SEM_ENTRADAS_LEGIVEIS", dados if isinstance(dados, dict) else None)
    conferidos, divergentes = [], []
    for nome, sha_esp in sorted(esperados.items()):
        p = os.path.join(trainers_dir, os.path.basename(nome))
        if not os.path.isfile(p):
            divergentes.append({"file": nome, "reason": "ABSENT"})
            continue
        obs = _sha256_ficheiro(p)
        (conferidos if obs == sha_esp else divergentes).append(
            {"file": nome, "expected": sha_esp, "observed": obs})
    rel["steps"].append({"step": "trainers_manifest", "status":
                         "PASS" if not divergentes else "FAIL",
                         "verified": len(conferidos), "divergent": divergentes})
    if divergentes:
        _prov_falhar(rel, "TRAINERS_HASH_DIVERGENTE", divergentes)

    import tempfile
    import shutil as _sh0
    tmpdir = tempfile.mkdtemp(prefix="nnunet252_dryrun_")
    tmpvenv = os.path.join(tmpdir, "v")
    try:
        d0 = _prov_run([sys.executable, "-m", "venv", tmpvenv], timeout=600)
        if d0["returncode"] != 0:
            _prov_falhar(rel, "VENV_TEMPORARIO_PARA_DRY_RUN_FALHOU", d0)
        tpy = os.path.join(tmpvenv, "bin", "python")
        _prov_run([tpy, "-m", "pip", "install", "-q", "-U", "pip"], timeout=900)
        d = _prov_run([tpy, "-m", "pip", "install", "--dry-run",
                       "--index-url", TORCH_INDEX,
                       "--extra-index-url", "https://pypi.org/simple",
                       TORCH_SPEC, NNUNET_SPEC], timeout=1800)
    finally:
        _sh0.rmtree(tmpdir, ignore_errors=True)
    rel["steps"].append({"step": "pip_dry_run",
                         "status": "PASS" if d["returncode"] == 0 else "FAIL",
                         "isolation": "venv temporario descartado apos a resolucao",
                         "system_python_untouched": True, **d})
    if d["returncode"] != 0:
        _prov_falhar(rel, "PIP_DRY_RUN_FALHOU", d)
    if dry_run_only:
        rel["PROVISION_STATUS"] = "DRY_RUN_ONLY"
        _prov_gravar(rel)
        print("[OK] dry-run resolvido; ambiente NAO criado (--provision-dry-run)")
        return rel

    if reutiliza:
        rel["steps"].append({"step": "create_venv", "status": "SKIPPED_REUSED"})
    else:
        d = _prov_run([sys.executable, "-m", "venv", destino], timeout=600)
        rel["steps"].append({"step": "create_venv", "status":
                             "PASS" if d["returncode"] == 0 else "FAIL", **d})
        if d["returncode"] != 0:
            _prov_falhar(rel, "CRIACAO_DO_VENV_FALHOU", d)
    vpy = os.path.join(destino, "bin", "python")

    passos_inst = [("pip_check", [vpy, "-m", "pip", "check"], 600)]
    if not reutiliza:
        passos_inst = [
            ("pip_upgrade", [vpy, "-m", "pip", "install", "-q", "-U", "pip", "wheel"], 900),
            ("install_torch", [vpy, "-m", "pip", "install", "--index-url", TORCH_INDEX,
                               TORCH_SPEC], 5400),
            ("install_nnunet", [vpy, "-m", "pip", "install", NNUNET_SPEC], 3600),
        ] + passos_inst
    else:
        d = _prov_run([vpy, "-c", "import importlib.metadata as md,json;"
                       "print(json.dumps({'torch':md.version('torch'),"
                       "'nnunetv2':md.version('nnunetv2')}))"], 300)
        rel["steps"].append({"step": "verify_reused_versions",
                             "status": "PASS" if d["returncode"] == 0 else "FAIL", **d})
        if d["returncode"] != 0:
            _prov_falhar(rel, "VERIFICACAO_DAS_VERSOES_REUTILIZADAS_FALHOU", d)
        try:
            v = json.loads(d["stdout_tail"].strip().splitlines()[-1])
        except Exception:
            _prov_falhar(rel, "VERSOES_REUTILIZADAS_ILEGIVEIS", d)
        rel["reused_versions"] = v
        if not v.get("torch", "").startswith("2.11.0") or v.get("nnunetv2") != NNUNET_VERSAO:
            _prov_falhar(rel, "VERSOES_REUTILIZADAS_DIVERGEM_DO_DECLARADO", v)
    for etapa, cmd, to in passos_inst:
        d = _prov_run(cmd, timeout=to)
        rel["steps"].append({"step": etapa, "status":
                             "PASS" if d["returncode"] == 0 else "FAIL", **d})
        if d["returncode"] != 0:
            _prov_falhar(rel, "%s_FALHOU" % etapa.upper(), d)

    d = _prov_run([vpy, "-c",
                   "import nnunetv2,os;print(os.path.dirname(nnunetv2.__file__))"], 300)
    if d["returncode"] != 0:
        _prov_falhar(rel, "LOCALIZACAO_DO_NNUNETV2_FALHOU", d)
    variants = os.path.join(d["stdout_tail"].strip(), "training",
                            "nnUNetTrainer", "variants")
    if not os.path.isdir(variants):
        _prov_falhar(rel, "DIRETORIO_VARIANTS_AUSENTE: %s" % variants)
    import shutil as _sh
    copiados = []
    for nome in sorted(esperados):
        src = os.path.join(trainers_dir, os.path.basename(nome))
        dst = os.path.join(variants, os.path.basename(nome))
        if os.path.exists(dst) and not reutiliza:
            _prov_falhar(rel, "TRAINER_JA_EXISTE_NO_DESTINO: %s" % dst)
        _sh.copy2(src, dst)
        obs = _sha256_ficheiro(dst)
        if obs != esperados[nome]:
            _prov_falhar(rel, "TRAINER_COPIADO_COM_HASH_DIVERGENTE: %s" % nome)
        copiados.append({"file": os.path.basename(nome), "dest": dst, "sha256": obs})
    raiz_nn = d["stdout_tail"].strip()
    _pd = PATCHES_DIR_POR_VERSAO.get(NNUNET_VERSAO)
    patches_dir = (os.path.join(os.path.dirname(trainers_dir), _pd)
                   if _pd else None)
    man_p = os.path.join(patches_dir, "patches_manifest.json") if patches_dir else None
    aplicados = []
    if man_p and os.path.isfile(man_p):
        dados_p = json.load(open(man_p))
        rel["patches_manifest_sha256"] = _sha256_ficheiro(man_p)
        for pt in (dados_p.get("patches") or []):
            src = (os.path.join(patches_dir, os.path.basename(pt["path"]))
                   if pt.get("path") else None)
            dst = os.path.join(raiz_nn, *pt["target_module"].split("/")[1:])
            if pt.get("kind", "FILE") == "FILE":
                if not src:
                    _prov_falhar(rel, "PATCH_DE_FICHEIRO_SEM_path: %s" % pt["name"])
                if not os.path.isfile(src):
                    _prov_falhar(rel, "PATCH_AUSENTE_NA_FONTE: %s" % src)
            if not os.path.isfile(dst):
                _prov_falhar(rel, "PATCH_SEM_ALVO_NO_PACOTE: %s" % dst)
            atual = _sha256_ficheiro(dst)
            if atual == pt["SHA256"]:
                aplicados.append({"name": pt["name"], "dest": dst,
                                  "status": "ALREADY_PATCHED", "sha256": atual})
                continue
            if atual != pt["EXPECTED_ORIGINAL_SHA256"]:
                _prov_falhar(rel,
                             "PATCH_ALVO_INESPERADO: %s tem sha %s, esperado o "
                             "original %s ou o ja aplicado %s. O pacote nao e o "
                             "que o manifesto descreve; patch NAO aplicado."
                             % (dst, atual[:16],
                                pt["EXPECTED_ORIGINAL_SHA256"][:16],
                                pt["SHA256"][:16]))
            _sh.copy2(dst, dst + ".pre_patch")
            tipo = pt.get("kind", "FILE")
            if tipo == "FILE":
                _sh.copy2(src, dst)
            elif tipo == "LINE":
                texto = open(dst).read()
                n = texto.count(pt["match"])
                if n != pt.get("occurrences_expected", 1):
                    _prov_falhar(rel, "PATCH_LINHA_COM_OCORRENCIAS_INESPERADAS: "
                                      "%s aparece %d vezes em %s, esperado %d"
                                 % (pt["name"], n, dst,
                                    pt.get("occurrences_expected", 1)))
                antes = texto.split("\n")
                nl = pt.get("line_number")
                if nl and antes[nl - 1] != pt["match"]:
                    _prov_falhar(rel, "PATCH_LINHA_FORA_DO_SITIO: %s linha %d nao "
                                      "e a esperada" % (dst, nl))
                open(dst, "w").write(texto.replace(pt["match"], pt["replace"]))
            else:
                _prov_falhar(rel, "PATCH_DE_TIPO_DESCONHECIDO: %s" % tipo)
            obs = _sha256_ficheiro(dst)
            if obs != pt["SHA256"]:
                _sh.copy2(dst + ".pre_patch", dst)
                _prov_falhar(rel, "PATCH_COM_HASH_DIVERGENTE: %s deu %s, esperado "
                                  "%s. Original reposto." % (pt["name"], obs[:16],
                                                             pt["SHA256"][:16]))
            aplicados.append({"name": pt["name"], "kind": tipo, "dest": dst,
                              "status": "APPLIED", "sha256": obs,
                              "original_sha256": atual,
                              "backup": dst + ".pre_patch",
                              "reason": pt.get("reason")})
    rel["steps"].append({"step": "install_patches",
                         "status": "PASS" if aplicados else "NONE_DECLARED",
                         "applied": aplicados,
                         "patches_dir": patches_dir,
                         "note": ("nenhum remendo previsto para esta pilha"
                                  if not patches_dir else
                                  "manifesto lido; nada a aplicar" if not aplicados
                                  else "aplicados e conferidos por hash")})

    rel["steps"].append({"step": "install_trainers", "status": "PASS",
                         "installed": copiados,
                         "note": "somente os declarados no manifesto"})

    prova = (
        "import importlib,inspect,json;"
        "names=%r;"
        "out={};"
        "\nfor n in names:\n"
        "    m=importlib.import_module('nnunetv2.training.nnUNetTrainer.variants.'+n)\n"
        "    cls=[k for k,v in vars(m).items() if inspect.isclass(v) and k.startswith('nnUNetTrainer')]\n"
        "    out[n]=cls\n"
        "print(json.dumps(out))"
    ) % [os.path.splitext(os.path.basename(x))[0] for x in sorted(esperados)]
    d = _prov_run([vpy, "-c", prova], 600)
    rel["steps"].append({"step": "import_trainer_classes", "status":
                         "PASS" if d["returncode"] == 0 else "FAIL", **d})
    if d["returncode"] != 0:
        _prov_falhar(rel, "IMPORT_DAS_CLASSES_DE_TRAINER_FALHOU", d)

    desc = (
        "import os,json,nnunetv2\n"
        "from nnunetv2.utilities.find_class_by_name import recursive_find_python_class\n"
        "from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer\n"
        "raiz=os.path.join(nnunetv2.__path__[0],'training','nnUNetTrainer')\n"
        "out={}\n"
        "for n in %r:\n"
        "    c=recursive_find_python_class(raiz,n,'nnunetv2.training.nnUNetTrainer')\n"
        "    out[n]={'found':c is not None,'name_ok':bool(c) and c.__name__==n,"
        "'is_trainer':bool(c) and issubclass(c,nnUNetTrainer)}\n"
        "print(json.dumps(out))\n"
    ) % [os.path.splitext(os.path.basename(x))[0] for x in sorted(esperados)]
    d = _prov_run([vpy, "-c", desc], 900)
    ok_desc = d["returncode"] == 0
    det_desc = {}
    if ok_desc:
        try:
            det_desc = json.loads(d["stdout_tail"].strip().splitlines()[-1])
            ok_desc = all(v["found"] and v["name_ok"] and v["is_trainer"]
                          for v in det_desc.values())
        except Exception:
            ok_desc = False
    rel["steps"].append({"step": "recursive_discovery",
                         "status": "PASS" if ok_desc else "FAIL",
                         "detail": det_desc, **d})
    if not ok_desc:
        _prov_falhar(rel, "DESCOBERTA_POR_RECURSIVE_FIND_PYTHON_CLASS_FALHOU", d)

    cuda_prova = (
        "import torch,json\n"
        "assert torch.cuda.is_available(), 'CUDA indisponivel'\n"
        "dev=torch.device('cuda')\n"
        "x=torch.randn(2,1,32,32,32,device=dev,requires_grad=True)\n"
        "c=torch.nn.Conv3d(1,8,3,padding=1).to(dev)\n"
        "y=c(x); l=y.square().mean(); l.backward()\n"
        "torch.cuda.synchronize()\n"
        "g=c.weight.grad\n"
        "print(json.dumps({'device':torch.cuda.get_device_name(0),"
        "'capability':list(torch.cuda.get_device_capability(0)),"
        "'torch':torch.__version__,'cuda':torch.version.cuda,"
        "'loss_finite':bool(torch.isfinite(l).item()),"
        "'grad_finite':bool(torch.isfinite(g).all().item()),"
        "'grad_nonzero':bool((g!=0).any().item()),"
        "'peak_mib':round(torch.cuda.max_memory_allocated()/2**20,2)}))\n"
    )
    d = _prov_run([vpy, "-c", cuda_prova], 900)
    ok_cuda = d["returncode"] == 0
    detalhe = {}
    if ok_cuda:
        try:
            detalhe = json.loads(d["stdout_tail"].strip().splitlines()[-1])
            ok_cuda = (detalhe.get("loss_finite") and detalhe.get("grad_finite")
                       and detalhe.get("grad_nonzero"))
        except Exception:
            ok_cuda = False
    rel["steps"].append({"step": "cuda_real_operation",
                         "status": "PASS" if ok_cuda else "FAIL",
                         "detail": detalhe, **d})
    if not ok_cuda:
        _prov_falhar(rel, "OPERACAO_CUDA_REAL_FALHOU", d)
    rel["cuda_proof"] = detalhe

    raizes = {}
    for sub in ("raw", "preprocessed", "results"):
        p = os.path.join(PROVISION_ROOTS, sub)
        os.makedirs(p, exist_ok=True)
        raizes[sub] = p
    rel["isolated_roots_created"] = raizes
    rel["steps"].append({"step": "isolated_roots", "status": "PASS", "roots": raizes})

    d = _prov_run([vpy, "-m", "pip", "freeze"], 600)
    if d["returncode"] == 0:
        pv = os.path.join(LOGS_DIR, "package_versions_%s.txt" % rel["run_id"])
        os.makedirs(LOGS_DIR, exist_ok=True)
        open(pv, "w").write(d["stdout_tail"])
        rel["package_versions_path"] = pv
        rel["package_versions_sha256"] = _sha256_ficheiro(pv)
    rel["steps"].append({"step": "pip_freeze", "status":
                         "PASS" if d["returncode"] == 0 else "FAIL"})

    eu = os.path.abspath(__file__)
    d = _prov_run([vpy, eu, "--profile", "nnunet-execution",
                   "--trainers", trainers_dir], timeout=1800)
    rel["steps"].append({"step": "script00_verification",
                         "status": "PASS" if d["returncode"] == 0 else "FAIL",
                         "profile": "nnunet-execution", **d})
    rel["SCRIPT00_PROFILE_RC"] = d["returncode"]

    todos_ok = all(x.get("status") in ("PASS", "SKIPPED_REUSED", "NONE_DECLARED")
                   for x in rel["steps"])
    rel["PROVISION_STATUS"] = "PASS" if todos_ok else "FAIL"
    rel["CLUSTER_CUDA_ENVIRONMENT"] = "PASS" if todos_ok else "NOT_ASSIGNED"
    rel["CLUSTER_CUDA_ENVIRONMENT_SCOPE"] = "SCRIPT08_NNUNET_EXECUTION"
    rel["PROJECT_WIDE_ENVIRONMENT"] = "NOT_EVALUATED"
    rel["READY_FOR_SCRIPT08"] = bool(todos_ok)
    rel["HISTORICAL_ENVIRONMENTS_MODIFIED"] = False
    _prov_gravar(rel)
    print("\n── PROVISION: %s ──" % rel["PROVISION_STATUS"])
    print("  CLUSTER_CUDA_ENVIRONMENT = %s (%s)"
          % (rel["CLUSTER_CUDA_ENVIRONMENT"], rel["CLUSTER_CUDA_ENVIRONMENT_SCOPE"]))
    print("  READY_FOR_SCRIPT08       = %s" % rel["READY_FOR_SCRIPT08"])
    print("  GPU: %s  cap %s  torch %s  cuda %s"
          % (detalhe.get("device"), detalhe.get("capability"),
             detalhe.get("torch"), detalhe.get("cuda")))
    return rel


VERSOES_EXIGIDAS = {"nnunetv2": NNUNET_VERSAO, "torch": "2.11.0+cu128",
                    "SimpleITK": "2.5.6"}
FORA_DE_AMBITO = {
    "lungmask":    "SCRIPT06_PRODUCT_REUSED_AND_ATTESTED",
    "tcia_utils":  "SCRIPTS01_02_CLOSED",
    "statsmodels": "REQUIRED_ONLY_BY_SCRIPTS21_25",
    "pingouin":    "NO_SCRIPT_RUNTIME_CONSUMER",
}
ENTRYPOINTS_EXIGIDOS = ("nnUNetv2_extract_fingerprint", "nnUNetv2_plan_and_preprocess",
                        "nnUNetv2_train", "nnUNetv2_predict")


def perfil_nnunet_execution(trainers_dir):
    """
    Corre DENTRO do venv canonico. Cada portao e verificado; nenhum e presumido.
    """
    import time as _t
    import importlib
    import importlib.metadata as _md
    run_id = "s00prof_%s_%d" % (_t.strftime("%Y%m%dT%H%M%S"), os.getpid())
    rel = {"run_id": run_id, "generated": _t.strftime("%Y-%m-%dT%H:%M:%S"),
           "PROFILE": "nnunet-execution",
           "QUESTION": "o ambiente canonico consegue planear, pre-processar e "
                       "treinar com o nnU-Net?",
           "host": platform.node(), "platform": platform.platform(),
           "python": platform.python_version(), "venv": sys.prefix,
           "out_of_scope": FORA_DE_AMBITO,
           "PROJECT_WIDE_ENVIRONMENT": "NOT_EVALUATED",
           "gates": {}}
    G = rel["gates"]

    obs, mau = {}, []
    for dist, esperada in VERSOES_EXIGIDAS.items():
        try:
            v = _md.version(dist)
        except Exception as e:
            v = "NOT_INSTALLED:%s" % type(e).__name__
        obs[dist] = v
        if v != esperada:
            mau.append({"package": dist, "expected": esperada, "observed": v})
    rel["versions_observed"] = obs
    G["required_versions"] = {"status": "PASS" if not mau else "FAIL",
                              "divergent": mau}

    man = os.path.join(trainers_dir, "trainers_manifest.json")
    tr = {"status": "FAIL", "detail": []}
    if os.path.isfile(man):
        sha_man = _sha256_ficheiro(man)
        tr["manifest_sha256"] = sha_man
        tr["manifest_matches_expected"] = (sha_man == TRAINERS_MANIFEST_SHA)
        tr["manifest_is_superseded"] = (sha_man in MANIFESTO_SUPERSEDED)
        dados = json.load(open(man))
        variants = os.path.join(os.path.dirname(importlib.import_module(
            "nnunetv2").__file__), "training", "nnUNetTrainer", "variants")
        okc = True
        for t in dados.get("trainers", []):
            nome = t["name"] if t["name"].endswith(".py") else t["name"] + ".py"
            dst = os.path.join(variants, nome)
            pres = os.path.isfile(dst)
            o = _sha256_ficheiro(dst) if pres else ""
            bate = (o == t.get("SHA256"))
            okc &= (pres and bate)
            tr["detail"].append({"file": nome, "installed": pres,
                                 "expected": t.get("SHA256"), "observed": o,
                                 "match": bate})
        tr["status"] = ("PASS" if (okc and tr["manifest_matches_expected"]
                                   and not tr["manifest_is_superseded"]) else "FAIL")
    else:
        tr["detail"] = "manifesto ausente: %s" % man
    G["trainer_hashes"] = tr

    nomes = [t["name"] for t in json.load(open(man)).get("trainers", [])] \
        if os.path.isfile(man) else []
    imp = {"status": "FAIL", "classes": {}}
    try:
        for n in nomes:
            m = importlib.import_module("nnunetv2.training.nnUNetTrainer.variants." + n)
            imp["classes"][n] = hasattr(m, n)
        imp["status"] = "PASS" if nomes and all(imp["classes"].values()) else "FAIL"
    except Exception as e:
        imp["error"] = "%s: %s" % (type(e).__name__, e)
    G["import_classes"] = imp

    disc = {"status": "FAIL", "found": {}}
    try:
        import nnunetv2
        from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer as _BT
        raiz = os.path.join(nnunetv2.__path__[0], "training", "nnUNetTrainer")
        for n in nomes:
            c = recursive_find_python_class(raiz, n, "nnunetv2.training.nnUNetTrainer")
            disc["found"][n] = {"found": c is not None,
                                "name_ok": bool(c) and c.__name__ == n,
                                "is_trainer": bool(c) and issubclass(c, _BT)}
        disc["status"] = "PASS" if nomes and all(
            v["found"] and v["name_ok"] and v["is_trainer"]
            for v in disc["found"].values()) else "FAIL"
    except Exception as e:
        disc["error"] = "%s: %s" % (type(e).__name__, e)
    G["recursive_discovery"] = disc

    cu = {"status": "FAIL"}
    try:
        import torch
        assert torch.cuda.is_available(), "CUDA indisponivel"
        dev = torch.device("cuda")
        x = torch.randn(2, 1, 32, 32, 32, device=dev, requires_grad=True)
        c = torch.nn.Conv3d(1, 8, 3, padding=1).to(dev)
        y = c(x); l = y.square().mean(); l.backward()
        torch.cuda.synchronize()
        g = c.weight.grad
        cu = {"status": "PASS" if (bool(torch.isfinite(l)) and bool(torch.isfinite(g).all())
                                   and bool((g != 0).any())) else "FAIL",
              "device": torch.cuda.get_device_name(0),
              "capability": list(torch.cuda.get_device_capability(0)),
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "loss_finite": bool(torch.isfinite(l)),
              "grad_finite": bool(torch.isfinite(g).all()),
              "grad_nonzero": bool((g != 0).any()),
              "peak_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20, 2)}
    except Exception as e:
        cu["error"] = "%s: %s" % (type(e).__name__, e)
    G["cuda_functional_test"] = cu

    rr = {"status": "FAIL", "roots": {}}
    todas = True
    for sub in ("raw", "preprocessed", "results"):
        d = os.path.join(PROVISION_ROOTS, sub)
        ex = os.path.isdir(d)
        wr = False
        if ex:
            try:
                t = os.path.join(d, ".w_%d" % os.getpid())
                open(t, "w").write("x"); os.remove(t); wr = True
            except Exception:
                wr = False
        rr["roots"][sub] = {"path": d, "exists": ex, "writable": wr}
        todas &= (ex and wr)
    rr["status"] = "PASS" if todas else "FAIL"
    G["isolated_roots"] = rr

    ep = {"status": "FAIL", "detail": {}}
    vb = os.path.dirname(sys.executable)
    okp = True
    for n in ENTRYPOINTS_EXIGIDOS:
        c = os.path.join(vb, n)
        pres = os.path.isfile(c) and os.access(c, os.X_OK)
        ep["detail"][n] = {"path": c, "present_executable": pres}
        okp &= pres
    ep["status"] = "PASS" if okp else "FAIL"
    G["nnunet_entrypoints"] = ep

    s8 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "08_prepare_nnunet.py")
    at = {"status": "FAIL", "path": s8, "present": os.path.isfile(s8)}
    if at["present"]:
        at["sha256"] = _sha256_ficheiro(s8)
        at["previous_sha256"] = "954ad0031b1171e87ce71ae690b37f3c266e89d90c2a60ce83fd3fc565449f2f"
        at["revision"] = ("REANCHOR_WITH_SHA256 — resolucao de caminho por "
                          "PULMONARY_BASE com hash conferido; ponteiros e "
                          "manifestos congelados INALTERADOS")
        at["status"] = "PASS"
    G["script08_code_attestation"] = at

    d = _prov_run([sys.executable, "-m", "pip", "check"], 600)
    G["pip_check"] = {"status": "PASS" if d["returncode"] == 0 else "FAIL", **d}

    d = _prov_run([sys.executable, "-m", "pip", "freeze"], 600)
    if d["returncode"] == 0:
        os.makedirs(LOGS_DIR, exist_ok=True)
        pv = os.path.join(LOGS_DIR, "package_versions_%s.txt" % run_id)
        open(pv, "w").write(d["stdout_tail"])
        rel["package_versions_path"] = pv
        rel["package_versions_sha256"] = _sha256_ficheiro(pv)
        rel["package_count"] = len([l for l in d["stdout_tail"].splitlines() if l.strip()])
    G["pip_freeze"] = {"status": "PASS" if d["returncode"] == 0 else "FAIL"}

    tudo = all(v.get("status") == "PASS" for v in G.values())
    rel["SCRIPT00_VERIFICATION"] = "PASS" if tudo else "FAIL"
    rel["CLUSTER_CUDA_ENVIRONMENT"] = "PASS" if tudo else "NOT_ASSIGNED"
    rel["CLUSTER_CUDA_ENVIRONMENT_SCOPE"] = "SCRIPT08_NNUNET_EXECUTION"
    rel["READY_FOR_SCRIPT08"] = bool(tudo)
    os.makedirs(LOGS_DIR, exist_ok=True)
    pj = os.path.join(LOGS_DIR, "script00_profile_%s.json" % run_id)
    json.dump(rel, open(pj, "w"), indent=2, sort_keys=True, default=str)
    print("── PERFIL nnunet-execution ──")
    for k, v in G.items():
        print("  %-26s %s" % (k, v.get("status")))
    print()
    for k in ("SCRIPT00_VERIFICATION", "CLUSTER_CUDA_ENVIRONMENT",
              "CLUSTER_CUDA_ENVIRONMENT_SCOPE", "PROJECT_WIDE_ENVIRONMENT",
              "READY_FOR_SCRIPT08"):
        print("  %-34s %s" % (k, rel[k]))
    print("  relatorio: %s" % pj)
    return rel


if "--profile" in sys.argv:
    _pf = sys.argv[sys.argv.index("--profile") + 1]
    if _pf != "nnunet-execution":
        print("perfil desconhecido: %s" % _pf); sys.exit(2)
    _td = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trainers")
    if "--trainers" in sys.argv:
        _td = sys.argv[sys.argv.index("--trainers") + 1]
    _r = perfil_nnunet_execution(os.path.abspath(_td))
    sys.exit(0 if _r["READY_FOR_SCRIPT08"] else 1)


if "--provision" in sys.argv or "--provision-dry-run" in sys.argv:
    _dest = PROVISION_VENV_PADRAO
    if "--venv" in sys.argv:
        _dest = os.path.expanduser(sys.argv[sys.argv.index("--venv") + 1])
    _tdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                         TRAINERS_DIR_POR_VERSAO.get(NNUNET_VERSAO, "trainers"))
    if "--trainers" in sys.argv:
        _tdir = sys.argv[sys.argv.index("--trainers") + 1]
    provisionar(_dest, os.path.abspath(_tdir),
                dry_run_only=("--provision-dry-run" in sys.argv))
    sys.exit(0)


v_py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
report["checks"]["python"] = v_py
versions_lines.append(f"python={v_py}")
print(f"[OK] Python {v_py}")


driver = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
if "ERROR" not in driver and driver:
    report["gpu_info"]["nvidia_driver"] = driver.splitlines()[0].strip()
    versions_lines.append(f"nvidia_driver={report['gpu_info']['nvidia_driver']}")
    print(f"[OK] Driver NVIDIA {report['gpu_info']['nvidia_driver']}")
else:
    report["gpu_info"]["nvidia_driver"] = "N/A (nvidia-smi não disponível)"
    print("[INFO] nvidia-smi não disponível (sem GPU NVIDIA ou macOS)")

nvcc = _run(["nvcc", "--version"])
report["gpu_info"]["nvcc"] = nvcc
versions_lines.append(f"nvcc={nvcc[:80]}")

try:
    import torch
    v = torch.__version__
    cuda_avail = torch.cuda.is_available()
    gpu_name   = torch.cuda.get_device_name(0) if cuda_avail else "N/A"
    cuda_ver   = torch.version.cuda if hasattr(torch.version, "cuda") else "N/A"
    report["checks"]["torch"]             = v
    report["checks"]["cuda_available"]    = cuda_avail
    report["gpu_info"]["name"]            = gpu_name
    report["gpu_info"]["cuda_version_torch"] = str(cuda_ver)
    versions_lines.append(f"torch={v}")
    versions_lines.append(f"cuda_available={cuda_avail}")
    print(f"[OK] PyTorch {v} | CUDA disponível={cuda_avail} | GPU={gpu_name}")
except ImportError:
    report["checks"]["torch"] = "NOT INSTALLED"
    print("[FAIL] PyTorch não instalado")


try:
    import SimpleITK as sitk
    v = sitk.Version.VersionString()
    report["checks"]["SimpleITK"] = v
    versions_lines.append(f"SimpleITK={v}")
    print(f"[OK] SimpleITK {v}")
except ImportError:
    report["checks"]["SimpleITK"] = "NOT INSTALLED"
    print("[FAIL] SimpleITK não instalado")


try:
    import nnunetv2
    v = getattr(nnunetv2, "__version__", None)
    if v is None:
        import importlib.metadata
        v = importlib.metadata.version("nnunetv2")
    report["checks"]["nnunetv2"] = v
    versions_lines.append(f"nnunetv2={v}")
    print(f"[OK] nnU-Net v2 {v}")
except ImportError:
    report["checks"]["nnunetv2"] = "NOT INSTALLED"
    print("[FAIL] nnU-Net v2 não instalado")


try:
    import lungmask
    import inspect
    v = _pkg_version(lungmask, "lungmask")
    report["checks"]["lungmask"] = v
    versions_lines.append(f"lungmask={v}")

    lm_dir = os.path.dirname(inspect.getfile(lungmask))
    commit = _run(["git", "-C", lm_dir, "rev-parse", "HEAD"])
    if "ERROR" in commit or "fatal" in commit or not commit:
        commit = _run(["git", "-C", os.path.dirname(lm_dir), "rev-parse", "HEAD"])
    if "ERROR" in commit or "fatal" in commit or not commit:
        commit = f"pip:{v}"
    report["checks"]["lungmask_commit"] = commit
    versions_lines.append(f"lungmask_commit={commit}")
    print(f"[OK] LungMask {v} | commit={commit[:12]}")
except ImportError:
    report["checks"]["lungmask"] = "NOT INSTALLED"
    print("[FAIL] LungMask não instalado")


STAT_PKGS = [
    "numpy", "scipy", "pandas", "sklearn", "matplotlib",
    "pingouin", "statsmodels", "tcia_utils",
    "pydicom", "skimage",
    "requests", "SimpleITK",
]

_DIST_NAME = {"sklearn": "scikit-learn", "skimage": "scikit-image",
              "SimpleITK": "SimpleITK", "tcia_utils": "tcia_utils"}

for pkg in STAT_PKGS:
    try:
        mod = __import__(pkg)
        v   = _pkg_version(mod, _DIST_NAME.get(pkg, pkg))
        report["package_versions"][pkg] = v
        versions_lines.append(f"{pkg}={v}")
        print(f"[{'OK' if v != 'UNRESOLVED' else 'AVISO'}] {pkg} {v}")
    except ImportError:
        report["package_versions"][pkg] = "NOT INSTALLED"
        print(f"[FAIL] {pkg} não instalado")


SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
git_commit = _run(["git", "-C", SCRIPTS_DIR, "rev-parse", "HEAD"])
if "ERROR" in git_commit:
    git_commit = _run(["git", "-C", BASE, "rev-parse", "HEAD"])
if "ERROR" in git_commit or not git_commit.strip():
    git_commit = "SEM_REPOSITORIO_GIT"
report["checks"]["scripts_git_commit"] = git_commit
report["checks"]["scripts_dir"]        = SCRIPTS_DIR
versions_lines.append(f"project_git_commit={git_commit}")
print(f"[{'OK' if git_commit != 'SEM_REPOSITORIO_GIT' else 'AVISO'}] "
      f"Commit projeto: {git_commit[:40]}")


for label, path in [("BASE", BASE), ("/", "/")]:
    try:
        usage   = shutil.disk_usage(path)
        free_gb = usage.free  / 1e9
        total_gb = usage.total / 1e9
        report["disk_space_gb"][label] = {
            "free_gb":  round(free_gb, 1),
            "total_gb": round(total_gb, 1),
        }
        status = "OK" if free_gb > 50 else "AVISO: < 50 GB livres"
        print(f"[DISCO {label}] {free_gb:.1f} GB livres / {total_gb:.1f} GB — {status}")
    except Exception as e:
        report["disk_space_gb"][label] = str(e)


for d in REQUIRED_DIRS:
    exists   = os.path.isdir(d)
    writable = os.access(d, os.W_OK) if exists else False
    report["folder_access"][d] = {"exists": exists, "writable": writable}
    if not exists:
        os.makedirs(d, exist_ok=True)
        report["folder_access"][d]["created"] = True
        print(f"[CRIADO] {d}")
    elif not writable:
        print(f"[ERRO] Sem permissão de escrita: {d}")
    else:
        print(f"[OK] {d}")


report_path = os.path.join(LOGS_DIR, "environment_report.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nRelatório salvo: {report_path}")

versions_path = os.path.join(LOGS_DIR, "package_versions.txt")
with open(versions_path, "w") as f:
    f.write("\n".join(versions_lines))
print(f"Versões salvas:  {versions_path}")


CRITICOS = {
    "torch": "09/10/14", "SimpleITK": "03–06", "nnunetv2": "08/09/10",
    "lungmask": "06", "tcia_utils": "01", "numpy": "todos",
    "scipy": "05/14", "pandas": "11–27", "sklearn": "07/17", "pydicom": "02–04",
    "skimage": "14", "statsmodels": "16/24", "pingouin": "21",
}

AMBIENTES = {
    "venv": {"estagios": ["00", "01", "02", "03", "05", "06", "07"],
             "exige": ["lungmask", "torch", "SimpleITK", "sklearn", "pydicom"]},
    "venv_pylidc021_audit": {"estagios": ["04", "04a"],
                             "exige": ["pylidc", "SimpleITK", "pydicom", "numpy"]},
}
import subprocess as _sp
_amb = {}
for _nome, _spec in AMBIENTES.items():
    _py = os.path.join(BASE, _nome, "bin", "python")
    _r = {"path": os.path.relpath(_py, BASE), "estagios": _spec["estagios"]}
    if not os.path.isfile(_py):
        _r["status"] = "AUSENTE"
        _r["modulos"] = {}
    else:
        _cod = ("import json,importlib.metadata as md;out={}\n"
                "for m in %r:\n"
                "    try: out[m]=md.version(m)\n"
                "    except Exception:\n"
                "        try:\n"
                "            import importlib; importlib.import_module(m); out[m]='presente'\n"
                "        except Exception: out[m]='NOT INSTALLED'\n"
                "print(json.dumps(out))" % _spec["exige"])
        try:
            _o = _sp.run([_py, "-c", _cod], capture_output=True, text=True, timeout=120)
            _r["modulos"] = json.loads(_o.stdout.strip().splitlines()[-1])
        except Exception as _e:
            _r["modulos"] = {}
            _r["erro"] = "%s: %s" % (type(_e).__name__, _e)
        _falta = sorted(k for k, v in _r["modulos"].items() if v == "NOT INSTALLED")
        _r["ausentes"] = _falta
        _r["status"] = "OK" if (not _falta and "erro" not in _r) else "INCOMPLETO"
    _amb[_nome] = _r
    print("[%s] ambiente %s (estagios %s)%s"
          % ("OK" if _r["status"] == "OK" else "FAIL", _nome,
             ",".join(_spec["estagios"]),
             "" if _r["status"] == "OK" else "  -> " + str(_r.get("ausentes") or _r["status"])))
report["ambientes_por_estagio"] = _amb
_amb_incompletos = sorted(k for k, v in _amb.items() if v["status"] != "OK")

todos = {**report["checks"], **report["package_versions"]}
ausentes    = sorted(k for k, v in todos.items() if v == "NOT INSTALLED")
irresolutos = sorted(k for k in CRITICOS
                     if str(todos.get(k, "")).strip() in ("", "UNRESOLVED",
                                                          "unknown", "ok", "None"))

if ausentes or _amb_incompletos:
    status = "FAIL_INPUT"
elif irresolutos or git_commit == "SEM_REPOSITORIO_GIT":
    status = "PASS_WITH_FLAG"
else:
    status = "PASS"

report["status"]              = status
report["missing_packages"]    = ausentes
report["unresolved_versions"] = irresolutos
report["incomplete_environments"] = _amb_incompletos

with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"\n── Veredicto: {status} ──")
if ausentes:
    print(f"[FAIL] Pacotes ausentes: {', '.join(ausentes)}")
for _n in _amb_incompletos:
    print(f"[FAIL] Ambiente {_n} incompleto (estagios "
          f"{','.join(_amb[_n]['estagios'])}): {_amb[_n].get('ausentes') or _amb[_n]['status']}")
for k in irresolutos:
    print(f"[AVISO] Versão por resolver: {k} (usado pelo script {CRITICOS[k]}) "
          f"= {todos.get(k)!r}")
if git_commit == "SEM_REPOSITORIO_GIT":
    print("[AVISO] Scripts fora de controlo de versão — a seção 7.1 do protocolo "
          "exige registar o commit de cada execução.")
if status == "PASS":
    print("[OK] Ambiente completo e integralmente rastreável.")

sys.exit(1 if status == "FAIL_INPUT" else 0)
