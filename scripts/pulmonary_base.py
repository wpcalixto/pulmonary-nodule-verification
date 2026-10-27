#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    pulmonary_base.py

File type:
    Module

Purpose:
    Shared module that resolves the project base directory from the PULMONARY_BASE environment variable.

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
import csv
import json
import math


_DEFAULT_BASE = ("<external>/Arquivos Pacheco 02_05_2022/"
                 "Programacao/Projeto_Pulmonar/Ultima_Versao/Pulmonary_Project")

BASE = os.environ.get("PULMONARY_BASE", _DEFAULT_BASE)

_SENTINEL_DIRS = ["data", "config", "analysis"]


def check_base(strict=True):
    """
    Confirma que BASE existe e contém a árvore esperada.
    strict=True levanta RuntimeError; strict=False devolve (ok, reason).
    """
    if not os.path.isdir(BASE):
        reason = f"BASE não existe: {BASE}"
        if strict:
            raise RuntimeError(
                f"{reason}\n"
                f"Defina PULMONARY_BASE ou corrija _DEFAULT_BASE em pulmonary_base.py."
            )
        return False, reason

    missing = [d for d in _SENTINEL_DIRS if not os.path.isdir(os.path.join(BASE, d))]
    if missing:
        reason = f"BASE existe mas faltam subpastas {missing}: {BASE}"
        if strict:
            raise RuntimeError(reason)
        return False, reason
    return True, "ok"


def compare_sets(expected, observed):
    """
    Compara conjuntos de identificadores.
    Devolve dict com ausentes, extras e contagens. Aprovar só com ambos vazios.
    """
    exp, obs = set(expected), set(observed)
    missing, extra = sorted(exp - obs), sorted(obs - exp)
    return {
        "n_expected": len(exp),
        "n_observed": len(obs),
        "missing":    missing,
        "extra":      extra,
        "ok":         not missing and not extra,
    }


def _fmt_set_diff(cmp_result, label="ids"):
    if cmp_result["ok"]:
        return "ok"
    parts = []
    if cmp_result["missing"]:
        head = ", ".join(cmp_result["missing"][:3])
        parts.append(f"{len(cmp_result['missing'])} {label} ausentes ({head}…)")
    if cmp_result["extra"]:
        head = ", ".join(cmp_result["extra"][:3])
        parts.append(f"{len(cmp_result['extra'])} {label} extras ({head}…)")
    return "; ".join(parts)


def nifti_info(path):
    """
    Lê SÓ o cabeçalho do NIfTI (sem carregar voxels). Rápido.
    Devolve dict com size/spacing/origin/direction, ou None se ilegível.
    """
    try:
        import SimpleITK as sitk
        r = sitk.ImageFileReader()
        r.SetFileName(path)
        r.ReadImageInformation()
        return {
            "size":      tuple(r.GetSize()),
            "spacing":   tuple(round(float(v), 6) for v in r.GetSpacing()),
            "origin":    tuple(round(float(v), 4) for v in r.GetOrigin()),
            "direction": tuple(round(float(v), 6) for v in r.GetDirection()),
        }
    except Exception:
        return None


def geom_close(a, b, tol=1e-3):
    """Compara duas geometrias devolvidas por nifti_info(). None nunca casa."""
    if a is None or b is None:
        return False
    if tuple(a["size"]) != tuple(b["size"]):
        return False
    for key in ("spacing", "origin", "direction"):
        va, vb = a[key], b[key]
        if len(va) != len(vb):
            return False
        if any(not math.isclose(x, y, abs_tol=tol) for x, y in zip(va, vb)):
            return False
    return True


def verify_gzip_stream(path):
    """
    Lê o fluxo gzip até EOF, validando CRC-32 e o marcador final.

    O módulo `gzip` só verifica CRC e ISIZE quando a descompressão chega ao fim;
    ler apenas o cabeçalho não deteta truncamento. Dois NIfTI do LIDC passaram em
    todas as verificações de geometria — cabeçalho e posições corretos — e só
    foram apanhados por comparação de conteúdo, porque o ficheiro estava
    fisicamente truncado.

    Devolve (ok, motivo, n_bytes_descomprimidos).
    """
    if not str(path).endswith(".gz"):
        return True, "não comprimido", os.path.getsize(path)
    import gzip as _gzip
    total = 0
    try:
        with _gzip.open(path, "rb") as f:
            for bloco in iter(lambda: f.read(1 << 22), b""):
                total += len(bloco)
    except EOFError:
        return False, "FAIL_PREMATURE_EOF: fluxo termina antes do marcador final", total
    except Exception as e:
        return False, f"FAIL_GZIP_STREAM: {e}", total
    return True, "ok", total


def verify_nifti_payload(path, n_bytes_descomprimidos=None):
    """
    Confirma que o tamanho descomprimido corresponde à carga útil declarada no
    cabeçalho NIfTI-1: `vox_offset` mais o produto das dimensões pelo `bitpix`.
    """
    h = nifti_header_fields(path)
    if h is None:
        return False, "cabeçalho NIfTI-1 ilegível", None
    import struct, gzip as _gzip
    try:
        abre = _gzip.open if str(path).endswith(".gz") else open
        with abre(path, "rb") as f:
            cab = f.read(348)
        e = "<" if struct.unpack("<i", cab[:4])[0] == 348 else ">"
        bitpix = struct.unpack(e + "h", cab[72:74])[0]
        vox_offset = struct.unpack(e + "f", cab[108:112])[0]
    except Exception as ex:
        return False, f"cabeçalho ilegível: {ex}", None

    dim = h["dim"]
    n = 1
    for i in range(1, min(dim[0], 7) + 1):
        n *= max(int(dim[i]), 1)
    esperado = int(vox_offset) + n * (bitpix // 8)
    if n_bytes_descomprimidos is None:
        ok, motivo, n_bytes_descomprimidos = verify_gzip_stream(path)
        if not ok:
            return False, motivo, esperado
    if n_bytes_descomprimidos < esperado:
        return False, (f"carga útil incompleta: {n_bytes_descomprimidos} B de "
                       f"{esperado} B declarados no cabeçalho"), esperado
    return True, "ok", esperado


def verify_nifti(path, ref_geom=None, expect_binary=False,
                 expect_nonempty=None, hu_range=None, min_bytes=1024,
                 check_stream=True):
    """
    Verifica um NIfTI de saída pelos VALORES, não só pela existência.

    ref_geom        dict de nifti_info() que a geometria deve reproduzir
    expect_binary   True  → todos os voxels devem estar em {0, 1}
    expect_nonempty True  → exige ao menos um voxel positivo
                    False → exige máscara integralmente vazia
                    None  → não avalia (vazio é resultado legítimo possível)
    hu_range        (lo, hi) plausível para intensidade; usa-se em CT
    min_bytes       tamanho mínimo; apanha ficheiro truncado ou de tamanho zero

    Ler voxels custa ~0,35 s por CT. É deliberado: sem isso não se distingue
    ficheiro válido de ficheiro corrompido que abre.
    """
    if not os.path.isfile(path):
        return False, "ausente"

    size_bytes = os.path.getsize(path)
    if size_bytes < min_bytes:
        return False, f"tamanho {size_bytes} B abaixo do mínimo ({min_bytes} B)"

    if check_stream:
        ok, motivo, n_bytes = verify_gzip_stream(path)
        if not ok:
            return False, motivo
        ok, motivo, _ = verify_nifti_payload(path, n_bytes)
        if not ok:
            return False, motivo

    info = nifti_info(path)
    if info is None:
        return False, "ilegível (cabeçalho NIfTI não abre)"

    if any(d <= 0 for d in info["size"]):
        return False, f"dimensão inválida {info['size']}"
    if any(s <= 0 for s in info["spacing"]):
        return False, f"espaçamento inválido {info['spacing']}"
    if any(not math.isfinite(v) for v in info["origin"] + info["direction"]):
        return False, "origem ou direção não finita"

    if ref_geom is not None and not geom_close(info, ref_geom):
        return False, (f"geometria divergente: size={info['size']} vs "
                       f"{ref_geom['size']}, spacing={info['spacing']} vs "
                       f"{ref_geom['spacing']}")

    need_values = expect_binary or expect_nonempty is not None or hu_range
    if need_values:
        try:
            import SimpleITK as sitk
            import numpy as np
            arr = sitk.GetArrayFromImage(sitk.ReadImage(path))
        except Exception as e:
            return False, f"voxels ilegíveis: {e}"

        if not np.all(np.isfinite(arr)):
            return False, "contém NaN ou infinito"

        if expect_binary:
            uniq = np.unique(arr)
            if not set(uniq.tolist()) <= {0, 1}:
                head = uniq[:5].tolist()
                return False, f"não binária (valores {head}…)"

        if expect_nonempty is True and arr.max() == 0:
            return False, "máscara vazia onde se esperava conteúdo"
        if expect_nonempty is False and arr.max() != 0:
            return False, "máscara não vazia onde se esperava vazio"

        if hu_range is not None:
            lo, hi = hu_range
            if arr.min() < lo or arr.max() > hi:
                return False, (f"intensidade fora de [{lo},{hi}]: "
                               f"[{arr.min()},{arr.max()}]")

    return True, "ok"


def read_csv_rows(path):
    """Lê um CSV em lista de dicts. Devolve None se ausente ou ilegível."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return None


def verify_csv(path, required_cols=(), min_rows=1, expected_ids=None,
               id_col=None, unique_key=None, numeric_cols=()):
    """
    Verifica um CSV de saída pelos VALORES.

    required_cols  colunas obrigatórias
    min_rows       número mínimo de linhas
    expected_ids   conjunto de identificadores que devem estar presentes
    id_col         coluna que carrega o identificador (exigida com expected_ids)
    unique_key     tupla de colunas cuja combinação deve ser única
    numeric_cols   colunas que devem ser numéricas e finitas em todas as linhas
    """
    rows = read_csv_rows(path)
    if rows is None:
        return False, "ausente ou ilegível"
    if len(rows) < min_rows:
        return False, f"{len(rows)} linhas, mínimo {min_rows}"

    cols = set(rows[0].keys())
    missing_cols = [c for c in required_cols if c not in cols]
    if missing_cols:
        return False, f"colunas ausentes: {missing_cols}"

    if expected_ids is not None:
        if not id_col:
            return False, "expected_ids fornecido sem id_col"
        observed = {r.get(id_col, "").strip() for r in rows if r.get(id_col, "").strip()}
        cmp_result = compare_sets(expected_ids, observed)
        if not cmp_result["ok"]:
            return False, _fmt_set_diff(cmp_result, id_col)

    if unique_key:
        seen, dupes = set(), 0
        for r in rows:
            k = tuple(r.get(c, "") for c in unique_key)
            if k in seen:
                dupes += 1
            seen.add(k)
        if dupes:
            return False, f"{dupes} linhas duplicadas por {unique_key}"

    for c in numeric_cols:
        for i, r in enumerate(rows):
            raw = (r.get(c, "") or "").strip()
            if raw == "":
                continue
            try:
                v = float(raw)
            except ValueError:
                return False, f"coluna {c} não numérica na linha {i + 2}: {raw!r}"
            if not math.isfinite(v):
                return False, f"coluna {c} não finita na linha {i + 2}"

    return True, "ok"


def verify_json(path, required_keys=(), expect=None):
    """
    Verifica um JSON de saída.
    expect  dict {chave: valor} que deve bater exatamente.
    """
    if not os.path.isfile(path):
        return False, "ausente"
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:
        return False, f"JSON inválido: {e}"

    if not isinstance(data, (dict, list)):
        return False, f"tipo inesperado: {type(data).__name__}"

    if isinstance(data, dict):
        missing = [k for k in required_keys if k not in data]
        if missing:
            return False, f"chaves ausentes: {missing}"
        for k, v in (expect or {}).items():
            if data.get(k) != v:
                return False, f"{k}={data.get(k)!r}, esperado {v!r}"

    return True, "ok"


def nifti_header_fields(path):
    """
    Lê `qform_code`, `sform_code`, as três linhas de `srow` e o `pixdim` de um
    NIfTI-1, interpretando os 348 bytes do cabeçalho diretamente.

    Feito sem `nibabel` para não acrescentar dependência ao ambiente. O SimpleITK
    não expõe estes campos, e sem eles não é possível demonstrar que a geometria
    física foi preservada na escrita.

    Devolve dict ou None se ilegível.
    """
    import gzip
    import struct
    try:
        abre = gzip.open if str(path).endswith(".gz") else open
        with abre(path, "rb") as f:
            h = f.read(348)
        if len(h) < 348:
            return None
        (sizeof_hdr,) = struct.unpack("<i", h[0:4])
        endian = "<" if sizeof_hdr == 348 else ">"
        if struct.unpack(endian + "i", h[0:4])[0] != 348:
            return None
        dim    = struct.unpack(endian + "8h", h[40:56])
        pixdim = struct.unpack(endian + "8f", h[76:108])
        qform_code, sform_code = struct.unpack(endian + "2h", h[252:256])
        quatern = struct.unpack(endian + "3f", h[256:268])
        qoffset = struct.unpack(endian + "3f", h[268:280])
        srow_x  = struct.unpack(endian + "4f", h[280:296])
        srow_y  = struct.unpack(endian + "4f", h[296:312])
        srow_z  = struct.unpack(endian + "4f", h[312:328])
        return {
            "dim": dim, "pixdim": pixdim, "qfac": pixdim[0],
            "qform_code": qform_code, "sform_code": sform_code,
            "quatern": quatern, "qoffset": qoffset,
            "srow": (srow_x, srow_y, srow_z),
        }
    except Exception:
        return None


def _cantos_fisicos(origem, direcao, espacamento, tamanho):
    """
    Os oito cantos do volume em coordenadas físicas.

    origem/direcao/espacamento/tamanho seguem a convenção do SimpleITK: direção
    como 9 valores em ordem de linha, tamanho em (x, y, z).
    """
    d = list(direcao)
    cantos = []
    for ix in (0, tamanho[0] - 1):
        for iy in (0, tamanho[1] - 1):
            for iz in (0, tamanho[2] - 1):
                idx = (ix, iy, iz)
                p = []
                for r in range(3):
                    p.append(origem[r] + sum(d[r * 3 + c] * espacamento[c] * idx[c]
                                             for c in range(3)))
                cantos.append(tuple(p))
    return sorted(cantos)


def verify_affine_equivalence(nifti_path, ref_origin, ref_direction,
                              ref_spacing, ref_size, tol_mm=0.5):
    """
    Demonstra que o NIfTI escrito preserva a geometria física da origem.

    Não compara a matriz DICOM-LPS com a matriz NIfTI-RAS diretamente: NIfTI usa
    RAS e DICOM usa LPS, e uma comparação literal classificaria a mudança de
    convenção como alteração anatômica. A equivalência é demonstrada no espaço
    físico — o SimpleITK devolve a geometria já em LPS ao reler o ficheiro, e a
    comparação incide sobre origem, direção, espaçamento, dimensões e os oito
    cantos do volume.

    `qform` e `sform` são lidos do cabeçalho para confirmar que o ficheiro
    carrega geometria declarada, e não apenas dimensões.

    Devolve (ok, motivo, detalhes).
    """
    info = nifti_info(nifti_path)
    if info is None:
        return False, "NIfTI ilegível", {}

    det = {}
    if tuple(info["size"]) != tuple(ref_size):
        return False, f"dimensões {info['size']} vs {tuple(ref_size)}", det

    for i, (a, b) in enumerate(zip(info["spacing"], ref_spacing)):
        if abs(a - b) > 1e-3:
            return False, f"espaçamento eixo {i}: {a} vs {b}", det

    for i, (a, b) in enumerate(zip(info["origin"], ref_origin)):
        if abs(a - b) > tol_mm:
            return False, f"origem eixo {i}: {a} vs {b} (tol {tol_mm} mm)", det

    ang = 0.0
    for a, b in zip(info["direction"], ref_direction):
        ang = max(ang, abs(a - b))
    if ang > 1e-3:
        return False, f"direção difere em {ang:.5f} (cosseno)", det
    det["max_direction_delta"] = round(ang, 6)

    c_out = _cantos_fisicos(info["origin"], info["direction"],
                            info["spacing"], info["size"])
    c_ref = _cantos_fisicos(ref_origin, ref_direction, ref_spacing, ref_size)
    pior = max(max(abs(x - y) for x, y in zip(p, q))
               for p, q in zip(c_out, c_ref))
    if pior > tol_mm:
        return False, f"cantos do volume divergem em {pior:.3f} mm", det
    det["max_corner_delta_mm"] = round(pior, 4)

    h = nifti_header_fields(nifti_path)
    if h is None:
        return False, "cabeçalho NIfTI-1 ilegível", det
    if h["qform_code"] == 0 and h["sform_code"] == 0:
        return False, "qform_code e sform_code ambos zero — sem geometria declarada", det
    det["qform_code"] = h["qform_code"]
    det["sform_code"] = h["sform_code"]

    return True, "ok", det


RIDER_SELECTION_JSON = os.path.join(BASE, "config", "rider_pair_selection.json")
RIDER_MANIFEST_CSV   = os.path.join(BASE, "analysis", "rider_pair_manifest.csv")


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_rider_selection(strict=True):
    """
    Carrega a âncora de seleção de pares RIDER e confirma que o manifesto em
    disco é exatamente aquele que a produziu, pelo SHA-256 completo.

    Qualquer script que consuma produtos RIDER deve chamar isto antes de
    trabalhar. Sem esta verificação, um manifesto regenerado com regra diferente
    passaria despercebido e os produtos antigos seriam reutilizados em silêncio.

    Devolve (config: dict, pares: list[dict]).
    Levanta RuntimeError quando strict=True e a verificação falha.
    """
    def falhar(msg):
        if strict:
            raise RuntimeError(f"FAIL_INPUT — âncora RIDER: {msg}")
        return None, None

    if not os.path.isfile(RIDER_SELECTION_JSON):
        return falhar(f"{RIDER_SELECTION_JSON} ausente — execute o script 02")
    with open(RIDER_SELECTION_JSON) as f:
        cfg = json.load(f)

    for chave in ("pair_selection_version", "pair_manifest_sha256"):
        if not cfg.get(chave):
            return falhar(f"campo {chave} ausente ou vazio")

    if not os.path.isfile(RIDER_MANIFEST_CSV):
        return falhar(f"{RIDER_MANIFEST_CSV} ausente")

    atual = sha256_file(RIDER_MANIFEST_CSV)
    if atual != cfg["pair_manifest_sha256"]:
        return falhar(
            f"o manifesto mudou desde a seleção.\n"
            f"  esperado: {cfg['pair_manifest_sha256']}\n"
            f"  em disco: {atual}\n"
            f"  Reexecute o script 02 e invalide os produtos derivados."
        )

    pares = read_csv_rows(RIDER_MANIFEST_CSV) or []
    return cfg, pares


def write_provenance(dest_json, **campos):
    """Grava o JSON lateral de proveniência de um produto."""
    os.makedirs(os.path.dirname(dest_json), exist_ok=True)
    with open(dest_json, "w") as f:
        json.dump(campos, f, indent=2, sort_keys=True)


def verify_provenance(prov_json, cfg, campos_obrigatorios=()):
    """
    Confirma que um produto existente foi gerado sob a seleção vigente.

    Um produto sem JSON de proveniência, ou com versão ou hash diferentes, é
    `INVALIDATED` — nunca reutilizável. Devolve (estado, motivo), em que estado
    é 'REUSABLE' ou 'INVALIDATED'.
    """
    if not os.path.isfile(prov_json):
        return "INVALIDATED", "sem ficheiro de proveniência"
    try:
        with open(prov_json) as f:
            p = json.load(f)
    except Exception as e:
        return "INVALIDATED", f"proveniência ilegível: {e}"

    falta = [c for c in campos_obrigatorios if not str(p.get(c, "")).strip()]
    if falta:
        return "INVALIDATED", f"campos ausentes na proveniência: {falta}"
    if p.get("pair_selection_version") != cfg["pair_selection_version"]:
        return "INVALIDATED", (f"versão {p.get('pair_selection_version')!r} "
                               f"!= {cfg['pair_selection_version']!r}")
    if p.get("pair_manifest_sha256") != cfg["pair_manifest_sha256"]:
        return "INVALIDATED", "hash do manifesto diferente do vigente"
    if p.get("conversion_status") != "OK":
        return "INVALIDATED", f"conversion_status={p.get('conversion_status')!r}"
    return "REUSABLE", "proveniência válida sob a seleção vigente"


def log_skip(log, label, ok, reason):
    """
    Regista a decisão de skip de forma uniforme e devolve `ok`.

        if log_skip(log, pid, *verify_nifti(out)):
            continue          # saída verificada — pular
        # senão: reprocessar
    """
    if ok:
        log.info(f"  [SKIP-OK] {label}")
    else:
        log.warning(f"  [REPROC]  {label} — saída inválida: {reason}")
    return ok


INDEPENDENT_READER_ID = "CUSTOM_NIFTI1_STREAM_READER"


def read_nifti1_independent(path):
    """
    Lê um NIfTI-1 interpretando o cabeçalho e os voxels diretamente do fluxo,
    sem SimpleITK e sem nibabel.

    Serve de segunda via de leitura para confrontar o que o SimpleITK devolve.
    Não substitui o `nibabel`: é uma implementação própria, e o resultado deve
    ser rotulado como tal.
    """
    import gzip as _gzip
    import struct
    import numpy as np
    abre = _gzip.open if str(path).endswith(".gz") else open
    with abre(path, "rb") as f:
        buf = f.read()
    e = "<" if struct.unpack("<i", buf[:4])[0] == 348 else ">"
    dim = struct.unpack(e + "8h", buf[40:56])
    dt_code = struct.unpack(e + "h", buf[70:72])[0]
    vox_offset = struct.unpack(e + "f", buf[108:112])[0]
    scl_slope, scl_inter = struct.unpack(e + "2f", buf[112:120])
    dt = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32,
          512: np.uint16, 256: np.int8}[dt_code]
    n = dim[1] * dim[2] * dim[3]
    a = np.frombuffer(buf, dtype=e + np.dtype(dt).char, count=n,
                      offset=int(vox_offset)).reshape(dim[3], dim[2], dim[1])
    if scl_slope not in (0.0, 1.0) or scl_inter != 0.0:
        a = a.astype(np.float64) * scl_slope + scl_inter
    return a


def independent_reader_hash():
    """SHA-256 do código-fonte do leitor próprio, para registo de proveniência."""
    import inspect
    import hashlib
    return hashlib.sha256(
        inspect.getsource(read_nifti1_independent).encode()).hexdigest()


ESQUEMA_PONTEIRO_ACTUAL = "pulmonary/v2"

_ALIASES_CAMINHO = ("manifest", "annotations_manifest", "lidc_final_manifest",
                    "manifest_path")
_ALIASES_HASH = ("manifest_body_sha256", "manifest_file_sha256",
                 "manifest_sha256", "manifest_file_sha")


def hash_corpo_manifesto(corpo, campo_assinatura="manifest_body_sha256"):
    """
    Assinatura de um manifesto sobre si próprio: sha256 do corpo excluindo o
    campo da assinatura. Aceita o nome antigo para poder verificar manifestos
    já publicados.
    """
    import json as _json
    import hashlib as _hashlib
    limpo = {k: v for k, v in corpo.items()
             if k not in (campo_assinatura, "manifest_sha256",
                          "manifest_body_sha256")}
    return _hashlib.sha256(
        _json.dumps(limpo, indent=1, sort_keys=True).encode()).hexdigest()


def assinar_manifesto(corpo):
    """Devolve o corpo com `manifest_body_sha256` preenchido."""
    m = dict(corpo)
    m.pop("manifest_sha256", None)
    m["manifest_body_sha256"] = hash_corpo_manifesto(m)
    return m


def escrever_ponteiro(caminho_ponteiro, caminho_manifesto, **campos):
    """
    Escreve um ponteiro no esquema actual. `campos` são os dados próprios do
    estágio; os de integridade são preenchidos aqui e não se passam à mão.
    """
    import json as _json
    corpo = _json.load(open(caminho_manifesto))
    p = dict(campos)
    p["pointer_schema"] = ESQUEMA_PONTEIRO_ACTUAL
    p["manifest"] = caminho_manifesto
    p["manifest_file_sha256"] = sha256_file(caminho_manifesto)
    p["manifest_body_sha256"] = corpo.get("manifest_body_sha256") or \
        corpo.get("manifest_sha256") or hash_corpo_manifesto(corpo)
    _json.dump(p, open(caminho_ponteiro, "w"), indent=1, sort_keys=True)
    return p


def ler_ponteiro(caminho_ponteiro, etiqueta, exigir_hash=True):
    """
    Lê e VERIFICA um ponteiro, do esquema actual ou de qualquer um dos antigos.

    Devolve (ponteiro, manifesto, caminho_do_manifesto, diagnostico), em que o
    diagnóstico diz que esquema foi lido e a que corresponde cada campo de hash
    declarado — `FILE`, `BODY` ou `UNRECOGNISED`.

    Levanta SystemExit se algum hash declarado não corresponder a nenhum dos
    dois valores reais. Um campo cujo nome mente sobre o que contém deixa de
    passar despercebido: ou bate com o ficheiro, ou bate com o corpo, ou reprova.
    """
    import json as _json
    if not os.path.isfile(caminho_ponteiro):
        raise SystemExit("[FAIL] sem ponteiro de %s em %s"
                         % (etiqueta, caminho_ponteiro))
    p = _json.load(open(caminho_ponteiro))

    caminho = next((p[k] for k in _ALIASES_CAMINHO if p.get(k)), None)
    if not caminho:
        raise SystemExit("[FAIL] ponteiro de %s nao declara caminho do manifesto "
                         "(procurado em %s)" % (etiqueta, list(_ALIASES_CAMINHO)))
    if not os.path.isabs(caminho):
        caminho = os.path.join(BASE, caminho)
    if not os.path.isfile(caminho):
        raise SystemExit("[FAIL] manifesto de %s inacessivel: %s"
                         % (etiqueta, caminho))

    corpo = _json.load(open(caminho))
    h_ficheiro = sha256_file(caminho)
    h_corpo = hash_corpo_manifesto(corpo)

    diagnostico = {"schema": p.get("pointer_schema", "legado"),
                   "manifest_path_field":
                       next(k for k in _ALIASES_CAMINHO if p.get(k)),
                   "campos": {}}
    declarados = {k: p[k] for k in _ALIASES_HASH if p.get(k)}
    for k, v in declarados.items():
        diagnostico["campos"][k] = ("FILE" if v == h_ficheiro else
                                    "BODY" if v == h_corpo else "UNRECOGNISED")
    maus = [k for k, v in diagnostico["campos"].items() if v == "UNRECOGNISED"]
    if maus:
        raise SystemExit(
            "[FAIL] ponteiro de %s: %s nao corresponde nem ao ficheiro (%s) nem "
            "ao corpo (%s) do manifesto. O produto nao e o que o ponteiro diz."
            % (etiqueta, maus, h_ficheiro[:12], h_corpo[:12]))
    if exigir_hash and not declarados:
        raise SystemExit("[FAIL] ponteiro de %s nao declara hash nenhum do "
                         "manifesto: nao ha como provar que aponta para o "
                         "produto certo" % etiqueta)
    return p, corpo, caminho, diagnostico
