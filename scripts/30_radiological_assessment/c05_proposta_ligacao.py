#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    c05_proposta_ligacao.py

File type:
    Script

Purpose:
    Proposes the link between the radiological records and the markers from the anatomical description.
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
    - c05_proposta_ligacao.md

Outputs:
    - c02b_referencias.csv
    - c04_esferas_verificador.csv
    - c04_referencias_verificador.csv
    - c05_proposta_ligacao.csv

Dependencies:
    - pandas

Execution:
    cd <project_dir> && python scripts/30_radiological_assessment/c05_proposta_ligacao.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import csv
import pandas as pd

O = "resultados/radiological_reanalysis"
esf = pd.read_csv(f"{O}/c04_esferas_verificador.csv", dtype={"exame": str}).set_index(["exame", "esfera"])
ref = pd.read_csv(f"{O}/c04_referencias_verificador.csv", dtype={"exame": str}).set_index(["exame", "ref_obj"])
lat = pd.read_csv(f"{O}/c02b_referencias.csv", dtype={"exame": str}).set_index(["exame", "ref_obj"])["lado_pulmao"]

def dr(e, g):
    r = ref.loc[(e, g)]
    f = r["3D_F_estado"] + ("" if r["3D_F_estado"] == "FN" else " " + r["3D_F_decisao"])
    m = r["3D_FREM_estado"] + ("" if r["3D_FREM_estado"] == "FN" else " " + r["3D_FREM_decisao"])
    return f, m

def ds(e, s):
    r = esf.loc[(e, s)]
    out = []
    for a in ("3D_F", "3D_FREM"):
        if r[f"{a}_ate5mm"] and r[f"{a}_estado"] == "FP":
            out.append(f"FP {r[f'{a}_decisao']}")
        elif r["ref_ate5mm"]:
            out.append(f"ref {int(r['ref_obj'])} {r['ref_estado_3D_F']}")
        else:
            out.append("sem objeto")
    return tuple(out)

def multi(e, alvos, fn):
    res = [fn(e, a) for a in alvos]
    if len(set(res)) == 1:
        return res[0][0] + " (idêntico)", res[0][1] + " (idêntico)"
    return " / ".join(r[0] for r in res), " / ".join(r[1] for r in res)

L = [
 ("0061","D1","Nódulo 1","TP","língula; nota 8","ref",[3,4,6],"ambígua","3 referências esquerdas de altura média/superior compatíveis","Não auditável"),
 ("0061","D1","Nódulo 2","TP","segmento anterior do LSE; nota 10","ref",[4,6],"ambígua","referências esquerdas superiores (altura 0,74–0,75)","Não auditável"),
 ("0061","D1","Nódulo 3","TP","paramediastinal esquerda; nota 10","ref",[5],"provável","única esquerda junto à pleura e próxima do plano médio (ref 5)","Não auditável"),
 ("0061","D1","Nódulo 4","TP","lingular inferior; nota 10","ref",[3],"provável","esquerda, altura 0,52, abaixo das demais candidatas","Não auditável"),
 ("0061","D1","Nódulo 5","TP","paramediastinal direita; nota 1 (delimitou só uma porção)","ref",[2],"única","única referência no pulmão direito, junto à pleura mediastinal; IoU atual 0,17","Provável"),
 ("0061","D1","Nódulo 6","TP","justadiafragmática esquerda; nota 8","ref",[1],"única","única esquerda basal (altura 0,20) com contacto pleural","Não auditável"),
 ("0061","D1","Nódulo 7","TP","LID; nota 10","-",[],"sem referência atual","não há referência no lobo inferior direito; a única direita é paramediastinal (Nódulo 5)","Não auditável"),
 ("0061","D1","Nódulo 8","TP","lobo médio; nota 5 (junto a vaso)","-",[],"sem referência atual","a única referência direita já corresponde ao Nódulo 5; a análise pelas figuras indicou um TP (discordância a rever na imagem)","Provável"),
 ("0080","D1","Nódulo 1","TP","LID, espiculado; nota 7","ref",[2],"única","única referência detectada (direita, basal, 27 mm)","Não auditável"),
 ("0097","D2","Nódulo 1","TP","justafissural direita; nota 5","ref",[1],"única","única referência do exame","Confirmada"),
 ("0235","D1","Nódulo 1","TP","segmento anterior do LSD; nota 10","ref",[3],"provável","referência direita mais superior (altura 0,64)","Não auditável"),
 ("0235","D1","Nódulo 2","TP","periferia do LID; nota 10","ref",[1,2],"ambígua","duas referências direitas inferiores (altura 0,21 e 0,46)","Não auditável"),
 ("0620","D1","Nódulo 1","TP","campo médio direito, peribroncovascular; nota 10","ref",[3,4],"ambígua","duas referências direitas detectadas","Não auditável"),
 ("0620","D1","Nódulo 2","TP","segmento anterior do LSD; nota 10","ref",[3,4],"ambígua","duas referências direitas detectadas","Não auditável"),
 ("0879","D1","Nódulo 1","TP","ápice direito; nota 10","ref",[8],"única","única referência direita apical (altura 0,93)","Não auditável"),
 ("0879","D1","Nódulo 2","TP","ápice esquerdo; nota 10","ref",[5,6],"ambígua","duas referências esquerdas apicais detectadas","Não auditável"),
 ("0879","D1","Nódulo 3","TP","LSE, justapleural; nota 10","ref",[5,6],"ambígua","duas referências esquerdas apicais detectadas","Não auditável"),
 ("0879","D1","Nódulo 4","TP","língula; nota 10","ref",[3],"provável","única esquerda de altura média (0,60)","Não auditável"),
 ("0879","D1","Nódulo 5","TP","lobo médio, maior nódulo; nota 10","ref",[1],"única","maior referência do exame (16,4 mm), direita","Não auditável"),
 ("0922","D1","Nódulo 1","TP","campo médio esquerdo; nota 10","ref",[1],"única","única referência detectada","Não auditável"),
 ("0925","D1","Nódulo 1","TP","nota 10","ref",[1],"única","única referência do exame","Não auditável"),
 ("1008","D2","Nódulo 1","TP","segmento posterior do LSD; nota 10","ref",[5],"provável","referência direita detectada mais superior (altura 0,67)","Não auditável"),
 ("1008","D2","Nódulo 2","TP","lobo médio; nota 10","ref",[2,4],"ambígua","duas referências direitas de altura média","Não auditável"),
 ("1008","D2","Nódulo 3","TP","LIE; nota 10","ref",[1],"única","única referência esquerda","Não auditável"),
 ("0061","D1","Caso 1","FP","pulmão direito, vasos; dificuldade 1","-",[],"não localizável","esferas do 0061 excluídas (marcação não declarada)","Não auditável"),
 ("0061","D1","Caso 2","FP","justapleural direita; dificuldade 1","-",[],"não localizável","esferas do 0061 excluídas (marcação não declarada)","Não auditável"),
 ("0097","D2","Caso 1","FP","diminutos esparsos, via aérea distal; dificuldade 8","esf",[1,2,3,4,5,6],"única","registo único para os 6 marcadores do exame","Não auditável"),
 ("0080","D1","Caso 1","FP","cicatriz, campo superior esquerdo; dificuldade 5","esf",[1],"provável","2 marcadores esquerdos; concordante com a análise pelas figuras","Confirmada"),
 ("0080","D1","Caso 2","FP","justapleural direita, atelectasia; dificuldade 1","-",[],"não localizável","sem marcador à direita","Não auditável"),
 ("0080","D1","Caso 3","FP","periferia superior esquerda, justapleural; dificuldade 10","esf",[2],"provável","segundo marcador esquerdo","Não auditável"),
 ("0080","D1","Caso 4","FP","periferia direita; dificuldade 10","-",[],"não localizável","sem marcador à direita","Não auditável"),
 ("0925","D1","Caso 1","FP","dois micronódulos no LSD, cicatriz; dificuldade 7","esf",[5,6],"provável","dois marcadores direitos superiores, junto à pleura (altura 0,70 e 0,76)","Ambígua"),
 ("0925","D1","Caso 2","FP","opacidades adjacentes à cicatriz","esf",[5,6],"provável","mesma topografia do Caso 1","Ambígua"),
 ("0925","D1","Caso 3","FP","micronódulo no LSE inferior; nódulo real","esf",[4],"única","único marcador esquerdo","Não auditável"),
 ("0925","D1","Caso 4","FP","micronódulo justafissural no lobo médio; real (provável linfonodo)","esf",[3],"provável","direito, altura 0,42, afastado da pleura (10,7 mm)","Não auditável"),
 ("0925","D1","Caso 5","FP","justadiafragmático direito, atelectasia; dificuldade 1","esf",[1],"provável","direito basal (altura 0,14) junto à pleura","Não auditável"),
 ("0925","D1","Caso 6","FP","ínfimo no lobo médio, vaso; dificuldade 1","esf",[2],"provável","marcador direito restante (altura 0,23)","Não auditável"),
 ("0620","D1","Caso 1","FP","três diminutos no LID; dificuldade 8","-",[],"não localizável","marcação do 0620 não preservada","Não auditável"),
 ("0879","D1","Caso 1","FP","LSD, vaso; dificuldade 1","esf",[2],"única","único marcador direito superior","Não auditável"),
 ("0879","D1","Caso 2","FP","LID; nódulo real ausente da referência","esf",[1],"única","único marcador direito inferior","Confirmada"),
 ("1008","D2","Caso 1","FP","semissólido peribroncovascular direito; real","esf",[5],"provável","direito, central (24,9 mm da pleura), HU médio −253","Provável"),
 ("1008","D2","Caso 2","FP","vaso próximo ao Caso 1; dificuldade 1","esf",[6],"provável","marcador vizinho ao do Caso 1 (altura 0,66)","Não auditável"),
 ("1008","D2","Caso 3","FP","micronódulo justapleural no lobo médio (provável linfonodo)","esf",[3],"provável","direito, altura 0,41, a 4,4 mm da pleura","Não auditável"),
 ("1008","D2","Caso 4","FP","micronódulo no LID; real","esf",[1],"provável","direito inferior (altura 0,28)","Provável"),
 ("0235","D1","Caso 1","FP","campo superior esquerdo, vaso; dificuldade 1","esf",[7],"provável","único marcador esquerdo superior (altura 0,70)","Não auditável"),
 ("0235","D1","Caso 2","FP","justapleural direita, linfonodo; dificuldade 1","esf",[4,5],"ambígua","dois marcadores direitos com contacto pleural","Não auditável"),
 ("0235","D1","Caso 3","FP","esparsos (vasos ou via aérea); dificuldade 10","esf",[1,2,3,6],"provável","marcadores restantes","Não auditável"),
 ("0922","D1","Caso 1","FP","campo médio esquerdo, vaso; dificuldade 1","esf",[2],"única","único marcador esquerdo fora do ápice","Não auditável"),
 ("0922","D1","Caso 2","FP","campo médio direito anterior, vaso; dificuldade 1","esf",[1],"única","único marcador direito","Não auditável"),
 ("0620","D1","Caso 1","FN","segmento superior do LSE, justapleural; dificuldade 1","ref",[5],"única","única referência esquerda superior (altura 0,90) junto à pleura","Não auditável"),
 ("0620","D1","Caso 2","FN","espiculado, basal do LIE","ref",[1],"única","única referência esquerda basal (12,7 mm)","Não auditável"),
 ("0879","D1","Caso 1","FN","micronódulo no ápice esquerdo; dificuldade 7","esf",[4],"provável","marcador esquerdo mais apical, sobre a ref 7","Não auditável"),
 ("0879","D1","Caso 2","FN","micronódulo superior esquerdo; dificuldade 7","esf",[3],"provável","segundo marcador esquerdo superior, sobre a ref 4","Não auditável"),
 ("0922","D1","Caso 1","FN","ápice esquerdo","esf",[3],"única","único marcador apical, sobre a ref 2","Não auditável"),
 ("0080","D1","Caso 1","FN","justadiafragmática direita; dificuldade 1","ref",[1],"única","única referência não detectada (direita, basal, 25 mm)","Não encontrada"),
 ("1008","D2","Caso 1","FN","micronódulo justafissural direito (provável linfonodo); dificuldade 1","ref",[3,6],"ambígua","duas referências direitas não detectadas","Não encontrada"),
 ("1008","D2","Caso 2","FN","opacidade nodular no lobo médio, junto a atelectasia; dificuldade 8","ref",[3,6],"ambígua","duas referências direitas não detectadas; marcador 4 sobre a ref 3","Não encontrada"),
 ("1008","D2","Caso 3","FN*","opacidade no LIE composta por vasos (o modelo marcou)","esf",[2],"única","único marcador esquerdo; objeto atual é FP","Não auditável"),
]
assert len(L) == 58
rows = []
for e, cj, reg, cl, desc, tipo, alvos, grau, just, fig in L:
    if tipo == "ref":
        f, m = multi(e, alvos, dr); alvo = "ref " + "/".join(map(str, alvos))
    elif tipo == "esf":
        f, m = multi(e, alvos, ds); alvo = "marcador " + "/".join(map(str, alvos))
    else:
        f, m, alvo = "--", "--", "--"
    rows.append(dict(exame=e, conjunto=cj, registo=reg, classe=cl, descricao=desc, alvo=alvo, ligacao=grau,
                     justificativa=just, desfecho_3D_F=f, desfecho_3D_FREM=m, analise_figuras=fig))
df = pd.DataFrame(rows)
df.to_csv(f"{O}/c05_proposta_ligacao.csv", index=False)
with open(f"{O}/c05_proposta_ligacao.md", "w") as fh:
    fh.write("# Proposta de ligação dos 58 registos radiológicos (para confirmação)\n\n")
    fh.write("Ligação por descrição anatômica; lado pelo pulmão mais próximo; altura 0 = base, 1 = ápice. "
             "'(idêntico)': todas as candidatas têm o mesmo desfecho. *Registo classificado como FN na lista original, mas descrito como objeto marcado pelo modelo.\n\n")
    fh.write("| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n")
    for r in df.itertuples(index=False):
        fh.write("| " + " | ".join(str(v) for v in r) + " |\n")
print(df.ligacao.value_counts().to_string())
print(pd.crosstab(df.classe, df.ligacao).to_string())
