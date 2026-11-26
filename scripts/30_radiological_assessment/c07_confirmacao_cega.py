#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    c07_confirmacao_cega.py

File type:
    Script

Purpose:
    Builds the blinded confirmation table of the link between radiological records and markers.
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
    - c07_confirmacao_cega.md

Outputs:
    - c02b_esferas.csv
    - c02b_referencias.csv
    - c07_confirmacao_cega.csv

Dependencies:
    - pandas

Execution:
    cd <project_dir> && python scripts/30_radiological_assessment/c07_confirmacao_cega.py

Portability notes:
    Paths are relative to the project directory (the working directory at execution), which reproduces the layout described in docs/post_v1_stages.md. No machine-specific path is required.

Reproducibility notes:
    Code archived as executed; only comments, this header and absolute path prefixes were changed (abstract syntax tree verified). Seeds and frozen parameters are those of the article (Table 1) and of the run records under data/.

Warnings:
    Source images (LIDC-IDRI, LNDb, RIDER Lung CT) and the radiologist's original marking files are not included.
"""

import pandas as pd
O = "resultados/radiological_reanalysis"
esf = pd.read_csv(f"{O}/c02b_esferas.csv", dtype={"exame": str}).set_index(["exame", "esfera"])
ref = pd.read_csv(f"{O}/c02b_referencias.csv", dtype={"exame": str}).set_index(["exame", "ref_obj"])
n = lambda v: f"{v:.1f}".replace(".", ",")
def at_ref(e, k):
    r = ref.loc[(e, k)]
    return f"ref {k}: {r.lado_pulmao[:3]}, alt {n(r.altura_0base_1apice)}, pleura {n(r.dist_pleura_mm)} mm, plano médio {n(r.dist_plano_medio_mm)} mm, {n(r.diam_mm)} mm"
def at_esf(e, k):
    r = esf.loc[(e, k)]
    return f"m{k}: {r.lado_pulmao[:3]}, alt {n(r.altura_0base_1apice)}, pleura {n(r.dist_pleura_mm)} mm, HU {int(r.hu_medio)}"
L = [
 ("0061","D1","Nódulo 1","TP","língula; nota 8","ref",[3,4,6],"ambígua","3 referências esquerdas de altura média/superior compatíveis","Não auditável"),
 ("0061","D1","Nódulo 2","TP","segmento anterior do LSE; nota 10","ref",[4,6],"ambígua","referências esquerdas superiores (altura 0,74–0,75)","Não auditável"),
 ("0061","D1","Nódulo 3","TP","paramediastinal esquerda; nota 10","ref",[5],"ambígua*","proposta: ref 5 (esquerda, contacto pleural, a mais próxima do plano médio); lobo não verificável sem imagem","Não auditável"),
 ("0061","D1","Nódulo 4","TP","lingular inferior; nota 10","ref",[3],"ambígua*","proposta: ref 3 (esquerda, altura 0,52, abaixo das demais); lobo só pela altura","Não auditável"),
 ("0061","D1","Nódulo 5","TP","paramediastinal direita; nota 1 (delimitou só uma porção)","ref",[2],"única","única referência no pulmão direito (9,3 mm do plano médio, contacto pleural); é também a única anotação LIDC à direita","Provável"),
 ("0061","D1","Nódulo 6","TP","justadiafragmática esquerda; nota 8","ref",[1],"única","única esquerda basal (altura 0,20) com contacto pleural","Não auditável"),
 ("0061","D1","Nódulo 7","TP","LID; nota 10","-",[],"sem referência atual","nenhum dos 4 leitores LIDC anotou nódulo no pulmão direito além da ref 2 (Nódulo 5); conferido em per_reader","Não auditável"),
 ("0061","D1","Nódulo 8","TP","lobo médio; nota 5 (junto a vaso)","-",[],"sem referência atual","idem Nódulo 7: nenhuma anotação LIDC no lobo médio (per_reader)","Provável"),
 ("0080","D1","Nódulo 1","TP","LID, espiculado; nota 7","ref",[2],"única","referência direita basal (27 mm)","Não auditável"),
 ("0097","D2","Nódulo 1","TP","justafissural direita; nota 5","ref",[1],"única","única referência do exame","Confirmada"),
 ("0235","D1","Nódulo 1","TP","segmento anterior do LSD; nota 10","ref",[3],"ambígua*","proposta: ref 3, direita mais superior (altura 0,64); segmento só pela altura","Não auditável"),
 ("0235","D1","Nódulo 2","TP","periferia do LID; nota 10","ref",[1,2],"ambígua","duas referências direitas inferiores (altura 0,21 e 0,46)","Não auditável"),
 ("0620","D1","Nódulo 1","TP","campo médio direito, peribroncovascular; nota 10","ref",[3,4],"ambígua","duas referências direitas compatíveis","Não auditável"),
 ("0620","D1","Nódulo 2","TP","segmento anterior do LSD; nota 10","ref",[3,4],"ambígua","duas referências direitas compatíveis","Não auditável"),
 ("0879","D1","Nódulo 1","TP","ápice direito; nota 10","ref",[8],"única","única referência direita apical (altura 0,93)","Não auditável"),
 ("0879","D1","Nódulo 2","TP","ápice esquerdo; nota 10","ref",[5,6],"ambígua","duas referências esquerdas apicais compatíveis","Não auditável"),
 ("0879","D1","Nódulo 3","TP","LSE, justapleural; nota 10","ref",[5,6],"ambígua","duas referências esquerdas apicais compatíveis","Não auditável"),
 ("0879","D1","Nódulo 4","TP","língula; nota 10","ref",[3],"ambígua*","proposta: ref 3, única esquerda de altura média (0,60); lobo só pela altura","Não auditável"),
 ("0879","D1","Nódulo 5","TP","lobo médio, maior nódulo; nota 10","ref",[1],"única","maior referência do exame (16,4 mm), direita","Não auditável"),
 ("0922","D1","Nódulo 1","TP","campo médio esquerdo; nota 10","ref",[1],"única","única referência compatível","Não auditável"),
 ("0925","D1","Nódulo 1","TP","nota 10","ref",[1],"única","única referência do exame","Não auditável"),
 ("1008","D2","Nódulo 1","TP","segmento posterior do LSD; nota 10","ref",[5],"ambígua*","proposta: ref 5, direita mais superior (altura 0,67); segmento só pela altura","Não auditável"),
 ("1008","D2","Nódulo 2","TP","lobo médio; nota 10","ref",[2,4],"ambígua","duas referências direitas de altura média","Não auditável"),
 ("1008","D2","Nódulo 3","TP","LIE; nota 10","ref",[1],"única","única referência esquerda","Não auditável"),
 ("0061","D1","Caso 1","FP","pulmão direito, vasos; dificuldade 1","esf",[2,3,5,8,9],"ambígua","5 marcadores direitos afastados da pleura (7,5–23,3 mm); a descrição (vasos) não distingue entre eles","Não auditável"),
 ("0061","D1","Caso 2","FP","justapleural direita; dificuldade 1","esf",[1,6,4],"ambígua","3 marcadores direitos junto à pleura (1,8; 3,7; 5,5 mm)","Não auditável"),
 ("0097","D2","Caso 1","FP","diminutos esparsos, via aérea distal; dificuldade 8","esf",[1,2,3,4,5,6],"múltipla","1 registo ligado aos 6 marcadores do exame; resultados por objeto","Não auditável"),
 ("0080","D1","Caso 1","FP","cicatriz, campo superior esquerdo; dificuldade 5","esf",[1],"provável","2 marcadores esquerdos; o 1 é o do campo superior","Confirmada"),
 ("0080","D1","Caso 2","FP","justapleural direita, atelectasia; dificuldade 1","-",[],"não localizável","sem marcador à direita","Não auditável"),
 ("0080","D1","Caso 3","FP","periferia superior esquerda, justapleural; dificuldade 10","esf",[2],"provável","segundo marcador esquerdo","Não auditável"),
 ("0080","D1","Caso 4","FP","periferia direita; dificuldade 10","-",[],"não localizável","sem marcador à direita","Não auditável"),
 ("0925","D1","Caso 1","FP","dois micronódulos no LSD, cicatriz; dificuldade 7","esf",[5,6],"múltipla","dois micronódulos: os 2 marcadores direitos superiores junto à pleura (altura 0,70 e 0,76)","Ambígua"),
 ("0925","D1","Caso 2","FP","opacidades adjacentes à cicatriz","esf",[5,6],"ambígua","partilha os marcadores 5/6 com o Caso 1; não contado como ligação própria","Ambígua"),
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
 ("0235","D1","Caso 3","FP","esparsos (vasos ou via aérea); dificuldade 10","esf",[1,2,3,6],"múltipla","esparsos: os 4 marcadores restantes do exame; resultados por objeto","Não auditável"),
 ("0922","D1","Caso 1","FP","campo médio esquerdo, vaso; dificuldade 1","esf",[2],"única","único marcador esquerdo fora do ápice","Não auditável"),
 ("0922","D1","Caso 2","FP","campo médio direito anterior, vaso; dificuldade 1","esf",[1],"única","único marcador direito","Não auditável"),
 ("0620","D1","Caso 1","FN","segmento superior do LSE, justapleural; dificuldade 1","ref",[5],"única","única referência esquerda superior (altura 0,90) junto à pleura","Não auditável"),
 ("0620","D1","Caso 2","FN","espiculado, basal do LIE","ref",[1],"única","única referência esquerda basal (12,7 mm)","Não auditável"),
 ("0879","D1","Caso 1","FN","micronódulo no ápice esquerdo; dificuldade 7","esf",[4],"provável","marcador esquerdo mais apical, coincide com a ref 7","Não auditável"),
 ("0879","D1","Caso 2","FN","micronódulo superior esquerdo; dificuldade 7","esf",[3],"provável","segundo marcador esquerdo superior, coincide com a ref 4","Não auditável"),
 ("0922","D1","Caso 1","FN","ápice esquerdo","esf",[3],"única","único marcador apical, coincide com a ref 2","Não auditável"),
 ("0080","D1","Caso 1","FN","justadiafragmática direita; dificuldade 1","ref",[1],"única","referência direita basal justadiafragmática (25 mm)","Não encontrada"),
 ("1008","D2","Caso 1","FN","micronódulo justafissural direito (provável linfonodo); dificuldade 1","ref",[3,6],"ambígua","duas referências direitas candidatas","Não encontrada"),
 ("1008","D2","Caso 2","FN","opacidade nodular no lobo médio, junto a atelectasia; dificuldade 8","ref",[3,6],"ambígua","duas referências direitas candidatas; marcador 4 coincide com a ref 3","Não encontrada"),
 ("1008","D2","Caso 3","FN","opacidade no LIE composta por vasos (o modelo marcou)","esf",[2],"única","único marcador esquerdo. DISCREPÂNCIA: classe original FN, mas o texto descreve objeto marcado pelo modelo; classe mantida","Não auditável"),
]
assert len(L) == 58
rows = []
for e, cj, reg, cl, desc, tipo, alvos, grau, just, _fig in L:
    if tipo == "ref":
        alvo, attr = "ref " + "/".join(map(str, alvos)), [at_ref(e, a) for a in alvos]
        if grau == "ambígua*":
            lado = ref.loc[(e, alvos[0])].lado_pulmao
            alt = [k for (x, k) in ref.index if x == e and k not in alvos and ref.loc[(x, k)].lado_pulmao == lado]
            alvo += " (alternativas: " + ("/".join(map(str, alt)) or "nenhuma") + ")"
            attr += [at_ref(e, a) for a in alt]
    elif tipo == "esf":
        alvo, attr = "marcador " + "/".join(map(str, alvos)), [at_esf(e, a) for a in alvos]
    else:
        alvo, attr = "--", []
    rows.append(dict(exame=e, conjunto=cj, registo=reg, classe=cl, descricao=desc, alvo=alvo,
                     atributos_alvo=" | ".join(attr), ligacao=grau, justificativa=just, confirmacao=""))
df = pd.DataFrame(rows)
df.insert(0, "n", range(1, 59))
df.to_csv(f"{O}/c07_confirmacao_cega.csv", index=False)
OBS = """
## Observações
- **Fonte dos 58 registos:** tabelas radiologist-1/2/3 do manuscrito original do artigo, conferidas com a planilha (24 TP + 25 FP + 9 FN). A ligação registo↔marcação foi feita na altura pelo analista Thiago e confirmada pela radiologista. Esta tabela só transfere esses registos, por coordenadas físicas e regras fixas, para os objetos da execução refeita.
- **Tabela cega:** sem desfechos do modelo, do verificador ou da análise pelas figuras. Estes só se juntam depois de fechada a coluna `confirmacao`.
- **ambígua\\*:** a ligação depende só da altura (lobo/segmento). Fica ambígua, a não ser que seja confirmada na imagem.
- **múltipla:** 1 registo ligado a vários objetos; resultados apresentados por objeto.
- **0925, Casos 1 e 2:** partilham os marcadores 5/6. O Caso 1 é a ligação; o Caso 2 é ambíguo, para que a mesma marcação não conte duas vezes.
- **1008, Caso 3:** a classe original FN foi mantida. Discrepância: o texto descreve um objeto marcado pelo modelo.
- **0061, Nódulos 7 e 8:** nas anotações por leitor (per_reader) há 6 agrupamentos, todos sobrepostos aos 6 objetos de consenso. O único à direita é o da ref 2, anotado por 1 dos 4 leitores. Não existe anotação LIDC no LID nem no lobo médio, portanto a ausência não vem de uma mudança do consenso. O exame tem 8 registos TP para 6 anotações LIDC.
- **0061, esferas (11, incluídas):** 8 à direita e 3 à esquerda. As esquerdas (m7, m10, m11) ficam a ≥24,9 mm de qualquer referência; m10 tem HU −994 (ar), e m11 tem HU +559 e altura 1,01, acima do ápice pulmonar. As esferas m1, m8 e m11 têm 2, 2 e 4 vezes o volume padrão (esferas sobrepostas). O desacordo entre a planilha e as marcações fica registado como observação, não como motivo de exclusão.
"""
with open(f"{O}/c07_confirmacao_cega.md", "w") as fh:
    fh.write("# Confirmação cega da transferência dos 58 registos radiológicos\n\n"
             "Lado: pulmão mais próximo; alt: 0 = base, 1 = ápice; pleura e plano médio em mm. "
             "Preencher `confirmacao` com ok ou com a correção.\n\n")
    cols = [c for c in df.columns if c != "confirmacao"]
    fh.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
    for r in df[cols].itertuples(index=False):
        fh.write("| " + " | ".join(str(v) for v in r) + " |\n")
    fh.write(OBS)
print(pd.crosstab(df.classe, df.ligacao, margins=True).to_string())
