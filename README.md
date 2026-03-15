# Análise de Risco de Defasagem Educacional

### Datathon -- Fase 5 \| Pós‑Tech Data Analytics (FIAP)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red) ![Machine
Learning](https://img.shields.io/badge/Machine%20Learning-MLPClassifier-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

------------------------------------------------------------------------

## Aplicação Preditiva

A solução final do projeto foi disponibilizada como uma aplicação
interativa em Streamlit.

Acesse a aplicação:

https://fase5datathon.streamlit.app/

A ferramenta permite inserir indicadores educacionais de um aluno e
calcular a **probabilidade de risco de defasagem educacional**.

------------------------------------------------------------------------

## Visão Geral do Projeto

Este projeto foi desenvolvido no **Datathon -- Fase 5 da Pós‑Tech em
Data Analytics da FIAP** utilizando dados educacionais da **Associação
Passos Mágicos**.

O objetivo é aplicar **análise exploratória de dados e Machine
Learning** para:

-   identificar padrões educacionais
-   entender fatores associados à defasagem
-   prever risco educacional antes da queda de desempenho

A solução proposta combina **análise estatística, modelagem preditiva e
aplicação prática de dados**.

------------------------------------------------------------------------

## Problema de Negócio

A Associação Passos Mágicos acompanha o desenvolvimento educacional de
alunos em situação de vulnerabilidade social por meio de diversos
indicadores.

A partir desses dados, o desafio do Datathon foi responder perguntas
estratégicas como:

-   Qual é o perfil de defasagem educacional dos alunos?
-   O desempenho acadêmico melhora ao longo do tempo?
-   O engajamento influencia diretamente o desempenho escolar?
-   Indicadores psicossociais antecipam quedas de desempenho?
-   É possível prever risco de defasagem antes que ela ocorra?

------------------------------------------------------------------------

## Indicadores Educacionais Utilizados

O dataset inclui diversos indicadores educacionais e psicossociais:

  Indicador   Descrição
  ----------- --------------------------------
  IAN         Índice de Adequação de Nível
  IDA         Índice de Desempenho Acadêmico
  IEG         Índice de Engajamento
  IAA         Índice de Autoavaliação
  IPS         Índice Psicossocial
  IPP         Índice Psicopedagógico
  IPV         Índice de Ponto de Virada
  INDE        Índice Global do Aluno

Esses indicadores permitem analisar o desenvolvimento educacional de
forma **multidimensional**.

------------------------------------------------------------------------

## Metodologia

O projeto foi desenvolvido em etapas:

### 1️⃣ Análise Exploratória de Dados

Foram analisados padrões de desempenho educacional considerando:

-   evolução temporal dos indicadores
-   relação entre engajamento e desempenho
-   influência de fatores psicossociais
-   coerência entre autoavaliação e desempenho real

Também foi investigada a evolução educacional nas fases pedagógicas:

-   Quartzo
-   Ágata
-   Ametista
-   Topázio

------------------------------------------------------------------------

### 2️⃣ Modelagem Preditiva

Foi desenvolvido um modelo de **Machine Learning baseado em rede neural
artificial (MLPClassifier)**.

Configuração do modelo:

    MLPClassifier(
        hidden_layer_sizes=(100,50),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    )

Pipeline de modelagem:

1.  Limpeza e tratamento dos dados
2.  Feature engineering
3.  Criação da variável target de risco
4.  Separação dos dados em treino e teste
5.  Padronização com StandardScaler
6.  Treinamento do modelo
7.  Avaliação com métricas de classificação
8.  Geração de probabilidades de risco

O modelo treinado é salvo e utilizado pela aplicação Streamlit.

------------------------------------------------------------------------

## Estrutura do Projeto
```
  ├── data/
  │ └── raw/
  │ └── BASE_DE_DADOS_PEDE_2024_DATATHON.xlsx
  ├── docs/
  │ └── documeto complemtar/
  │ └── Dicionário Dados Datathon.pdf
  │
  ├── models/
  │ └── modelo_risco_defasagem_mlp.joblib
  │
  ├── notebooks/
  │ └── analise_risco_defasagem.ipynb
  │
  ├── presentation/
  │ └── apresentação.pptx
  │
  ├── video/
  │ └── link_da_apresntação_gravada.txt
  │
  ├── app.py
  ├── requirements.txt
  └── README.md
```

------------------------------------------------------------------------

## Resultados

A análise identificou padrões relevantes:

-   baixo engajamento está associado a queda de desempenho acadêmico
-   fatores psicossociais podem antecipar dificuldades educacionais
-   a combinação de múltiplos indicadores melhora a previsão de risco

O modelo desenvolvido permite **antecipar possíveis casos de
defasagem**, possibilitando intervenções educacionais mais eficazes.

------------------------------------------------------------------------

## Equipe

Projeto desenvolvido por:

-   Fabiana Sampaio Luz (RM362180)
-   Giovanna Salgado Stancati de Carvalho (RM361545)
-   Juliana Albuquerque Vitoriano (RM362853)
-   Lais Stefanie Dias Pereira (RM363219)

Pós‑Tech em Data Analytics -- FIAP

------------------------------------------------------------------------

## Conclusão

Este projeto demonstra como **análise de dados e Machine Learning podem
apoiar organizações educacionais na identificação precoce de riscos
acadêmicos**, permitindo intervenções pedagógicas mais eficazes.

A aplicação desenvolvida transforma análise estatística em **ferramenta
prática de apoio à tomada de decisão educacional**.

------------------------------------------------------------------------

## Licença

Projeto distribuído sob licença MIT.
