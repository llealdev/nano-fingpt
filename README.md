# nano-fingpt

Reimplementação enxuta ("nano") da camada de **retrieval** proposta pelo paper [**FinGPT: Open-Source Financial Large Language Models** (Yang et al., 2023)](https://arxiv.org/abs/2306.06031), focada em estudar a fundo a ciência por trás da **busca vetorial moderna** aplicada a documentos financeiros.

O FinGPT original descreve um framework end-to-end de cinco camadas (Data Source → Data Engineering → LLMs → Tasks → Applications) que cobre desde a coleta de notícias, social media e filings até fine-tuning com LoRA e RLSP. Este projeto isola e aprofunda uma fatia específica desse stack:

- **Data Source**: apenas *Company Filings* (SEC EDGAR — 10-K e 10-Q).
- **Data Engineering**: chunking semântico via clustering, vector embedding multi-modal.
- **LLMs Layer**: exclusivamente o componente de **Retrieval-Augmented Generation** (seção 3.3.3 do paper).

A motivação é simples: para que um FinLLM produza respostas confiáveis sobre documentos da SEC, a etapa de retrieval precisa ser cirúrgica. O paper trata RAG em alto nível; este repositório implementa essa camada com técnicas estado-da-arte e mostra, na prática, como cada componente impacta a qualidade do contexto entregue ao LLM.

---

## Motivação

A maioria dos tutoriais de RAG resolve o problema com `text_splitter.split_text() + embed() + similarity_search()` e para por aí. Essa abordagem ignora os principais fatores que determinam a qualidade real de um sistema de retrieval em produção:

- **Como** os chunks são formados (boundary semântica vs. recorte arbitrário).
- **O que** é vetorizado e **como** é vetorizado (dense, sparse, multi-vector).
- **Como** os scores de diferentes recuperadores são combinados.
- **Como** se faz o re-rank fino para maximizar precisão no top-K.

No contexto financeiro essa exigência é amplificada pelas características apontadas pelo próprio paper FinGPT: **granularidade** dos filings, **baixa razão sinal-ruído**, **terminologia técnica densa** e **alto custo de erro factual**. Este repositório explora cada uma dessas dimensões aplicadas aos itens mais carregados de sinal financeiro do 10-K e 10-Q.

---

## Arquitetura

```
                 ┌──────────────────────┐
                 │   SEC EDGAR API      │
                 │   (10-K / 10-Q)      │
                 └──────────┬───────────┘
                            │  EdgarClient
                            ▼
                 ┌──────────────────────┐
                 │  Semantic Chunker    │
                 │  (HDBSCAN clustering)│
                 └──────────┬───────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
         ┌─────────┐  ┌──────────┐  ┌──────────┐
         │  Dense  │  │  Sparse  │  │ ColBERT  │
         │ MiniLM  │  │  BM25    │  │  v2.0    │
         └────┬────┘  └─────┬────┘  └─────┬────┘
              │             │             │
              └──────┬──────┘             │
                     ▼                    │
            ┌────────────────┐            │
            │  RRF Fusion    │ top-20     │
            │ (dense+sparse) │────────────┤
            └────────────────┘            │
                                          ▼
                              ┌────────────────────────┐
                              │ ColBERT Re-rank (MaxSim)│
                              │     top-K final        │
                              └────────────────────────┘
```

Todo o índice é mantido no **Qdrant** usando *named vectors* e *multi-vector configuration* (MAX_SIM comparator) para suportar dense, sparse e multi-vector em uma única collection.

---

## Stack Técnica

| Camada              | Tecnologia                                                |
| ------------------- | --------------------------------------------------------- |
| Linguagem           | Python 3.13                                               |
| Gerenciador         | `uv`                                                      |
| Vector DB           | Qdrant (cloud)                                            |
| Dense embeddings    | `sentence-transformers/all-MiniLM-L6-v2` (384 dim)        |
| Sparse embeddings   | `Qdrant/bm25`                                             |
| Late interaction    | `colbert-ir/colbertv2.0` (128 dim, multi-vector)          |
| Clustering          | `hdbscan`                                                 |
| Tokenização         | `transformers` (`AutoTokenizer`)                          |
| Ingestão de filings | `edgartools`                                              |
| Inferência          | `fastembed` (ONNX runtime)                                |

---

## A Ciência Por Trás do Projeto

### 1. Semantic Chunking via HDBSCAN

A primeira etapa de qualquer RAG é fatiar o documento. Recortes por tamanho fixo (`chunk_size=500`) quebram ideias no meio e degradam o sinal semântico do embedding. A solução adotada:

1. Quebra o texto em parágrafos significativos (≥ 10 palavras).
2. Embeda cada parágrafo com MiniLM.
3. Roda **HDBSCAN** sobre os vetores para agrupar parágrafos por similaridade semântica (densidade no espaço vetorial, sem precisar definir `k`).
4. Concatena parágrafos do mesmo cluster até atingir um orçamento de tokens (`max_tokens=300`).
5. Parágrafos *orphan* (ruído, label `-1`) passam por uma segunda rodada de clustering com `min_cluster_size` mais permissivo; o que sobra vira chunk individual.

Resultado: chunks **temáticos**, não posicionais, com fronteiras que respeitam a estrutura argumentativa do documento.

Implementação: `utils/semantic_chunker.py`

### 2. Dense Retrieval (Bi-Encoder)

Embeddings densos capturam **similaridade semântica** — sinônimos, paráfrases, contexto. MiniLM foi escolhido por equilíbrio entre custo (384 dim, ONNX via fastembed) e qualidade. Distância: cosseno.

Limitação: sofre com **out-of-vocabulary** e termos raros/numéricos comuns em documentos financeiros (CIKs, tickers, valores).

### 3. Sparse Retrieval (BM25)

Para cobrir o ponto cego do dense, adiciona-se **BM25** como vetor esparso nativo do Qdrant. BM25 captura **correspondência lexical exata**, essencial para queries que mencionam termos específicos do domínio.

### 4. Hybrid Search com Reciprocal Rank Fusion (RRF)

Em vez de tentar combinar scores heterogêneos (cosseno do dense vs. BM25 do sparse — escalas completamente diferentes), usa-se **RRF**:

```
RRF(d) = Σ 1 / (k + rank_i(d))
```

A fusão é feita pelo Qdrant via `FusionQuery(Fusion.RRF)`, combinando as listas top-10 de cada recuperador em um único top-20. Essa abordagem é robusta porque depende apenas da **ordem** dos resultados, não da magnitude dos scores.

### 5. Re-ranking com ColBERT v2 (Late Interaction)

Bi-encoders são rápidos mas perdem granularidade ao colapsar a query inteira em um único vetor. Cross-encoders são precisos mas inviáveis em escala. **ColBERT** entrega o melhor dos dois mundos:

- Cada token da query e do documento vira um vetor (multi-vector).
- A pontuação é calculada via **MaxSim**: para cada token da query, pega o maior cosseno contra todos os tokens do documento; somam-se os máximos.

No pipeline, ColBERT atua como **estágio de re-rank**: recebe os 20 candidatos do RRF e devolve o top-3 final. No Qdrant isso é configurado com `MultiVectorConfig(comparator=MAX_SIM)`.

### 6. Normalização de Scores

Os scores do ColBERT são somas de MaxSims (não estão em `[0,1]`). Para apresentação ao usuário/LLM, divide-se pelo score máximo do batch, gerando uma escala relativa interpretable.

---

## Evolução do Projeto (commit por commit)

O repositório foi construído de forma incremental, cada commit adicionando uma camada de sofisticação:

| Commit     | Etapa                                                                              |
| ---------- | ---------------------------------------------------------------------------------- |
| `acdbf47`  | Setup inicial do projeto com `uv` e Python 3.13.                                   |
| `10ea7e0`  | **Baseline**: chunking por `split("\n\n")`, dense embedding e busca cosseno pura.  |
| `1d3e445`  | **Busca híbrida**: adiciona sparse BM25 + RRF fusion.                              |
| `a821bf4`  | **Re-rank**: integra ColBERT v2 como estágio final (late interaction).             |
| `de4f450`  | **Semantic chunking**: substitui split ingênuo por clustering HDBSCAN.             |
| `d745fd2`  | Refator do chunker: extrai pipeline de clustering em método reutilizável, segunda passada para orphans. |
| `c76062a`  | **`EdgarClient`**: abstração sobre `edgartools` para ingerir 10-K e 10-Q direto da SEC com metadados. |
| `8df8627`  | Separa script de query (`test-query.py`) do pipeline de ingestão.                  |
| `5978a8e`  | Separa criação da collection (`create_collection.py`) — idempotência operacional. |
| `863fda8`  | Refator final do `ingestion.py` consumindo `EdgarClient` + `SemanticChunker`.      |

A separação em três scripts (`create_collection.py` / `ingestion.py` / `test-query.py`) reflete o ciclo de vida real de um índice vetorial: **schema → ingest → query**.

---

## Estrutura

```
nano-fingpt/
├── create_collection.py     # cria a collection no Qdrant com os 3 named vectors
├── ingestion.py             # baixa filings, chunka, embeda e indexa
├── test-query.py            # roda query híbrida com re-rank
├── utils/
│   ├── edgar_clinet.py      # wrapper sobre edgartools (10-K, 10-Q)
│   └── semantic_chunker.py  # chunker HDBSCAN + budget de tokens
├── pyproject.toml
└── .env                     # QDRANT_URL, QDRANT_API_KEY
```

---

## Como rodar

Pré-requisitos: Python 3.13, `uv` instalado, uma instância Qdrant (local ou cloud).

```bash
uv sync
cp .env.example .env   # preencha QDRANT_URL e QDRANT_API_KEY

uv run create_collection.py
uv run ingestion.py
uv run test-query.py
```

Saída típica de `test-query.py`:

```
Score: 1.0
Texto: The Company's business, reputation, results of operations, financial condition...
--------------------------------------------------------------------------------
Score: 0.94
Texto: Macroeconomic conditions, including inflation, changes in interest rates...
--------------------------------------------------------------------------------
```

---

## O que este projeto demonstra

- Capacidade de **ler um paper** (FinGPT, arXiv:2306.06031), isolar uma camada (RAG) e implementá-la com profundidade.
- Domínio prático de **information retrieval moderno**: dense, sparse, hybrid, late interaction.
- Capacidade de **modelar problemas de IA** indo além de chamadas de API — entendendo trade-offs de cada componente.
- Engenharia: separação de responsabilidades, abstrações finas (`EdgarClient`, `SemanticChunker`), uso idiomático do Qdrant (named vectors, multi-vector, fusion queries).
- **Domain awareness**: escolha consciente de itens específicos do 10-K/10-Q (`1`, `1A`, `7`, `8`, `9A`) que concentram o sinal financeiro relevante.

---

## Próximos passos (rumo ao FinGPT completo)

O escopo atual cobre Data Source (filings) + Data Engineering + camada de RAG. Para fechar o loop com o framework original:

- **Tasks Layer**: avaliação quantitativa com Q&A rotulado sobre filings (Recall@K, MRR, nDCG) e tarefas fundamentais do paper (sumarização, NER, numerical reasoning).
- **LLMs Layer (geração)**: acoplar um LLM consumindo o top-K do re-rank para respostas grounded.
- **Data Source Layer (expansão)**: adicionar notícias financeiras e social media via FinNLP, como no paper.
- **Fine-tuning**: LoRA/QLoRA sobre o LLM gerador usando pares (query, contexto recuperado, resposta).
- **Operacional**: atualização incremental de filings sem reindexar, quantização dos vetores ColBERT.

---

## Referência

```bibtex
@article{yang2023fingpt,
  title={FinGPT: Open-Source Financial Large Language Models},
  author={Yang, Hongyang and Liu, Xiao-Yang and Wang, Christina Dan},
  journal={arXiv preprint arXiv:2306.06031},
  year={2023}
}
```
