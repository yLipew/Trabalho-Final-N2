![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-%23C52B24.svg?style=for-the-badge&logo=qdrant&logoColor=white)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue?style=for-the-badge)

# 🔍 Hybrid RAG — Implementação com Qdrant

**Trabalho Final N2 — Implementação de Arquitetura RAG**  
**SENAI Goiás**

| Integrantes |
|---|---|
| Lauro Lobo
| Gustavo de Carvalho
| Felipe Mendonça

---

## 📌 Tipo de RAG: Hybrid RAG

Combinação de busca **vetorial (semântica)** + busca por **palavras-chave (BM25/sparse)** com fusão dos resultados via **Reciprocal Rank Fusion (RRF)**.

---

## 🧠 O que é Hybrid RAG?

O RAG básico (naive RAG) recupera documentos usando apenas **similaridade semântica** (embeddings). Isso funciona bem para consultas com linguagem natural, mas falha em casos como:

- Termos técnicos específicos que o modelo de embedding "diluiu" no espaço vetorial
- Nomes próprios, siglas ou identificadores exatos
- Queries curtas ou ambíguas

O **Hybrid RAG** resolve isso combinando **duas estratégias complementares**:

| Método | O que captura | Limitação |
|---|---|---|
| Busca densa (embedding) | Significado semântico | Pode perder termos exatos |
| Busca esparsa (BM25) | Termos exatos e frequência | Não entende sinônimos |
| **Hybrid (RRF)** | **Ambos** | Mais robusto que qualquer um isolado |

### Como funciona o pipeline

```
Pergunta do usuário
       │
       ├──► Embedding (vetor denso) ──► Busca semântica no Qdrant ──► Top-K docs
       │
       └──► Tokenização BM25 (vetor esparso) ──► Busca por palavras no Qdrant ──► Top-K docs
                                                            │
                                               Fusão via RRF (Reciprocal Rank Fusion)
                                                            │
                                                    Top-K docs combinados
                                                            │
                                               Contexto enviado para a LLM
                                                            │
                                                     Resposta gerada
```

---

## 🗄️ Banco Vetorial: Qdrant

O **Qdrant** foi escolhido por:
- Suporte nativo a **vetores densos e esparsos na mesma coleção**
- API de **fusão RRF integrada** (`FusionQuery`)
- Fácil execução local via Docker
- Alta performance com índice HNSW
- SDK Python completo e bem documentado

### Estrutura da coleção

```
Collection: hybrid_rag_docs
├── Vector "dense"  → paraphrase-multilingual-MiniLM-L12-v2 (384 dims, cosine)
└── Vector "sparse" → BM25 (vetor esparso com índices e valores de TF-IDF)
Payload por ponto:
  ├── text         → conteúdo do chunk
  ├── source       → arquivo de origem
  └── chunk_index  → posição no documento original
```

---

## 📚 Base de Dados

Artigos sobre IA e LLMs criados para o projeto, cobrindo:

| Arquivo | Conteúdo |
|---|---|
| `intro_llm.txt` | O que são LLMs, treinamento, RLHF |
| `rag_overview.txt` | RAG básico: conceito, etapas, limitações |
| `hybrid_rag.txt` | Hybrid RAG, RRF, suporte no Qdrant |
| `embeddings.txt` | Embeddings e algoritmo BM25 |
| `vector_db.txt` | Bancos vetoriais, HNSW, Qdrant |
| `chunking.txt` | Estratégias de chunking |

**Total:** 12 chunks indexados.

---

## ⚙️ Como executar

### 1. Pré-requisitos

- Python 3.10+
- Docker

### 2. Subir o Qdrant localmente

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Configurar a chave da Maritaca

```bash
export MARITACA_API_KEY="sua_chave_aqui"
```

### 5. Executar

```bash
python hybrid_rag.py
```

O script vai:
1. Carregar o modelo de embeddings
2. Indexar os 12 chunks no Qdrant (vetores densos + esparsos)
3. Executar 5 perguntas de exemplo mostrando os chunks recuperados e a resposta

---

## 💬 Exemplos de Perguntas e Respostas

### Pergunta 1
**"O que é RAG e como ele funciona?"**

*Chunks recuperados:* `rag_overview.txt` (x2), `hybrid_rag.txt`

*Resposta:*
> RAG (Retrieval-Augmented Generation) combina recuperação de informação com geração de texto. O modelo busca documentos relevantes em uma base externa e usa essas informações para gerar respostas mais precisas e atualizadas. O pipeline envolve três etapas: indexação em embeddings, recuperação dos chunks mais similares e geração condicionada ao contexto.

---

### Pergunta 2
**"Qual a diferença entre busca vetorial e BM25?"**

*Chunks recuperados:* `embeddings.txt` (x2), `hybrid_rag.txt`

*Resposta:*
> A busca vetorial usa embeddings para capturar o significado semântico do texto, aproximando textos similares no espaço vetorial. Já o BM25 é um algoritmo clássico baseado em frequência de termos (TF-IDF), que pontua documentos pela presença e frequência dos termos da query. O BM25 é mais eficaz para termos técnicos e nomes próprios, enquanto embeddings capturam sinônimos e conceitos relacionados.

---

### Pergunta 3
**"Como o Qdrant suporta busca híbrida?"**

*Chunks recuperados:* `hybrid_rag.txt` (x2), `vector_db.txt`

*Resposta:*
> O Qdrant suporta nativamente vetores esparsos e densos na mesma coleção, permitindo buscas híbridas sem múltiplos bancos de dados. A query de fusão com RRF é configurada diretamente via API, tornando a implementação simples e eficiente.

---

### Pergunta 4
**"O que é RRF e para que serve no Hybrid RAG?"**

*Chunks recuperados:* `hybrid_rag.txt` (x2), `rag_overview.txt`

*Resposta:*
> RRF (Reciprocal Rank Fusion) é um método de fusão que combina os rankings da busca densa e da busca esparsa. Ele atribui pontuações inversamente proporcionais à posição de cada documento nos dois rankings, tornando o sistema mais robusto que qualquer método isolado.

---

### Pergunta 5
**"Quais são as limitações do RAG básico?"**

*Chunks recuperados:* `rag_overview.txt`, `hybrid_rag.txt`, `embeddings.txt`

*Resposta:*
> O RAG básico depende apenas de similaridade semântica via embeddings, o que pode falhar quando a pergunta contém termos técnicos específicos, nomes próprios ou siglas que o embedding não captura bem. O Hybrid RAG resolve esse problema combinando busca semântica com busca por palavras-chave.

---

## 📊 Principais Resultados

- A busca híbrida recuperou chunks mais relevantes do que a busca vetorial isolada em perguntas com termos técnicos como "RRF", "BM25" e "HNSW"
- O modelo `paraphrase-multilingual-MiniLM-L12-v2` gerou embeddings de boa qualidade para português
- A fusão RRF do Qdrant funcionou sem necessidade de código extra de reranking
- Tempo médio de indexação: ~3s para 12 chunks | Tempo médio de query: ~200ms

---

## ⚠️ Limitações

- **Base pequena**: 12 chunks é insuficiente para produção; em produção, milhares de documentos são esperados
- **BM25 in-memory**: O vocabulário BM25 é construído na memória e precisa ser reconstruído ao reiniciar; em produção, usar Qdrant sparse vectors com modelo SPLADE seria melhor
- **Sem reranking explícito**: O RRF é um reranker implícito, mas um reranker neural (ex: cross-encoder) melhoraria ainda mais a precisão
- **Chunking fixo**: Não há sobreposição entre chunks; em documentos reais, overlap de ~20% é recomendado
- **LLM externa**: Depende da API Maritaca; em produção, seria ideal ter fallback

---

## 💡 Aprendizados

- A principal vantagem do Hybrid RAG é a **robustez**: nenhum método isolado é ótimo para todos os tipos de query
- O Qdrant simplifica muito a implementação ao unificar vetores densos e esparsos na mesma API
- A qualidade do chunking impacta diretamente a qualidade da recuperação — é tão importante quanto o modelo de embedding
- O Hybrid RAG é especialmente útil em domínios técnicos com vocabulário específico (medicina, direito, engenharia)
- Em produção, adicionar reranking neural sobre os resultados do Hybrid RAG (RAG com reranking) seria o próximo passo natural

---

## 🏭 Impacto em Cenário Real de Mercado

O Hybrid RAG seria ideal para:
- **Suporte técnico**: Documentações extensas com termos técnicos exatos
- **Jurídico/Compliance**: Busca precisa por artigos, incisos e parágrafos
- **E-commerce**: Catálogos com códigos de produto, SKUs e especificações
- **Healthcare**: Prontuários com CIDs, medicamentos e procedimentos

Em todos esses casos, depender apenas de embeddings resultaria em buscas imprecisas para termos específicos.

---

## 🛠️ Tecnologias

| Tecnologia | Versão | Uso |
|---|---|---|
| Python | 3.10+ | Linguagem principal |
| Qdrant | 1.9+ | Banco vetorial (denso + esparso) |
| sentence-transformers | 2.7+ | Geração de embeddings |
| rank-bm25 | 0.2+ | Vetores esparsos BM25 |
| httpx | 0.27+ | Chamadas à API Maritaca |
| Maritaca (Sabiá-3) | — | Geração de respostas |
