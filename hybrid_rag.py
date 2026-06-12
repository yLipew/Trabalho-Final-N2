"""
Hybrid RAG — Implementação com Qdrant
Grupo: Lauro Lobo, Gustavo de Carvalho, Felipe Mendonça
SENAI Goiás — Trabalho Final N2

Arquitetura: Hybrid RAG
Combina busca vetorial (semântica) + busca por palavras-chave (BM25/sparse)
para recuperar os chunks mais relevantes antes de gerar a resposta com a LLM.
"""

import os
import uuid
from typing import Optional
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    SparseVector,
    NamedVector,
    NamedSparseVector,
    SearchRequest,
    Prefetch,
    FusionQuery,
    Fusion,
)
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import httpx
import json

# ─────────────────────────────────────────────
# Configurações
# ─────────────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
MARITACA_API_KEY = "106731539683870700861_3120f713d0518a3e"
MARITACA_API_URL = "https://chat.maritaca.ai/api/chat/completions"
COLLECTION_NAME = "hybrid_rag_docs"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 5


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    chunk_index: int


# ─────────────────────────────────────────────
# Base de conhecimento (artigos sobre IA/LLMs)
# ─────────────────────────────────────────────
KNOWLEDGE_BASE = [
    {
        "source": "intro_llm.txt",
        "text": (
            "Grandes Modelos de Linguagem (LLMs) são redes neurais treinadas em vastos "
            "corpora de texto. Eles aprendem padrões estatísticos da linguagem e são capazes "
            "de gerar texto coerente, responder perguntas e realizar tradução automática. "
            "Exemplos populares incluem GPT-4, Claude e Gemini."
        ),
    },
    {
        "source": "intro_llm.txt",
        "text": (
            "O processo de treinamento de um LLM envolve pré-treinamento em grandes "
            "quantidades de dados não rotulados, seguido de fine-tuning supervisionado "
            "e alinhamento via RLHF (Reinforcement Learning from Human Feedback). "
            "Esse processo garante que o modelo seja útil e seguro."
        ),
    },
    {
        "source": "rag_overview.txt",
        "text": (
            "RAG (Retrieval-Augmented Generation) é uma arquitetura que combina recuperação "
            "de informação com geração de texto. Em vez de depender apenas do conhecimento "
            "memorizado durante o treinamento, o modelo busca documentos relevantes em uma "
            "base de conhecimento externa e usa essas informações para gerar respostas mais "
            "precisas e atualizadas."
        ),
    },
    {
        "source": "rag_overview.txt",
        "text": (
            "O RAG básico (naive RAG) segue três etapas: (1) indexação dos documentos em "
            "embeddings vetoriais; (2) recuperação dos chunks mais similares à pergunta; "
            "(3) geração da resposta condicionada ao contexto recuperado. Sua principal "
            "limitação é depender apenas de similaridade semântica, o que pode falhar "
            "quando a pergunta contém termos técnicos específicos."
        ),
    },
    {
        "source": "hybrid_rag.txt",
        "text": (
            "O Hybrid RAG combina busca densa (vetorial/semântica) com busca esparsa "
            "(por palavras-chave, como BM25). A busca densa captura o significado geral "
            "da consulta, enquanto a busca esparsa garante que termos exatos e nomes "
            "técnicos sejam encontrados mesmo que o embedding não os capture bem."
        ),
    },
    {
        "source": "hybrid_rag.txt",
        "text": (
            "A fusão dos resultados no Hybrid RAG pode ser feita por Reciprocal Rank Fusion (RRF) "
            "ou por score relativo. O RRF combina os rankings das duas buscas atribuindo "
            "pontuações inversamente proporcionais à posição de cada documento. Isso torna "
            "o sistema mais robusto do que qualquer método isolado."
        ),
    },
    {
        "source": "hybrid_rag.txt",
        "text": (
            "O Qdrant suporta nativamente vetores esparsos e densos na mesma coleção. "
            "Isso permite executar buscas híbridas sem necessidade de múltiplos bancos de dados. "
            "A query de fusão com RRF pode ser configurada diretamente via API, tornando "
            "a implementação de Hybrid RAG simples e eficiente."
        ),
    },
    {
        "source": "embeddings.txt",
        "text": (
            "Embeddings são representações vetoriais de texto em espaço de alta dimensão. "
            "Textos semanticamente similares ficam próximos nesse espaço. Modelos como "
            "sentence-transformers geram embeddings de boa qualidade para português e inglês. "
            "O tamanho do vetor (dimensão) varia de 384 a 1536 dependendo do modelo."
        ),
    },
    {
        "source": "embeddings.txt",
        "text": (
            "BM25 é um algoritmo clássico de recuperação baseado em frequência de termos (TF) "
            "e frequência inversa de documentos (IDF). Ele pontua documentos baseando-se em "
            "quantas vezes os termos da query aparecem no documento, penalizando documentos "
            "muito longos. É altamente eficaz para termos técnicos e nomes próprios."
        ),
    },
    {
        "source": "vector_db.txt",
        "text": (
            "Bancos de dados vetoriais como Qdrant, Chroma e Pinecone são otimizados para "
            "armazenar e buscar vetores de alta dimensão com baixa latência. Eles usam "
            "índices como HNSW (Hierarchical Navigable Small World) para realizar buscas "
            "aproximadas de vizinhos mais próximos (ANN) de forma eficiente."
        ),
    },
    {
        "source": "vector_db.txt",
        "text": (
            "O Qdrant é um banco vetorial open-source escrito em Rust, com suporte a "
            "filtros, payload, vetores múltiplos por ponto e busca híbrida nativa. "
            "Pode ser usado localmente via Docker ou na nuvem. Sua API é compatível com "
            "Python, JavaScript e REST, facilitando a integração em qualquer stack."
        ),
    },
    {
        "source": "chunking.txt",
        "text": (
            "Chunking é o processo de dividir documentos em pedaços menores (chunks) "
            "antes de indexar. Chunks menores aumentam a precisão da recuperação, mas "
            "podem perder contexto. Chunks maiores preservam contexto, mas reduzem a "
            "precisão. O tamanho ideal depende do tipo de documento e do modelo de embedding."
        ),
    },
    {
        "source": "chunking.txt",
        "text": (
            "Estratégias comuns de chunking incluem: divisão por tamanho fixo com sobreposição, "
            "divisão por parágrafos ou seções, chunking semântico (baseado em coesão temática) "
            "e chunking hierárquico. O Hierarchical RAG usa múltiplos níveis de chunks, "
            "buscando primeiro em resumos e depois nos detalhes."
        ),
    },
]


# ─────────────────────────────────────────────
# Indexação
# ─────────────────────────────────────────────

class HybridRAGIndexer:
    def __init__(self):
        print("Carregando modelo de embeddings...")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.dense_dim = self.embedder.get_sentence_embedding_dimension()
        self.client = QdrantClient(url=QDRANT_URL)
        self.chunks: list[Chunk] = []
        self.bm25: Optional[BM25Okapi] = None

    def _prepare_chunks(self) -> list[Chunk]:
        chunks = []
        for doc in KNOWLEDGE_BASE:
            chunk = Chunk(
                id=str(uuid.uuid4()),
                text=doc["text"],
                source=doc["source"],
                chunk_index=len(chunks),
            )
            chunks.append(chunk)
        return chunks

    def _compute_sparse_vectors(self, texts: list[str]) -> list[dict[int, float]]:
        """Gera vetores esparsos BM25 para cada texto."""
        tokenized = [t.lower().split() for t in texts]
        bm25 = BM25Okapi(tokenized)
        self.bm25 = bm25

        sparse_vecs = []
        # Vocabulário: mapeando tokens para índices
        vocab = {}
        idx = 0
        for tokens in tokenized:
            for token in tokens:
                if token not in vocab:
                    vocab[token] = idx
                    idx += 1

        for i, tokens in enumerate(tokenized):
            scores = bm25.get_scores(tokens)
            # Pegamos apenas os tokens do próprio documento com score > 0
            indices = []
            values = []
            for token in set(tokens):
                if token in vocab:
                    score = float(bm25.idf.get(token, 0) * bm25.get_scores([token])[i])
                    if score > 0:
                        indices.append(vocab[token])
                        values.append(score)
            sparse_vecs.append({"indices": indices, "values": values})

        self.vocab = vocab
        return sparse_vecs

    def create_collection(self):
        """Cria a coleção no Qdrant com suporte a vetores densos e esparsos."""
        # Remove coleção existente se houver
        try:
            self.client.delete_collection(COLLECTION_NAME)
            print(f"Coleção '{COLLECTION_NAME}' removida.")
        except Exception:
            pass

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(size=self.dense_dim, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            },
        )
        print(f"Coleção '{COLLECTION_NAME}' criada.")

    def index(self):
        """Indexa todos os chunks no Qdrant."""
        self.chunks = self._prepare_chunks()
        texts = [c.text for c in self.chunks]

        print(f"Gerando embeddings densos para {len(texts)} chunks...")
        dense_vecs = self.embedder.encode(texts, show_progress_bar=True).tolist()

        print("Gerando vetores esparsos (BM25)...")
        sparse_vecs = self._compute_sparse_vectors(texts)

        self.create_collection()

        points = []
        for chunk, dense, sparse in zip(self.chunks, dense_vecs, sparse_vecs):
            points.append(
                PointStruct(
                    id=chunk.id,
                    vector={
                        "dense": dense,
                        "sparse": SparseVector(
                            indices=sparse["indices"],
                            values=sparse["values"],
                        ),
                    },
                    payload={
                        "text": chunk.text,
                        "source": chunk.source,
                        "chunk_index": chunk.chunk_index,
                    },
                )
            )

        self.client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"{len(points)} chunks indexados com sucesso!")
        return self


# ─────────────────────────────────────────────
# Recuperação híbrida
# ─────────────────────────────────────────────

class HybridRAGRetriever:
    def __init__(self, indexer: HybridRAGIndexer):
        self.client = indexer.client
        self.embedder = indexer.embedder
        self.vocab = indexer.vocab
        self.bm25 = indexer.bm25

    def _query_sparse(self, query: str) -> SparseVector:
        """Gera vetor esparso para a query usando o vocabulário do índice."""
        tokens = query.lower().split()
        indices = []
        values = []
        for token in set(tokens):
            if token in self.vocab:
                idx = self.vocab[token]
                # IDF score simples como peso
                score = self.bm25.idf.get(token, 0.1)
                if score > 0:
                    indices.append(idx)
                    values.append(float(score))
        return SparseVector(indices=indices, values=values)

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """
        Busca híbrida: combina busca densa (semântica) + esparsa (BM25)
        usando Reciprocal Rank Fusion (RRF) do Qdrant.
        """
        dense_vec = self.embedder.encode(query).tolist()
        sparse_vec = self._query_sparse(query)

        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                Prefetch(
                    query=dense_vec,
                    using="dense",
                    limit=top_k * 2,
                ),
                Prefetch(
                    query=sparse_vec,
                    using="sparse",
                    limit=top_k * 2,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        chunks = []
        for r in results.points:
            chunks.append({
                "text": r.payload["text"],
                "source": r.payload["source"],
                "score": r.score,
            })
        return chunks

# ─────────────────────────────────────────────
# Geração com Maritaca
# ─────────────────────────────────────────────


def generate_answer(query: str, context_chunks: list[dict]) -> str:
    """Envia o contexto recuperado + a pergunta para a LLM Maritaca."""
    context = "\n\n".join(
        f"[Fonte: {c['source']}]\n{c['text']}" for c in context_chunks
    )

    system_prompt = (
        "Você é um assistente especialista em IA. "
        "Responda à pergunta do usuário usando APENAS as informações do contexto fornecido. "
        "Se a resposta não estiver no contexto, diga que não encontrou informações suficientes. "
        "Seja objetivo e claro."
    )

    user_message = f"""Contexto:
{context}

Pergunta: {query}

Resposta:"""

    payload = {
        "model": "sabia-4",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": 512,
        "temperature": 0.2,
    }

    # Alterado de "Key" para "Bearer" para alinhar com o endpoint 'chat/completions' da Maritaca
    headers = {
        "Authorization": f"Bearer {MARITACA_API_KEY}",
        "Content-Type": "application/json",
    }

    response = httpx.post(
        MARITACA_API_URL, json=payload, headers=headers, timeout=30
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


# ─────────────────────────────────────────────
# Pipeline completo
# ─────────────────────────────────────────────

class HybridRAGPipeline:
    def __init__(self):
        self.indexer = HybridRAGIndexer()
        self.retriever: Optional[HybridRAGRetriever] = None

    def build_index(self):
        self.indexer.index()
        self.retriever = HybridRAGRetriever(self.indexer)
        return self

    def ask(self, query: str, verbose: bool = True) -> str:
        assert self.retriever, "Execute build_index() antes de fazer perguntas."

        if verbose:
            print(f"\n{'='*60}")
            print(f"Pergunta: {query}")
            print(f"{'='*60}")

        chunks = self.retriever.retrieve(query)

        if verbose:
            print(f"\n📚 Chunks recuperados ({len(chunks)}):")
            for i, c in enumerate(chunks, 1):
                print(f"  [{i}] Score: {c['score']:.4f} | Fonte: {c['source']}")
                print(f"       {c['text'][:5000]}...")

        answer = generate_answer(query, chunks)

        if verbose:
            print(f"\n🤖 Resposta:\n{answer}")

        return answer


# ─────────────────────────────────────────────
# Ponto de entrada
# ─────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = HybridRAGPipeline()
    pipeline.build_index()

    perguntas = [
        "O que é RAG e como ele funciona?",
        "Qual a diferença entre busca vetorial e BM25?",
        "Como o Qdrant suporta busca híbrida?",
        "O que é RRF e para que serve no Hybrid RAG?",
        "Quais são as limitações do RAG básico?",
    ]

    for pergunta in perguntas:
        resposta = pipeline.ask(pergunta)
        print()
