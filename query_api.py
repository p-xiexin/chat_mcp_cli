from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
from langchain_milvus import Milvus
from pymilvus import utility

from lib.db.base import get_db
from lib.db.kb import get_kb_by_project_id
from lib.db.chunks import get_chunks_by_project_id
from utils import logger
from config import settings

def normalize_scores(docs):
    """将文档列表的 score 归一化到 [0,1]"""
    if not docs:
        return docs
    scores = [doc.metadata.get("score", 1.0) for doc in docs]
    max_score = max(scores)
    min_score = min(scores)
    if max_score == min_score:
        for doc in docs:
            doc.metadata["score_norm"] = 1.0
    else:
        for doc, score in zip(docs, scores):
            doc.metadata["score_norm"] = (score - min_score) / (max_score - min_score)
    return docs


def weighted_fusion(dense_docs, sparse_docs, dense_weight=0.5, sparse_weight=0.5, top_k=5):
    """基于归一化 score 的加权融合（企业级推荐方案）"""
    dense_docs = normalize_scores(dense_docs)
    sparse_docs = normalize_scores(sparse_docs)

    doc_map = {}
    for doc in dense_docs:
        doc_id = str(doc.metadata.get("id") or id(doc))
        doc_map[doc_id] = doc
        doc.metadata["final_score"] = doc.metadata["score_norm"] * dense_weight

    for doc in sparse_docs:
        doc_id = str(doc.metadata.get("id") or id(doc))
        if doc_id in doc_map:
            doc_map[doc_id].metadata["final_score"] += doc.metadata["score_norm"] * sparse_weight
        else:
            doc.metadata["final_score"] = doc.metadata["score_norm"] * sparse_weight
            doc_map[doc_id] = doc

    sorted_docs = sorted(doc_map.values(), key=lambda x: x.metadata["final_score"], reverse=True)
    return sorted_docs[:top_k]

def get_vector_store(project_id: str, embeddings: OpenAIEmbeddings):
    """获取项目对应的向量检索器（仅查询用）"""
    collection_name = f"proj_{project_id.replace('-', '_')}"

    try:
        exists = utility.has_collection(collection_name)
        if not exists:
            print(f"⚠️ 向量集合 '{collection_name}' 不存在，请先进行向量化。")
            return None
    except Exception as e:
        print(f"⚠️ 检查 Milvus 集合时出错: {e}")
        return None

    # 返回 Milvus 检索器（只读）
    vector_store = Milvus(
        embedding_function=embeddings,
        collection_name=collection_name,
        connection_args={
            "host": settings.MILVUS_HOST,
            "port": settings.MILVUS_PORT,
        },
    )
    return vector_store

# ------------------------------
# 🔹 FastAPI Router
# ------------------------------
router = APIRouter(prefix="/querySimple", tags=["querySimple"])


class QueryRequest(BaseModel):
    query: str = Field(..., description="用户查询内容")
    retrieval_mode: str = Field("hybrid", description="检索模式: 'hybrid', 'dense', 'sparse'")
    top_k: int = Field(5, description="返回前K条结果")
    retrieval_weight: float = Field(0.5, description="稠密检索权重(0-1)")


class RetrievedDoc(BaseModel):
    id: str
    filename: str
    content: str


class QueryResponse(BaseModel):
    query: str
    mode: str
    results: List[RetrievedDoc]


def create_query_routes():
    @router.post("/{project_id}", response_model=QueryResponse)
    async def query_project(
        project_id: str,
        body: QueryRequest = Body(...),
        db: AsyncSession = Depends(get_db)
    ):
        try:
            # 1️⃣ 获取知识库与 Embedding 模型
            kb = await get_kb_by_project_id(project_id)
            if not kb:
                raise HTTPException(status_code=404, detail="Knowledge base not found")
            if not kb.model:
                raise HTTPException(status_code=400, detail="Embedding model not configured")

            embeddings = OpenAIEmbeddings(
                model=kb.model.name,
                base_url=kb.model.url,
                api_key=kb.model.api_key
            )

            # 2️⃣ 稀疏检索 (BM25)
            chunks = await get_chunks_by_project_id(project_id)
            documents = [
                Document(
                    page_content=chunk["content"],
                    metadata={
                        "id": chunk["id"],
                        "name": chunk["name"],
                        "file_id": chunk["fileId"],
                        "file_name": chunk["fileName"],
                        "summary": chunk["summary"],
                    }
                )
                for chunk in chunks
                if chunk.get("content")  # 避免空内容
            ]
            bm25_retriever = BM25Retriever.from_documents(documents, k=body.top_k)
            

            # 3️⃣ 稠密检索 (Milvus)
            vector_store = get_vector_store(project_id, embeddings)
            dense_retriever = vector_store.as_retriever(search_kwargs={"k": body.top_k})
            # 4️⃣ 执行检索
            if body.retrieval_mode == "dense":
                results = await dense_retriever.ainvoke(body.query)
            elif body.retrieval_mode == "sparse":
                results = bm25_retriever.get_relevant_documents(body.query)
            elif body.retrieval_mode == "hybrid":
                # sparse_docs = bm25_retriever.get_relevant_documents(body.query)
                sparse_docs = bm25_retriever.invoke(body.query)
                dense_docs = await dense_retriever.ainvoke(body.query)

                # 部分 BM25 结果可能没有 score，这里加 rank-based score
                for rank, doc in enumerate(sparse_docs):
                    doc.metadata["score"] = 1.0 / (rank + 1)

                for doc in dense_docs:
                    if not doc.metadata.get("score"):
                        doc.metadata["score"] = 1.0

                results = weighted_fusion(
                    dense_docs=dense_docs,
                    sparse_docs=sparse_docs,
                    dense_weight=body.retrieval_weight,
                    sparse_weight=1.0 - body.retrieval_weight,
                    top_k=body.top_k
                )
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported retrieval_mode: {body.retrieval_mode}")

            # 5️⃣ 构建响应
            response_docs = [
                RetrievedDoc(
                    id=doc.metadata.get("id", f"doc_{i}"),
                    filename=doc.metadata.get("fileName", f"document_{i}.txt"),
                    content=doc.page_content
                )
                for i, doc in enumerate(results)
            ]

            return QueryResponse(
                query=body.query,
                mode=body.retrieval_mode,
                results=response_docs
            )

        except Exception as e:
            logger.error(f"RAG 查询异常: {e}")
            raise HTTPException(status_code=500, detail=f"查询失败: {e}")

    return router
