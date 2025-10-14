from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Optional, Any

from lib.core.rag_store import RAGStore
from lib.core.build_whoosh_index import search_whoosh_index
from lib.db.kb import get_kb_by_project_id
from lib.db.base import get_db
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from utils import logger

# ------------------------------
# 辅助函数
# ------------------------------
def rrf_fusion(result_lists, k=60, weights=None):
    """加权 Reciprocal Rank Fusion 算法"""
    from collections import defaultdict

    scores = defaultdict(float)
    doc_map = {}

    if weights is None:
        weights = [1.0] * len(result_lists)

    for list_idx, result_list in enumerate(result_lists):
        weight = weights[list_idx]
        for rank, doc in enumerate(result_list):
            doc_id = doc.metadata.get("id") if hasattr(doc, "metadata") else str(hash(doc))
            rrf_score = weight / (k + rank + 1)
            scores[doc_id] += rrf_score
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[doc_id] for doc_id, _ in sorted_docs]


def dict_to_document(doc_dict):
    """将字典转为 LangChain Document 对象"""
    return Document(
        page_content=doc_dict.get("content", ""),
        metadata={
            "id": doc_dict.get("id", ""),
            "fileName": doc_dict.get("fileName", ""),
            "projectId": doc_dict.get("projectId", "")
        }
    )



router = APIRouter(
    prefix="/querySimple",
    tags=["querySimple"]
)

# 请求模型
class QueryRequest(BaseModel):
    query: str = Field(..., description="用户查询内容")
    retrieval_mode: str = Field("hybrid", description="检索模式: 'hybrid', 'dense', 'sparse'")
    top_k: int = Field(5, description="返回前K条结果")
    retrieval_weight: float = Field(0.5, description="稠密检索权重(0-1)")

# 响应模型
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
        """
        最简版 RAG 查询接口（无大模型，仅检索）
        """
        try:
            # 读取知识库配置
            kb = await get_kb_by_project_id(project_id)
            if not kb:
                raise HTTPException(status_code=404, detail="Knowledge base not found")

            if not kb.model:
                raise HTTPException(status_code=400, detail="Embedding model not configured")

            # 初始化 embedding 模型
            embeddings = OpenAIEmbeddings(
                model=kb.model.name,
                base_url=kb.model.url,
                api_key=kb.model.api_key
            )

            # 创建 RAGStore 并获取向量检索器
            rag_store = RAGStore()
            vector_store = rag_store.get_vector_store(project_id, embeddings)
            retriever = vector_store.as_retriever(search_kwargs={"k": body.top_k})

            # 执行检索逻辑
            if body.retrieval_mode == "hybrid":
                whoosh_results = search_whoosh_index(project_id, body.query, top_k=body.top_k)
                whoosh_docs = [dict_to_document(doc) for doc in whoosh_results]
                vector_results = await retriever.ainvoke(body.query)

                combined_results = rrf_fusion(
                    [whoosh_docs, vector_results],
                    k=60,
                    weights=[1.0 - body.retrieval_weight, body.retrieval_weight]
                )[:body.top_k]

            elif body.retrieval_mode == "dense":
                combined_results = await retriever.ainvoke(body.query)
            elif body.retrieval_mode == "sparse":
                whoosh_results = search_whoosh_index(project_id, body.query, top_k=body.top_k)
                combined_results = [dict_to_document(doc) for doc in whoosh_results]
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported retrieval_mode: {body.retrieval_mode}")

            # 构建响应
            results = [
                RetrievedDoc(
                    id=doc.metadata.get("id", f"doc_{i}"),
                    filename=doc.metadata.get("fileName", f"document_{i}.txt"),
                    content=doc.page_content
                )
                for i, doc in enumerate(combined_results)
            ]

            return QueryResponse(
                query=body.query,
                mode=body.retrieval_mode,
                results=results
            )

        except Exception as e:
            logger.error(f"RAG 查询异常: {e}")
            raise HTTPException(status_code=500, detail=f"查询失败: {e}")

    return router


