from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline   
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Any

class QueryRequest(BaseModel):
    query: str = Field(..., examples=["What is Partitioning?"])
    top_k: int = Field(default=3, ge=1, le=10)

class QueryResponse(BaseModel):
    query: str
    message: List[Dict[Any, Any]] | str

class SearchResultItem(BaseModel):
    # score: float
    metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResultItem]


app = FastAPI(title="DocuMon", description="DocuMon API", version="1.0.0")


# docs = load_all_documents("documents")
# emb_pipe = EmbeddingPipeline()
# chunks = emb_pipe.chunk_documents(docs)
# embeddings = emb_pipe.embed_chunks(chunks)
# store = FaissVectorStore("faiss_store")
# store.build_from_documents(docs)
# store.load()
# print(store.query("What is Partitioning", top_k=3))
# rag_search = RAGSearch()
# query = "What is Partitioning"
# endpoint for get result from user query
try:
    rag_coordinator = RAGSearch(
        persist_dir="faiss_store",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        llm_model="groq/compound-mini"
    )
except Exception as e:
    print(f"[CRITICAL] Failed to initialize RAG Search engine: {e}")
    rag_coordinator = None

@app.get("/health")
def health_check():
    """Simple check to verify the API server is alive."""
    return {"status": "healthy", "engine": "active"}

@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(payload: QueryRequest):
    """
    **RAG Query Endpoint:** Searches your documents and utilizes the Groq LLM
    to generate a clean, coherent natural language answer summary.
    """
    if not rag_coordinator:
        raise HTTPException(status_code=500, detail="Search engine database is unavailable.")
    
    try:
        # Calls the method we fixed earlier
        summary = rag_coordinator.search_and_summarize(payload.query, top_k=payload.top_k)
        
        return QueryResponse(query=payload.query, message=str(summary))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM compilation generation failed: {str(e)}")


@app.get("/api/search", response_model=SearchResponse)
async def search_endpoint(
    q: str = Query(..., description="The search string terms to locate inside the vectors"),
    top_k: int = Query(default=5, ge=1, le=20)
):
    """
    **Raw Search Endpoint:** Returns pure semantic context matches, vector scores, 
    and document chunks from the FAISS database without calling the LLM.
    """
    if not rag_coordinator:
        raise HTTPException(status_code=500, detail="Search engine database is unavailable.")
    
    try:
        # Directly queries your vector store
        raw_results = rag_coordinator.vectorstore.query(q, top_k=top_k)
        
        # Formats output values to match the Pydantic schema structure safely
        formatted_results = []
        for item in raw_results:
            formatted_results.append(
                SearchResultItem(
                    # score=float(item.get("score", 0.0)),
                    metadata=item.get("metadata") or {}
                )
            )
            
        return SearchResponse(query=q, results=formatted_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector store search failed{str(e)}")


