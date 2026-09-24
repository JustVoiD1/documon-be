

from fastapi import Header
from fastapi import Form
from fastapi import File
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Query, UploadFile
from typing import Optional, Dict, Any, List
from pydantic import Field
from pydantic import BaseModel
app = FastAPI(title="DocuMon", description="DocuMon API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)




# Pydantic Schemas with { success: true, message: str, ...usual_fields }
from pydantic import field_validator

class HealthResponse(BaseModel):
    success: bool = True
    message: str = "API server is healthy"
    status: str = "healthy"
    engine: str = "active"

class QueryRequest(BaseModel):
    query: str = Field(..., examples=["What is Partitioning?"])
    chat_id: str = Field(description="Optional chat session UUID to scope document context search")
    top_k: int = Field(default=3, ge=1, le=10)

   


class QueryResponse(BaseModel):
    success: bool = True
    message: str
    query: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchResultItem(BaseModel):
    content: str
    metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    success: bool = True
    query: str
    results: List[SearchResultItem]

class DocumentItem(BaseModel):
    filename: str
    file_size: int
    pages: int
    uploaded_by: str
    created_at: str
    doc_type: str

class DocumentUploadResponse(BaseModel):
    success: bool = True
    document: DocumentItem


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Simple check to verify the API server is alive."""
    return HealthResponse(
        success=True,
        message="API server is healthy",
        status="healthy",
        engine="active"
    )

@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    """
    **RAG Query Endpoint:** Searches your documents and utilizes the Gemini LLM
    to generate a clean, coherent natural language answer summary.
    """
    
    return {"query": req.query, "top_k": req.top_k, "message": "Query received successfully"}
    


@app.get("/api/search", response_model=SearchResponse)
async def search_endpoint(
    query: str = Query(..., description="The search string terms to locate inside the vectors"),
    chat_id: str = Header(..., alias="X-Chat-ID"),
    top_k: int = Query(default=5, ge=1, le=20)
):
    print({
        "query":query, "chat_id":chat_id, "top_k":top_k
    })
    return {"success": True, "query": query, "results": []}
    

@app.post("/api/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    chat_id: Optional[str] = Form(None),
    uploaded_by: Optional[str] = Form(None)
):
    """
    **Document Upload Endpoint:**
    Uploads a file directly to the S3 bucket under the mapped subfolder, records document metadata
    in PostgreSQL, and parses/ingests document text in-memory without persistent local storage.
    """
    return { "name":file.filename, "chat_id": chat_id, "uploaded_by": uploaded_by }




    
    

