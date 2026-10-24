from datetime import datetime
import math
import uvicorn

from fastapi import Header
import io
from pathlib import Path
from uuid import UUID, uuid4
from typing import Dict, List, Any, Optional
from env import settings
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.search import RAGSearch
from src.s3_loader import S3DataLoader, FOLDER_MAPPING
from src.data_loader import DataLoader
from src.vectorstore.pgvector import PgVectorStore

app = FastAPI(title="Smriti", description="Smriti API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom Exception Handlers for Unified Error Responses: { "success": false, "error": str }
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    error_msg = "; ".join([f"{' -> '.join(str(l) for l in err.get('loc', []))}: {err.get('msg', '')}" for err in errors])
    print(f"[VALIDATION ERROR] Request to {request.url.path} failed: {error_msg}")
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": f"Validation Error: {error_msg}"
        }
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    print(f"[INTERNAL SERVER ERROR] {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": str(exc)
        }
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
    chat_id: str = Field(description="Chat session UUID to scope document context search")
    top_k: int = Field(default=3, ge=1, le=10)

class DeleteDocumentRequest(BaseModel):
    chat_id: str = Field(description="Chat session UUID to delete documents from")
    document_id: str = Field(description="Document UUID to delete")
    doc_type: str = Field(description="Type of the document")

class QueryResponse(BaseModel):
    success: bool = True
    message: str
    query: str

class SearchResultItem(BaseModel):
    content: str
    metadata: Dict[str, Any]
    distance: float

class SearchResponse(BaseModel):
    success: bool = True
    query: str
    results: List[SearchResultItem]
class DocumentResponse(BaseModel):
    id: UUID
    name: str
    doc_type: str
    download_url: str
    
class DocumentUploadResponse(BaseModel):
    success: bool = True
    message: str
    document: DocumentResponse


class DocumentDeleteResponse(BaseModel):
    success: bool = True
    message: str

try:
    rag_coordinator = RAGSearch(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        llm_model=settings.MODEL
    )
except Exception as e:
    print(f"[CRITICAL] Failed to initialize RAG Search engine: {e}")
    rag_coordinator = None

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
    if not rag_coordinator:
        print(f"[ERROR] Search engine database is unavailable.")
        raise HTTPException(status_code=500, detail="Search engine database is unavailable.")
    
    parsed_chat_id = UUID(req.chat_id)

    try:
        summary = rag_coordinator.search_and_summarize(
            query=req.query,
            chat_id=parsed_chat_id,
            top_k=req.top_k
        )
        
        return QueryResponse(
            success=True,
            message=str(summary),
            query=req.query
        )
    except Exception as e:
        print(f"[ERROR] LLM compilation generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LLM compilation generation failed: {str(e)}")


@app.get("/api/search", response_model=SearchResponse)
async def search_endpoint(
    query: str = Query(..., description="The search string terms to locate inside the vectors"),
    chat_id: str = Header(..., alias="X-Chat-ID"),
    top_k: int = Query(default=5, ge=1, le=20)
):
    """
    **Raw Search Endpoint:** Returns pure semantic context matches, vector scores, 
    and document chunks from the database without calling the LLM.
    """
    if not rag_coordinator:
        raise HTTPException(status_code=500, detail="Search engine database is unavailable.")
    
    try:
        raw_results = rag_coordinator.vectorstore.query(
            query_text=query,
            chat_id=UUID(chat_id),
            top_k=top_k
        )
        
        formatted_results = []
        for item in raw_results:
            formatted_results.append(
                SearchResultItem(
                    content=item.get("content") or "",
                    metadata=item.get("metadata") or {},
                    distance=float(item.get("distance") or math.inf)
                )
            )
            
        return SearchResponse(
            success=True,
            query=query,
            chat_id=chat_id,
            results=formatted_results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector store search failed: {str(e)}")

@app.post("/api/documents/upload", response_model=DocumentUploadResponse)
async def document_upload_endpoint(
    file: UploadFile = File(...),
    chat_id: Optional[str] = Form(None),
    uploaded_by: Optional[str] = Form(None)
):
    """
    **Document Upload Endpoint:**
    Uploads a file directly to the S3 bucket under the mapped subfolder, records document metadata
    in PostgreSQL, and parses/ingests document text in-memory without persistent local storage.
    """

    # store them as bucket_name/chat_id/doc_id.extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file filename provided.")

    filename = file.filename
    ext = Path(filename).suffix.lower()
    doc_type = ext.lstrip(".") if ext else "unknown"

    parsed_chat_id: Optional[UUID] = None
    if chat_id and chat_id.strip() not in ("", "null", "undefined", "None"):
        try:
            parsed_chat_id = UUID(chat_id.strip())
        except ValueError:
            parsed_chat_id = None

    parsed_uploaded_by: Optional[UUID] = None
    if uploaded_by and uploaded_by.strip() not in ("", "null", "undefined", "None"):
        try:
            parsed_uploaded_by = UUID(uploaded_by.strip())
        except ValueError:
            parsed_uploaded_by = None

    extension_folder = FOLDER_MAPPING.get(ext, (doc_type + "s") if doc_type != "unknown" else "others")
    subfolder = f"{parsed_chat_id}/{extension_folder}" if parsed_chat_id else extension_folder

    doc_id = uuid4()

    # Read uploaded file bytes into memory
    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {str(e)}")

    # 1. Upload file stream directly to S3 object storage
    s3_key = f"{subfolder}/{doc_id}{ext}"
    try:
        s3_loader = S3DataLoader()
        download_url = s3_loader.upload_fileobj(
            fileobj=io.BytesIO(file_bytes),
            key=s3_key,
            content_type=file.content_type
        )
    except Exception as s3_err:
        print(f"[WARN] S3 Upload failed/bypassed: {s3_err}")
        download_url = f"/documents/{subfolder}/{doc_id}{ext}"

    # 2. Record document metadata in PostgreSQL & parse/ingest directly from memory bytes
    chunks_count = 0
    pg_store = PgVectorStore(embedding_model="all-MiniLM-L6-v2")
    try:

        pg_store.create_document(
            document_id=doc_id,
            name=filename,
            doc_type=doc_type,
            download_url=download_url,
            chat_id=parsed_chat_id,
            uploaded_by=parsed_uploaded_by,
        )

        data_loader = DataLoader("documents")
        parsed_docs = data_loader.load_from_bytes(file_bytes=file_bytes, filename=filename)

        if parsed_docs:
            chunks_count = pg_store.build_from_documents(documents=parsed_docs, document_id=doc_id)
    except Exception as db_err:
        print(f"[ERROR] Ingestion / DB update failed: {db_err}")
        try:
            pg_store.delete_document(doc_id)
        except Exception as cleanup_err:
            print(f"[WARN] Failed to clean up orphaned document record {doc_id}: {cleanup_err}")
        raise HTTPException(status_code=500, detail=f"Document storage or vector ingestion failed: {str(db_err)}")

    return DocumentUploadResponse(
        success=True,
        message=f"Successfully uploaded '{filename}' to bucket and ingested {chunks_count} document segment(s).",
        document={
            "id": str(doc_id),
            "name": filename,
            "doc_type": doc_type,
            "download_url": download_url,
        }
    )
    
@app.delete("/api/chat/{chat_id}/documents", response_model=DocumentDeleteResponse)
async def document_delete_all_endpoint(chat_id: UUID):
    """
    **Document Delete Endpoint:**
    Deletes documents from S3.
    """
    

    s3_loader = S3DataLoader()
    s3_loader.delete_documents_by_chat_id(chat_id)
    return DocumentDeleteResponse(
        success=True,
        message=f"Documents in chat: {chat_id} deleted successfully"
    )

@app.delete("/api/chat/documents", response_model=DocumentDeleteResponse)
async def document_delete_endpoint(req: DeleteDocumentRequest):
    """
    **Document Delete Endpoint:**
    Deletes documents from S3.
    """
    
    try:
        parsed_chat_id = UUID(req.chat_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid chat_id.")
    try:
        parsed_document_id = UUID(req.document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document_id.")
    parsed_doc_type = req.doc_type
        

    s3_loader = S3DataLoader()
    pg_store = PgVectorStore()
    s3_loader.delete_document(parsed_document_id, parsed_chat_id, parsed_doc_type)
    pg_store.delete_document(parsed_document_id)

    return DocumentDeleteResponse(
        success=True,
        message=f"Documents in chat: {parsed_chat_id} deleted successfully"
    )





    




    
    

