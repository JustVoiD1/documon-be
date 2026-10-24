from typing import Dict
from typing import TypedDict
from langchain_core.documents import Document
from src.db.postgres import get_connection
from typing import List, Any
from uuid import UUID, uuid4
from pgvector.psycopg2 import register_vector
from psycopg2.extras import register_uuid, Json
from src.embedding import EmbeddingPipeline
from src.sanitizer import sanitize_metadata, sanitize_text
from pymilvus import model as embedding_model

class SearchResultItem(TypedDict):
    id: str  # or int / uuid.UUID depending on your DB schema
    document_id: str  # or int / uuid.UUID
    chunk_index: int
    content: str
    metadata: Dict[str, Any]  # Storing arbitrary key-value metadata
    page_number: int          # or Optional[int] if some chunks don't have pages
    distance: float

class PgVectorStore:
    def __init__(self, model_name: str = 'GPTCache/paraphrase-albert-onnx', chunk_size: int = 1000, chunk_overlap: int = 200):
        self.model_name = model_name
        self._model = None
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @property
    def model(self):
        if self._model is None:
            print(f"[INFO] Lazily loading embedding model: {self.model_name}")
            self._model = embedding_model.DefaultEmbeddingFunction(self.model_name)
        return self._model


    def init_tables(self):
        """Creates 'documents' and 'document_chunks' tables in PostgreSQL if they do not exist."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE EXTENSION IF NOT EXISTS vector;

                    CREATE TABLE IF NOT EXISTS documents (
                        id UUID PRIMARY KEY,
                        name TEXT,
                        type VARCHAR(50),
                        download_url TEXT,
                        chat_id UUID,
                        uploaded_by UUID,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE TABLE IF NOT EXISTS document_chunks (
                        id UUID PRIMARY KEY,
                        document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
                        chunk_index INT,
                        content TEXT,
                        embedding vector(768),
                        metadata JSONB,
                        page_number INT
                    );
                """)
            conn.commit()
        print("[INFO] PostgreSQL tables initialized.")

    def get_documents_by_chat_id(self, chat_id: UUID):
        with get_connection() as conn:
            register_uuid(conn)
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM documents WHERE chat_id = %s", (chat_id,))
                return cur.fetchall()

    def create_document(
        self,
        document_id: UUID,
        name: str | None = None,
        doc_type: str | None = None,
        download_url: str | None = None,
        chat_id: UUID | None = None,
        uploaded_by: UUID | None = None,
    ):
        """Inserts a new document record into the 'documents' table prior to chunking."""
        self.init_tables()
        clean_name = sanitize_text(name) if name else None
        clean_doc_type = sanitize_text(doc_type) if doc_type else None
        clean_download_url = sanitize_text(download_url) if download_url else None
        with get_connection() as conn:
            register_uuid(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents (id, name, doc_type, download_url, chat_id, uploaded_by, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, NOW())
                    """,
                    (document_id, clean_name, clean_doc_type, clean_download_url, chat_id, uploaded_by),
                )
            conn.commit()
        print(f"[INFO] Inserted document record into 'documents' table: {document_id} (name: {clean_name}, type: {clean_doc_type})")

    def get_doctype_by_id(self, document_id: UUID, chat_id: UUID):
        """Retrieves a document record from the 'documents' table by ID."""
        with get_connection() as conn:
            register_uuid(conn)
            with conn.cursor() as cur:
                cur.execute("SELECT doc_type FROM documents WHERE id = %s AND chat_id = %s", (document_id, chat_id,))
                return cur.fetchone()

    def delete_document(self, document_id: UUID):
        """Deletes a document record and its cascading chunks from PostgreSQL."""
        with get_connection() as conn:
            register_uuid(conn)
            with conn.cursor() as cur:
                cur.execute("DELETE FROM documents WHERE id = %s", (document_id,))
            conn.commit()
        print(f"[INFO] Deleted document {document_id} and its associated chunks from database.")

    def build_from_documents(self, documents: List[Document], document_id: UUID) -> int:
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        emb_pipe = EmbeddingPipeline(
            model_name=self.model_name, 
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)
        self.add_embeddings(
            embeddings,
            chunks,
            document_id
        )
        print(f"[INFO] Vector store built and {len(chunks)} embeddings saved to PostgresDB")
        return len(chunks)
    

    def add_embeddings(
        self, embeddings, chunks, document_id: UUID
    ):
        with get_connection() as conn:
            register_vector(conn)
            register_uuid(conn)
            with conn.cursor() as curr:
                for index, (embedding, chunk) in enumerate(
                    zip(embeddings, chunks)
                ):

                    metadata = getattr(
                        chunk,
                        "metadata",
                        {}
                    )
                    clean_meta = sanitize_metadata(metadata)
                    page_number = clean_meta.get("page")
                    clean_content = sanitize_text(getattr(chunk, "page_content", ""))

                    curr.execute(
                        """
                        INSERT INTO document_chunks
                        (
                            id,
                            document_id,
                            chunk_index,
                            content,
                            embedding,
                            metadata,
                            page_number
                        )
                        VALUES
                        (
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s
                        )
                        """,
                        (
                            uuid4(),
                            document_id,
                            index,
                            clean_content,
                            embedding,
                            Json(clean_meta),
                            page_number,
                        ),
                    )

            conn.commit()

        print(
            f"[INFO] Added {len(chunks)} chunks to PostgreSQL"
        )

    def search(
        self,
        query_embedding,
        chat_id: UUID,
        top_k: int = 5
    ) -> List[SearchResultItem]:
        with get_connection() as conn:
            register_vector(conn)
            register_uuid(conn)

            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        c.id,
                        c.document_id,
                        c.chunk_index,
                        c.content,
                        c.metadata,
                        c.page_number,
                        c.embedding <=> %s AS distance
                    FROM document_chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE d.chat_id = %s
                    ORDER BY c.embedding <=> %s
                    LIMIT %s
                    """,
                    (
                        query_embedding,
                        chat_id,
                        query_embedding,
                        top_k,
                    ),
                )
                rows = cur.fetchall()

        return [
            {
                "id": row[0],
                "document_id": row[1],
                "chunk_index": row[2],
                "content": row[3],
                "metadata": row[4],
                "page_number": row[5],
                "distance": row[6],
            }
            for row in rows
        ]

    def query(
        self,
        query_text: str,
        chat_id: UUID,
        top_k: int = 5
    ):

        print(
            f"[INFO] Querying vector store: '{query_text}' (chat_id: {chat_id})"
        )

        query_embedding = self.model.encode_queries(
            [query_text]
        )[0]

        return self.search(
            query_embedding=query_embedding,
            chat_id=chat_id,
            top_k=top_k
        )


# Example usage
# if __name__ == "__main__":
#     from data_loader import load_all_documents
#     docs = load_all_documents("data")
#     store = FaissVectorStore("faiss_store")
#     store.build_from_documents(docs)
#     store.load()
#     print(store.query("What is Partitioning", top_k=3))