from typing import Dict
from typing import TypedDict
from langchain_core.documents import Document
from src.db.postgres import get_connection
from typing import List, Any
from uuid import UUID, uuid4
from pgvector.psycopg2 import register_vector
from psycopg2.extras import register_uuid, Json
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline

class SearchResultItem(TypedDict):
    id: str  # or int / uuid.UUID depending on your DB schema
    document_id: str  # or int / uuid.UUID
    chunk_index: int
    content: str
    metadata: Dict[str, Any]  # Storing arbitrary key-value metadata
    page_number: int          # or Optional[int] if some chunks don't have pages
    distance: float

class PgVectorStore:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.embedding_model = embedding_model
        self._model = None
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @property
    def model(self):
        if self._model is None:
            print(f"[INFO] Lazily loading embedding model: {self.embedding_model}")
            self._model = SentenceTransformer(self.embedding_model)
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
                        embedding vector(384),
                        metadata JSONB,
                        page_number INT
                    );
                """)
            conn.commit()
        print("[INFO] PostgreSQL tables initialized.")

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
        with get_connection() as conn:
            register_uuid(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents (id, name, doc_type, download_url, chat_id, uploaded_by, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, NOW())
                    """,
                    (document_id, name, doc_type, download_url, chat_id, uploaded_by),
                )
            conn.commit()
        print(f"[INFO] Inserted document record into 'documents' table: {document_id} (name: {name}, type: {doc_type})")

    def build_from_documents(self, documents: List[Document], document_id: UUID):
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model, 
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

                    page_number = metadata.get("page")

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
                            chunk.page_content,
                            embedding,
                            Json(metadata),
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

        query_embedding = self.model.encode(
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