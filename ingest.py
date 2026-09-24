from uuid import uuid4

from src.data_loader import DataLoader
from src.s3_loader import S3DataLoader
from src.vectorstore.pgvector import PgVectorStore


def ingest_documents(chat_id=None, uploaded_by=None):
    print("[INFO] Downloading documents from Object Storage (S3)...")
    s3_loader = S3DataLoader()
    file_infos = s3_loader.download_all_files()

    if not file_infos:
        print("[WARN] No documents found in Object Storage to ingest.")
        return

    data_loader = DataLoader("documents")
    vectorstore = PgVectorStore("all-MiniLM-L6-v2")

    for file_info in file_infos:
        doc_id = uuid4()
        file_name = file_info["name"]
        file_type = file_info["type"]
        download_url = file_info["download_url"]
        local_path = file_info["local_path"]

        print(f"\n[INFO] --- Processing File: '{file_name}' (ID: {doc_id}) ---")

        # 1. Insert document metadata into `documents` table FIRST
        vectorstore.create_document(
            document_id=doc_id,
            name=file_name,
            doc_type=file_type,
            download_url=download_url,
            chat_id=chat_id,
            uploaded_by=uploaded_by,
        )

        # 2. Parse pages/docs from this specific file
        docs = data_loader.load_single_file(local_path)
        if not docs:
            print(f"[WARN] No text content extracted from file '{file_name}'. Skipping vector store build.")
            continue

        # 3. Chunk & store vector embeddings in `document_chunks` using doc_id
        vectorstore.build_from_documents(
            documents=docs,
            document_id=doc_id
        )
        print(f"[SUCCESS] Successfully ingested '{file_name}' into PgVector with document ID: {doc_id}")


if __name__ == "__main__":
    ingest_documents()
