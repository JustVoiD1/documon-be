import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import Any
from src.vectorstore.pgvector import PgVectorStore
from uuid import UUID
from env import settings
load_dotenv()



os.environ["HF_HUB_OFFLINE"] = "1"

class RAGSearch:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = settings.MODEL):
        self.vectorstore = PgVectorStore(embedding_model)
        self.llm = ChatGroq(
            model=llm_model,
            temperature=0.05,
        )
        print(f"[INFO] LLM initialized: {llm_model}")

    def search_and_summarize(self,
        query: str,
        chat_id: UUID,
        top_k: int = 5
       ) -> list[dict[Any, Any] | str] | str:
        results = self.vectorstore.query(query_text=query, chat_id=chat_id, top_k=top_k)
        print(results)
        texts = [r["content"] for r in results if r.get("content")]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        instructions = """You are an advanced assistant analyzing document context.
Your goal is to answer the query based ONLY on the provided context below.
CRITICAL RULES:
1. If the context does not contain the answer to the query, or if the query is completely unrelated to the context topic, DO NOT make things up and DO NOT summarize the context. Instead, state clearly: "I cannot find the answer to that question in the uploaded documents."
2. Do not use outside knowledge unless normal greeting."""

        prompt = f"""{instructions}
        
        Context: {context}
        
        Query: '{query}'"""
        response = self.llm.invoke([prompt])
        return response.content

# Example usage
# if __name__ == "__main__":
#     rag_search = RAGSearch()
#     query = "What is Partitioning"
#     summary = rag_search.search_and_summarize(query, top_k=3)
#     print("Summary:", summary)