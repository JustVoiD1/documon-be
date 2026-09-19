import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import Any
from src.vectorstore import FaissVectorStore
load_dotenv()
os.environ["HF_HUB_OFFLINE"] = "1"
class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "groq/compound-mini"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) ->  list[dict[Any, Any] | str] | str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        instructions = """You are an advanced assistant analyzing document context.
Your goal is to answer the query based ONLY on the provided context below.
CRITICAL RULES:
1. If the context does not contain the answer to the query, or if the query is completely unrelated to the context topic, DO NOT make things up and DO NOT summarize the context. Instead, state clearly: "I cannot find the answer to that question in the uploaded documents."
2. Do not use outside knowledge."""

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