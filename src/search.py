import os
from env import settings
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import Any
from src.vectorstore.pgvector import PgVectorStore
from uuid import UUID
from env import settings
load_dotenv()



os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_TOKEN"] = settings.HF_TOKEN

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
        texts = [r["content"] for r in results if r.get("content")]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        instructions = """
You are a document question-answering assistant.

Your task is to answer the user's question using ONLY the information
contained in the provided document context.

RULES:

1. CONTEXT-ONLY
   Use only information explicitly stated in the provided context.
   Do not use your general knowledge, training data, assumptions,
   or information from outside the documents.

2. ANSWERABILITY
   Before answering, determine whether the provided context contains
   enough relevant information to answer the question.

   If the context does not contain enough information to answer the
   question, respond EXACTLY with:
   "I cannot find the answer to that question in the uploaded documents."

3. NO HALLUCINATION
   Never invent facts, numbers, names, dates, explanations, or conclusions
   that are not supported by the context.

4. RELEVANCE
   Use only the parts of the context that are relevant to the question.
   Do not summarize or discuss unrelated parts of the documents.

5. REASONING
   You may combine information from multiple parts of the context when
   necessary to answer the question, but every part of the answer must
   be supported by the provided context.

6. UNCERTAINTY
   If the context provides incomplete or ambiguous information, clearly
   state what can and cannot be determined from the documents.

7. DIRECTNESS
   Answer the question directly and concisely. Do not add unnecessary
   background information.

8. GREETINGS
   If the user sends a normal greeting such as "hello" or "hi", respond
   naturally without requiring document context.
"""

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