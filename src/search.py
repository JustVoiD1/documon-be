from src.db.postgres import get_connection
from langchain_core.messages import BaseMessage
from typing import List, Dict, Any
import os
from env import settings
from dotenv import load_dotenv
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from typing import Any
from src.vectorstore.pgvector import PgVectorStore
from uuid import UUID
from env import settings
load_dotenv()



os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_TOKEN"] = settings.HF_TOKEN


def get_chat_history_from_db(chat_id: UUID, limit: int = 10) -> List[Dict[str, str]]:
    """
    Fetches the most recent `limit` messages for the given chat_id from PostgreSQL,
    returned in chronological order (oldest to newest).
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT role, content
                    FROM messages
                    WHERE chat_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (chat_id, limit),
                )
                rows = cur.fetchall()
        return [{"role": str(row[0]), "content": str(row[1])} for row in reversed(rows)]
    except Exception as e:
        print(f"[WARN] Failed to fetch message history from DB for chat {chat_id}: {e}")
        return []


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

        history = get_chat_history_from_db(chat_id=chat_id, limit=10)
        print("[INFO] Chat History: \n\n", history)

        # If the last message in history is the exact current user query, avoid duplicating it
        if (
            history
            and history[-1].get("role") in ("user", "human")
            and history[-1].get("content", "").strip() == query.strip()
        ):
            history = history[:-1]

        # Keep within the sliding context window
        if len(history) > 10:
            history = history[-10:]

        # 3. Construct system instruction
        system_instruction = (
            "You are Documon, an AI assistant specialized in answering questions based on the provided documents and conversation context.\n\n"
            "Guidelines:\n"
            "1. Conversation History: You have access to previous messages in this conversation. Use them to understand context, follow-ups, clarifications, and questions referring to past exchanges.\n"
            "2. Document Grounding: For factual questions about the documents, base your answers on the provided Document Context. Do not hallucinate facts outside the provided documents or conversation.\n"
            "3. If the user asks about previous messages (e.g., 'what was my last question?', 'summarize what you just said'), answer directly using the conversation history.\n"
            "4. If a question is specifically about the documents, but neither the Document Context nor the conversation history contains the answer, respond with:\n"
            "\"I cannot find the answer to that question in the uploaded documents.\""
        )

        messages: List[BaseMessage] = [SystemMessage(content=system_instruction)]

        # 4. Append historical turns into the context window
        for msg in history:
            role = (msg.get("role") or "").lower()
            content = msg.get("content") or ""
            if not content:
                continue
            if role in ("user", "human"):
                messages.append(HumanMessage(content=content))
            elif role in ("assistant", "ai"):
                messages.append(AIMessage(content=content))
            elif role == "system":
                messages.append(SystemMessage(content=content))

        # 5. Append current turn with document context
        if context:
            user_prompt = f"Document Context:\n{context}\n\nUser Question: {query}"
        else:
            user_prompt = f"User Question: {query}"

        messages.append(HumanMessage(content=user_prompt))

        # 6. Invoke LLM
        response = self.llm.invoke(messages)
        return response.content

# Example usage
# if __name__ == "__main__":
#     rag_search = RAGSearch()
#     query = "What is Partitioning"
#     summary = rag_search.search_and_summarize(query, top_k=3)
#     print("Summary:", summary)