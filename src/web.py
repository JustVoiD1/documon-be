from tavily import TavilyClient
from env import settings
tavily_client = TavilyClient(settings.TAVILY_API_KEY)

def get_web_context(query: str, top_k: int = 3) -> str:
    """
    Fetches clean web context for a RAG prompt while optimizing credit usage.
    """
    try:
        response = tavily_client.search(
            query=query,
            search_depth="basic",     
            max_results=top_k,  
            include_answer=False,     
            include_raw_content=False
        )
        
        # Format the results cleanly into a single context string for your LLM
        context_segments = []
        for result in response.get("results", []):
            title = result.get("title", "No Title")
            url = result.get("url", "")
            content = result.get("content", "")
            
            context_segments.append(f"Source: {title} ({url})\nContent: {content}\n---")
            
        return "\n\n".join(context_segments)

    except Exception as e:
        print(f"Tavily API Error: {e}")
        return ""