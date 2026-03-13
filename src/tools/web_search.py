import os
from tavily import TavilyClient

class WebSearchTool:
    """Wrapper around Tavily API for semantic web search."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key or self.api_key == "your_tavily_api_key_here":
            self.client = None
            print("Warning: No valid TAVILY_API_KEY found. Web search will be mocked.")
        else:
            self.client = TavilyClient(api_key=self.api_key)

    def search(self, query: str) -> str:
        """Searches the web for recent context relevant to Medicaid, healthcare compliance, or payer updates."""
        if not self.client:
            return (f"MOCK WEB SEARCH RESULT for '{query}':\n"
                    "Recent Medicaid FFS policy updates emphasize stricter requirements "
                    "for CMS compliance validation.")
            
        try:
            response = self.client.search(
                query=query, 
                search_depth="advanced", 
                max_results=3, 
                include_answer=True
            )
            
            # If Tavily returns an aggregated answer, use it, else stringify the snippets
            if "answer" in response and response["answer"]:
                return f"Web Search Answer: {response['answer']}"
                
            snippets = [
                f"- {res['title']} ({res['url']}): {res['content']}" 
                for res in response.get("results", [])
            ]
            return "Web Search Results:\n" + "\n".join(snippets)
            
        except Exception as e:
            return f"Web search failed: {str(e)}"
