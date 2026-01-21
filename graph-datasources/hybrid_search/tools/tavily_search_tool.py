"""
Tavily Search Tool for LlamaIndex Agent.

This module provides web search functionality using the Tavily caching proxy.
It enables real-time web search and research capabilities for the hybrid search agent.
"""

import os
import sys
import json
import logging
from typing import Optional, Dict, Any
import httpx

# Add parent directory to path for config import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import config

logger = logging.getLogger(__name__)

# Tavily proxy base URL
TAVILY_PROXY_BASE_URL = os.getenv(
    "TAVILY_PROXY_BASE_URL", 
    "https://api.blacksmith.deerfieldgreen.com/tavily"
)


def tavily_search_func(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    include_answer: bool = False,
    include_domains: Optional[list] = None,
    exclude_domains: Optional[list] = None,
    topic: Optional[str] = None,
    time_range: Optional[str] = None
) -> str:
    """
    Perform web search using Tavily proxy API.
    
    This tool searches the web for real-time information, news, and research.
    Results are cached by the proxy for performance.
    
    Args:
        query: Search query string (required)
        max_results: Maximum number of results to return (default: 5, max: 20)
        search_depth: Search depth - "basic" or "advanced" (default: "basic")
        include_answer: Include AI-generated answer summary (default: False)
        include_domains: List of domains to restrict search to (optional)
        exclude_domains: List of domains to exclude from search (optional)
        topic: Optional topic filter (optional)
        time_range: Optional time range filter (optional)
    
    Returns:
        Formatted string with search results, or error message if search fails
    """
    try:
        # Validate inputs
        if not query or not query.strip():
            return "Error: Query cannot be empty"
        
        max_results = min(max(1, max_results), 20)  # Clamp between 1 and 20
        
        # Prepare request payload
        payload = {
            "query": query.strip(),
            "max_results": max_results,
            "search_depth": search_depth
        }
        
        # Add optional parameters
        if include_answer:
            payload["include_answer"] = True
        
        if include_domains:
            payload["include_domains"] = include_domains
        
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains
        
        if topic:
            payload["topic"] = topic
        
        if time_range:
            payload["time_range"] = time_range
        
        # Make request to Tavily proxy
        logger.info(f"Searching Tavily for query: {query[:100]}...")
        
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{TAVILY_PROXY_BASE_URL}/search",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Format results for agent consumption
                formatted_results = []
                
                # Add answer if available
                if data.get("answer"):
                    formatted_results.append(f"AI Answer: {data['answer']}\n")
                
                # Format search results
                results = data.get("results", [])
                if results:
                    formatted_results.append(f"Found {len(results)} search results:\n")
                    
                    for i, result in enumerate(results, 1):
                        title = result.get("title", "No title")
                        url = result.get("url", "No URL")
                        content = result.get("content", "")
                        
                        result_str = f"\n[{i}] {title}\n"
                        result_str += f"URL: {url}\n"
                        if content:
                            # Truncate content to reasonable length
                            content_preview = content[:500] + "..." if len(content) > 500 else content
                            result_str += f"Content: {content_preview}\n"
                        
                        formatted_results.append(result_str)
                else:
                    formatted_results.append("No results found for this query.")
                
                return "\n".join(formatted_results)
            
            elif response.status_code == 429:
                return "Error: Rate limit exceeded. Please try again later."
            elif response.status_code == 401:
                return "Error: Authentication failed. Check API configuration."
            else:
                logger.error(f"Tavily API error: {response.status_code} - {response.text}")
                return f"Error: Tavily search failed with status {response.status_code}"
                
    except httpx.TimeoutException:
        logger.error("Tavily search request timed out")
        return "Error: Search request timed out. Please try again."
    except httpx.RequestError as e:
        logger.error(f"Tavily search request error: {e}")
        return f"Error: Failed to connect to Tavily proxy: {repr(e)}"
    except Exception as e:
        import traceback
        logger.error(f"Error in Tavily search: {e}")
        logger.error(traceback.format_exc())
        return f"Error executing Tavily search: {repr(e)}"


# Create the Tool object
from llama_index.core.tools import FunctionTool

tavily_search_tool = FunctionTool.from_defaults(
    fn=tavily_search_func,
    name="tavily_search",
    description="""Web search tool that searches the internet for real-time information, news, and research using Tavily.
    
    This tool provides access to current web content, news articles, and research materials.
    Results are cached by the proxy for performance.
    
    Use this tool when:
    - You need real-time information or current events
    - You're looking for recent news articles or press releases
    - You need information not available in the graph or vector databases
    - You want to research external sources or industry trends
    - You need company announcements or market updates
    
    The tool automatically:
    - Searches the web using intelligent query processing
    - Returns relevant results with titles, URLs, and content snippets
    - Optionally provides AI-generated answer summaries
    - Supports domain filtering and time-based filtering
    
    Parameters:
    - query: Natural language search query (required)
    - max_results: Number of results to return (default: 5, max: 20)
    - search_depth: "basic" for faster results or "advanced" for deeper search (default: "basic")
    - include_answer: Include AI-generated answer summary (default: False)
    - include_domains: Restrict search to specific domains (optional)
    - exclude_domains: Exclude specific domains from search (optional)
    - topic: Filter by topic (optional)
    - time_range: Filter by time range (optional)
    
    Example queries:
    - "Latest Tesla earnings announcement"
    - "Recent news about Apple stock"
    - "What happened in the market today?"
    - "Company press releases about AI"
    """
)

if __name__ == "__main__":
    # Test the Tavily search
    import logging
    logging.basicConfig(level=logging.INFO)
    
    test_query = "Tesla stock news today"
    print(f"\n=== Testing Tavily Search ===\n")
    print(f"Query: '{test_query}'\n")
    
    result = tavily_search_func(test_query, max_results=3)
    print(result)
    print("\n=== Test Complete ===")

