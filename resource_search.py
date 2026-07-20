"""Resource search using Tavily API for YouTube videos and Research papers."""

import logging
from tavily_helper import tavily_client

logger = logging.getLogger(__name__)


def search_youtube_tavily(query: str, max_results: int = 2) -> list[dict]:
    """Search YouTube videos via Tavily public API by targeting youtube.com."""
    tavily_query = f"{query} site:youtube.com"
    resources = []
    try:
        search_results = tavily_client.search(query=tavily_query, search_depth="basic")
        for r in search_results.get("results", []):
            url = r.get("url", "")
            # Filter specifically for youtube links
            if "youtube.com/watch" in url or "youtu.be/" in url or "youtube.com/embed" in url:
                resources.append({
                    "title": r.get("title", "YouTube Video"),
                    "url": url,
                    "source": "youtube",
                    "thumbnail": None,
                    "description": r.get("content", "")[:200] if r.get("content") else None
                })
        
        # If no specific YouTube URLs found, fall back to general search results
        if not resources:
            for r in search_results.get("results", [])[:max_results]:
                resources.append({
                    "title": r.get("title", "Resource"),
                    "url": r.get("url", ""),
                    "source": "youtube",
                    "thumbnail": None,
                    "description": r.get("content", "")[:200] if r.get("content") else None
                })
                
        logger.info(f"Tavily found {len(resources)} YouTube resources for query '{query}'")
        return resources[:max_results]
    except Exception as e:
        logger.warning(f"Tavily YouTube search failed for '{query}': {e}")
        # Fallback search link
        return [{
            "title": f"Search YouTube: {query}",
            "url": f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}",
            "source": "youtube",
            "thumbnail": None,
            "description": f"Search YouTube for: {query}"
        }]


def search_research_papers_tavily(query: str, max_results: int = 2) -> list[dict]:
    """Search academic research papers via Tavily."""
    tavily_query = f"{query} academic research paper pdf site:arxiv.org OR site:semanticscholar.org OR site:ieee.org"
    resources = []
    try:
        search_results = tavily_client.search(query=tavily_query, search_depth="basic")
        for r in search_results.get("results", []):
            resources.append({
                "title": r.get("title", "Research Paper"),
                "url": r.get("url", ""),
                "source": "research_paper",
                "thumbnail": None,
                "description": r.get("content", "")[:200] if r.get("content") else None
            })
        logger.info(f"Tavily found {len(resources)} research paper resources for query '{query}'")
        return resources[:max_results]
    except Exception as e:
        logger.warning(f"Tavily research paper search failed for '{query}': {e}")
        # Fallback search link
        return [{
            "title": f"Search Research Papers: {query}",
            "url": f"https://scholar.google.com/scholar?q={query.replace(' ', '+')}",
            "source": "research_paper",
            "thumbnail": None,
            "description": f"Search Google Scholar for: {query}"
        }]


def search_documentation_tavily(query: str, max_results: int = 2) -> list[dict]:
    """Search official documentation and reference guides via Tavily."""
    tavily_query = f"{query} official documentation reference guide manual"
    resources = []
    try:
        search_results = tavily_client.search(query=tavily_query, search_depth="basic")
        for r in search_results.get("results", []):
            resources.append({
                "title": r.get("title", "Official Documentation"),
                "url": r.get("url", ""),
                "source": "documentation",
                "thumbnail": None,
                "description": r.get("content", "")[:200] if r.get("content") else None
            })
        logger.info(f"Tavily found {len(resources)} documentation resources for query '{query}'")
        return resources[:max_results]
    except Exception as e:
        logger.warning(f"Tavily documentation search failed for '{query}': {e}")
        # Fallback search link
        return [{
            "title": f"Search Documentation: {query}",
            "url": f"https://duckduckgo.com/?q={query.replace(' ', '+')}+documentation",
            "source": "documentation",
            "thumbnail": None,
            "description": f"Search for documentation: {query}"
        }]

