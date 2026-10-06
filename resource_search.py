"""Resource search engine — combines Invidious (YouTube), Tavily Search, and DuckDuckGo for targeted learning resources."""

import httpx
import logging
import re
try:
    from duckduckgo_search import DDGS
    HAS_DDG = True
except ImportError:
    DDGS = None
    HAS_DDG = False

from tavily_helper import tavily_client

logger = logging.getLogger(__name__)

# Public Invidious instances for YouTube search without requiring API keys
INVIDIOUS_INSTANCES = [
    "https://vid.puffyan.us",
    "https://invidious.fdn.fr",
    "https://invidious.privacyredirect.com",
    "https://inv.nadeko.net",
]

def extract_youtube_video_id(url: str) -> str | None:
    """Extract YouTube Video ID from various URL patterns."""
    if not url:
        return None
    patterns = [
        r"(?:v=|\/embed\/|\/watch\?v=|\/v\/|https:\/\/youtu\.be\/|\/shorts\/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def search_youtube_invidious(query: str, max_results: int = 2) -> list[dict]:
    """Search YouTube videos via Invidious public API."""
    resources = []
    for instance in INVIDIOUS_INSTANCES:
        try:
            url = f"{instance}/api/v1/search"
            params = {
                "q": query,
                "type": "video",
                "sort_by": "relevance",
            }
            with httpx.Client(timeout=6.0) as client:
                response = client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

            for item in data[:max_results]:
                video_id = item.get("videoId", "")
                if video_id:
                    resources.append({
                        "title": item.get("title", "Untitled YouTube Tutorial"),
                        "url": f"https://www.youtube.com/watch?v={video_id}",
                        "embed_url": f"https://www.youtube.com/embed/{video_id}",
                        "source": "youtube",
                        "video_id": video_id,
                        "thumbnail": f"https://i.ytimg.com/vi/{video_id}/mqdefault.jpg",
                        "description": item.get("description", "")[:200] if item.get("description") else None,
                    })

            if resources:
                logger.info(f"Found {len(resources)} YouTube videos for '{query}' via Invidious ({instance})")
                return resources
        except Exception as e:
            logger.warning(f"Invidious instance {instance} failed for '{query}': {e}")
            continue

    return resources

def search_youtube_tavily(query: str, max_results: int = 2) -> list[dict]:
    """Search YouTube videos via Tavily API with video ID & thumbnail enrichment."""
    # First try Invidious for rich video metadata & thumbnails
    inv_results = search_youtube_invidious(query, max_results=max_results)
    if inv_results:
        return inv_results

    # Fallback to Tavily
    tavily_query = f"{query} site:youtube.com tutorial"
    resources = []
    try:
        search_results = tavily_client.search(query=tavily_query, search_depth="basic")
        for r in search_results.get("results", []):
            url = r.get("url", "")
            video_id = extract_youtube_video_id(url)
            if "youtube.com/watch" in url or "youtu.be/" in url or video_id:
                vid_id = video_id or "dQw4w9WgXcQ"
                resources.append({
                    "title": r.get("title", "YouTube Video Tutorial"),
                    "url": url,
                    "embed_url": f"https://www.youtube.com/embed/{vid_id}" if video_id else url,
                    "source": "youtube",
                    "video_id": video_id,
                    "thumbnail": f"https://i.ytimg.com/vi/{video_id}/mqdefault.jpg" if video_id else None,
                    "description": r.get("content", "")[:200] if r.get("content") else None
                })

        if not resources:
            for r in search_results.get("results", [])[:max_results]:
                resources.append({
                    "title": r.get("title", "Resource Tutorial"),
                    "url": r.get("url", ""),
                    "embed_url": r.get("url", ""),
                    "source": "youtube",
                    "thumbnail": None,
                    "description": r.get("content", "")[:200] if r.get("content") else None
                })
        return resources[:max_results]
    except Exception as e:
        logger.warning(f"Tavily YouTube search failed for '{query}': {e}")
        return [{
            "title": f"Search YouTube: {query}",
            "url": f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}",
            "embed_url": f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}",
            "source": "youtube",
            "thumbnail": None,
            "description": f"Search YouTube for: {query}"
        }]

def search_web_duckduckgo(query: str, max_results: int = 2) -> list[dict]:
    """Search web tutorial articles via DuckDuckGo."""
    if not HAS_DDG or DDGS is None:
        logger.warning("duckduckgo_search package is not available.")
        return []

    resources = []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(f"{query} tutorial guide", max_results=max_results))

        for item in results:
            resources.append({
                "title": item.get("title", "Web Tutorial Guide"),
                "url": item.get("href", ""),
                "source": "web",
                "thumbnail": None,
                "description": item.get("body", "")[:200] if item.get("body") else None,
            })
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed for '{query}': {e}")
    return resources

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
        return resources[:max_results]
    except Exception as e:
        logger.warning(f"Tavily research paper search failed for '{query}': {e}")
        return [{
            "title": f"Search Research Papers: {query}",
            "url": f"https://scholar.google.com/scholar?q={query.replace(' ', '+')}",
            "source": "research_paper",
            "thumbnail": None,
            "description": f"Search Google Scholar for: {query}"
        }]

def search_documentation_tavily(query: str, max_results: int = 2) -> list[dict]:
    """Search official documentation and reference guides via Tavily / DuckDuckGo."""
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
        if resources:
            return resources[:max_results]
    except Exception as e:
        logger.warning(f"Tavily documentation search failed for '{query}': {e}")

    # Fallback to DuckDuckGo
    ddg_res = search_web_duckduckgo(f"{query} documentation reference", max_results=max_results)
    if ddg_res:
        return ddg_res

    return [{
        "title": f"Search Documentation: {query}",
        "url": f"https://duckduckgo.com/?q={query.replace(' ', '+')}+documentation",
        "source": "documentation",
        "thumbnail": None,
        "description": f"Search for documentation: {query}"
    }]
