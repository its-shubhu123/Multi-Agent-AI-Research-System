"""Two custom LangChain tools: web_search (Tavily) and scrape_url (BeautifulSoup).

Install:
    pip install langchain-core tavily-python requests beautifulsoup4 python-dotenv

Put your key in a .env file next to this script:
    TAVILY_API_KEY=tvly-...
"""

import os
import re

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.tools import tool
from tavily import TavilyClient
from rich import print

load_dotenv()  # reads TAVILY_API_KEY from a .env file in the working directory

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the internet for live, up-to-date information using Tavily. Returns titles, URLs
    and snippet.

    Args:
        query: The search query.
        max_results: Number of results to return (1-10).
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY environment variable is not set."

    try:
        client = TavilyClient(api_key=api_key)
        data = client.search(
            query=query,
            max_results=max(1, min(int(max_results), 10)),
            search_depth="basic",
            include_answer=True,
        )
    except Exception as e:
        return f"Error: Tavily request failed: {e}"

    parts = []
    if data.get("answer"):
        parts.append(f"Summary: {data['answer']}\n")

    for i, r in enumerate(data.get("results", []), 1):
        parts.append(
            f"[{i}] {r.get('title', 'Untitled')}\n"
            f"URL: {r.get('url', '')}\n"
            f"{r.get('content', '').strip()}\n"
        )

    return "\n".join(parts) if parts else "No results found."


@tool
def scrape_url(url: str, max_chars: int = 3000) -> str:
    """Visit a web page and extract its clean, readable text.

    Args:
        url: The full URL of the page (including http:// or https://).
        max_chars: Maximum number of characters of text to return.
    """
    if not url.startswith(("http://", "https://")):
        return "Error: URL must start with http:// or https://"

    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=8)
        resp.raise_for_status()
    except requests.RequestException as e:
        return f"Error: could not fetch {url}: {e}"

    soup = BeautifulSoup(resp.text, "html.parser")

    # Drop non-content elements
    for tag in soup(["script", "style", "noscript", "nav", "footer",
                     "header", "aside", "form", "iframe", "svg"]):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else ""
    text = soup.get_text(separator="\n")

    # Collapse whitespace and blank lines
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    clean = "\n".join(ln for ln in lines if ln)

    if len(clean) > max_chars:
        clean = clean[:max_chars].rstrip() + "\n...[truncated]"

    return f"Title: {title}\nURL: {url}\n\n{clean}" if clean else "No readable text found."

