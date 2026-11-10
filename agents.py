import re
from typing import List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from tools import scrape_url, web_search

load_dotenv()

# model setup
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")


# 1st agent
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt=(
            "You are a search agent. Use the web_search tool to find current, relevant "
            "information on the user's topic. Run more than one query if results are thin. "
            "Return the best sources as a list, each with its title, the full URL exactly "
            "as returned by the tool, and a one-line note on what it covers. "
            "Never invent or shorten URLs."
        ),
    )


# 2nd agent
def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt=(
            "You are a reader agent. Use the scrape_url tool to open the URLs you are "
            "given and read them. Summarize what the pages say as clear bullet points, "
            "and put the source URL in brackets after each point. Only use information "
            "from the scraped text. If a page fails to load, say so and move on."
        ),
    )


# --- link extraction helpers -------------------------------------------------------
_URL_RE = re.compile(r"https?://[^\s<>\")\]]+")


def get_text(result: dict) -> str:
    """Final message text of an agent result."""
    content = result["messages"][-1].content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return content


def extract_urls(result: dict) -> List[str]:
    """Collect URLs from an agent result, de-duplicated and in order.

    Reads the raw tool messages first (web_search output, which always has the real
    URLs), then the agent's final answer, so links survive even if the model
    forgets to repeat them.
    """
    texts = [str(m.content) for m in result["messages"] if getattr(m, "type", "") == "tool"]
    texts.append(get_text(result))

    urls: List[str] = []
    for text in texts:
        for url in _URL_RE.findall(text):
            url = url.rstrip(".,;:)")
            if url not in urls:
                urls.append(url)
    return urls


def build_research(findings: str, urls: List[str]) -> str:
    """Append the collected links to the research text so the writer always sees them."""
    if not urls:
        return findings
    return findings + "\n\nSource URLs:\n" + "\n".join(f"- {u}" for u in urls)


# writer chain
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list every URL from the 'Source URLs' section and any other URL found in the research, copied exactly)

If the research contains URLs, never say that no URLs were provided.
Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()

# critic chain
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()