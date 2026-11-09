"""Two Gemini-powered agents plus two LCEL chains (prompt | llm | parser).

Install:
    pip install langchain langchain-core langchain-google-genai python-dotenv pydantic

.env should contain:
    GOOGLE_API_KEY=...
    TAVILY_API_KEY=tvly-...
    GEMINI_MODEL=gemini-2.5-flash   # optional, change to any Gemini model you have access to
"""

import os
from typing import List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from tools import scrape_url, web_search

load_dotenv()

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def _llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=temperature,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


# --- Agent 1: finds relevant sources on the web ---------------------------------
search_agent = create_agent(
    model=_llm(),
    tools=[web_search],
    system_prompt=(
        "You are a search agent. Use the web_search tool to find current, relevant "
        "information for the user's topic. Run more than one query if the first "
        "results are thin. Return a short list of the best sources, each with its "
        "title, URL and a one-line note on why it is useful. Do not invent URLs."
    ),
)

# --- Agent 2: reads a page and writes up what it says ---------------------------
reader_agent = create_agent(
    model=_llm(),
    tools=[scrape_url],
    system_prompt=(
        "You are a reader agent. Use the scrape_url tool to open the URLs you are "
        "given and read their content. Then write a clear, factual summary of what "
        "the pages say, with key points as bullets. Only use information from the "
        "scraped text. If a page fails to load, say so and move on."
    ),
)


# --- Output schemas --------------------------------------------------------------
class WriterOutput(BaseModel):
    title: str = Field(description="A concise title for the piece")
    article: str = Field(description="The full written article in clear prose")
    key_points: List[str] = Field(description="3-5 key takeaways")


class ReaderOutput(BaseModel):
    summary: str = Field(description="A short, factual summary of the text")
    key_points: List[str] = Field(description="3-6 key points from the text")
    topics: List[str] = Field(description="Main topics or keywords covered")


writer_parser = PydanticOutputParser(pydantic_object=WriterOutput)
reader_parser = PydanticOutputParser(pydantic_object=ReaderOutput)

# --- Prompts ---------------------------------------------------------------------
writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a skilled writer. Using only the research notes provided, write "
            "a well-structured, engaging article on the topic.\n{format_instructions}",
        ),
        ("human", "Topic: {topic}\n\nResearch notes:\n{notes}"),
    ]
).partial(format_instructions=writer_parser.get_format_instructions())

reader_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a careful reader. Read the text provided and extract a faithful "
            "summary. Use only information from the text.\n{format_instructions}",
        ),
        ("human", "Text to read:\n{text}"),
    ]
).partial(format_instructions=reader_parser.get_format_instructions())

# --- Chains (pipe operator) -------------------------------------------------------
writer_chain = writer_prompt | _llm(temperature=0.7) | writer_parser
reader_chain = reader_prompt | _llm(temperature=0.1) | reader_parser
                   