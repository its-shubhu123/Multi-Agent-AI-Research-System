from agents import (
    build_reader_agent,
    build_research,
    build_search_agent,
    critic_chain,
    extract_urls,
    get_text,
    writer_chain,
)

# (step number -> label shown in the UI)
STEP_LABELS = {
    1: "finding sources",
    2: "scraping top resources",
    3: "drafting the report",
    4: "reviewing the report",
}


def run_research_pipeline(topic: str, on_step=None) -> dict:
    """Run search -> reader -> writer -> critic, keeping everything in one state dict.

    on_step: optional callback with the signature on_step(step, label, output):
      - called as on_step(n, label, None) when step n (1-4) starts
      - called as on_step(n, label, output_text) when it finishes
    """
    state = {"topic": topic}

    def started(n: int):
        print("\n" + "=" * 50)
        print(f"step {n} - {STEP_LABELS[n]} ...")
        print("=" * 50)
        if on_step:
            on_step(n, STEP_LABELS[n], None)

    def finished(n: int, output: str):
        if on_step:
            on_step(n, STEP_LABELS[n], output)

    # step 1 - search agent
    started(1)
    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
    })
    state["search_results"] = get_text(search_result)
    state["urls"] = extract_urls(search_result)  # links taken from the raw tool output

    print("\nsearch result:\n", state["search_results"])
    print("\nlinks found:")
    for u in state["urls"]:
        print("  -", u)

    links_md = "\n".join(f"- {u}" for u in state["urls"]) or "_No links found._"
    finished(1, f"{state['search_results']}\n\n**Links found:**\n{links_md}")

    # step 2 - reader agent
    started(2)
    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages": [("user",
            f"Based on the following search results about '{topic}', "
            f"pick the most relevant URLs (up to 2) and scrape them for deeper content.\n\n"
            f"URLs:\n" + "\n".join(state["urls"]) + "\n\n"
            f"Search Results:\n{state['search_results']}"
        )]
    })
    state["scraped_content"] = get_text(reader_result)
    # also keep any extra links the reader touched
    for u in extract_urls(reader_result):
        if u not in state["urls"]:
            state["urls"].append(u)

    print("\nscraped content:\n", state["scraped_content"])
    finished(2, state["scraped_content"])

    # step 3 - writer chain
    started(3)
    research_combined = build_research(
        f"SEARCH RESULTS:\n{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}",
        state["urls"],
    )

    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined,
    })

    print("\nfinal report:\n", state["report"])
    finished(3, state["report"])

    # step 4 - critic chain
    started(4)
    state["feedback"] = critic_chain.invoke({"report": state["report"]})

    print("\ncritic report:\n", state["feedback"])
    finished(4, state["feedback"])

    return state


if __name__ == "__main__":
    topic = input("\nEnter a research topic: ")
    run_research_pipeline(topic)