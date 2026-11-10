import streamlit as st

from pipeline import run_research_pipeline

st.set_page_config(page_title="AI Research Assistant", page_icon="🔎", layout="wide")

STEP_NAMES = {
    1: "Search agent",
    2: "Reader agent",
    3: "Writer",
    4: "Critic",
}

# ---------- sidebar ----------
with st.sidebar:
    st.header("About")
    st.write(
        "This app runs a 4-step research pipeline:\n\n"
        "1. **Search agent** finds recent sources\n"
        "2. **Reader agent** scrapes the best one\n"
        "3. **Writer** drafts a report\n"
        "4. **Critic** reviews the report"
    )
    if st.button("Clear results", use_container_width=True):
        st.session_state.pop("result", None)
        st.session_state.pop("topic", None)
        st.rerun()

# ---------- main ----------
st.title("🔎 AI Research Assistant")
st.caption("Enter a topic and let the agents research it for you.")

with st.form("topic_form"):
    topic = st.text_input(
        "Research topic",
        placeholder="e.g. Latest advances in solid-state batteries",
    )
    submitted = st.form_submit_button("Run research", type="primary")

if submitted:
    if not topic.strip():
        st.warning("Please enter a topic first.")
    else:
        st.session_state.pop("result", None)
        st.session_state["topic"] = topic.strip()

        st.subheader("Progress")
        status_boxes = {}

        def on_step(step, label, output):
            name = STEP_NAMES[step]
            if output is None:
                # step started
                status_boxes[step] = st.status(f"Step {step}: {name} - {label}...", state="running", expanded=False)
            else:
                # step finished
                box = status_boxes[step]
                box.update(label=f"Step {step}: {name} - {label}", state="complete", expanded=False)
                with box:
                    st.markdown(output)

        try:
            st.session_state["result"] = run_research_pipeline(topic.strip(), on_step=on_step)
        except Exception as e:
            st.error(f"Pipeline failed: {e}")

# ---------- results ----------
result = st.session_state.get("result")
if result:
    st.divider()
    st.subheader(f"Results for: {st.session_state.get('topic', '')}")

    tab_report, tab_feedback, tab_raw = st.tabs(["📄 Report", "🧐 Critic feedback", "🗂 Raw research"])

    with tab_report:
        st.markdown(result["report"])
        st.download_button(
            "Download report (.md)",
            data=result["report"],
            file_name="research_report.md",
            mime="text/markdown",
        )

    with tab_feedback:
        st.markdown(result["feedback"])

    with tab_raw:
        with st.expander("Search results", expanded=False):
            st.markdown(result["search_results"])
        with st.expander("Scraped content", expanded=False):
            st.markdown(result["scraped_content"])