import os
from pathlib import Path

import pandas as pd
import streamlit as st

from agentic_engine import AgentConfig, AgenticEngine

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "sample_docs"
CHROMA_DIR = PROJECT_DIR / "data" / "chroma_db"

st.set_page_config(
    page_title="Agentic RAG: Autonomous AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.05rem; color: #4B5563; margin-bottom: 1.5rem; }
    .plan-box { background-color: #EFF6FF; border-left: 4px solid #3B82F6; padding: 14px; border-radius: 6px; margin: 12px 0; }
    .tool-tag { background-color: #E0E7FF; color: #3730A3; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 0.85rem; }
    .tool-obs-box { background-color: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 6px; padding: 12px; margin: 8px 0; font-family: monospace; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Initializing Agentic RAG Engine & Vector Store...")
def get_agent_engine() -> AgenticEngine:
    config = AgentConfig(
        data_dir=DATA_DIR,
        chroma_dir=CHROMA_DIR,
    )
    return AgenticEngine(config)


engine = get_agent_engine()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bot.png", width=64)
    st.title("Agent Mission Control")
    st.caption("Agentic RAG with Autonomous Tool Dispatching")

    st.markdown("---")
    st.subheader("1. LLM Reasoning Engine")
    llm_choice = st.radio(
        "Provider",
        ["Fast Deterministic Mode (Local)", "Google Gemini API"],
        help="Gemini API enables dynamic multi-step natural language reasoning and tool execution.",
    )

    if llm_choice == "Google Gemini API":
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="AIzaSy...",
            help="Free API key from Google AI Studio",
        )
        if api_key:
            success = engine.update_gemini_key(api_key)
            if success:
                st.success("Agent connected to Gemini! ⚡")
            else:
                st.error("Failed to connect. Check your key.")
    else:
        engine.update_gemini_key("")

    st.markdown("---")
    st.subheader("2. Agent Tool Registry")
    st.markdown(
        """
        - 🔍 `document_search`: Semantic Vector Search
        - 🧮 `calculator`: Precise Math & Ratios
        - 📄 `document_summary`: Document Overview
        - ⚖️ `comparative_analyzer`: Multi-Entity Comparison
        - 📊 `table_data_analyzer`: CSV Table Querying
        """
    )

    st.markdown("---")
    st.subheader("3. Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload Docs or Datasets (PDF, DOCX, PPTX, TXT, CSV)",
        type=["pdf", "docx", "pptx", "txt", "csv"],
        accept_multiple_files=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Ingest Uploads", type="primary", use_container_width=True):
            if uploaded_files:
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                saved_paths = []
                for file in uploaded_files:
                    target_path = DATA_DIR / file.name
                    target_path.write_bytes(file.read())
                    saved_paths.append(target_path)
                with st.spinner("Processing & indexing..."):
                    report = engine.ingest_files(saved_paths)
                st.success(f"Indexed {report['chunks']} chunks.")
                st.rerun()
            else:
                st.warning("Upload files first.")

    with col2:
        if st.button("Load Samples", use_container_width=True):
            with st.spinner("Ingesting academic & data samples..."):
                report = engine.ingest_files()
            st.success(f"Indexed {report['chunks']} chunks.")
            st.rerun()

    # Metrics
    chunk_count = engine.collection.count()
    st.metric("Indexed Knowledge Chunks", chunk_count)
    st.metric("Active CSV Data Tables", len(engine.dataframes))
    if chunk_count == 0:
        st.info("Click 'Load Samples' to populate the knowledge base.")


# --- MAIN PAGE ---
st.markdown('<div class="main-header">🤖 Agentic RAG: Autonomous Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Understands complex questions, plans retrieval strategies, '
    'invokes specialized tools (Search, Math, Summarization, Tabular Analysis), and synthesizes grounded answers.</div>',
    unsafe_allow_html=True,
)

# Quick Prompts
st.markdown("**💡 Try these multi-hop and tool-requiring queries:**")
q_cols = st.columns(4)
sample_q = ""
with q_cols[0]:
    if st.button("📊 Cloud Table (Data)", use_container_width=True):
        sample_q = "What is the cheapest cloud GPU instance in the pricing table, what is its hourly cost, and what GPU does it offer?"
with q_cols[1]:
    if st.button("⚖️ H100 vs A100 (Compare)", use_container_width=True):
        sample_q = "Compare NVIDIA H100 vs NVIDIA A100 architecture, tensor cores, and memory bandwidth."
with q_cols[2]:
    if st.button("🧮 Compute Speedup (Math)", use_container_width=True):
        sample_q = "If H100 delivers 989 TFLOPS in FP16 and A100 delivers 312 TFLOPS, calculate the percentage increase in compute throughput."
with q_cols[3]:
    if st.button("🔍 RAG Limits (Search)", use_container_width=True):
        sample_q = "What are the core failure modes and limitations of Naive RAG in enterprise applications?"

with st.form("agent_form"):
    user_query = st.text_area(
        "Enter your query:",
        value=sample_q,
        height=90,
        placeholder="Ask a complex question, request a comparison, calculation, or table lookup...",
    )
    submit_btn = st.form_submit_button("Launch Agent 🚀", type="primary")

if submit_btn and user_query.strip():
    if engine.collection.count() == 0 and not engine.dataframes:
        st.error("The knowledge base is empty! Please click 'Load Samples' in the sidebar first.")
    else:
        with st.spinner("Agent is reasoning, planning, and executing tools..."):
            trace = engine.run_agentic_pipeline(user_query)

        st.markdown("---")

        # --- STEP 1: AGENT'S PLAN ---
        st.subheader("1. 📋 Agent's Decomposed Plan")
        st.markdown(
            f'<div class="plan-box">'
            f'<b>Goal:</b> <i>"{trace["query"]}"</i><br>'
            f'<b>Execution Time:</b> {trace["execution_time"]}s<br><br>'
            f'<b>Planned Steps:</b><br>'
            + "<br>".join(f"• <b>Step {p['step']}</b>: Use <span class='tool-tag'>{p['tool']}</span> ➡️ {p['intent']}" for p in trace["plan"])
            + '</div>',
            unsafe_allow_html=True,
        )

        # --- STEP 2: TOOL EXECUTION TIMELINE (ReAct Scratchpad) ---
        st.subheader("2. 🛠️ Tool Execution & Observations (ReAct Trace)")
        for step in trace["steps"]:
            with st.expander(f"Step {step['step']}: Tool `{step['tool']}` ➡️ {step['intent']}", expanded=True):
                st.markdown(f"**Arguments passed:** `{step['args']}`")
                st.markdown("**Observation / Result:**")
                st.markdown(f'<div class="tool-obs-box">{step["observation"]}</div>', unsafe_allow_html=True)

        # --- STEP 3: FINAL SYNTHESIZED ANSWER ---
        st.subheader("3. 📝 Final Synthesized Answer")
        st.markdown(trace["final_answer"])

        # --- STEP 4: SOURCE CITATIONS ---
        if trace["sources"]:
            st.subheader("4. 📚 Grounded Source References")
            src_df = pd.DataFrame(trace["sources"])
            st.dataframe(
                src_df,
                column_config={
                    "source": "Document / Dataset",
                    "page": "Page / Slide",
                    "similarity": st.column_config.ProgressColumn("Similarity", min_value=0.0, max_value=1.0, format="%.2f"),
                },
                hide_index=True,
                use_container_width=True,
            )
