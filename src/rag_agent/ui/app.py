"""
app.py
======
Streamlit user interface for the Deep Learning RAG Interview Prep Agent.

Three-panel layout:
  - Left sidebar: Document ingestion and corpus browser
  - Centre: Document viewer
  - Right: Chat interface

API contract with the backend (agree this with Pipeline Engineer
before building anything):

  ingest(file_paths: list[Path]) -> IngestionResult
  list_documents() -> list[dict]
  get_document_chunks(source: str) -> list[DocumentChunk]
  chat(query: str, history: list[dict], filters: dict) -> AgentResponse

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import streamlit as st
from langchain_core.messages import HumanMessage

from rag_agent.agent.graph import get_compiled_graph
from rag_agent.agent.state import AgentResponse
from rag_agent.config import get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager

import json
from langchain_core.messages import HumanMessage
from rag_agent.agent.prompts import QUESTION_GENERATION_PROMPT, ANSWER_EVALUATION_PROMPT
from rag_agent.config import LLMFactory, get_settings


# ---------------------------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------------------------
# Use st.cache_resource for objects that should persist across reruns
# and be shared across all user sessions. This prevents re-initialising
# ChromaDB and reloading the embedding model on every button click.


@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    """
    Return the singleton VectorStoreManager.

    Cached so ChromaDB connection is initialised once per application
    session, not on every Streamlit rerun.
    """
    return VectorStoreManager()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    """Return the singleton DocumentChunker."""
    return DocumentChunker()


@st.cache_resource
def get_graph():
    """Return the compiled LangGraph agent."""
    return get_compiled_graph()


# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------


def initialise_session_state() -> None:
    """
    Initialise all st.session_state keys on first run.

    Must be called at the top of main() before any UI is rendered.
    Without this, state keys referenced in callbacks will raise KeyError.

    Interview talking point: Streamlit reruns the entire script on every
    user interaction. session_state is the mechanism for persisting data
    (chat history, ingestion results) across reruns.
    """
    defaults = {
        "chat_history": [],           # list of {"role": "user"|"assistant", "content": str}
        "ingested_documents": [],     # list of dicts from list_documents()
        "selected_document": None,    # source filename currently in viewer
        "last_ingestion_result": None,
        "thread_id": "default-session",  # LangGraph conversation thread
        "topic_filter": None,
        "difficulty_filter": None,
        "uploader_key": 0, ## Reset logic: added this to keep up with cleaning the stack of uploaded files
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Ingestion Panel (Sidebar)
# ---------------------------------------------------------------------------


def render_ingestion_panel(
    store: VectorStoreManager,
    chunker: DocumentChunker,
) -> None:
    """
    Render the document ingestion panel in the sidebar.

    Allows multi-file upload of PDF and Markdown files. Displays
    ingestion results (chunks added, duplicates skipped, errors).
    Updates the ingested documents list after successful ingestion.

    Parameters
    ----------
    store : VectorStoreManager
    chunker : DocumentChunker
    """
    st.sidebar.header("📂 Corpus Ingestion")

    # TODO: implement
    # 1. st.sidebar.file_uploader(
    #        "Upload study materials",
    #        type=["pdf", "md"],
    #        accept_multiple_files=True
    #    )
    #
    # 2. "Ingest Documents" button — only enabled when files are selected
    #
    # 3. On button click:
    #    a. Save uploaded files to a temp directory
    #    b. chunker.chunk_files(file_paths)
    #    c. store.ingest(chunks) → IngestionResult
    #    d. Display result: st.success / st.warning / st.error
    #       Show: "{result.ingested} chunks added, {result.skipped} duplicates skipped"
    #    e. Refresh ingested documents list in session_state
    #
    # 4. Render ingested documents list below the uploader
    #    For each document: show source name, topic, chunk count
    #    Add a small "🗑 Remove" button per document that calls store.delete_document()

    st.sidebar.info("Upload .pdf or .md files to populate the corpus.")
    uploaded_files = st.sidebar.file_uploader(
        "Upload study materials",
        type=["pdf", "md"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}" # NEW RESET LOGIC
    )

    if st.sidebar.button("Ingest Documents", disabled=not uploaded_files):
        with st.spinner("Processing files..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                file_paths = []
                for uploaded_file in uploaded_files:
                    tmp_path = Path(tmpdir) / uploaded_file.name
                    tmp_path.write_bytes(uploaded_file.getvalue())
                    file_paths.append(tmp_path)
                
                try:
                    chunks = chunker.chunk_files(file_paths)
                    result = store.ingest(chunks)
                    
                    st.sidebar.success(f"✅ {result.ingested} added, {result.skipped} skipped.")
                    
                    st.session_state.ingested_documents = store.list_documents()
                    # --- NEW RESET LOGIC ---
                    # Increment the key to destroy the old uploader widget
                    st.session_state.uploader_key += 1
                    # Force a rerun to instantly clear the UI
                    st.rerun()
                    # -----------------------
                except Exception as e:
                    st.sidebar.error(f"Ingestion failed: {e}")

    st.sidebar.divider()
    
    st.sidebar.subheader("Ingested Files")
    docs = store.list_documents()
    
    if not docs:
        st.sidebar.caption("No documents ingested yet.")
    else:
        for doc in docs:
            col1, col2 = st.sidebar.columns([4, 1])
            with col1:
                st.caption(f"📄 **{doc['source']}** ({doc['chunk_count']} chunks)")
            with col2:
                if st.button("🗑️", key=f"del_{doc['source']}", help="Delete"):
                    store.delete_document(doc["source"])
                    st.rerun()

def render_interview_panel(store) -> None:
    """Render a dedicated mock interview panel in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Mock Interview Mode")
    
    settings = get_settings()
    llm = LLMFactory(settings).create()

    # Button to generate a question
    if st.sidebar.button("Generate Interview Question"):
        with st.sidebar.status("Analyzing corpus..."):
            # 1. Grab a random relevant chunk (we'll just query a broad term)
            chunks = store.query("deep learning neural networks")
            
            if not chunks:
                st.sidebar.error("Ingest some documents first!")
                return
                
            context = chunks[0].chunk_text
            
            # 2. Format and send the QUESTION prompt
            prompt = QUESTION_GENERATION_PROMPT.format(
                context=context, 
                difficulty="intermediate"
            )
            
            try:
                # Force JSON response
                response = llm.invoke([HumanMessage(content=prompt)])
                
                # Strip markdown code blocks if the LLM added them
                clean_json = response.content.replace("```json", "").replace("```", "").strip()
                q_data = json.loads(clean_json)
                
                # 3. Save to session state so we can answer it
                st.session_state.current_question = q_data["question"]
                st.session_state.question_context = context
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Failed to generate: {e}")

    # Display the generated question and answer box
    if st.session_state.get("current_question"):
        st.sidebar.info(f"**Question:** {st.session_state.current_question}")
        
        user_answer = st.sidebar.text_area("Your Answer:")
        if st.sidebar.button("Submit Answer"):
            with st.sidebar.status("Grading..."):
                # Format and send the EVALUATION prompt
                eval_prompt = ANSWER_EVALUATION_PROMPT.format(
                    question=st.session_state.current_question,
                    candidate_answer=user_answer,
                    context=st.session_state.question_context
                )
                
                try:
                    eval_response = llm.invoke([HumanMessage(content=eval_prompt)])
                    clean_eval = eval_response.content.replace("```json", "").replace("```", "").strip()
                    grade_data = json.loads(clean_eval)
                    
                    # Display the scorecard!
                    st.sidebar.metric("Score", f"{grade_data['score']}/10")
                    st.sidebar.success(f"**What you got right:** {grade_data['what_was_correct']}")
                    st.sidebar.warning(f"**What was missing:** {grade_data['what_was_missing']}")
                    st.sidebar.write(f"**Verdict:** {grade_data['interview_verdict'].upper()}")
                    
                except Exception as e:
                    st.sidebar.error(f"Grading failed: {e}")

def render_corpus_stats(store: VectorStoreManager) -> None:
    """
    Render a compact corpus health summary in the sidebar.

    Shows total chunks, topics covered, and whether bonus topics
    are present. Used during Hour 3 to demonstrate corpus completeness.

    Parameters
    ----------
    store : VectorStoreManager
    """
    # TODO: implement
    # stats = store.get_collection_stats()
    # st.sidebar.metric("Total Chunks", stats["total_chunks"])
    # st.sidebar.write("Topics:", ", ".join(stats["topics"]))
    # if stats["bonus_topics_present"]:
    #     st.sidebar.success("✅ Bonus topics present")
    # else:
    #     st.sidebar.warning("⚠️ No bonus topics yet")
    st.sidebar.divider()
    st.sidebar.subheader("📊 Corpus Stats")
    
    stats = store.get_collection_stats()
    if not stats or stats.get("total_chunks", 0) == 0:
        st.sidebar.caption("Corpus is empty.")
        return

    col1, col2 = st.sidebar.columns(2)
    col1.metric("Total Chunks", stats["total_chunks"])
    col2.metric("Unique Docs", len(stats.get("sources", [])))
    
    st.sidebar.caption(f"**Topics:** {', '.join(stats['topics'])}")
    
    if stats.get("bonus_topics_present"):
        st.sidebar.success("✅ Bonus topics present")
    else:
        st.sidebar.warning("⚠️ No bonus topics yet")


# ---------------------------------------------------------------------------
# Document Viewer Panel (Centre)
# ---------------------------------------------------------------------------


def render_document_viewer(store: VectorStoreManager) -> None:
    """
    Render the document viewer in the main centre column.

    Displays a selectable list of ingested documents. When a document
    is selected, renders its chunk content in a scrollable pane.

    Parameters
    ----------
    store : VectorStoreManager
    """
    st.subheader("📄 Document Viewer")

    # TODO: implement
    # 1. If no documents ingested: show placeholder message
    #
    # 2. st.selectbox("Select document", options=[doc["source"] for doc in docs])
    #    Store selection in st.session_state["selected_document"]
    #
    # 3. On selection change: store.get_document_chunks(selected_source)
    #
    # 4. Render chunks in a scrollable container (st.container with fixed height)
    #    For each chunk:
    #    - Show metadata badge: topic | difficulty | type
    #    - Show chunk text
    #    - Show similarity score if this chunk was used in last response
    #
    # 5. Display chunk count and coverage summary below viewer

    docs = store.list_documents()
    if not docs:
        st.info("Ingest documents using the sidebar to view content here.")
        return

    doc_names = [doc["source"] for doc in docs]
    
    selected_doc = st.selectbox(
        "Select document", 
        options=doc_names,
        index=0 if st.session_state.selected_document not in doc_names else doc_names.index(st.session_state.selected_document)
    )
    st.session_state.selected_document = selected_doc

    if selected_doc:
        chunks = store.get_document_chunks(selected_doc)
        
        with st.container(height=600):
            for i, chunk in enumerate(chunks):
                with st.expander(f"Chunk {i+1} | Topic: {chunk.metadata.topic}", expanded=False):
                    st.markdown(f"**Difficulty:** `{chunk.metadata.difficulty}`")
                    st.divider()
                    st.markdown(chunk.chunk_text)


# ---------------------------------------------------------------------------
# Chat Interface Panel (Right)
# ---------------------------------------------------------------------------


def render_chat_interface(graph) -> None:
    """
    Render the chat interface in the right column.

    Supports multi-turn conversation with the LangGraph agent.
    Displays source citations with every response.
    Shows a clear "no relevant context" indicator when the
    hallucination guard fires.

    Parameters
    ----------
    graph : CompiledStateGraph
        The compiled LangGraph agent from get_compiled_graph().
    """
    st.subheader("💬 Interview Prep Chat")

    # Filters
    col_topic, col_diff = st.columns(2)
    with col_topic:
        st.session_state.topic_filter = st.selectbox(
            "Filter by Topic", 
            options=["All", "ANN", "CNN", "RNN", "LSTM"]
        )
    with col_diff:
        st.session_state.difficulty_filter = st.selectbox(
            "Filter by Difficulty", 
            options=["All", "beginner", "intermediate", "advanced"]
        )

    # Chat history display
    chat_container = st.container(height=520)
    # with chat_container:
    #     for message in st.session_state.chat_history:
    #         with st.chat_message(message["role"]):
    #             st.markdown(message["content"])
    #             if message.get("sources"):
    #                 with st.expander("📎 Sources"):
    #                     for source in message["sources"]:
    #                         st.caption(source)
    #             if message.get("no_context_found"):
    #                 st.warning("⚠️ No relevant content found in corpus.")

    with chat_container:
        if not st.session_state.chat_history:
            st.caption("Start the interview by asking a deep learning question.")
            
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                if message.get("sources"):
                    with st.expander("📎 Sources"):
                        for source in message["sources"]:
                            st.caption(source)
                            
                if message.get("no_context"):
                    st.warning("⚠️ Guardrail Triggered: No relevant content found.")

    # Chat input
    # TODO: implement
    # 1. query = st.chat_input("Ask about a deep learning topic...")
    #
    # 2. On submit:
    #    a. Append user message to chat_history
    #    b. Display user message immediately (st.rerun or direct render)
    #    c. Build LangGraph input:
    #       {"messages": [HumanMessage(content=query)]}
    #    d. config = {"configurable": {"thread_id": st.session_state.thread_id}}
    #    e. result = graph.invoke(input, config=config)
    #    f. response = result["final_response"]
    #    g. Append assistant message with answer, sources, no_context_found flag
    #
    # STRETCH GOAL — streaming:
    # Replace graph.invoke with graph.stream() and use st.write_stream()
    # to display tokens as they arrive. Significant "wow factor" in Hour 3.
    
    if query := st.chat_input("Ask about a deep learning topic..."):
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        with chat_container:
            with st.chat_message("user"):
                st.markdown(query)
                
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    inputs = {
                        "messages": [HumanMessage(content=query)],
                        "topic_filter": None if st.session_state.topic_filter == "All" else st.session_state.topic_filter,
                        "difficulty_filter": None if st.session_state.difficulty_filter == "All" else st.session_state.difficulty_filter
                    }
                    config = {"configurable": {"thread_id": st.session_state.thread_id}}
                    
                    try:
                        result = graph.invoke(inputs, config=config)
                        response = result.get("final_response", {})
                        
                        # Handle LangGraph's state serialization safely
                        if isinstance(response, dict):
                            ans = response.get("answer", "I encountered an error.")
                            srcs = response.get("sources", [])
                            no_ctx = response.get("no_context_found", False)
                        else:
                            ans = getattr(response, "answer", "I encountered an error.")
                            srcs = getattr(response, "sources", [])
                            no_ctx = getattr(response, "no_context_found", False)
                            
                        st.markdown(ans)
                        if srcs:
                            with st.expander("📎 Sources"):
                                for source in srcs:
                                    st.caption(source)
                        if no_ctx:
                            st.warning("⚠️ Guardrail Triggered: No relevant content found.")
                            
                        st.session_state.chat_history.append({
                            "role": "assistant", 
                            "content": ans,
                            "sources": srcs,
                            "no_context": no_ctx
                        })
                    except Exception as e:
                        st.error(f"Agent error: {e}")
# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Application entry point.

    Sets page config, initialises session state, instantiates shared
    resources, and renders all UI panels.

    Run with: uv run streamlit run src/rag_agent/ui/app.py
    """
    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_title,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(f"🧠 {settings.app_title}")
    st.caption(
        "RAG-powered interview preparation — built with LangChain, LangGraph, and ChromaDB"
    )

    initialise_session_state()

    # Instantiate shared backend resources
    store = get_vector_store()
    chunker = get_chunker()
    graph = get_graph()

    # Sidebar
    render_ingestion_panel(store, chunker)
    render_interview_panel(store)
    render_corpus_stats(store)

    # Main content area — two columns
    viewer_col, chat_col = st.columns([1, 1], gap="large")

    with viewer_col:
        render_document_viewer(store)

    with chat_col:
        render_chat_interface(graph)


if __name__ == "__main__":
    main()
