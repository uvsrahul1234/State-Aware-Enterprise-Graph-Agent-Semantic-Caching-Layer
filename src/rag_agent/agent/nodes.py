"""
nodes.py
========
LangGraph node functions for the RAG interview preparation agent.

Each function in this module is a node in the agent state graph.
Nodes receive the current AgentState, perform their operation,
and return a dict of state fields to update.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from loguru import logger
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages

from rag_agent.agent.prompts import (
    QUESTION_GENERATION_PROMPT,
    SYSTEM_PROMPT,
)
from rag_agent.agent.state import AgentResponse, AgentState, RetrievedChunk
from rag_agent.config import LLMFactory, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Node: Query Rewriter
# ---------------------------------------------------------------------------


def query_rewrite_node(state: AgentState) -> dict:
    """
    Rewrite the user's query to maximise retrieval effectiveness.

    Natural language questions are often poorly suited for vector
    similarity search. This node rephrases the query into a form
    that produces better embedding matches against the corpus.

    Example
    -------
    Input:  "I'm confused about how LSTMs remember things long-term"
    Output: "LSTM long-term memory cell state forget gate mechanism"

    Interview talking point: query rewriting is a production RAG pattern
    that significantly improves retrieval recall. It acknowledges that
    users do not phrase queries the way documents are written.

    Parameters
    ----------
    state : AgentState
        Current graph state. Reads: messages (for context).

    Returns
    -------
    dict
        Updates: original_query, rewritten_query.
    """
    # TODO: implement
    # 1. Extract the latest HumanMessage from state.messages as original_query
    # 2. Build a short prompt instructing the LLM to rewrite for vector search
    #    Keep the rewriting prompt lightweight — this adds latency
    # 3. Call llm.invoke() with the rewrite prompt
    # 4. Return {"original_query": original_query, "rewritten_query": rewritten}
    #
    # Fallback: if rewriting fails (API error, timeout), return the original
    # query unchanged so the graph continues gracefully
    settings = get_settings()
    llm = LLMFactory(settings).create()

    # 1. Extract the latest HumanMessage from state.messages as original_query
    messages = state.get("messages", [])
    if not messages:
        return {"original_query": "", "rewritten_query": ""}
        
    last_message = messages[-1]
    original_query = last_message.content

    # 2. Build a short prompt instructing the LLM to rewrite for vector search
    rewrite_prompt = (
        "You are an expert search query optimizer for a vector database containing "
        "Deep Learning study materials.\n"
        "Convert the user's conversational input into a dense, keyword-rich search "
        "query optimized for cosine similarity retrieval.\n"
        "Focus on core ML concepts (e.g., ANN, CNN, RNN, backpropagation).\n"
        "Respond with ONLY the optimized query string. No preamble or explanations.\n\n"
        f"User Input: {original_query}"
    )

    try:
        # 3. Call llm.invoke() with the rewrite prompt
        response = llm.invoke([HumanMessage(content=rewrite_prompt)])
        rewritten_query = response.content.strip()
        logger.info(f"Query rewritten: '{original_query}' -> '{rewritten_query}'")
        
        # 4. Return {"original_query": original_query, "rewritten_query": rewritten}
        return {
            "original_query": original_query, 
            "rewritten_query": rewritten_query
        }
    except Exception as e:
        logger.error(f"Query rewrite failed: {e}. Falling back to original query.")
        # Fallback: return the original query unchanged
        return {
            "original_query": original_query, 
            "rewritten_query": original_query
        }


# ---------------------------------------------------------------------------
# Node: Retriever
# ---------------------------------------------------------------------------


def retrieval_node(state: AgentState) -> dict:
    """
    Retrieve relevant chunks from ChromaDB based on the rewritten query.

    Sets the no_context_found flag if no chunks meet the similarity
    threshold. This flag is checked by generation_node to trigger
    the hallucination guard.

    Interview talking point: separating retrieval into its own node
    makes it independently testable and replaceable — you could swap
    ChromaDB for Pinecone or Weaviate by changing only this node.

    Parameters
    ----------
    state : AgentState
        Current graph state.
        Reads: rewritten_query, topic_filter, difficulty_filter.

    Returns
    -------
    dict
        Updates: retrieved_chunks, no_context_found.
    """
    # TODO: implement
    # 1. Instantiate VectorStoreManager (consider caching this)
    # 2. manager.query(
    #        query_text=state.rewritten_query,
    #        topic_filter=state.topic_filter,
    #        difficulty_filter=state.difficulty_filter
    #    )
    # 3. If result is empty: return {"retrieved_chunks": [], "no_context_found": True}
    # 4. Otherwise: return {"retrieved_chunks": chunks, "no_context_found": False}
    settings = get_settings()
    
    # 1. Instantiate VectorStoreManager
    manager = VectorStoreManager(settings)
    
    query_to_use = state.get("rewritten_query") or state.get("original_query", "")
    if not query_to_use:
        return {"retrieved_chunks": [], "no_context_found": True}

    logger.info(f"Retrieving chunks for: {query_to_use}")

    # 2. Query ChromaDB
    chunks = manager.query(
        query_text=query_to_use,
        topic_filter=state.get("topic_filter"),
        difficulty_filter=state.get("difficulty_filter")
    )

    # 3 & 4. Evaluate results and set flags
    if not chunks:
        logger.warning("No context found above similarity threshold.")
        return {"retrieved_chunks": [], "no_context_found": True}
        
    logger.info(f"Retrieved {len(chunks)} relevant chunks.")
    return {"retrieved_chunks": chunks, "no_context_found": False}


# ---------------------------------------------------------------------------
# Node: Generator
# ---------------------------------------------------------------------------


def generation_node(state: AgentState) -> dict:
    """
    Generate the final response using retrieved chunks as context.

    Implements the hallucination guard: if no_context_found is True,
    returns a clear "no relevant context" message rather than allowing
    the LLM to answer from parametric memory.

    Implements token-aware conversation memory trimming: when the
    message history approaches max_context_tokens, the oldest
    non-system messages are removed.

    Interview talking point: the hallucination guard is the most
    commonly asked about production RAG pattern. Interviewers want
    to know how you prevent the model from confidently making up
    information when the retrieval step finds nothing relevant.

    Parameters
    ----------
    state : AgentState
        Current graph state.
        Reads: retrieved_chunks, no_context_found, messages,
               original_query, topic_filter.

    Returns
    -------
    dict
        Updates: final_response, messages (with new AIMessage appended).
    """
    settings = get_settings()
    llm = LLMFactory(settings).create()

    # ---- Hallucination Guard -----------------------------------------------
    if state.get("no_context_found", False):
        no_context_message = (
            "I was unable to find relevant information in the corpus for your query. "
            "This may mean the topic is not yet covered in the study material, or "
            "your query may need to be rephrased. Please try a more specific "
            "deep learning topic such as 'LSTM forget gate' or 'CNN pooling layers'."
        )
        response = AgentResponse(
            answer=no_context_message,
            sources=[],
            confidence=0.0,
            no_context_found=True,
            rewritten_query=state.get("rewritten_query", ""),
        )
        return {
            "final_response": response,
            "messages": [AIMessage(content=no_context_message)],
        }

    # ---- Build Context from Retrieved Chunks --------------------------------
    # TODO: implement
    # 1. Format retrieved chunks into a context string with citations
    #    Each chunk should appear as: "[SOURCE: topic | file]\n{chunk_text}\n"
    # 2. Calculate average confidence score from chunk scores
    # 3. Build the full prompt:
    #    - SystemMessage with SYSTEM_PROMPT
    #    - Context message with formatted chunks
    #    - Trimmed conversation history (trim to max_context_tokens)
    #    - HumanMessage with original_query
    # 4. llm.invoke(messages)
    # 5. Construct AgentResponse with answer, sources (list of citations), confidence
    # 6. Append AIMessage to messages
    # 7. Return {"final_response": response, "messages": [new_ai_message]}
    chunks = state.get("retrieved_chunks", [])
    
    # 1. Format retrieved chunks into a context string with citations
    context_parts = []
    unique_sources = set()
    total_score = 0.0
    
    for chunk in chunks:
        citation = chunk.to_citation()
        unique_sources.add(citation)
        total_score += chunk.score
        context_parts.append(f"{citation}\n{chunk.chunk_text}\n")
        
    context_string = "\n".join(context_parts)
    
    # 2. Calculate average confidence score from chunk scores
    avg_confidence = total_score / len(chunks) if chunks else 0.0

    # 3. Build the full prompt:
    sys_message = SystemMessage(content=SYSTEM_PROMPT)
    context_message = HumanMessage(
        content=f"Here is the retrieved context from the study materials:\n\n{context_string}"
    )
    
    # Trim conversation history to avoid max token overflow
    history_messages = state.get("messages", [])[:-1] 
    if len(history_messages) > 6: 
        history_messages = history_messages[-6:]

    final_human_message = HumanMessage(content=state.get("original_query", ""))

    messages_for_llm = [sys_message, context_message] + history_messages + [final_human_message]

    try:
        # 4. Call the LLM
        logger.info("Generating final response with LLM...")
        llm_response = llm.invoke(messages_for_llm)
        answer_text = llm_response.content
        
        # 5. Construct AgentResponse
        agent_response = AgentResponse(
            answer=answer_text,
            sources=list(unique_sources),
            confidence=avg_confidence,
            no_context_found=False,
            rewritten_query=state.get("rewritten_query", ""),
        )
        
        # 6 & 7. Append AIMessage and return updates
        return {
            "final_response": agent_response,
            "messages": [AIMessage(content=answer_text)],
        }
        
    except Exception as e:
        logger.error(f"LLM Generation failed: {e}")
        error_msg = "I encountered an error while trying to generate a response. Please try again."
        return {
            "final_response": AgentResponse(
                answer=error_msg, sources=[], confidence=0.0, 
                no_context_found=False, rewritten_query=""
            ),
            "messages": [AIMessage(content=error_msg)],
        }

# ---------------------------------------------------------------------------
# Routing Function
# ---------------------------------------------------------------------------


def should_retry_retrieval(state: AgentState) -> str:
    """
    Conditional edge function: decide whether to retry retrieval or generate.

    Called by the graph after retrieval_node. If no context was found,
    the graph routes back to query_rewrite_node for one retry with a
    broader query before triggering the hallucination guard.

    Interview talking point: conditional edges in LangGraph enable
    agentic behaviour — the graph makes decisions about its own
    execution path rather than following a fixed sequence.

    Parameters
    ----------
    state : AgentState
        Current graph state. Reads: no_context_found, retrieved_chunks.

    Returns
    -------
    str
        "generate" — proceed to generation_node.
        "end"      — skip generation, return no_context response directly.

    Notes
    -----
    Retry logic should be limited to one attempt to prevent infinite loops.
    Track retry count in AgentState if implementing retry behaviour.
    """
    # TODO: implement
    # Simple version: if no_context_found → "end", else → "generate"
    # Advanced version: track retry count, allow one retry with broader query
    if state.get("no_context_found", False):
        # We MUST route to generate here so the generation_node's 
        # Hallucination Guard has a chance to execute and overwrite the state!
        logger.info("Routing: no context found -> triggering hallucination guard in generation")
        return "generate"
        
    logger.info("Routing: context found -> generate")
    return "generate"
