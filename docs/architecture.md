# System Architecture
## Team: RAG REGEN
## Date: 03/17/2026
## Members and Roles:
- Corpus Architect: Anisha
- Pipeline Engineer: Mounika
- UX Lead: Sowmya
- Prompt Engineer: Viral
- QA Lead: Rahul

---

## Architecture Diagram

Replace this section with your team's completed flow chart.
Export from FigJam, Miro, or draw.io and embed as an image,
or describe the architecture as an ASCII diagram.

The diagram must show:
- [ ] How a corpus file becomes a chunk
- [ ] How a chunk becomes an embedding
- [ ] How duplicate detection fires
- [ ] How a user query flows through LangGraph to a response
- [ ] Where the hallucination guard sits in the graph
- [ ] How conversation memory is maintained across turns

[Corpus (.md/.pdf)] 
        │
        ▼
[Document Chunker] ──(Splits into 100-300 word semantic concepts)──┐
        │                                                          │
        ▼                                                          ▼
[Content Hasher] ──(Generates SHA-256 hash for idempotency)──> [ChromaDB]
        │                                                      (Vector Store)
        ▼                                                          ▲
[Embedding Factory] ──(all-MiniLM-L6-v2)───────────────────────────┘


                                ┌─────────────────────────┐
[User UI / Streamlit] ────────> │ LangGraph Orchestrator  │
        ▲                       └─────────────────────────┘
        │                               │     ▲
        │       (Query)                 ▼     │ (Top K Chunks)
        │                       [Retrieval Node] 
        │                               │
        │                               ▼
        │                       [Guardrail Evaluator] ──(Threshold < 0.65?) ──> Return "Off-Topic"
        │                               │
        │                               ▼
        └───────────(JSON/Text)─ [Generation Node] (Groq Llama-3.1-8b)


---

## Component Descriptions

### Corpus Layer

- **Source files location:** `data/corpus/`
- **File formats used:**
  .md (for custom drafted concepts) and .pdf (for landmark papers).

- **Landmark papers ingested:**
  - Rumelhart, Hinton & Williams (1986) - Backpropagation
  - LeCun et al. (1998) - LeNet
  - Hochreiter & Schmidhuber (1997) - LSTM

- **Chunking strategy:**
  We utilized semantic chunking targeting 100-300 words with a 50-character overlap. Instead of arbitrary character cutoffs, we chunked by atomic ideas (e.g., separating the LSTM "Forget Gate" from the "Input Gate") to ensure the LLM retrieves precise, targeted context for interview questions.

- **Metadata schema:**
  | Field | Type | Purpose |
  |---|---|---|
  | topic | string | Categorizes the chunk for UI filtering (e.g., "ANN", "CNN"). |
  | difficulty | string | Enables the Prompt Engineer to calibrate question generation. |
  | type | string | Identifies if the chunk is a "concept_explanation" or "code_example". |
  | source | string | Required for the LLM to generate accurate source citations. |
  | related_topics | list | Allows for cross-topic interview questions (e.g., CNN + RNN). |
  | is_bonus | bool | Flags stretch topics like GANs or SOMs. |

- **Duplicate detection approach:**
  We generate a SHA-256 hash of the chunk_text and store it as the document ID. This content-based hashing ensures pipeline idempotency—a crucial requirement for robust data engineering workflows—preventing duplicate vectors even if a file is renamed and re-uploaded.

- **Corpus coverage:**
  - [ ] ANN
  - [ ] CNN
  - [ ] RNN
  - [ ] LSTM
  - [ ] Seq2Seq
  - [ ] Autoencoder
  - [ ] SOM *(bonus)*
  - [ ] Boltzmann Machine *(bonus)*
  - [ ] GAN *(bonus)*

---

### Vector Store Layer

- **Database:** 
  *ChromaDB — PersistentClient*

- **Local persistence path:** 
  *./data/chroma_db*

- **Embedding model:**
  *all-MiniLM-L6-v2 via sentence-transformers*

- **Why this embedding model:**
  *It is a lightweight, open-source local model that balances embedding quality with processing speed. Running it locally avoids API rate limits during bulk ingestion.*

- **Similarity metric:**
  *Cosine similarity. It handles vector magnitude variations well, which is important since our chunk lengths vary between 100 and 300 words.*

- **Retrieval k:**
  *This provides enough context for the LLM to synthesize a complete answer without overflowing the context window or introducing distracting, loosely related concepts. K = 3*

- **Similarity threshold:**
  *Calibrated by manually testing off-topic queries (e.g., "History of Rome") versus highly specific ML queries.*

- **Metadata filtering:**
  *Users can filter by the topic metadata field via a dropdown in the Streamlit UI, which passes a where clause to the ChromaDB query.*

---

### Agent Layer

- **Framework:** LangGraph

- **Graph nodes:**
  *(describe what each node does in one sentence)*
  | Node | Responsibility |
  |---|---|
  | query_rewrite_node | Refines conversational user input into a dense search query. |
  | retrieval_node | Interfaces with VectorStoreManager to fetch the top k chunks. |
  | generation_node | Passes chunks to the LLM and formats the final JSON/Text response. |

- **Conditional edges:**
  *After retrieval, a conditional edge evaluates the similarity score. If the max score is below 0.65, the graph routes directly to the END node, skipping generation.*

- **Hallucination guard:**
  *I cannot find relevant information in the provided study materials to answer this question. Let's stick to the deep learning corpus.*

- **Query rewriting:**
  - Raw query: What about the vanishing gradient one?
  - Rewritten query: How do LSTMs solve the vanishing gradient problem in RNNs?

- **Conversation memory:**
  *Maintained in Streamlit's st.session_state["messages"] and passed into the LangGraph state on each turn*

- **LLM provider:**
  *Groq (llama-3.1-8b-instant)*

- **Why this provider:**
  *Groq's LPU architecture provides near-instantaneous inference. For an interactive UI like an interview prep agent, low latency is critical for user experience.*

---

### Prompt Layer

- **System prompt summary:**
  *The agent assumes the persona of a strict but encouraging Senior ML Interviewer. The absolute constraint is that it must only evaluate answers or generate questions based on the provided context chunks, and it must append a [SOURCE: ...] tag to every technical claim*

- **Question generation prompt:**
  *Takes {context} and {difficulty} as inputs. It returns a strictly formatted JSON object containing the question, model answer, and a follow-up prompt.*

- **Answer evaluation prompt:**
  *Takes the {question}, the student's {candidate_answer}, and the {context}. It uses a 10-point rubric to penalize hallucinations and reward accurate terminology.*

- **JSON reliability:**
  *Appended the instruction: "Respond with the JSON object only. No preamble, explanation, or markdown code fences." We also implemented a lightweight Python try/except JSON parsing block in the backend to strip accidental backticks.*

- **Failure modes identified:**
  - Mode: LLM occasionally complimented the user before delivering the JSON (breaking the parser). Fix: Enforced strict JSON-only output in the system prompt.

  - Mode: Generated questions were too broad (e.g., "What is a neural network?"). Fix: Forced the prompt to focus specifically on the atomic concept in the chunk.
---

### Interface Layer

- **Framework:** *Streamlit*
- **Deployment platform:** *Streamlit Community Cloud*
- **Public URL:** *(paste your deployed app URL here once live)*

- **Ingestion panel features:**
  *A sidebar with st.file_uploader (accepting multiple files), a progress bar during ingestion, and a success metric showing the number of unique chunks added to ChromaDB.*

- **Document viewer features:**
  *An expander widget that lists all ingested sources. Clicking a source reveals the individual chunk texts and their associated metadata.*

- **Chat panel features:**
  *A scrollable st.chat_message container. Source citations are rendered in bold markdown. If the hallucination guard trips, it renders a warning UI element.*

- **Session state keys:**
  *(list the st.session_state keys your app uses and what each stores)*
  | Key | Stores |
  |---|---|
  | messages | The ongoing list of user/assistant chat dictionaries. |
  | vector_store | The initialized VectorStoreManager instance to prevent re-instantiation. |
  | processed_files | A set of filename hashes to prevent redundant UI ingestion attempts. |

- **Stretch features implemented:**
  *(streaming responses, async ingestion, hybrid search, re-ranking, other)*

---

## Design Decisions

Document at least three deliberate decisions your team made.
These are your Hour 3 interview talking points — be specific.
"We used the default settings" is not a design decision.

1. **Decision:**
   *Using SHA-256 content hashing for chunk IDs instead of sequential numbers or filenames.*
   **Rationale:**
   *This guarantees idempotency. If a team member modifies a single typo in a markdown file and re-uploads it, only the modified chunk gets a new hash and is ingested. The rest are skipped.*
   **Interview answer:**
   *To ensure the ingestion pipeline is robust and idempotent, we implemented content-based hashing; this prevents vector duplication at the data layer, regardless of how many times a user attempts to re-ingest the same file.*

2. **Decision:**
   Choosing Groq over local Ollama inference for the Agent layer.
   **Rationale:**
   While local embeddings (all-MiniLM) are fast, local LLM generation can bottleneck the chat interface. Groq's LPU speed keeps the interview agent conversational and responsive.
   **Interview answer:**
   We decoupled our compute requirements by running embeddings locally for privacy and cost, while utilizing Groq's API for generation to guarantee sub-second latency in the chat UI.

3. **Decision:**
   Implementing a hard similarity threshold conditional edge in LangGraph.
   **Rationale:**
   Rather than relying entirely on the LLM's system prompt to say "I don't know," severing the graph edge completely before generation guarantees zero hallucination for off-topic queries and saves API tokens.
   **Interview answer:**
   We implemented a deterministic retrieval threshold in our LangGraph orchestrator; if the vector search fails to meet our confidence score, we short-circuit the LLM generation entirely to strictly enforce our hallucination guardrails.

---

## QA Test Results

*(QA Lead fills this in during Phase 2 of Hour 2)*

| Test | Expected | Actual | Pass / Fail |
|---|---|---|---|
| Normal query | Relevant chunks, source cited | Relevant chunks retrieved, correct citation generated | Pass |
| Off-topic query | No context found message | Graph short-circuited; exact guardrail message returned | Pass |
| Duplicate ingestion | Second upload skipped | ChromaDB reported 0 new elements added | Pass |
| Empty query | Graceful error, no crash | UI surfaced a "Please enter a query" toast message | Pass |
| Cross-topic query | Multi-topic retrieval | | |


**Critical failures fixed before Hour 3:**
- falure 1
-  falirue 2

**Known issues not fixed (and why):**
- issue 1
-

---

## Known Limitations

Be honest. Interviewers respect candidates who understand
the boundaries of their own system.

- The current document chunker relies on markdown headers (##). Ingesting raw PDFs with complex multi-column layouts occasionally introduces formatting noise into the chunk_text.
- Conversation memory is stored in st.session_state, meaning a hard browser refresh clears the user's interview history.

---

## What We Would Do With More Time

- Implement a hybrid search pipeline, combining our dense vector search (ChromaDB) with a sparse keyword search (BM25) to better handle queries looking for exact acronyms or specific author names.
- Migrate the ingestion pipeline to run asynchronously so that uploading a massive PDF textbook doesn't freeze the Streamlit frontend.

---

## Hour 3 Interview Questions

*(QA Lead fills this in — these are the questions your team
will ask the opposing team during judging)*

**Question 1:** You are building an image classification model and notice the network has an exorbitant number of parameters, leading to severe overfitting. Why might you switch from a standard fully connected Artificial Neural Network (ANN) to a Convolutional Neural Network (CNN)?

Model answer: CNNs introduce parameter sharing and local receptive fields through convolutional filters. This drastically reduces the total number of parameters compared to a fully connected ANN, where every input pixel connects to every neuron, making the CNN much less prone to overfitting on image data.

**Question 2:** Explain how the architecture of an LSTM directly addresses the vanishing gradient problem found in standard RNNs.

Model answer: Standard RNNs struggle with long sequences because gradients shrink exponentially during backpropagation through time. LSTMs solve this by introducing a cell state (the "conveyor belt") and three specific gates (forget, input, output). This structure allows gradients to flow unchanged through the cell state, preventing them from vanishing.

**Question 3:** If we inject identical study materials into our RAG system twice, how does the architecture guarantee that we don't retrieve duplicate context chunks during a query?

Model answer: The ingestion pipeline uses a cryptographic hash (like SHA-256) of the chunk's actual content to generate its unique database ID. ChromaDB will natively reject or overwrite inserts with identical IDs, ensuring the vector database remains perfectly deduplicated.

---

## Team Retrospective

*(fill in after Hour 3)*

**What clicked:**
-

**What confused us:**
-

**One thing each team member would study before a real interview:**
- Corpus Architect:
- Pipeline Engineer:
- UX Lead:
- Prompt Engineer:
- QA Lead:
