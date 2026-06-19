# State-Aware Enterprise Graph Agent & Semantic Caching Layer

An enterprise-grade AI routing backend designed to translate natural language into complex graph database queries (Cypher) for navigating highly interconnected internal ecosystems (e.g., microservices, organizational charts, deployment pipelines). 

This system implements a distributed semantic caching layer to intercept redundant queries, bypassing LLM generation and drastically optimizing system latency. Built with Python, FastAPI, LangGraph, Neo4j, and Redis.

---

## 🚀 Key Features

* **Natural Language to Cypher:** Utilizes a deterministic state machine to parse user intent and generate valid Neo4j graph queries.
* **Semantic Caching Layer:** Leverages vector embeddings and Redis to intercept semantically similar conversational queries, routing around expensive LLM calls.
* **LLM Self-Reflection & Auto-Correction:** If the LLM generates invalid Cypher syntax, the state machine catches the database driver error, feeds the traceback back to the model, and autonomously rewrites the query before failing.
* **Contextual Memory Pruning:** Manages agent state across multi-turn interactions without overflowing the LLM context window.
* **Scalable Infrastructure:** Fully containerized backend deployment using Docker Compose for reproducible execution environments.

---

## 🏗️ System Architecture

1. **Routing Engine (FastAPI):** Receives user natural language queries.
2. **Vector Cache Check (Redis):** Embeds the query and checks for a cosine similarity match > 95% against historical queries.
   * *Cache Hit:* Returns the stored graph data instantly (**<15ms**).
   * *Cache Miss:* Forwards the request to the Agentic State Machine.
3. **Agent Orchestration (LangGraph):** Manages the prompt execution, injecting the Neo4j schema into the context.
4. **Graph Execution (Neo4j):** Executes the Cypher query. Triggers the self-correction loop if syntax is invalid.
5. **Cache Hydration:** Stores the successful query and result back into Redis for future semantic matches.

---

## 📊 Benchmarking & Performance

System performance was validated using a synthetic enterprise dataset consisting of Teams, Employees, and Microservice dependencies.

| Query Type | Execution Route | Avg. Latency | Cost / Token Usage |
| :--- | :--- | :--- | :--- |
| **Cold Query** (First time) | LLM Generation + Neo4j Execution | ~2,450 ms | High (Full Prompt) |
| **Warm Query** (Semantic Match) | Redis Vector Search | **< 15 ms** | **Zero (Bypassed)** |

*Result: Achieved a **98% reduction in latency** and a **40% drop in token consumption** under simulated repetitive enterprise workloads.*

---

## 💻 Local Setup & Quickstart

### Prerequisites
* Docker & Docker Compose
* Python 3.10+
* OpenAI API Key

### 1. Clone the Repository
```bash
git clone [https://github.com/uvsrahul1234/enterprise-graph-agent.git](https://github.com/uvsrahul1234/enterprise-graph-agent.git)
cd enterprise-graph-agent
