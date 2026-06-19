# State-Aware Enterprise Graph Agent & Semantic Caching Layer

An enterprise-grade AI routing backend designed to translate natural language into complex graph database queries (Cypher) for navigating highly interconnected internal ecosystems (e.g., microservices, organizational charts, deployment pipelines). 

This system implements a distributed semantic caching layer to intercept redundant queries, bypassing LLM generation and drastically optimizing system latency. Built with Python, FastAPI, LangGraph, Neo4j, and Redis.

---

## 🛠️ Tech Stack
Backend: Python, FastAPI

Agentic Framework: LangChain, LangGraph

Databases: Neo4j (Graph), Redis Stack (Vector/Cache)

Infrastructure: Docker, Docker Compose

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
```

### 2. Spin Up Infrastructure
Start the Neo4j graph database and Redis cache locally.
```bash
docker-compose up -d
```

### 3. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 4. Seed the Graph Database
Generate the synthetic enterprise ecosystem (Microservices, Teams, Employees) to populate Neo4j.
```bash
python seed_graph.py
```

### 5. Run the FastAPI Server
```bash
export OPENAI_API_KEY="your-api-key-here"
uvicorn main:app --reload --port 8000
```

## 🧪 API Usage
Endpoint: POST /api/v1/query

Request:
```bash
{
  "query": "Which team owns the BillingService?"
}
```

Response (Cache Miss - 2.4s):
```bash
{
  "answer": "The Platform Engineering team owns the BillingService.",
  "source": "neo4j_graph",
  "cypher_executed": "MATCH (t:Team)-[:OWNS]->(s:Service {name: 'BillingService'}) RETURN t.name",
  "latency_ms": 2453
}
```

Request (Semantically Similar):
```bash
{
  "query": "Can you tell me who runs the billing service?"
}
```

Response (Cache Hit - 12ms):
```bash
{
  "answer": "The Platform Engineering team owns the BillingService.",
  "source": "redis_semantic_cache",
  "latency_ms": 12
}
```
