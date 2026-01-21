# 🤖 SaaS ARR Revenue Strategy RAG Agent with Thinking & Web Research

## Overview

A production-ready **Retrieval-Augmented Generation (RAG) Agent** built with LlamaIndex that intelligently analyzes SaaS revenue strategy patterns, ARR trends, and business models. The system employs a dual-agent architecture with advanced reasoning, web research capabilities, and hybrid vector search to deliver comprehensive research outputs.

---

## 🎯 Purpose & Use Cases

This application helps teams and researchers:
- **Mine web knowledge** on SaaS ARR revenue strategies from Substack publications, industry blogs, and internet resources
- **Conduct hybrid vector searches** across accumulated domain knowledge
- **Generate research insights** with advanced reasoning and thinking models
- **Ask sophisticated questions** via CLI with intelligent answer generation
- **Maintain local, reproducible infrastructure** with Docker-based vector database

---

## 🏗️ Architecture

### Dual-Agent System

#### **Agent 1: Knowledge Mining Agent** 🕷️
- **Role:** Datamines websites, Substack publications, and internet resources
- **Target Domain:** SaaS ARR Revenue Strategy, pricing models, business metrics
- **Process:**
  1. Discovers and extracts knowledge from configured sources
  2. Processes documents through QwenMax reasoning models
  3. Generates embeddings (dense + sparse) via Qwen embeddings
  4. Persists enriched documents to Milvus vector collection
  5. Builds a comprehensive knowledge base over time

#### **Agent 2: Query & Research Agent** 🔍
- **Role:** Retrieves and synthesizes research from the knowledge base
- **Capabilities:**
  1. Performs **hybrid vector search** (dense + sparse embeddings)
  2. Accesses internal tools and web research via Tavily
  3. Uses QwenMax with extended thinking for reasoning
  4. Generates detailed, context-aware research outputs
  5. Answers complex multi-part questions

### Key Components

| Component | Technology | Role |
|-----------|-----------|------|
| **LLM** | Qwen3-Max (Novita API) | Core reasoning & generation model |
| **Embeddings** | Qwen Embeddings | Dense + sparse vector generation |
| **Vector Database** | Milvus (Local Docker) | Hybrid search & knowledge storage |
| **Web Research** | Tavily API | Real-time internet research capability |
| **RAG Framework** | LlamaIndex | Orchestration & query pipeline |
| **Question Generation** | QwenMax | Synthetic question creation |
| **Interface** | Python CLI | Command-line interaction |

---

## 🛠️ Tech Stack

```
Language:        Python 3.10+
RAG Framework:   LlamaIndex
Vector DB:       Milvus (Docker)
LLM Provider:    Novita API (Qwen3-Max)
Embeddings:      Qwen Text Embeddings
Web Search:      Tavily Research API
Deployment:      Docker Compose (local)
```

---

## 🚀 Features

### ✨ Advanced Reasoning
- **Extended Thinking:** QwenMax models process queries with deep reasoning
- **Question Generator:** Synthetic question creation for comprehensive coverage
- **Hybrid Vector Search:** Combines dense semantic + sparse keyword matching

### 🌐 Web Research
- **Tavily Integration:** Live web research during query time
- **Knowledge Mining:** Automated datamining of target websites
- **Multi-source Aggregation:** Consolidates insights from diverse sources

### 📚 Local-First Infrastructure
- **Dockerized Milvus:** Vector database runs entirely locally
- **No External Dependencies:** Complete privacy and reproducibility
- **Version Control:** Vector DB snapshots stored in GitHub
- **Easy Setup:** Single `docker-compose up` to start

### 🎮 User-Friendly Interface
- **CLI-Based:** Full command-line interaction
- **Query Flexibility:** Complex multi-part questions supported
- **Research Output:** Structured, formatted research responses
- **Configuration:** Easy agent parameter tuning

---

## 📋 Installation & Setup

### Prerequisites
```bash
- Python 3.10 or higher
- Docker & Docker Compose
- Git
```

### Quick Start

```bash
# Navigate to the project directory
cd saas-revenue-strategy

# Install Python dependencies
pip install -r requirements.txt

# Set up environment variables
# Create .env file with your API keys:
# NOVITA_API_KEY=<your-key>
# TAVILY_API_KEY=<your-key>
# See QUICKSTART.md for detailed setup instructions

# Start Milvus vector database
docker-compose up -d

# Wait for Milvus to be ready (30-60 seconds)
# Check with: docker-compose ps

# Initialize vector collection (first run)
python scripts/setup_milvus.py

# Verify setup
python scripts/verify_setup.py
```

---

## 💻 Usage

### Main CLI Interface

```bash
# Interactive mode (recommended)
python main.py --interactive

# Show menu
python main.py
```

### Run Knowledge Mining Agent

```bash
# Mine from configured sources (via main.py)
python main.py --mine

# Or use standalone entry point
python knowledge_miner_main.py

# Mine specific topics
python knowledge_miner_main.py --topics "SaaS ARR growth" "pricing models"

# Mine with custom sources config
python knowledge_miner_main.py --sources config/sources.yaml
```

### Run Query Agent

```bash
# Single query (via main.py - recommended)
python main.py --query "What are the latest trends in SaaS ARR revenue strategies?"

# Interactive mode (via main.py - recommended)
python main.py --interactive

# Or use standalone entry point
python query_agent_main.py "What are the latest trends in SaaS ARR revenue strategies?"
python query_agent_main.py --interactive
```

### Using Main CLI

```bash
# Mine knowledge
python main.py --mine

# Query agent
python main.py --query "What are emerging SaaS pricing models?"

# Interactive mode
python main.py --interactive
```

---

## 🗄️ Vector Database Management

### Backup Vector DB
```bash
# Create timestamped backup (saved to scripts/backups/)
python scripts/backup_vector_db.py

# Custom output path
python scripts/backup_vector_db.py --output ./backups/milvus_snapshot.tar.gz

# Verify backup integrity
python scripts/verify_backup.py --input scripts/backups/milvus_snapshot_YYYYMMDD_HHMMSS.tar.gz
```

### Store in GitHub
```bash
# Compress and commit snapshot
git add scripts/backups/milvus_snapshot_*.tar.gz
git commit -m "Update Milvus vector DB snapshot"
git push origin main
```

### Restore from Snapshot
```bash
# Restore from timestamped backup
python scripts/restore_vector_db.py --input scripts/backups/milvus_snapshot_YYYYMMDD_HHMMSS.tar.gz

# Or from custom path
python scripts/restore_vector_db.py --input ./backups/milvus_snapshot.tar.gz
```

---

## 📊 Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE MINING AGENT                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Internet Resources  →  Web Scraper  →  QwenMax Thinking  │
│  (Substack, Blogs)       (Tavily)          (Process)       │
│                                                │            │
│                                                ↓            │
│                                    Dense + Sparse Embeddings│
│                                    (Qwen Embeddings)        │
│                                                │            │
└────────────────────────────────────┬───────────┘            │
                                     │                        │
                          ┌──────────↓──────────┐            │
                          │  Milvus Vector DB  │            │
                          │  (Local Docker)    │            │
                          │  Hybrid Collection │            │
                          └──────────┬──────────┘            │
                                     │                        │
┌────────────────────────────────────↓──────────────────────┐
│                   QUERY & RESEARCH AGENT                  │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  User Question  →  Hybrid Search  →  Context Retrieval │
│  (CLI)              (Dense+Sparse)     (Top-K)          │
│                                            │             │
│                                            ↓             │
│                                   QwenMax Extended      │
│                                   Thinking Reasoning    │
│                                   + Tavily Web Research │
│                                            │             │
│                                            ↓             │
│                              Structured Research Output │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## 🔑 API Keys Required

| Service | Purpose | Where to Get |
|---------|---------|--------------|
| **Novita API** | QwenMax LLM & Embeddings | [novita.ai](https://novita.ai) |
| **Tavily** | Web Research Tool | [tavily.com](https://tavily.com) |

---

## 📁 Project Structure

```
saas-revenue-strategy/
├── agents/                      # Core agent implementations
│   ├── __init__.py             # Package initialization
│   ├── knowledge_miner.py      # Mining agent logic
│   ├── query_agent.py          # Query & research agent
│   ├── novita_embedding.py     # Novita embedding wrapper
│   └── logging_utils.py        # Logging utilities
├── scripts/                     # Utility scripts
│   ├── __init__.py             # Package initialization
│   ├── setup_milvus.py         # Vector DB initialization
│   ├── backup_vector_db.py     # Backup utilities
│   ├── restore_vector_db.py    # Restore from snapshot
│   ├── verify_setup.py         # Setup verification
│   ├── verify_backup.py        # Backup verification
│   └── backups/                # Backup storage
│       └── milvus_snapshot_*.tar.gz  # Timestamped backups
├── config/                      # Configuration files
│   ├── agents.yaml             # Agent parameters
│   └── sources.yaml            # Data sources config
├── backups/                     # Vector DB backups directory
│   └── .gitkeep                # Directory placeholder
├── docker-compose.yml           # Milvus + dependencies
├── requirements.txt             # Python dependencies
├── .env                         # Your environment (gitignored)
├── main.py                      # Main CLI entry point
├── knowledge_miner_main.py      # Standalone mining entry point
├── query_agent_main.py          # Standalone query entry point
├── novita_docs.txt             # Novita API documentation
├── README.md                    # This file
├── QUICKSTART.md               # Quick start guide
├── PROJECT_STRUCTURE.md        # Detailed project structure
├── CONTRIBUTING.md             # Contribution guidelines
└── LICENSE                      # MIT License
```

---

## 🧠 Technical Highlights

### Hybrid Vector Search
- **Dense Embeddings:** Semantic similarity via Qwen embeddings (Novita API)
- **Sparse Embeddings:** Keyword-based BM25 matching
- **Combined:** Retrieves most relevant + most keyword-matching documents
- **Custom Embedding Wrapper:** `NovitaEmbedding` class for OpenAI-compatible API integration

### Extended Thinking
- QwenMax models use internal reasoning before generating answers
- Improves answer quality for complex SaaS strategy questions
- Supports multi-step logical reasoning

### Reproducibility
- Versioned vector database snapshots in GitHub
- Deterministic embedding generation
- Full configuration in version control
- Backup verification utilities for data integrity

### Logging & Observability
- Structured logging with payload tracking
- Configurable log levels via environment variables
- Secret redaction for security
- Comprehensive error tracking

---

## 🎓 Example Queries

```
1. "What are the top SaaS companies optimizing for ARR growth and how do their revenue strategies differ?"

2. "Compare pricing models across Atlassian, HubSpot, and Datadog. What patterns emerge?"

3. "Analyze the relationship between customer acquisition cost and lifetime value in modern SaaS businesses."

4. "What changes in SaaS revenue strategy are emerging post-2024? Cite recent trends."

5. "How do vertical SaaS companies approach ARR differently than horizontal platforms?"
```

---

## 🔄 Workflow

### For Knowledge Researchers
1. Configure knowledge sources in `config/sources.yaml`
2. Run mining agent daily/weekly to keep knowledge fresh:
   ```bash
   python main.py --mine
   # or
   python knowledge_miner_main.py
   ```
3. Monitor vector DB size and performance
4. Backup vector DB periodically:
   ```bash
   python scripts/backup_vector_db.py
   ```
5. Commit backups to GitHub for version control

### For Query Users
1. Start interactive mode or run single queries:
   ```bash
   python main.py --interactive
   # or
   python main.py --query "Your question here"
   ```
2. Receive researched, reasoning-based answers
3. Get citations to source documents
4. Continue conversation in interactive mode for follow-up questions

---

## 📦 Dependencies Overview

```yaml
Core RAG Framework:
  - llama-index>=0.10.0
  - llama-index-core>=0.10.0
  - llama-index-llms-openai>=0.1.0
  - llama-index-llms-openai-like
  - llama-index-embeddings-openai>=0.1.0
  - llama-index-vector-stores-milvus>=0.1.0
  
Vector Database:
  - pymilvus>=2.4.0
  
LLM/Embeddings:
  - openai>=1.0.0  # OpenAI-compatible API (Novita)
  
Web Research:
  - tavily-python>=0.3.0
  
Infrastructure:
  - docker
  - docker-compose
  
Utilities:
  - python-dotenv>=1.0.0
  - pyyaml>=6.0.0
  - requests>=2.31.0
  - beautifulsoup4>=4.12.0
  - lxml>=4.9.0
  
CLI and Display:
  - rich>=13.0.0
  - click>=8.1.0
  
Type Hints:
  - typing-extensions>=4.0.0
```

---

## 🚨 Important Notes

- **API Keys:** Store securely in `.env` (never commit to version control)
- **Docker Required:** Milvus runs in Docker (must be installed and running)
- **Rate Limits:** Be mindful of API rate limits for Novita and Tavily
- **Initial Mining:** First knowledge mine may take time depending on source volume
- **Vector DB Size:** Monitor local storage for vector snapshots
- **Backup Location:** Backups are stored in `scripts/backups/` with timestamps
- **Entry Points:** Use `main.py` for unified CLI, or standalone scripts for direct access

---

## 🔧 Configuration

### Agent Configuration (config/agents.yaml)

Customize agent behavior:
- LLM model and parameters
- Embedding model
- Search parameters (top_k, similarity threshold)
- Web research settings

### Sources Configuration (config/sources.yaml)

Configure data sources:
- Substack publications
- Blog sources
- Research topics
- Seed queries

---

## 🐛 Troubleshooting

### Milvus Connection Issues

```bash
# Check if Milvus is running
docker-compose ps

# Restart Milvus
docker-compose restart

# Check logs
docker-compose logs standalone
```

### Python Package Issues

```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Check installed packages
pip list | grep llama-index
```

### API Key Issues

```bash
# Verify environment variables
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('NOVITA:', bool(os.getenv('NOVITA_API_KEY'))); print('TAVILY:', bool(os.getenv('TAVILY_API_KEY')))"
```

---

## 🤝 Contributing

Contributions welcome! Areas:
- Additional data sources (Crunchbase, AngelList, etc.)
- Query optimization techniques
- Vector DB performance tuning
- Additional output formats

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🎯 Roadmap

- [ ] Multi-language support for international SaaS analysis
- [ ] Real-time dashboard for vector DB insights
- [ ] API endpoint for remote queries
- [ ] Fine-tuned embedding models for SaaS domain
- [ ] Batch query processing
- [ ] Advanced filtering and faceted search

---

## 📧 Support & Questions

For issues, questions, or feature requests, please open a GitHub issue with:
- **Clear description** of the problem/request
- **Steps to reproduce** (if applicable)
- **Your environment** (Python version, OS, Docker version)
- **Expected vs. actual behavior**
- **Logs** (with sensitive information redacted)

### Additional Resources

- **Quick Start Guide:** See [QUICKSTART.md](QUICKSTART.md) for step-by-step setup
- **Project Structure:** See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed architecture
- **Contributing:** See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines

---

## 🙏 Acknowledgments

Built with:
- [LlamaIndex](https://www.llamaindex.ai/) - RAG framework
- [Milvus](https://milvus.io/) - Vector database
- [Qwen](https://github.com/QwenLM/Qwen) - LLM & embeddings
- [Tavily](https://tavily.com/) - Web research
- Inspired by SaaS research from [Clouded Judgement](https://cloudedjudgement.substack.com/) & [Growth Stack Mafia](https://austinhay.substack.com/)

---

**Made with ❤️ for the SaaS community**
