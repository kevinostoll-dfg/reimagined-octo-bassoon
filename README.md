# reimagined-octo-bassoon

> **Research repository exploring AI, Chat, LangChain, and LlamaIndex applications**

A comprehensive research and development repository containing two major projects focused on advanced AI applications: graph-based knowledge processing for financial documents and RAG (Retrieval-Augmented Generation) systems for domain-specific research.

---

## 🎯 Overview

This repository serves as a research and experimentation platform for exploring cutting-edge AI technologies, specifically:

- **Graph-based Knowledge Processing**: Building and querying knowledge graphs from financial and regulatory documents using Memgraph
- **RAG Systems**: Implementing advanced retrieval-augmented generation with hybrid search capabilities for domain-specific knowledge mining
- **Production-Ready Pipelines**: End-to-end document processing, entity extraction, and intelligent query systems

---

## 📁 Repository Structure

```
reimagined-octo-bassoon/
├── graph-datasources/          # Graph database processing pipelines
│   ├── graph-earnings-announcement-transcripts/
│   ├── graph-fomc-transcripts/
│   ├── graph-sec-10k-filings/
│   ├── graph-sec-F4-filings/
│   ├── hybrid_search/          # FastAPI agent with graph RAG + vector search
│   ├── memgraph_docker/        # Local Memgraph development setup
│   ├── publish_schema/         # Schema introspection and publishing
│   ├── gcs_bucket/             # GCS bucket policy management
│   ├── finetune_model/         # Model fine-tuning infrastructure
│   └── compute_node/           # GPU compute node management
│
└── saas-revenue-strategy/      # SaaS revenue strategy RAG agent
    ├── agents/                 # Core agent implementations
    ├── scripts/                # Utility scripts (backup, restore, setup)
    ├── config/                 # Configuration files
    └── main.py                 # Unified CLI entry point
```

---

## 🚀 Projects

### 1. Graph Datasources

**Location**: [`graph-datasources/`](./graph-datasources/)

**Purpose**: Comprehensive graph database processing pipelines for ingesting, analyzing, and storing financial and regulatory documents in Memgraph graph databases.

#### Key Features

- **Document Ingestion**: Download and process financial documents from various sources
- **Graph Processing**: Extract entities, relationships, and metadata to build knowledge graphs
- **Storage & Retrieval**: Store processed data in Memgraph with optimized queries and indexes
- **Hybrid Search**: Combine graph-based RAG with vector search for advanced querying
- **Infrastructure**: Docker setups, GCS bucket management, and model fine-tuning utilities

#### Supported Document Types

1. **Earnings Announcement Transcripts** (`graph-earnings-announcement-transcripts/`)
   - Company metrics, financial statements, executive commentary
   - Entity relationships (PERSON, ORG, PRODUCT, CONCEPT, METRIC)
   - Temporal context and sentiment analysis

2. **FOMC Transcripts** (`graph-fomc-transcripts/`)
   - Federal Reserve policy discussions and decisions
   - Economic indicators and monetary policy analysis
   - RSS feed consumption for transcript updates

3. **SEC Form 10-K Filings** (`graph-sec-10k-filings/`)
   - Company annual financial statements and disclosures
   - Risk factors, business operations, and management discussion

4. **SEC Form 4 Filings** (`graph-sec-F4-filings/`)
   - Insider trading transactions
   - Company insider information and holdings

#### Architecture Highlights

- **Knowledge Graph**: Memgraph for storing structured relationships
- **Hybrid Search Agent**: FastAPI-based agent combining graph RAG with Milvus vector search
- **Schema Publishing**: Automated schema introspection stored in DragonFly
- **Checkpointing**: Resumable batch processing with checkpoint files
- **Vault Integration**: Secrets management via HashiCorp Vault

#### Quick Start

```bash
cd graph-datasources

# Start local Memgraph
cd memgraph_docker
docker-compose up -d

# Process earnings announcements
cd ../graph-earnings-announcement-transcripts
pip install -r requirements.txt
python batch_process_ea.py
```

**For detailed documentation**, see [`graph-datasources/README.md`](./graph-datasources/README.md)

---

### 2. SaaS Revenue Strategy RAG Agent

**Location**: [`saas-revenue-strategy/`](./saas-revenue-strategy/)

**Purpose**: Production-ready Retrieval-Augmented Generation (RAG) Agent built with LlamaIndex that intelligently analyzes SaaS revenue strategy patterns, ARR trends, and business models.

#### Key Features

- **Dual-Agent Architecture**:
  - **Knowledge Mining Agent**: Datamines websites, Substack publications, and internet resources
  - **Query & Research Agent**: Retrieves and synthesizes research from the knowledge base

- **Advanced Reasoning**: QwenMax models with extended thinking for deep reasoning
- **Hybrid Vector Search**: Combines dense semantic + sparse keyword matching
- **Web Research**: Tavily integration for live web research during query time
- **Local-First Infrastructure**: Dockerized Milvus vector database runs entirely locally

#### Tech Stack

```
Language:        Python 3.10+
RAG Framework:   LlamaIndex
Vector DB:       Milvus (Docker)
LLM Provider:    Novita API (Qwen3-Max)
Embeddings:      Qwen Text Embeddings
Web Search:      Tavily Research API
Deployment:      Docker Compose (local)
```

#### Quick Start

```bash
cd saas-revenue-strategy

# Install dependencies
pip install -r requirements.txt

# Configure environment variables (.env)
# NOVITA_API_KEY=your-key
# TAVILY_API_KEY=your-key

# Start Milvus
docker-compose up -d

# Initialize vector database
python scripts/setup_milvus.py

# Run interactive mode
python main.py --interactive
```

**For detailed documentation**, see [`saas-revenue-strategy/README.md`](./saas-revenue-strategy/README.md)

---

## 🛠️ Common Technologies

Both projects share several core technologies and patterns:

### Core Frameworks

- **Python 3.8+**: Primary development language
- **Docker & Docker Compose**: Containerized infrastructure
- **LlamaIndex**: RAG framework and orchestration (used in SaaS project)

### Data Storage

- **Memgraph**: Graph database for structured relationships (graph-datasources)
- **Milvus**: Vector database for embeddings and similarity search (both projects)
- **GCS (Google Cloud Storage)**: Document storage and backup

### AI/ML Components

- **Qwen Models**: LLM and embeddings (via Novita API)
- **spaCy**: NLP processing for entity extraction
- **Tavily**: Web research and content extraction

### Infrastructure

- **HashiCorp Vault**: Secrets management
- **DragonFly**: Schema storage and retrieval
- **FastAPI**: API framework (hybrid_search agent)

---

## 📋 Prerequisites

### General Requirements

- **Python**: 3.8 or higher
- **Docker & Docker Compose**: For running vector databases and graph databases locally
- **Git**: Version control

### Service Access

Depending on which project you're using:

- **Graph Datasources**:
  - GCP account with access to GCS buckets
  - Access to Memgraph database (local or remote)
  - HashiCorp Vault access (for secrets)
  - DragonFly access (for schema storage)

- **SaaS Revenue Strategy**:
  - **Novita API Key**: For QwenMax LLM & embeddings ([novita.ai](https://novita.ai))
  - **Tavily API Key**: For web research ([tavily.com](https://tavily.com))

---

## 🏗️ Architecture Patterns

### 1. Hybrid Search

Both projects implement hybrid search strategies:

- **Graph-datasources**: Combines Memgraph graph queries with Milvus vector search
- **SaaS Revenue Strategy**: Combines dense embeddings with sparse keyword matching (BM25)

### 2. Checkpointing & Resumability

Processing pipelines support checkpoint-based resumable processing for long-running operations.

### 3. Schema Management

Graph schemas are automatically introspected, versioned, and published for programmatic access.

### 4. Local-First Development

Both projects prioritize local development with Docker containers, enabling:
- Privacy and data sovereignty
- Reproducible environments
- Easy backup and version control of databases

---

## 📚 Documentation

### Project-Specific Documentation

- **Graph Datasources**: [`graph-datasources/README.md`](./graph-datasources/README.md)
  - Comprehensive guide to all pipelines
  - Architecture overview
  - Query examples and schema documentation

- **SaaS Revenue Strategy**: [`saas-revenue-strategy/README.md`](./saas-revenue-strategy/README.md)
  - Quick start guide
  - Architecture details
  - Usage examples and troubleshooting

### Additional Resources

- **SaaS Revenue Strategy**:
  - [`QUICKSTART.md`](./saas-revenue-strategy/QUICKSTART.md) - Step-by-step setup
  - [`PROJECT_STRUCTURE.md`](./saas-revenue-strategy/PROJECT_STRUCTURE.md) - Detailed architecture
  - [`CONTRIBUTING.md`](./saas-revenue-strategy/CONTRIBUTING.md) - Development guidelines

---

## 🔍 Use Cases

### Graph Datasources Use Cases

- **Financial Research**: Query relationships between companies, executives, and financial metrics
- **Regulatory Analysis**: Analyze SEC filings, FOMC decisions, and earnings calls
- **Knowledge Discovery**: Explore connections and patterns across financial documents
- **Policy Research**: Track Federal Reserve policy discussions and economic indicators

### SaaS Revenue Strategy Use Cases

- **Revenue Strategy Research**: Mine insights on SaaS ARR growth strategies
- **Competitive Analysis**: Compare pricing models and business strategies
- **Trend Analysis**: Identify emerging patterns in SaaS revenue strategies
- **Knowledge Mining**: Build a comprehensive knowledge base from web sources

---

## 🔐 Security & Best Practices

- **Secrets Management**: All secrets stored in HashiCorp Vault (graph-datasources) or environment variables (saas-revenue-strategy)
- **Never Commit Secrets**: Use `.env` files (gitignored) for local development
- **Network Security**: Use SSL/TLS for production database connections
- **Access Control**: Follow principle of least privilege for GCP and database access
- **Local-First**: Both projects prioritize local execution to maintain data privacy

---

## 🚦 Project Status

| Project | Status | Last Updated |
|---------|--------|--------------|
| graph-earnings-announcement-transcripts | ✅ Production | 2025 |
| graph-fomc-transcripts | ✅ Production | 2025 |
| graph-sec-10k-filings | ✅ Production | 2025 |
| graph-sec-F4-filings | ✅ Production | 2025 |
| hybrid_search | ✅ Production | 2025 |
| publish_schema | ✅ Production | 2025 |
| saas-revenue-strategy | ✅ Production | 2025 |

---

## 🤝 Contributing

When contributing to either project:

1. **Follow existing patterns**: Maintain consistency with current architecture
2. **Implement checkpointing**: For long-running processes
3. **Document changes**: Update relevant README files and documentation
4. **Test thoroughly**: Verify local setup and functionality
5. **Use proper secrets management**: Never commit API keys or credentials

See individual project READMEs for specific contribution guidelines.

---

## 📄 License

This repository contains multiple projects with different licenses:

- **graph-datasources**: See individual subdirectories for license information
- **saas-revenue-strategy**: MIT License (see [`saas-revenue-strategy/LICENSE`](./saas-revenue-strategy/LICENSE))
- **Root**: See [`LICENSE`](./LICENSE) file (GNU GPL v3)

---

## 🎓 Learning Resources

### Related Technologies

- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Memgraph Documentation](https://memgraph.com/docs/)
- [Milvus Documentation](https://milvus.io/docs/)
- [LangChain Documentation](https://python.langchain.com/) (referenced in repository description)

### Research Inspiration

- **SaaS Revenue Strategy**: Inspired by research from [Clouded Judgement](https://cloudedjudgement.substack.com/) & [Growth Stack Mafia](https://austinhay.substack.com/)
- **Graph Knowledge Processing**: Patterned after modern knowledge graph architectures and RAG systems

---

## 🔄 Future Directions

### Potential Enhancements

- **Multi-language Support**: Extend to international documents and analysis
- **Real-time Dashboards**: Visual interfaces for graph exploration and vector DB insights
- **API Endpoints**: Remote access to query systems
- **Fine-tuned Models**: Domain-specific embedding models
- **Advanced Filtering**: Faceted search and metadata filtering
- **Batch Processing**: Bulk query and analysis capabilities

---

## 📧 Support & Questions

For issues, questions, or feature requests:

1. **Check Documentation**: Review project-specific READMEs first
2. **Open an Issue**: Provide clear description, reproduction steps, and environment details
3. **Review Logs**: Check Docker logs, application logs, and error messages

### Getting Help

- **Graph Datasources**: See [`graph-datasources/README.md`](./graph-datasources/README.md) troubleshooting section
- **SaaS Revenue Strategy**: See [`saas-revenue-strategy/README.md`](./saas-revenue-strategy/README.md) and [`QUICKSTART.md`](./saas-revenue-strategy/QUICKSTART.md)

---

## 🙏 Acknowledgments

### Technologies Used

- [LlamaIndex](https://www.llamaindex.ai/) - RAG framework
- [Memgraph](https://memgraph.com/) - Graph database
- [Milvus](https://milvus.io/) - Vector database
- [Qwen](https://github.com/QwenLM/Qwen) - LLM & embeddings
- [Tavily](https://tavily.com/) - Web research
- [LangChain](https://www.langchain.com/) - Referenced in repository description

### Inspiration

Built for research and experimentation with modern AI technologies, particularly focusing on:
- Knowledge graph construction and querying
- Retrieval-augmented generation systems
- Hybrid search strategies
- Production-ready AI applications

---

## 📝 Notes

- All pipelines support checkpoint-based resumable processing
- Graph schemas are automatically published and versioned
- Both projects are designed for production use with proper error handling and logging
- Local-first architecture ensures data privacy and reproducibility
- Vector database snapshots can be version-controlled for reproducibility

---

**Last Updated**: January 2026  
**Repository Status**: ✅ Active Development  
**Primary Focus**: AI Research, RAG Systems, Knowledge Graphs
