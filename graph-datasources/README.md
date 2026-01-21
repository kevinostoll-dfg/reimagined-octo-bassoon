# Graph Datasources

> **Comprehensive graph database processing pipelines for financial and regulatory documents**

This directory contains end-to-end processing pipelines for ingesting, analyzing, and storing financial and regulatory documents in Memgraph graph databases. The pipelines process various document types including earnings announcements, FOMC transcripts, and SEC filings to create queryable knowledge graphs.

## 🎯 Overview

The graph-datasources project provides:

- **Document Ingestion**: Download and process financial documents from various sources
- **Graph Processing**: Extract entities, relationships, and metadata to build knowledge graphs
- **Storage & Retrieval**: Store processed data in Memgraph with optimized queries and indexes
- **Hybrid Search**: Combine graph-based RAG with vector search for advanced querying
- **Infrastructure**: Docker setups, GCS bucket management, and model fine-tuning utilities

## 📁 Directory Structure

### Core Processing Pipelines

#### 📈 `graph-earnings-announcement-transcripts/`
Processes earnings call transcripts to extract:
- Company metrics, financial statements, and executive commentary
- Entity relationships (PERSON, ORG, PRODUCT, CONCEPT, METRIC)
- Temporal context and sentiment analysis
- Knowledge graph storage in Memgraph

**Key Features:**
- Batch processing with checkpointing
- GCS integration for transcript storage
- Schema publishing to DragonFly
- Graph operations (indexes, views, query optimization)

#### 🏛️ `graph-fomc-transcripts/`
Processes Federal Reserve Open Market Committee (FOMC) transcripts:
- Federal Reserve policy discussions and decisions
- Economic indicators and monetary policy analysis
- Historical FOMC meeting documentation
- RSS feed consumption for transcript updates

**Key Features:**
- RSS feed processing for new transcripts
- spaCy-based NLP processing
- Graph operations for policy analysis queries

#### 📄 `graph-sec-10k-filings/`
Processes SEC Form 10-K annual reports:
- Company annual financial statements and disclosures
- Risk factors, business operations, and management discussion
- Regulatory compliance data

#### 💼 `graph-sec-F4-filings/`
Processes SEC Form 4 insider trading filings:
- Insider trading transactions
- Company insider information and holdings
- Transaction details and timing

**Key Features:**
- Historical data processing (2009+)
- Batch processing with checkpoint validation
- Document size analysis and optimization

### Infrastructure & Utilities

#### 🔍 `hybrid_search/`
FastAPI-based agent that combines graph RAG with vector search:

**Features:**
- Graph-based retrieval from Memgraph
- Vector search via Milvus
- Multi-tool agent pattern (GraphRAG, Tavily, FMP, DateTime)
- Completion routing and benchmarking
- Example weekly market analysis workflows

**Architecture:**
- `agent/` - Agent configuration and completion routing
- `tools/` - GraphRAG, Milvus search, Tavily search, FMP data tools
- `prompts/` - Prompt templates and utilities
- `scripts/` - Benchmarking, schema fetching, exploration tools

#### 🗄️ `memgraph_docker/`
Docker Compose setup for local Memgraph development:
- Memgraph database instance
- Memgraph Lab UI (port 3000)
- Persistent volume storage

#### 📦 `publish_schema/`
Automated schema introspection and publishing service:
- Queries Memgraph to extract complete graph schemas
- Tags schema elements by service (ea, fomc, sec-10k, sec-f4)
- Stores schemas in DragonFly for programmatic access
- Generates service-specific schema views

#### ☁️ `gcs_bucket/`
GCS bucket policy management:
- Lifecycle and retention policy auditing
- Automated policy application
- Backup and restore functionality
- Cost optimization (Nearline → Coldline → Archive)

#### 🤖 `finetune_model/`
Model fine-tuning infrastructure:
- Classification model training
- Benchmark and comparison utilities
- GPU node creation and validation
- Model upload to GCS

#### 🔧 `compute_node/`
GPU compute node management:
- Automated GPU node creation scripts
- GPU validation and health checks
- Compute infrastructure automation

#### 📚 `spacy_model/`
spaCy model management:
- Model upload to GCS
- NLP model versioning and distribution

## 🏗️ Architecture

### Data Flow

```
Document Source
    ↓
Download/Ingest (GCS)
    ↓
Process & Extract Entities
    ↓
Build Knowledge Graph (Memgraph)
    ↓
Index & Optimize
    ↓
Query & Retrieve (Hybrid Search)
```

### Common Patterns

#### 1. **Checkpointing**
All processing pipelines use checkpoint files to:
- Resume processing after interruptions
- Track processing progress
- Validate data integrity

#### 2. **Graph Operations**
Each pipeline includes `graph-operations/` with:
- `create_indexes.py` - Performance optimization
- `create_views.py` - Query simplification
- `optimize_queries.md` - Query optimization guidelines

#### 3. **Vault Integration**
Secrets management via HashiCorp Vault:
- Environment variable discovery from code
- Automated secret storage
- Service-specific secret paths

#### 4. **Schema Publishing**
All pipelines publish schemas to DragonFly:
- Complete cross-service schema view
- Service-specific filtered schemas
- Programmatic schema access

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Docker & Docker Compose
- GCP account with access to GCS buckets
- Access to Memgraph database (local or remote)
- HashiCorp Vault access (for secrets)

### Quick Start

1. **Start Local Memgraph:**
```bash
cd memgraph_docker
docker-compose up -d
```

2. **Install Dependencies:**
```bash
# Each pipeline has its own requirements.txt
cd graph-earnings-announcement-transcripts
pip install -r requirements.txt
```

3. **Configure Environment:**
```bash
# Set required environment variables
export MEMGRAPH_HOST=localhost
export MEMGRAPH_PORT=7687
export GCS_BUCKET=your-bucket-name
# ... other variables
```

4. **Process Documents:**
```bash
# Example: Process earnings announcements
cd graph-earnings-announcement-transcripts
python batch_process_ea.py
```

### Individual Pipeline Setup

Each pipeline directory contains detailed documentation:

- **graph-earnings-announcement-transcripts/** - See vault_operations/README.md
- **graph-fomc-transcripts/** - See load_data_gcs/README.md
- **graph-sec-F4-filings/** - See vault_operations/README.md
- **publish_schema/** - See README.md
- **gcs_bucket/** - See README.md

## 🔧 Configuration

### Environment Variables

Common environment variables across pipelines:

```bash
# Memgraph Configuration
MEMGRAPH_HOST=localhost
MEMGRAPH_PORT=7687
MEMGRAPH_USER=memgraphdb
MEMGRAPH_PASSWORD=memgraphdb

# GCS Configuration
GCS_BUCKET=blacksmith-sec-filings
GCP_PROJECT=your-project-id

# Vault Configuration
VAULT_TOKEN=your-vault-token
VAULT_ADDR=https://vault.zagreus.deerfieldgreen.com

# DragonFly Configuration
DRAGONFLY_HOST=dragonfly.blacksmith.deerfieldgreen.com
DRAGONFLY_PORT=6379
```

### Vault Secrets

Each pipeline stores secrets in HashiCorp Vault:

- **Path**: `blacksmith-project-secrets/{pipeline-name}`
- **Format**: KV v2
- **Management**: Use `vault_operations/push_to_vault.py` in each pipeline

## 📊 Graph Schemas

### Common Node Labels

- `PERSON` - People mentioned in documents
- `ORG` - Organizations and companies
- `PRODUCT` - Products and services
- `CONCEPT` - Abstract concepts
- `METRIC` - Financial and business metrics
- `METRIC_DEFINITION` - Definitions of metrics
- `STATEMENT` - Document statements
- `DATE`, `TIME`, `MONEY`, `PERCENT`, `CARDINAL` - Extracted entities

### Common Relationship Types

- `SVO_TRIPLE` - Subject-Verb-Object triplets
- `TEMPORAL_CONTEXT` - Time-based relationships
- `CO_MENTIONED` - Co-occurrence relationships
- `CAUSES`, `DRIVES`, `BOOSTS`, `HURTS` - Causal relationships
- `RESULTS_IN`, `LEADS_TO` - Outcome relationships
- `POSITIVE_ABOUT`, `NEGATIVE_ABOUT` - Sentiment relationships

### Schema Discovery

View complete schemas via the publish_schema service:

```bash
cd publish_schema
python store_schema_dragonfly.py
```

Then query DragonFly for schemas:
- `memgraph:cipher:full` - Complete schema
- `memgraph:cipher:ea` - Earnings announcements only
- `memgraph:cipher:fomc` - FOMC transcripts only
- `memgraph:cipher:sec-10k` - SEC 10-K filings only
- `memgraph:cipher:sec-f4` - SEC Form 4 filings only

## 🔍 Querying

### Hybrid Search Agent

The hybrid search agent provides a unified interface for querying:

```bash
cd hybrid_search
python main.py
```

**Available Tools:**
- `graphrag_tool` - Graph-based retrieval from Memgraph
- `milvus_search_tool` - Vector similarity search
- `tavily_search_tool` - Web search integration
- `fmp_tool` - Financial Modeling Prep API
- `datetime_tool` - Date/time manipulation

### Direct Memgraph Queries

Connect to Memgraph and query using CIPHER:

```python
from gqlalchemy import Memgraph

db = Memgraph(host="localhost", port=7687)

# Find companies mentioned in earnings calls
result = db.execute_and_fetch(
    "MATCH (org:ORG)-[:CO_MENTIONED]->(metric:METRIC) "
    "RETURN org, metric LIMIT 10"
)
```

## 🛠️ Development

### Adding a New Pipeline

1. Create a new directory following naming convention: `graph-{source}-{doc-type}/`
2. Implement processing scripts with checkpoint support
3. Add `graph-operations/` for indexes and views
4. Add `vault_operations/` for secret management
5. Update `publish_schema/store_schema_dragonfly.py` with new service definition

### Benchmarking

Use the hybrid search benchmarking tools:

```bash
cd hybrid_search/scripts
python benchmark_agent.py --config benchmark_config.json
```

## 📚 Documentation

### Pipeline-Specific Documentation

- **graph-earnings-announcement-transcripts/**
  - `memgraph_report.md` - Memgraph statistics and analysis
  - `graph-operations/optimize_queries.md` - Query optimization guide

- **graph-fomc-transcripts/**
  - `spacy_metadata.md` - NLP processing metadata
  - `load_data_gcs/README.md` - RSS consumption guide

- **graph-sec-F4-filings/**
  - `secapi-docs.md` - SEC API documentation
  - `notes.md` - Processing notes and findings

- **hybrid_search/**
  - `scripts/BENCHMARK_README.md` - Benchmarking guide

### Common Documentation

- **publish_schema/README.md** - Comprehensive schema publishing guide
- **gcs_bucket/README.md** - GCS policy management guide

## 🔒 Security

- **Secrets**: All secrets stored in HashiCorp Vault
- **Credentials**: Use environment variables, never commit secrets
- **Network**: Use SSL/TLS for production database connections
- **Access**: Follow principle of least privilege for GCP and database access

## 🚦 Status

| Pipeline | Status | Last Updated |
|----------|--------|--------------|
| graph-earnings-announcement-transcripts | ✅ Production | 2025 |
| graph-fomc-transcripts | ✅ Production | 2025 |
| graph-sec-10k-filings | ✅ Production | 2025 |
| graph-sec-F4-filings | ✅ Production | 2025 |
| hybrid_search | ✅ Production | 2025 |
| publish_schema | ✅ Production | 2025 |

## 📝 Notes

- All pipelines support checkpoint-based resumable processing
- Graph schemas are automatically published and versioned
- Processing pipelines are designed for batch operations
- Hybrid search provides real-time query capabilities

## 🤝 Contributing

When adding or modifying pipelines:

1. Follow existing directory structure and naming conventions
2. Implement checkpointing for long-running processes
3. Add graph operations (indexes, views) for performance
4. Document schema changes in publish_schema
5. Update this README with new pipelines

---

**Last Updated**: December 2025  
**Version**: 1.0  
**Status**: ✅ Production Ready