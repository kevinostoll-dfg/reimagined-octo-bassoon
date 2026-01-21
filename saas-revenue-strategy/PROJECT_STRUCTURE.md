# SaaS Revenue Strategy RAG Agent - Project Structure

## Overview

This document provides a complete overview of the project structure and file organization.

## Directory Structure

```
saas-revenue-strategy/
├── agents/                     # Core agent implementations
│   ├── __init__.py            # Package initialization
│   ├── knowledge_miner.py     # Mining agent (web scraping, indexing)
│   └── query_agent.py         # Query agent (retrieval, synthesis)
│
├── scripts/                    # Utility scripts
│   ├── __init__.py            # Package initialization
│   ├── setup_milvus.py        # Initialize Milvus collection
│   ├── backup_vector_db.py    # Backup vector database
│   ├── restore_vector_db.py   # Restore from backup
│   └── verify_setup.py        # Verify installation
│
├── config/                     # Configuration files
│   ├── agents.yaml            # Agent parameters
│   └── sources.yaml           # Data sources configuration
│
├── backups/                    # Vector database backups
│   └── .gitkeep               # Directory placeholder
│
├── main.py                     # Main CLI entry point
├── requirements.txt            # Python dependencies
├── docker-compose.yml          # Milvus Docker setup
├── .env.example               # Environment template
├── .gitignore                 # Git ignore rules
│
├── README.md                   # Main documentation
├── QUICKSTART.md              # Quick start guide
├── CONTRIBUTING.md            # Contribution guidelines
├── LICENSE                     # MIT License
└── PROJECT_STRUCTURE.md       # This file
```

## File Descriptions

### Core Application Files

#### `main.py`
- Main CLI entry point
- Provides unified interface for both agents
- Supports interactive and command-line modes
- Modes: mine knowledge, query, interactive

#### `agents/knowledge_miner.py`
- Knowledge mining agent implementation
- Web scraping via Tavily API
- Document processing and chunking
- Embedding generation
- Vector database indexing

#### `agents/query_agent.py`
- Query and research agent implementation
- Hybrid vector search
- ReAct agent with tools
- Web research integration
- Answer synthesis and formatting

### Utility Scripts

#### `scripts/setup_milvus.py`
- Initialize Milvus connection
- Create collection schema
- Set up indexes
- First-time setup

#### `scripts/backup_vector_db.py`
- Backup vector database to tar.gz
- Export collection metadata
- Export document data
- Version control support

#### `scripts/restore_vector_db.py`
- Restore from backup file
- Recreate collection schema
- Re-import documents
- Rebuild indexes

#### `scripts/verify_setup.py`
- Check environment variables
- Verify Python packages
- Test Milvus connection
- Validate Docker setup
- Check configuration files

### Configuration Files

#### `config/agents.yaml`
- Agent parameters
- LLM configuration
- Embedding settings
- Search parameters
- Temperature and max tokens

#### `config/sources.yaml`
- Data sources (Substack, blogs)
- Research topics
- Seed queries
- Source priorities

#### `.env.example`
- Environment variable template
- API key placeholders
- Milvus connection settings
- Default configurations

#### `docker-compose.yml`
- Milvus standalone setup
- etcd for metadata
- MinIO for object storage
- Volume mappings
- Port configurations

### Documentation

#### `README.md`
- Complete project documentation
- Architecture overview
- Installation instructions
- Usage examples
- Technical details
- Troubleshooting

#### `QUICKSTART.md`
- Step-by-step setup guide
- Basic usage examples
- Common workflows
- Troubleshooting tips
- Quick reference

#### `CONTRIBUTING.md`
- Contribution guidelines
- Development setup
- Code style standards
- Testing guidelines
- Review process

#### `LICENSE`
- MIT License
- Copyright information
- Usage permissions

## Component Interactions

### Knowledge Mining Flow

```
User → main.py → KnowledgeMiner
                      ↓
                 Tavily API (web search)
                      ↓
                 Document Processing
                      ↓
                 Embedding Generation (Novita)
                      ↓
                 Milvus Vector Store
```

### Query Flow

```
User → main.py → QueryAgent
                      ↓
                 Vector Search (Milvus)
                      ↓
                 ReAct Agent
                      ├→ Knowledge Base Tool
                      └→ Web Search Tool (Tavily)
                      ↓
                 Answer Synthesis (Novita LLM)
                      ↓
                 Formatted Response
```

## Key Dependencies

### Core Frameworks
- **LlamaIndex**: RAG framework and orchestration
- **PyMilvus**: Vector database client
- **OpenAI**: LLM client (Novita-compatible)

### Tools & APIs
- **Tavily**: Web research API
- **Novita**: LLM and embeddings API
- **Docker**: Milvus containerization

### Utilities
- **python-dotenv**: Environment management
- **PyYAML**: Configuration parsing
- **Rich**: CLI formatting
- **Click**: Command-line interface

## Configuration Management

### Environment Variables (`.env`)
```
NOVITA_API_KEY          # Required: Novita API key
TAVILY_API_KEY          # Required: Tavily API key
MILVUS_HOST             # Optional: Default localhost
MILVUS_PORT             # Optional: Default 19530
MILVUS_COLLECTION_NAME  # Optional: Collection name
```

### Agent Configuration (`config/agents.yaml`)
- LLM model and parameters
- Embedding model settings
- Search parameters
- Mining settings

### Sources Configuration (`config/sources.yaml`)
- Substack sources
- Blog sources
- Research topics
- Seed queries

## Data Storage

### Milvus Collections
- **Collection Name**: saas_revenue_knowledge (default)
- **Schema**: Text, metadata, embeddings
- **Index Type**: IVF_FLAT
- **Metric**: Cosine similarity

### Backup Format
- **Format**: tar.gz archive
- **Contents**: metadata.json + data.json
- **Location**: `backups/` directory
- **Naming**: milvus_snapshot_YYYYMMDD_HHMMSS.tar.gz

## Development Workflow

1. **Setup**: Install dependencies, configure environment
2. **Initialize**: Start Milvus, create collection
3. **Mine**: Gather and index knowledge
4. **Query**: Ask questions and get answers
5. **Maintain**: Backup, update, monitor

## Testing

- Python syntax validation
- Configuration validation
- Connection testing
- End-to-end workflow testing

## Security Considerations

- API keys in `.env` (not committed)
- Sensitive data not logged
- Docker containers isolated
- Local-first architecture

## Performance Optimization

- Batch processing for embeddings
- Efficient vector search
- Connection pooling
- Proper indexing strategies

---

**Last Updated**: January 2026
**Version**: 1.0.0
