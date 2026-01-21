# Quick Start Guide

## Prerequisites Setup

### 1. Install Docker

Make sure Docker and Docker Compose are installed on your system:

```bash
# Check Docker installation
docker --version
docker-compose --version
```

If not installed, visit [Docker's official site](https://docs.docker.com/get-docker/).

### 2. Get API Keys

You'll need two API keys:

1. **Novita API Key** - For LLM and embeddings
   - Sign up at https://novita.ai
   - Get your API key from the dashboard

2. **Tavily API Key** - For web research
   - Sign up at https://tavily.com
   - Get your API key from the dashboard

### 3. Install Python Dependencies

```bash
cd saas-revenue-strategy
pip install -r requirements.txt
```

## Initial Setup

### 1. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
nano .env  # or use your favorite editor
```

Update these lines in `.env`:
```
NOVITA_API_KEY=your-actual-novita-key-here
TAVILY_API_KEY=your-actual-tavily-key-here
```

### 2. Start Milvus

```bash
# Start Milvus in Docker
docker-compose up -d

# Wait 30-60 seconds for services to initialize
sleep 60

# Check if services are running
docker-compose ps
```

You should see three containers running:
- milvus-etcd
- milvus-minio
- milvus-standalone

### 3. Initialize Vector Database

```bash
# Create the collection schema
python scripts/setup_milvus.py
```

### 4. Verify Setup

```bash
# Run the verification script
python scripts/verify_setup.py
```

This will check:
- Environment variables are set
- Python packages are installed
- Docker is running
- Milvus is accessible
- Collection is created

## Using the System

### Option 1: Interactive Mode (Recommended for First Time)

```bash
python main.py --interactive
```

This starts an interactive session where you can:
1. Ask questions
2. Get detailed answers with citations
3. Continue the conversation

### Option 2: Single Query

```bash
python main.py --query "What are the top SaaS ARR growth strategies?"
```

### Option 3: Mine Knowledge First

```bash
# Mine knowledge from configured sources
python main.py --mine

# Then query
python main.py --interactive
```

## Example Workflow

### 1. Mine Knowledge

```bash
# Start knowledge mining
python agents/knowledge_miner.py

# Or mine specific topics
python agents/knowledge_miner.py --topics "SaaS pricing models" "ARR optimization"
```

This will:
- Search the web for relevant content
- Process and chunk the documents
- Generate embeddings
- Store in Milvus

### 2. Query the Knowledge Base

```bash
# Interactive mode
python agents/query_agent.py --interactive
```

Try these example queries:
```
1. What are the top SaaS companies optimizing for ARR growth?
2. Compare pricing models across different SaaS segments
3. What is the relationship between CAC and LTV?
4. What are emerging trends in SaaS revenue strategy?
```

## Maintenance

### Backup Vector Database

```bash
# Create a backup
python scripts/backup_vector_db.py

# This creates: backups/milvus_snapshot_<timestamp>.tar.gz
```

### Restore Vector Database

```bash
# Restore from a backup
python scripts/restore_vector_db.py --input backups/milvus_snapshot_YYYYMMDD_HHMMSS.tar.gz
```

### Update Knowledge Base

Run the mining agent periodically to keep knowledge fresh:

```bash
# Weekly or monthly
python agents/knowledge_miner.py
```

## Troubleshooting

### Milvus Connection Failed

```bash
# Check if containers are running
docker-compose ps

# Restart Milvus
docker-compose restart

# Check logs
docker-compose logs standalone
```

### Python Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### API Key Errors

```bash
# Verify environment variables are loaded
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('NOVITA:', bool(os.getenv('NOVITA_API_KEY'))); print('TAVILY:', bool(os.getenv('TAVILY_API_KEY')))"
```

### Collection Not Found

```bash
# Recreate the collection
python scripts/setup_milvus.py
```

## Advanced Usage

### Custom Configuration

Edit `config/agents.yaml` to customize:
- LLM temperature
- Number of results (top_k)
- Chunk size
- Search parameters

Edit `config/sources.yaml` to customize:
- Data sources
- Research topics
- Priority settings

### Batch Processing

Create a script to process multiple queries:

```python
from agents.query_agent import QueryAgent

agent = QueryAgent()

queries = [
    "Query 1",
    "Query 2",
    "Query 3"
]

for query in queries:
    response = agent.query(query)
    # Save or process response
```

## Next Steps

1. **Explore the codebase** - Understand how agents work
2. **Customize configurations** - Tailor to your needs
3. **Add more sources** - Expand the knowledge base
4. **Integrate with workflows** - Use in your research process

## Getting Help

- Check the main [README.md](README.md) for detailed documentation
- Run `python scripts/verify_setup.py` to diagnose issues
- Review configuration files in `config/` directory
- Check Docker logs: `docker-compose logs`

## Tips

1. **Start small** - Mine a few topics first to test
2. **Monitor costs** - Be aware of API usage
3. **Backup regularly** - Use the backup script
4. **Update often** - Keep knowledge base current
5. **Experiment** - Try different query formulations
