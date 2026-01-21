
import os
from dotenv import load_dotenv

# Load environment variables from potential locations
# Try loading from the sibling directory where existing scripts are
load_dotenv(os.path.join(os.path.dirname(__file__), '../graph-earnings-announcement-transcripts/.env'))
# Also try local .env
load_dotenv()

# Memgraph Connection Details
MEMGRAPH_HOST = os.getenv("MEMGRAPH_HOST", "test-memgraph.blacksmith.deerfieldgreen.com")
MEMGRAPH_PORT = int(os.getenv("MEMGRAPH_PORT", 7687))
MEMGRAPH_USER = os.getenv("MEMGRAPH_USER", "memgraphdb")
MEMGRAPH_PASSWORD = os.getenv("MEMGRAPH_PASSWORD", "")
MEMGRAPH_URI = f"bolt://{MEMGRAPH_HOST}:{MEMGRAPH_PORT}"

# Dragonfly (Redis) Configuration
DRAGONFLY_HOST = os.getenv("DRAGONFLY_HOST", "dragonfly.blacksmith.deerfieldgreen.com")
DRAGONFLY_PORT = int(os.getenv("DRAGONFLY_PORT", "6379"))
DRAGONFLY_PASSWORD = os.getenv("DRAGONFLY_PASSWORD")
DRAGONFLY_DB = int(os.getenv("DRAGONFLY_DB", "0"))
DRAGONFLY_SSL = os.getenv("DRAGONFLY_SSL", "false").lower() == "true"

# OpenAI API Key (using Novita.ai as per existing scripts)
OPENAI_API_KEY = os.getenv("NOVITA_API_KEY")
OPENAI_API_BASE = "https://api.novita.ai/v3/openai"

if not OPENAI_API_KEY:
    # Fallback to standard OpenAI if Novita is not set but OpenAI is
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_API_BASE = None

# Gemini API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Milvus Configuration
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "root:Milvus")  # Default: root:Milvus
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "default_collection")
MILVUS_DENSE_FIELD = os.getenv("MILVUS_DENSE_FIELD", "dense_vector")
MILVUS_SPARSE_FIELD = os.getenv("MILVUS_SPARSE_FIELD", "sparse_vector")
MILVUS_HYBRID_LIMIT = int(os.getenv("MILVUS_HYBRID_LIMIT", "10"))
MILVUS_DENSE_CANDIDATES = int(os.getenv("MILVUS_DENSE_CANDIDATES", "15"))
MILVUS_SPARSE_CANDIDATES = int(os.getenv("MILVUS_SPARSE_CANDIDATES", "15"))
MILVUS_RRF_K = int(os.getenv("MILVUS_RRF_K", "60"))
