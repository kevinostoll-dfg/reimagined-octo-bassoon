#!/usr/bin/env python3
"""
Knowledge Mining Agent for SaaS Revenue Strategy RAG Agent.
Datamines websites, Substack publications, and internet resources.
"""

import os
import sys
import time
import yaml
import json
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

from pymilvus import connections, Collection, utility
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai_like import OpenAILike
from agents.novita_embedding import NovitaEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore

from agents.logging_utils import log_payload, redact_secret

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    print("⚠ Warning: tavily-python not installed. Web search functionality limited.")

# Environment variables are read from the process environment.


class KnowledgeMiner:
    """Agent for mining and indexing SaaS revenue strategy knowledge."""
    
    def __init__(self, config_path: str = "config/agents.yaml"):
        """Initialize the Knowledge Mining Agent."""
        self.logger = logging.getLogger(self.__class__.__name__)
        log_payload(
            self.logger,
            "knowledge_miner.init.start",
            {"config_path": config_path}
        )

        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config.get("knowledge_miner", {})
        log_payload(
            self.logger,
            "knowledge_miner.config.loaded",
            {"config": self.config}
        )
        
        # Initialize LLM with Novita API
        api_key = os.getenv("NOVITA_API_KEY")
        if not api_key:
            raise ValueError("NOVITA_API_KEY not set in environment")
        
        api_base = os.getenv("NOVITA_API_BASE", "https://api.novita.ai/openai")
        log_payload(
            self.logger,
            "knowledge_miner.novita.settings",
            {
                "api_base": api_base,
                "api_key": redact_secret(api_key),
                "model": self.config.get("model", "qwen/qwen3-max"),
                "temperature": self.config.get("temperature", 0.3),
                "max_tokens": self.config.get("max_tokens", 2048),
            }
        )

        self.llm = OpenAILike(
            model=self.config.get("model", "qwen/qwen3-max"),
            api_base=api_base,
            api_key=api_key,
            is_chat_model=True,
            is_function_calling_model=True,
            temperature=self.config.get("temperature", 0.3),
            max_tokens=self.config.get("max_tokens", 2048),
        )
        
        # Initialize embeddings
        self.embedding_model = NovitaEmbedding(
            model=self.config.get("embedding_model", "qwen/qwen3-embedding-8b"),
            api_base=api_base,
            api_key=api_key,
            embed_batch_size=self.config.get("batch_size", 32)
        )
        log_payload(
            self.logger,
            "knowledge_miner.embedding.settings",
            {
                "model": self.config.get("embedding_model", "qwen/qwen3-embedding-8b"),
                "embed_batch_size": self.config.get("batch_size", 32),
            }
        )
        
        # Set global settings
        Settings.llm = self.llm
        Settings.embed_model = self.embedding_model
        Settings.chunk_size = self.config.get("chunk_size", 1024)
        Settings.chunk_overlap = self.config.get("chunk_overlap", 200)
        log_payload(
            self.logger,
            "knowledge_miner.llama_index.settings",
            {
                "chunk_size": Settings.chunk_size,
                "chunk_overlap": Settings.chunk_overlap,
            }
        )
        
        # Initialize Tavily client if available
        if TAVILY_AVAILABLE:
            tavily_key = os.getenv("TAVILY_API_KEY")
            if tavily_key:
                self.tavily_client = TavilyClient(api_key=tavily_key)
            else:
                self.tavily_client = None
                print("⚠ TAVILY_API_KEY not set")
        else:
            self.tavily_client = None
        log_payload(
            self.logger,
            "knowledge_miner.tavily.status",
            {
                "tavily_available": TAVILY_AVAILABLE,
                "tavily_configured": bool(os.getenv("TAVILY_API_KEY")),
            }
        )
        
        # Connect to Milvus
        self._connect_milvus()
        
        print(f"✓ Knowledge Mining Agent initialized")
    
    def _connect_milvus(self):
        """Connect to Milvus vector database."""
        host = os.getenv("MILVUS_HOST", "localhost")
        port = os.getenv("MILVUS_PORT", "19530")
        collection_name = os.getenv("MILVUS_COLLECTION_NAME", "saas_revenue_knowledge")
        milvus_uri = os.getenv("MILVUS_URI") or f"http://{host}:{port}"
        log_payload(
            self.logger,
            "knowledge_miner.milvus.connect",
            {
                "host": host,
                "port": port,
                "collection_name": collection_name,
                "milvus_uri": milvus_uri,
                "embedding_dimension": int(os.getenv("EMBEDDING_DIMENSION", "4096")),
            }
        )
        
        try:
            async def _init_vector_store():
                return MilvusVectorStore(
                    uri=milvus_uri,
                    collection_name=collection_name,
                    dim=int(os.getenv("EMBEDDING_DIMENSION", "4096")),
                    overwrite=False,
                )

            # Ensure a running event loop exists for async Milvus client init
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                self.vector_store = asyncio.run(_init_vector_store())
            else:
                self.vector_store = MilvusVectorStore(
                    uri=milvus_uri,
                    collection_name=collection_name,
                    dim=int(os.getenv("EMBEDDING_DIMENSION", "4096")),
                    overwrite=False,
                )
            print(f"✓ Connected to Milvus collection: {collection_name}")
            log_payload(
                self.logger,
                "knowledge_miner.milvus.connected",
                {"collection_name": collection_name}
            )
            
        except Exception as e:
            print(f"✗ Error connecting to Milvus: {e}")
            print("Make sure Milvus is running: docker-compose up -d")
            sys.exit(1)
    
    def mine_from_web_search(self, topics: List[str], max_results: int = 5) -> List[Document]:
        """Mine knowledge from web search results."""
        
        if not self.tavily_client:
            print("⚠ Tavily client not available, skipping web search")
            log_payload(
                self.logger,
                "knowledge_miner.web_search.skipped",
                {"reason": "tavily_client_unavailable"}
            )
            return []
        
        documents = []
        log_payload(
            self.logger,
            "knowledge_miner.web_search.start",
            {"topics": topics, "max_results": max_results}
        )
        
        for topic in topics:
            print(f"Searching for: {topic}")
            log_payload(
                self.logger,
                "knowledge_miner.web_search.request",
                {
                    "query": topic,
                    "search_depth": "advanced",
                    "max_results": max_results,
                }
            )
            
            try:
                # Perform web search
                results = self.tavily_client.search(
                    query=topic,
                    search_depth="advanced",
                    max_results=max_results
                )
                
                # Convert results to documents
                for result in results.get("results", []):
                    doc = Document(
                        text=result.get("content", ""),
                        metadata={
                            "source": "web_search",
                            "url": result.get("url", ""),
                            "title": result.get("title", ""),
                            "topic": topic,
                            "timestamp": int(datetime.now().timestamp())
                        }
                    )
                    documents.append(doc)
                    log_payload(
                        self.logger,
                        "knowledge_miner.web_search.document",
                        {
                            "topic": topic,
                            "title": result.get("title", ""),
                            "url": result.get("url", ""),
                            "text_length": len(result.get("content", "") or ""),
                            "metadata": doc.metadata,
                        }
                    )
                
                print(f"  ✓ Found {len(results.get('results', []))} results")
                log_payload(
                    self.logger,
                    "knowledge_miner.web_search.response",
                    {
                        "topic": topic,
                        "result_count": len(results.get("results", [])),
                    }
                )
                
                # Rate limiting
                time.sleep(self.config.get("crawl_delay", 1.0))
                
            except Exception as e:
                print(f"  ✗ Error searching for '{topic}': {e}")
                log_payload(
                    self.logger,
                    "knowledge_miner.web_search.error",
                    {"topic": topic, "error": str(e)}
                )
        
        return documents
    
    def mine_from_sources(self, sources_config_path: str = "config/sources.yaml") -> List[Document]:
        """Mine knowledge from configured sources."""
        
        # Load sources configuration
        with open(sources_config_path, 'r') as f:
            sources = yaml.safe_load(f)
        log_payload(
            self.logger,
            "knowledge_miner.sources.loaded",
            {"sources_config_path": sources_config_path, "sources": sources}
        )
        
        # Use research topics for web search
        topics = sources.get("research_topics", [])
        
        print(f"Mining knowledge from {len(topics)} topics...")
        log_payload(
            self.logger,
            "knowledge_miner.sources.topics",
            {"topics": topics}
        )
        documents = self.mine_from_web_search(topics)
        
        return documents
    
    def index_documents(self, documents: List[Document]):
        """Index documents into Milvus vector store."""
        
        if not documents:
            print("⚠ No documents to index")
            log_payload(
                self.logger,
                "knowledge_miner.index.skipped",
                {"reason": "no_documents"}
            )
            return
        
        print(f"Indexing {len(documents)} documents...")
        log_payload(
            self.logger,
            "knowledge_miner.index.start",
            {
                "document_count": len(documents),
                "chunk_size": Settings.chunk_size,
                "chunk_overlap": Settings.chunk_overlap,
            }
        )
        
        try:
            # Create index from documents
            index = VectorStoreIndex.from_documents(
                documents,
                vector_store=self.vector_store
            )
            
            print(f"✓ Successfully indexed {len(documents)} documents")
            log_payload(
                self.logger,
                "knowledge_miner.index.success",
                {"document_count": len(documents)}
            )
            
        except Exception as e:
            print(f"✗ Error indexing documents: {e}")
            log_payload(
                self.logger,
                "knowledge_miner.index.error",
                {"error": str(e)}
            )
            raise
    
    def run(self, sources_config_path: str = "config/sources.yaml"):
        """Run the knowledge mining process."""
        
        print("="*60)
        print("Knowledge Mining Agent - Starting")
        print("="*60)
        log_payload(
            self.logger,
            "knowledge_miner.run.start",
            {"sources_config_path": sources_config_path}
        )
        
        # Mine documents from sources
        documents = self.mine_from_sources(sources_config_path)
        
        if not documents:
            print("⚠ No documents mined. Check your configuration and API keys.")
            log_payload(
                self.logger,
                "knowledge_miner.run.no_documents",
                {"sources_config_path": sources_config_path}
            )
            return
        
        # Index documents
        self.index_documents(documents)
        
        print("="*60)
        print("Knowledge Mining Agent - Completed")
        print("="*60)
        print(f"Total documents indexed: {len(documents)}")
        log_payload(
            self.logger,
            "knowledge_miner.run.completed",
            {"document_count": len(documents)}
        )


def main():
    """Main entry point for the knowledge mining agent."""
    
    import argparse

    from agents.logging_utils import configure_logging
    configure_logging(verbose=True)
    
    parser = argparse.ArgumentParser(
        description="Knowledge Mining Agent for SaaS Revenue Strategy"
    )
    parser.add_argument(
        "--sources",
        default="config/sources.yaml",
        help="Path to sources configuration file"
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        help="Specific topics to mine (overrides config)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize agent
        miner = KnowledgeMiner()
        
        # Override topics if provided
        if args.topics:
            print(f"Mining specific topics: {args.topics}")
            documents = miner.mine_from_web_search(args.topics)
            miner.index_documents(documents)
        else:
            # Run with sources config
            miner.run(args.sources)
        
    except KeyboardInterrupt:
        print("\n⚠ Mining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
