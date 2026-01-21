#!/usr/bin/env python3
"""
Query & Research Agent for SaaS Revenue Strategy RAG Agent.
Retrieves and synthesizes research from the knowledge base.
"""

import os
import sys
import yaml
import asyncio
import logging
from typing import List, Optional
from datetime import datetime

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.openai_like import OpenAILike
from agents.novita_embedding import NovitaEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool

from agents.logging_utils import log_payload, redact_secret

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Environment variables are read from the process environment.


class QueryAgent:
    """Agent for querying and synthesizing research from the knowledge base."""
    
    def __init__(self, config_path: str = "config/agents.yaml"):
        """Initialize the Query & Research Agent."""
        self.logger = logging.getLogger(self.__class__.__name__)
        log_payload(
            self.logger,
            "query_agent.init.start",
            {"config_path": config_path}
        )

        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config.get("query_agent", {})
        log_payload(
            self.logger,
            "query_agent.config.loaded",
            {"config": self.config}
        )
        
        # Initialize console for rich output
        if RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None
        
        # Initialize LLM with Novita API
        api_key = os.getenv("NOVITA_API_KEY")
        if not api_key:
            raise ValueError("NOVITA_API_KEY not set in environment")
        
        api_base = os.getenv("NOVITA_API_BASE", "https://api.novita.ai/openai")
        log_payload(
            self.logger,
            "query_agent.novita.settings",
            {
                "api_base": api_base,
                "api_key": redact_secret(api_key),
                "model": self.config.get("model", "qwen/qwen3-max"),
                "temperature": self.config.get("temperature", 0.7),
                "max_tokens": self.config.get("max_tokens", 4096),
            }
        )

        self.llm = OpenAILike(
            model=self.config.get("model", "qwen/qwen3-max"),
            api_base=api_base,
            api_key=api_key,
            is_chat_model=True,
            is_function_calling_model=True,
            temperature=self.config.get("temperature", 0.7),
            max_tokens=self.config.get("max_tokens", 4096),
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
            "query_agent.embedding.settings",
            {
                "model": self.config.get("embedding_model", "qwen/qwen3-embedding-8b"),
                "embed_batch_size": self.config.get("batch_size", 32),
            }
        )
        
        # Set global settings
        Settings.llm = self.llm
        Settings.embed_model = self.embedding_model
        log_payload(
            self.logger,
            "query_agent.llama_index.settings",
            {"embed_model": str(self.embedding_model)}
        )
        
        # Initialize Tavily client if available
        self.web_research_enabled = self.config.get("enable_web_research", True)
        if self.web_research_enabled and TAVILY_AVAILABLE:
            tavily_key = os.getenv("TAVILY_API_KEY")
            if tavily_key:
                self.tavily_client = TavilyClient(api_key=tavily_key)
            else:
                self.tavily_client = None
                print("⚠ TAVILY_API_KEY not set, web research disabled")
        else:
            self.tavily_client = None
        log_payload(
            self.logger,
            "query_agent.tavily.status",
            {
                "web_research_enabled": self.web_research_enabled,
                "tavily_available": TAVILY_AVAILABLE,
                "tavily_configured": bool(os.getenv("TAVILY_API_KEY")),
            }
        )
        
        # Connect to Milvus and create query engine
        self._setup_query_engine()
        
        # Setup agent with tools
        self._setup_agent()
        
        print(f"✓ Query & Research Agent initialized")
    
    def _setup_query_engine(self):
        """Setup the vector store and query engine."""
        host = os.getenv("MILVUS_HOST", "localhost")
        port = os.getenv("MILVUS_PORT", "19530")
        collection_name = os.getenv("MILVUS_COLLECTION_NAME", "saas_revenue_knowledge")
        milvus_uri = os.getenv("MILVUS_URI") or f"http://{host}:{port}"
        log_payload(
            self.logger,
            "query_agent.milvus.connect",
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
            
            # Create index from existing vector store
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store
            )
            
            # Create retriever
            self.retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=self.config.get("top_k", 5),
            )
            log_payload(
                self.logger,
                "query_agent.retriever.settings",
                {"top_k": self.config.get("top_k", 5)}
            )
            
            # Create query engine
            self.query_engine = RetrieverQueryEngine(
                retriever=self.retriever,
            )
            
            print(f"✓ Connected to Milvus collection: {collection_name}")
            log_payload(
                self.logger,
                "query_agent.milvus.connected",
                {"collection_name": collection_name}
            )
            
        except Exception as e:
            print(f"✗ Error setting up query engine: {e}")
            print("Make sure Milvus is running and the collection exists")
            sys.exit(1)
    
    def _web_search_tool(self, query: str) -> str:
        """Tool for performing web research via Tavily."""
        
        if not self.tavily_client:
            log_payload(
                self.logger,
                "query_agent.web_search.skipped",
                {"reason": "tavily_client_unavailable"}
            )
            return "Web research is not available (Tavily API key not configured)"
        
        try:
            max_results = self.config.get("web_search_results", 3)
            log_payload(
                self.logger,
                "query_agent.web_search.request",
                {
                    "query": query,
                    "search_depth": "advanced",
                    "max_results": max_results,
                }
            )
            results = self.tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results
            )
            log_payload(
                self.logger,
                "query_agent.web_search.response",
                {
                    "query": query,
                    "result_count": len(results.get("results", [])),
                }
            )
            
            # Format results
            output = []
            for i, result in enumerate(results.get("results", []), 1):
                log_payload(
                    self.logger,
                    "query_agent.web_search.result",
                    {
                        "index": i,
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "content_length": len(result.get("content", "") or ""),
                    }
                )
                output.append(
                    f"{i}. {result.get('title', 'No title')}\n"
                    f"   URL: {result.get('url', 'No URL')}\n"
                    f"   {result.get('content', 'No content')[:200]}...\n"
                )
            
            return "\n".join(output) if output else "No results found"
            
        except Exception as e:
            return f"Error performing web search: {e}"
    
    def _knowledge_base_tool(self, query: str) -> str:
        """Tool for querying the internal knowledge base."""
        
        try:
            log_payload(
                self.logger,
                "query_agent.knowledge_base.query",
                {"query": query}
            )
            response = self.query_engine.query(query)
            log_payload(
                self.logger,
                "query_agent.knowledge_base.response",
                {"response_length": len(str(response))}
            )
            return str(response)
        except Exception as e:
            log_payload(
                self.logger,
                "query_agent.knowledge_base.error",
                {"error": str(e)}
            )
            return f"Error querying knowledge base: {e}"
    
    def _setup_agent(self):
        """Setup the ReAct agent with tools."""
        
        tools = [
            FunctionTool.from_defaults(
                fn=self._knowledge_base_tool,
                name="knowledge_base",
                description=(
                    "Search the internal knowledge base of SaaS revenue strategy information. "
                    "Use this to find information about ARR trends, pricing models, business metrics, "
                    "and revenue strategies from the indexed documents."
                )
            )
        ]
        
        # Add web search tool if available
        if self.tavily_client:
            tools.append(
                FunctionTool.from_defaults(
                    fn=self._web_search_tool,
                    name="web_search",
                    description=(
                        "Search the web for current information about SaaS companies, revenue trends, "
                        "and business strategies. Use this for up-to-date information not in the knowledge base."
                    )
                )
            )
        
        # Create ReAct agent
        self.agent = ReActAgent(
            tools=tools,
            llm=self.llm,
            verbose=True,
        )
        log_payload(
            self.logger,
            "query_agent.agent.setup",
            {
                "tool_names": [tool.metadata.name for tool in tools],
                "verbose": True,
            }
        )
    
    def query(self, question: str) -> str:
        """Query the agent with a question."""
        
        print("\n" + "="*60)
        print("Processing Query")
        print("="*60)
        print(f"Question: {question}\n")
        log_payload(
            self.logger,
            "query_agent.query.start",
            {
                "question": question,
                "max_iterations": self.config.get("max_iterations", 5),
            }
        )
        
        try:
            # Use agent to answer question
            async def _run_agent():
                handler = self.agent.run(
                    user_msg=question,
                    max_iterations=self.config.get("max_iterations", 5),
                )
                return await handler

            response = asyncio.run(_run_agent())

            # Format response
            answer = str(response)
            log_payload(
                self.logger,
                "query_agent.query.response",
                {"answer_length": len(answer)}
            )
            
            # Display with rich formatting if available
            if self.console and self.config.get("format_markdown", True):
                self.console.print("\n" + "="*60)
                self.console.print(Panel(
                    Markdown(answer),
                    title="[bold blue]Research Output[/bold blue]",
                    border_style="blue"
                ))
            else:
                print("\n" + "="*60)
                print("Research Output")
                print("="*60)
                print(answer)
            
            return answer
            
        except Exception as e:
            error_msg = f"Error processing query: {e}"
            print(f"\n✗ {error_msg}")
            log_payload(
                self.logger,
                "query_agent.query.error",
                {"error": str(e)}
            )
            return error_msg
    
    def interactive_mode(self):
        """Run the agent in interactive mode."""
        
        if self.console:
            self.console.print(Panel(
                "[bold green]SaaS Revenue Strategy Query Agent[/bold green]\n"
                "Ask questions about SaaS revenue strategies, ARR trends, and business models.\n"
                "Type 'exit' or 'quit' to end the session.",
                border_style="green"
            ))
        else:
            print("="*60)
            print("SaaS Revenue Strategy Query Agent")
            print("="*60)
            print("Ask questions about SaaS revenue strategies, ARR trends, and business models.")
            print("Type 'exit' or 'quit' to end the session.")
            print("="*60)
        
        while True:
            try:
                # Get user input
                if self.console:
                    self.console.print("\n[bold cyan]Your question:[/bold cyan] ", end="")
                else:
                    print("\nYour question: ", end="")
                
                question = input().strip()
                log_payload(
                    self.logger,
                    "query_agent.interactive.input",
                    {"question": question}
                )
                
                # Check for exit
                if question.lower() in ['exit', 'quit', 'q']:
                    print("\nGoodbye!")
                    break
                
                if not question:
                    continue
                
                # Process query
                self.query(question)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except EOFError:
                print("\n\nGoodbye!")
                break


def main():
    """Main entry point for the query agent."""
    
    import argparse

    from agents.logging_utils import configure_logging
    configure_logging(verbose=True)
    
    parser = argparse.ArgumentParser(
        description="Query & Research Agent for SaaS Revenue Strategy"
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Question to ask the agent"
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize agent
        agent = QueryAgent()
        
        if args.interactive:
            # Interactive mode
            agent.interactive_mode()
        elif args.query:
            # Single query mode
            question = " ".join(args.query)
            agent.query(question)
        else:
            print("Error: Please provide a query or use --interactive mode")
            parser.print_help()
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
