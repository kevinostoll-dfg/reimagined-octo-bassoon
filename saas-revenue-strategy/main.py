#!/usr/bin/env python3
"""
Main CLI entry point for the SaaS Revenue Strategy RAG Agent.
Provides a unified interface for both mining and querying agents.
"""

import sys
import logging
from pathlib import Path

# Add agents directory to path
sys.path.insert(0, str(Path(__file__).parent / "agents"))

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from agents.knowledge_miner import KnowledgeMiner
from agents.query_agent import QueryAgent
from agents.logging_utils import configure_logging, log_payload


def print_banner():
    """Print the application banner."""
    if RICH_AVAILABLE:
        console = Console()
        banner = """
[bold blue]🤖 SaaS ARR Revenue Strategy RAG Agent[/bold blue]

A dual-agent system for mining and querying SaaS revenue strategy knowledge.

[cyan]Modes:[/cyan]
  1. Mine Knowledge - Extract and index SaaS revenue insights
  2. Query Agent - Ask questions about SaaS strategies
  3. Interactive Mode - Chat with the query agent

[yellow]Tech Stack:[/yellow]
  • LlamaIndex RAG Framework
  • Milvus Vector Database
  • Qwen3-Max (Novita API)
  • Tavily Web Research
        """
        console.print(Panel(banner, border_style="blue"))
    else:
        print("="*60)
        print("🤖 SaaS ARR Revenue Strategy RAG Agent")
        print("="*60)
        print("\nA dual-agent system for mining and querying SaaS revenue strategy knowledge.")
        print("\nModes:")
        print("  1. Mine Knowledge - Extract and index SaaS revenue insights")
        print("  2. Query Agent - Ask questions about SaaS strategies")
        print("  3. Interactive Mode - Chat with the query agent")
        print("="*60)


def mine_knowledge_mode():
    """Run the knowledge mining agent."""
    logger = logging.getLogger("main.mine")
    print("\n" + "="*60)
    print("Knowledge Mining Mode")
    print("="*60)
    
    try:
        miner = KnowledgeMiner()
        
        # Ask user for input
        if RICH_AVAILABLE:
            console = Console()
            choice = Prompt.ask(
                "\n[cyan]Mine from:[/cyan]\n"
                "  1. Configured sources (config/sources.yaml)\n"
                "  2. Custom topics\n"
                "Enter choice",
                choices=["1", "2"],
                default="1"
            )
        else:
            print("\nMine from:")
            print("  1. Configured sources (config/sources.yaml)")
            print("  2. Custom topics")
            choice = input("Enter choice (1 or 2) [1]: ").strip() or "1"
        log_payload(
            logger,
            "main.mine.choice",
            {"choice": choice}
        )
        
        if choice == "2":
            # Custom topics
            if RICH_AVAILABLE:
                topics_input = Prompt.ask("\n[cyan]Enter topics (comma-separated)[/cyan]")
            else:
                topics_input = input("\nEnter topics (comma-separated): ")
            
            topics = [t.strip() for t in topics_input.split(",") if t.strip()]
            log_payload(
                logger,
                "main.mine.topics_input",
                {"topics": topics}
            )
            
            if topics:
                print(f"\nMining {len(topics)} topics...")
                documents = miner.mine_from_web_search(topics)
                miner.index_documents(documents)
            else:
                print("No topics provided.")
        else:
            # Use configured sources
            miner.run()
        
        print("\n✓ Knowledge mining completed successfully")
        
    except Exception as e:
        print(f"\n✗ Error during mining: {e}")
        import traceback
        traceback.print_exc()


def query_mode():
    """Run a single query."""
    logger = logging.getLogger("main.query")
    print("\n" + "="*60)
    print("Query Mode")
    print("="*60)
    
    try:
        agent = QueryAgent()
        
        # Get query from user
        if RICH_AVAILABLE:
            console = Console()
            question = Prompt.ask("\n[cyan]Enter your question[/cyan]")
        else:
            question = input("\nEnter your question: ")
        
        if question.strip():
            log_payload(
                logger,
                "main.query.question",
                {"question": question}
            )
            agent.query(question)
        else:
            print("No question provided.")
        
    except Exception as e:
        print(f"\n✗ Error during query: {e}")
        import traceback
        traceback.print_exc()


def interactive_mode():
    """Run the interactive query mode."""
    logger = logging.getLogger("main.interactive")
    try:
        log_payload(
            logger,
            "main.interactive.start",
            {"mode": "interactive"}
        )
        agent = QueryAgent()
        agent.interactive_mode()
    except Exception as e:
        print(f"\n✗ Error in interactive mode: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    import argparse

    configure_logging(verbose=True)
    log_payload(
        logging.getLogger("main"),
        "main.start",
        {"argv": sys.argv}
    )
    
    parser = argparse.ArgumentParser(
        description="SaaS Revenue Strategy RAG Agent - Main CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run in interactive mode
  python main.py --interactive
  
  # Mine knowledge from configured sources
  python main.py --mine
  
  # Query the agent
  python main.py --query "What are the top SaaS ARR growth strategies?"
  
  # Show menu
  python main.py
        """
    )
    
    parser.add_argument(
        "--mine",
        action="store_true",
        help="Run knowledge mining agent"
    )
    parser.add_argument(
        "--query",
        nargs="+",
        help="Query the agent with a question"
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    # Handle command line arguments
    if args.interactive:
        print_banner()
        interactive_mode()
        return
    
    if args.mine:
        print_banner()
        mine_knowledge_mode()
        return
    
    if args.query:
        print_banner()
        try:
            agent = QueryAgent()
            question = " ".join(args.query)
            agent.query(question)
        except Exception as e:
            print(f"\n✗ Error: {e}")
            sys.exit(1)
        return
    
    # No arguments - show menu
    print_banner()
    
    while True:
        try:
            if RICH_AVAILABLE:
                console = Console()
                console.print("\n[bold cyan]Select Mode:[/bold cyan]")
                console.print("  1. Mine Knowledge")
                console.print("  2. Query Agent")
                console.print("  3. Interactive Mode")
                console.print("  4. Exit")
                choice = Prompt.ask("\nEnter choice", choices=["1", "2", "3", "4"], default="3")
            else:
                print("\nSelect Mode:")
                print("  1. Mine Knowledge")
                print("  2. Query Agent")
                print("  3. Interactive Mode")
                print("  4. Exit")
                choice = input("\nEnter choice (1-4) [3]: ").strip() or "3"
            
            if choice == "1":
                mine_knowledge_mode()
            elif choice == "2":
                query_mode()
            elif choice == "3":
                interactive_mode()
            elif choice == "4":
                print("\nGoodbye!")
                break
            else:
                print("Invalid choice. Please enter 1-4.")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            print("\n\nGoodbye!")
            break


if __name__ == "__main__":
    main()
