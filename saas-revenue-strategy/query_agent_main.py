#!/usr/bin/env python3
"""
Dedicated entry point for the Query & Research Agent.
"""

import argparse
import logging
import os
import sys

from agents.query_agent import QueryAgent
from agents.logging_utils import configure_logging, log_payload, redact_secret


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query & Research Agent for SaaS Revenue Strategy"
    )
    parser.add_argument(
        "--config",
        default="config/agents.yaml",
        help="Path to agents configuration file"
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Question to ask the agent"
    )
    args = parser.parse_args()

    configure_logging(verbose=True)
    logger = logging.getLogger("query_agent_main")
    log_payload(
        logger,
        "query_agent_main.start",
        {
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "config_path": args.config,
            "interactive": args.interactive,
            "query_provided": bool(args.query),
        }
    )
    log_payload(
        logger,
        "query_agent_main.env",
        {
            "NOVITA_API_BASE": os.getenv("NOVITA_API_BASE", "https://api.novita.ai/openai"),
            "NOVITA_API_KEY": redact_secret(os.getenv("NOVITA_API_KEY", "")),
            "TAVILY_API_KEY": redact_secret(os.getenv("TAVILY_API_KEY", "")),
            "MILVUS_HOST": os.getenv("MILVUS_HOST", "localhost"),
            "MILVUS_PORT": os.getenv("MILVUS_PORT", "19530"),
            "MILVUS_COLLECTION_NAME": os.getenv("MILVUS_COLLECTION_NAME", "saas_revenue_knowledge"),
            "MILVUS_URI": os.getenv("MILVUS_URI") or None,
            "EMBEDDING_DIMENSION": os.getenv("EMBEDDING_DIMENSION", "4096"),
        }
    )

    try:
        agent = QueryAgent(config_path=args.config)

        if args.interactive:
            agent.interactive_mode()
        elif args.query:
            question = " ".join(args.query)
            log_payload(
                logger,
                "query_agent_main.query",
                {"question": question}
            )
            agent.query(question)
        else:
            logger.error("query_agent_main.no_input")
            parser.print_help()
            sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("query_agent_main.interrupted")
        sys.exit(1)
    except Exception as exc:
        log_payload(
            logger,
            "query_agent_main.error",
            {"error": str(exc)}
        )
        raise


if __name__ == "__main__":
    main()
