#!/usr/bin/env python3
"""
Dedicated entry point for the Knowledge Mining Agent.
"""

import argparse
import logging
import os
import sys

from agents.knowledge_miner import KnowledgeMiner
from agents.logging_utils import configure_logging, log_payload, redact_secret


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Knowledge Mining Agent for SaaS Revenue Strategy"
    )
    parser.add_argument(
        "--config",
        default="config/agents.yaml",
        help="Path to agents configuration file"
    )
    parser.add_argument(
        "--sources",
        default="config/sources.yaml",
        help="Path to sources configuration file"
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        help="Specific topics to mine (overrides sources config)"
    )
    args = parser.parse_args()

    configure_logging(verbose=True)
    logger = logging.getLogger("knowledge_miner_main")
    log_payload(
        logger,
        "knowledge_miner_main.start",
        {
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "config_path": args.config,
            "sources_path": args.sources,
            "topics_override": bool(args.topics),
        }
    )
    log_payload(
        logger,
        "knowledge_miner_main.env",
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
        miner = KnowledgeMiner(config_path=args.config)

        if args.topics:
            log_payload(
                logger,
                "knowledge_miner_main.topics_override",
                {"topics": args.topics}
            )
            documents = miner.mine_from_web_search(args.topics)
            miner.index_documents(documents)
        else:
            miner.run(args.sources)
    except KeyboardInterrupt:
        logger.warning("knowledge_miner_main.interrupted")
        sys.exit(1)
    except Exception as exc:
        log_payload(
            logger,
            "knowledge_miner_main.error",
            {"error": str(exc)}
        )
        raise


if __name__ == "__main__":
    main()
