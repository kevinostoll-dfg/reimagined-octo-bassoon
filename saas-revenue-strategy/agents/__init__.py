"""
SaaS Revenue Strategy RAG Agent - Agents Package

This package contains the dual-agent system:
- KnowledgeMiner: Mines and indexes SaaS revenue strategy knowledge
- QueryAgent: Retrieves and synthesizes research from the knowledge base
"""

from .knowledge_miner import KnowledgeMiner
from .query_agent import QueryAgent

__all__ = ["KnowledgeMiner", "QueryAgent"]
