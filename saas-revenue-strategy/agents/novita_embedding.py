#!/usr/bin/env python3
"""
Novita OpenAI-compatible embedding wrapper for LlamaIndex.
"""

from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI, OpenAI
from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.constants import DEFAULT_EMBED_BATCH_SIZE


class NovitaEmbedding(BaseEmbedding):
    """Embedding client for Novita's OpenAI-compatible API."""

    model: str = Field(description="Embedding model identifier.")
    api_key: str = Field(description="Novita API key.")
    api_base: str = Field(description="Novita API base URL.")
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Extra OpenAI-compatible params."
    )

    _client: Optional[OpenAI] = PrivateAttr(default=None)
    _aclient: Optional[AsyncOpenAI] = PrivateAttr(default=None)

    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: str,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            model_name=model,
            api_key=api_key,
            api_base=api_base,
            embed_batch_size=embed_batch_size,
            additional_kwargs=additional_kwargs or {},
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "NovitaEmbedding"

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
            )
        return self._client

    def _get_aclient(self) -> AsyncOpenAI:
        if self._aclient is None:
            self._aclient = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
            )
        return self._aclient

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return await self._aget_text_embedding(query)

    def _get_text_embedding(self, text: str) -> Embedding:
        text = text.replace("\n", " ")
        response = self._get_client().embeddings.create(
            input=[text],
            model=self.model,
            **self.additional_kwargs,
        )
        return response.data[0].embedding

    async def _aget_text_embedding(self, text: str) -> Embedding:
        text = text.replace("\n", " ")
        response = await self._get_aclient().embeddings.create(
            input=[text],
            model=self.model,
            **self.additional_kwargs,
        )
        return response.data[0].embedding

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        sanitized = [text.replace("\n", " ") for text in texts]
        response = self._get_client().embeddings.create(
            input=sanitized,
            model=self.model,
            **self.additional_kwargs,
        )
        return [item.embedding for item in response.data]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        sanitized = [text.replace("\n", " ") for text in texts]
        response = await self._get_aclient().embeddings.create(
            input=sanitized,
            model=self.model,
            **self.additional_kwargs,
        )
        return [item.embedding for item in response.data]
