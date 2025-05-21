
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from lancedb.embeddings import TextEmbeddingFunction
from lancedb.embeddings.registry import register
from typing import Optional, Dict, Tuple, List
from dotenv import load_dotenv
from openai import AzureOpenAI
from pydantic import BaseModel
from datetime import datetime

import threading
import logging
import openai
import time
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from configs import config

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

AZURE_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

# Fallback versions – use whichever is stable in your subscription
DEFAULT_CHAT_VERSION = "2024-02-01"
DEFAULT_EMBED_VERSION = "2024-02-01"


class ConnectionManager:
    """
    Keeps one AzureOpenAI client **per (thread_id, api_version)** and
    cleans up idle clients after `timeout_seconds`.
    """

    def __init__(self, timeout_seconds: int = 25):
        # key: (thread_id, api_version)  →  value: (AzureOpenAI client, last_used)
        self._connections: Dict[Tuple[int, str], Tuple[AzureOpenAI, datetime]] = {}
        self._lock = threading.Lock()
        self._timeout = timeout_seconds

        threading.Thread(target=self._cleanup_loop, daemon=True).start()
        logger.info("ConnectionManager initialised (idle-timeout: %ss)", timeout_seconds)

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10),
           stop=stop_after_attempt(3),
           retry=retry_if_exception_type((ConnectionError, TimeoutError)))
    def get_connection(self, *, api_version: str) -> AzureOpenAI:
        thread_id = threading.get_ident()
        key = (thread_id, api_version)

        with self._lock:
            if key in self._connections:
                client, _ = self._connections[key]
                self._connections[key] = (client, datetime.now())
                logger.debug("Re-using client [thread=%s, v=%s]", thread_id, api_version)
                return client

            # Create new client for this (thread, version)
            try:
                logger.info("Opening new client [thread=%s, v=%s]", thread_id, api_version)
                client = AzureOpenAI(
                    api_key=AZURE_KEY,
                    azure_endpoint=AZURE_ENDPOINT,
                    api_version=api_version
                )
                self._connections[key] = (client, datetime.now())
                return client
            except Exception:
                # Will be retried by tenacity wrapper
                logger.exception("Failed to create AzureOpenAI client")
                raise

    def close_connection(self, *, api_version: Optional[str] = None) -> None:
        """
        Close a specific version for the current thread, or **all** versions if
        `api_version is None`.
        """
        thread_id = threading.get_ident()
        with self._lock:
            keys = list(self._connections.keys())
            for key in keys:
                tid, ver = key
                if tid == thread_id and (api_version is None or ver == api_version):
                    client, _ = self._connections.pop(key)
                    try:
                        client.close()
                        logger.info("Closed client [thread=%s, v=%s]", tid, ver)
                    except Exception:
                        logger.exception("Error closing client [thread=%s, v=%s]", tid, ver)

    def _cleanup_loop(self) -> None:
        """Background thread: close idle clients."""
        while True:
            with self._lock:
                now = datetime.now()
                to_remove: List[Tuple[int, str]] = []

                for key, (client, last_used) in self._connections.items():
                    if (now - last_used).total_seconds() > self._timeout:
                        to_remove.append(key)
                        try:
                            client.close()
                            logger.info("Idle timeout – closed client %s", key)
                        except Exception:    # pragma: no-cover
                            logger.exception("Error closing idle client %s", key)

                for key in to_remove:
                    self._connections.pop(key, None)

            time.sleep(1)   # one-second granularity is enough

class AzureOpenAIClient:
    _manager = ConnectionManager()

    # Return a cached client for the requested version
    @classmethod
    def _client(cls, *, api_version: str) -> AzureOpenAI:
        return cls._manager.get_connection(api_version=api_version)

    # Make sure thread cleanup happens if an object is destroyed explicitly
    def __del__(self):
        try:
            self._manager.close_connection()
        except Exception:
            # Destructors must never raise
            logger.debug("Destructor cleanup failed", exc_info=True)


@register("OpenAIEmbeddings")
class AzureOpenAIEmbeddings(TextEmbeddingFunction, AzureOpenAIClient):
    name: str = ""
    _ndims: Optional[int] = None

    # ---- Embedding call -------------------------------------------------------
    @retry(wait=wait_exponential(multiplier=1, min=4, max=10),
           stop=stop_after_attempt(3),
           retry=retry_if_exception_type((openai.APIError, openai.APITimeoutError)))
    def generate_embeddings(self,
                            texts: List[str] | str,
                            *,
                            api_version: str = DEFAULT_EMBED_VERSION) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]

        thread_id = threading.get_ident()
        logger.info("Embedding request [thread=%s, v=%s, n_texts=%s]",
                    thread_id, api_version, len(texts))

        client = self._client(api_version=api_version)
        try:
            resp = client.embeddings.create(input=texts, model=config.model_embedding)
            return [d.embedding for d in resp.data]
        except openai.BadRequestError:
            logger.exception("Bad request while embedding")
            raise
        except Exception:
            logger.exception("Unexpected embedding error")
            raise

    def ndims(self) -> int:
        if self._ndims is None:
            self._ndims = len(self.generate_embeddings("probe-dims")[0])
        return self._ndims

    @property
    def function(self):
        return self.generate_embeddings

    # Let instances be callable
    def __call__(self, texts):
        return self.generate_embeddings(texts)


class AzureOpenAIChat(AzureOpenAIClient):
    SYS_PROMPT = '"""Help the user find the answer they are looking for."""'

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10),
           stop=stop_after_attempt(3),
           retry=retry_if_exception_type((openai.APIError, openai.APITimeoutError)))
    def generate_response(self,
                          input_data: str | list[dict],
                          *,
                          temperature: float = 0.0,
                          top_p: float = 0.1,
                          max_tokens: int = 4096,
                          seed: int = 42,
                          response_format: BaseModel | None = None,
                          model: str = config.model_chat,
                          api_version: str = DEFAULT_CHAT_VERSION) -> str:
        """
        Parameters
        ----------
        input_data : str | list[dict]
            Either a raw user message or a full chat history
            (each item must have 'role' & 'content').
        api_version : str
            **Switch Azure API version here.**  Each unique version in a thread
            gets its own cached client.
        """

        # Build chat messages ---------------------------------------------------
        if isinstance(input_data, str):
            messages = [
                {"role": "system", "content": self.SYS_PROMPT},
                {"role": "user", "content": input_data}
            ]
        elif isinstance(input_data, list):
            if not all(isinstance(m, dict) and {'role', 'content'} <= m.keys()
                       for m in input_data):
                raise ValueError("Chat history items must be dict(role, content)")
            messages = [{"role": "system", "content": self.SYS_PROMPT}, *input_data]
        else:
            raise ValueError("input_data must be str or list[dict]")

        params = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
        )
        if response_format:
            params["response_format"] = response_format

        client = self._client(api_version=api_version)
        logger.info("Chat completion [thread=%s, v=%s, model=%s, tokens<=%s]",
                    threading.get_ident(), api_version, model, max_tokens)
        try:
            resp = client.chat.completions.create(**params)
            return resp.choices[0].message.content
        except openai.BadRequestError:
            logger.exception("Bad request during chat completion")
            raise
        except Exception:
            logger.exception("Unexpected chat completion error")
            raise


if __name__ == "__main__":
    chat = AzureOpenAIChat()

    # Try two different versions in the same thread
    print("\n--- GPT-4o preview ---")
    print(chat.generate_response(
        "Summarise the major trends in GenAI this year.",
        model="gpt-4o",
        api_version="2024-02-15-preview"
    ))

    print("\n--- GPT-4 turbo (stable) ---")
    print(chat.generate_response(
        "Hello again – which API version am I on now?",
        model="gpt-4-turbo",
        api_version="2024-02-01"
    ))

    encoder = AzureOpenAIEmbeddings()
    vec = encoder.generate_embeddings("vector me!", api_version="2024-02-01")
    print("\nEmbedding dims:", len(vec[0]))
