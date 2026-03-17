from fastapi import HTTPException
import httpx
import asyncio
import logging
import os
from pydantic import BaseModel, Field
from typing import List

# --- KEPT server-related imports ---
from .process_management import (
    get_server_processes,
    get_server_configs,
)


logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    message: str
    port: int = 8081
    threads: int = Field(default_factory=lambda: os.cpu_count() or 1, gt=0)
    ctx_size: int = Field(default=2048, gt=0)
    n_predict: int = Field(default=128, gt=0)
    temperature: float = Field(default=0.8, gt=0.0, le=2.0)


class MultiChatRequestItem(ChatRequest):
    pass


class MultiChatRequest(BaseModel):
    requests: List[MultiChatRequestItem]


# --- Endpoint logic functions ---

def _clean_response(text: str) -> str:
    """Strip repetitive or meta-text patterns from raw model output."""
    import re
    # Take only the first meaningful answer block
    # Stop at patterns like "Question:", "Input:", "Output:", "(no answer)", "(end of answer)"
    cut_patterns = [
        r'\n\s*Question\s*:',
        r'\n\s*Input\s*:',
        r'\n\s*Output\s*:',
        r'\(no answer\)',
        r'\(end of answer\)',
        r'\(No answer required\)',
    ]
    for pat in cut_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            text = text[:m.start()]
    return text.strip()


async def handle_chat_with_bitnet_server(chat: ChatRequest):
    host = "127.0.0.1"
    key = (host, chat.port)
    proc_entry = get_server_processes().get(key)
    cfg = get_server_configs().get(key)
    if not (proc_entry and proc_entry["process"].returncode is None and cfg):
        logger.warning(f"Chat request to non-existent or stopped server on port {chat.port}.")
        raise HTTPException(status_code=404, detail=f"Server on port {chat.port} not running or not configured.")

    # Use the OpenAI-compatible chat completions endpoint so llama-server
    # applies the correct LLaMA 3 chat template automatically.
    server_url = f"http://{host}:{chat.port}/v1/chat/completions"

    system_prompt = cfg.get("system_prompt", "You are a helpful assistant.")
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chat.message},
        ],
        "n_predict": chat.n_predict,
        "temperature": chat.temperature,
        "stop": ["\nQuestion:", "\nInput:", "\nOutput:", "\n\nQuestion:", "\n\nInput:"],
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(server_url, json=payload, timeout=60.0)
            response.raise_for_status()
            response_data = response.json()

            # OpenAI-compatible format: choices[0].message.content
            choices = response_data.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
            else:
                # Fallback for raw /completion style response
                content = response_data.get("content", "")

            content = _clean_response(content)
            return {"response": content, "port": chat.port}
    except httpx.RequestError as e:
        logger.error(f"HTTP request error to server {host}:{chat.port}: {e}")
        raise HTTPException(status_code=503, detail=f"Error communicating with BitNet server on port {chat.port}: {e}")
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP status error from server {host}:{chat.port}: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Error from BitNet server on port {chat.port}: {e.response.text}")
    except Exception as e:
        logger.error(f"Unexpected error during chat with server {host}:{chat.port}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


async def handle_multichat_with_bitnet_server(data: MultiChatRequest):
    async def single_chat_wrapper(chat_request: MultiChatRequestItem):
        try:
            return await handle_chat_with_bitnet_server(chat_request)
        except HTTPException as e:
            return {"port": chat_request.port, "error": e.detail, "status_code": e.status_code}
        except Exception as e:
            return {"port": chat_request.port, "error": str(e), "status_code": 500}

    results = await asyncio.gather(*[single_chat_wrapper(req) for req in data.requests])
    return {"results": results}
