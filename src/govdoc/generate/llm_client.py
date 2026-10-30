"""OpenAI and Ollama compatible local LLM client for llama-server or Ollama."""

import argparse
import json
import time
from typing import Any, Dict, Optional
import requests

from govdoc.config import load_config


class LLMClient:
    """Client for querying locally hosted LLMs (llama-server or Ollama)."""

    def __init__(
        self,
        config_dict: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
    ):
        cfg = config_dict or load_config()
        llm_cfg = cfg.get("llm", {})
        self.server_url = llm_cfg.get("server_url", "http://127.0.0.1:11434").rstrip("/")
        # If server_url ends in /v1, also keep base_url
        if self.server_url.endswith("/v1"):
            self.base_url = self.server_url[:-3]
        else:
            self.base_url = self.server_url
        self.model_name = model_name or llm_cfg.get("model_name", "qwen2.5:1.5b")
        self.temperature = float(llm_cfg.get("temperature", 0.1))
        self.max_tokens = int(llm_cfg.get("max_tokens", 512))
        self.call_count = 0

    def chat(
        self,
        prompt: str,
        system_prompt: str = "You are a factual government document assistant. Answer truthfully based only on provided context.",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Send chat completion request to the local server."""
        self.call_count += 1
        temp = self.temperature if temperature is None else temperature
        num_predict = self.max_tokens if max_tokens is None else max_tokens

        t0 = time.time()
        # First try OpenAI standard /v1/chat/completions
        v1_url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temp,
            "max_tokens": num_predict,
            "stream": False,
        }

        try:
            resp = requests.post(v1_url, json=payload, timeout=90)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                elapsed = time.time() - t0
                return {
                    "content": content,
                    "latency_s": round(elapsed, 3),
                    "usage": data.get("usage", {}),
                    "call_count": self.call_count,
                    "protocol": "openai_v1",
                }
        except Exception:
            pass

        # Fallback to Ollama native /api/generate
        ollama_url = f"{self.base_url}/api/generate"
        ollama_payload = {
            "model": self.model_name,
            "prompt": f"{system_prompt}\n\nUser: {prompt}\nAssistant:",
            "options": {
                "temperature": temp,
                "num_predict": num_predict,
            },
            "stream": False,
        }
        resp = requests.post(ollama_url, json=ollama_payload, timeout=90)
        elapsed = time.time() - t0
        if resp.status_code != 200:
            raise RuntimeError(
                f"LLM request failed with status {resp.status_code}: {resp.text}"
            )

        data = resp.json()
        content = data.get("response", "")
        return {
            "content": content,
            "latency_s": round(elapsed, 3),
            "usage": {
                "prompt_eval_count": data.get("prompt_eval_count"),
                "eval_count": data.get("eval_count"),
            },
            "call_count": self.call_count,
            "protocol": "ollama_native",
        }

    def health_check(self) -> bool:
        """Check if LLM server is accessible."""
        # Try /api/tags (Ollama)
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        # Try /v1/models (llama-server)
        try:
            resp = requests.get(f"{self.base_url}/v1/models", timeout=3)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        return False


def main():
    parser = argparse.ArgumentParser(description="Test local LLM client.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="State the purpose of RTI Act 2005 in one brief sentence.",
        help="Prompt to send to LLM.",
    )
    parser.add_argument("--check-health", action="store_true", help="Check server health only.")
    args = parser.parse_args()

    client = LLMClient()
    print(f"Connecting to LLM server at: {client.base_url} (Model: {client.model_name})")

    if not client.health_check():
        print(f"WARNING: Server at {client.base_url} is not responding.")
        print("To start Ollama: run 'ollama serve' in a separate terminal.")
        print("To start llama-server: run 'llama-server.exe -m <gguf_path> --port 8080'")
        return

    print("Server health check: OK")
    if args.check_health:
        return

    print(f"Sending prompt: {args.prompt}")
    res = client.chat(args.prompt)
    print(f"Latency: {res['latency_s']}s")
    print(f"Protocol: {res['protocol']}")
    print(f"Response:\n{res['content']}")
    print(f"Total client calls: {res['call_count']}")


if __name__ == "__main__":
    main()
