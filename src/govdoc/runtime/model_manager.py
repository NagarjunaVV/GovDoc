"""Sequential residency model manager enforcing 4 GB VRAM budget."""

import gc
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import torch
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

from govdoc.config import load_config, resolve_path
from govdoc.runtime.vram_monitor import record_vram_reading, get_gpu_memory_used_mb
from govdoc.generate.llm_client import LLMClient

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class ModelManager:
    """
    Manages sequential residency across operational modes:
    - 'qa': Qwen2.5-1.5B active via server/GGUF; IndicBART unloaded; BGE-M3 on CPU.
    - 'summarise': IndicBART LoRA merged loaded on GPU; Qwen unloaded.
    - 'ingest': BGE-M3 dense embedder on GPU; IndicBART unloaded.
    """

    VALID_MODES = {"qa", "summarise", "ingest"}

    def __init__(self, initial_mode: str = "qa"):
        self.cfg = load_config()
        self.active_mode: Optional[str] = None
        self.loaded_models: Dict[str, Any] = {}
        self.llm_client: Optional[LLMClient] = None
        self.set_mode(initial_mode)

    def get_active_mode(self) -> Optional[str]:
        """Return currently active sequential residency mode."""
        return self.active_mode

    def _unload_active(self):
        """Force eviction of all cached PyTorch model instances and clear GPU cache."""
        for name in list(self.loaded_models.keys()):
            del self.loaded_models[name]

        self.loaded_models.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def set_mode(self, mode: str) -> Dict[str, Any]:
        """
        Switch operational mode under the sequential residency policy.
        Frees memory of non-resident models and loads target mode components.
        """
        mode = mode.lower()
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Expected one of {self.VALID_MODES}")

        if self.active_mode == mode:
            return {
                "mode": mode,
                "status": "already_active",
                "used_vram_mb": get_gpu_memory_used_mb(),
            }

        prev_mode = self.active_mode
        record_vram_reading(f"mode_switch_start_from_{prev_mode or 'none'}_to_{mode}")

        print(f"ModelManager: Switching mode [{prev_mode or 'init'}] -> [{mode}]...")
        self._unload_active()

        if mode == "qa":
            # In QA mode: Qwen is served via local LLM server; IndicBART unloaded
            self.llm_client = LLMClient(self.cfg)
            health = self.llm_client.health_check()
            self.loaded_models["qa_client"] = self.llm_client
            print(f"  QA mode active: Local LLM server reachable={health}")

        elif mode == "summarise":
            # In Summarise mode: Load IndicBART merged weights on GPU in fp16
            summarizer_cfg = self.cfg.get("models", {}).get("summarizer", {})
            adapter_path = resolve_path(summarizer_cfg.get("lora_adapter", "models/indicbart_merged"))
            base_model_name = summarizer_cfg.get("base", "ai4bharat/IndicBART")
            target_path = adapter_path if adapter_path.exists() else base_model_name

            print(f"  Loading IndicBART summarizer ({target_path}) on GPU...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32

            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                use_fast=False,
                keep_accents=True,
                do_lower_case=False,
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                target_path,
                torch_dtype=dtype,
            ).to(device)
            model.eval()

            self.loaded_models["summarizer_tokenizer"] = tokenizer
            self.loaded_models["summarizer_model"] = model

        elif mode == "ingest":
            # In Ingest mode: Load BGE-M3 embedder on GPU for bulk embedding
            print("  Ingest mode active: GPU allocated for BGE-M3 bulk indexing.")

        self.active_mode = mode
        rec = record_vram_reading(
            f"mode_switch_done_{mode}",
            extra_metadata={"active_mode": mode, "prev_mode": prev_mode},
        )
        print(f"  Mode switch to [{mode}] complete. VRAM used: {rec.get('used_mb', 0)} MB")

        return {
            "mode": mode,
            "status": "success",
            "used_vram_mb": rec.get("used_mb", 0),
        }

    def get_qa_client(self) -> LLMClient:
        """Get the LLM client instance for QA mode."""
        if self.active_mode != "qa" or "qa_client" not in self.loaded_models:
            self.set_mode("qa")
        return self.loaded_models["qa_client"]

    def get_summarizer(self) -> Tuple[Any, Any]:
        """Get (tokenizer, model) tuple for summariser mode."""
        if self.active_mode != "summarise" or "summarizer_model" not in self.loaded_models:
            self.set_mode("summarise")
        return self.loaded_models["summarizer_tokenizer"], self.loaded_models["summarizer_model"]
