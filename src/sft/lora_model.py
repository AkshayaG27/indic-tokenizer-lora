"""LoRA Adapter Injection & Parameter Efficiency Engine.

Configures Low-Rank Adaptation (LoRA) on Small Language Model (SLM)
using its NATIVE pre-trained tokenizer to eliminate the Embedding Misalignment Trap.

Injects rank r=8, alpha=16 adapters into attention projections (q_proj, v_proj).
Verifies that trainable parameters represent < 1.0% of total weights (AC-5).

Aligns with Chapter 08 (PEFT / LoRA Formulation) of course textbook.
"""

from __future__ import annotations

import os
from typing import Dict, Any, Tuple, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from peft import LoraConfig, get_peft_model, PeftModel


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def get_lora_config(
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None
) -> LoraConfig:
    """Instantiate standard LoRA configuration."""
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )


def load_model_and_tokenizer(
    model_name: str = DEFAULT_BASE_MODEL,
    device: str = "cpu"
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load base SLM and its native tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32 if device == "cpu" else torch.bfloat16
    )
    model.to(device)
    return model, tokenizer


def apply_lora_to_model(
    base_model: PreTrainedModel,
    config: Optional[LoraConfig] = None
) -> PeftModel:
    """Inject LoRA adapters and freeze all base weights W_0."""
    if config is None:
        config = get_lora_config()

    lora_model = get_peft_model(base_model, config)
    return lora_model


def audit_parameter_efficiency(model: PeftModel) -> Dict[str, Any]:
    """Verify AC-5: Trainable adapter parameters represent < 1.0% of total."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    trainable_pct = (trainable_params / max(1, all_params)) * 100.0

    return {
        "trainable_parameters": trainable_params,
        "total_parameters": all_params,
        "trainable_percentage": round(trainable_pct, 4),
        "parameter_budget_satisfied": trainable_pct < 1.0
    }


def setup_lora_pipeline(
    model_name: str = DEFAULT_BASE_MODEL,
    r: int = 8,
    alpha: int = 16,
    device: str = "cpu"
) -> Tuple[PeftModel, PreTrainedTokenizer, Dict[str, Any]]:
    """Complete factory function loading base model, tokenizer, and PEFT adapter."""
    base_model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    config = get_lora_config(r=r, alpha=alpha)
    lora_model = apply_lora_to_model(base_model, config)
    efficiency = audit_parameter_efficiency(lora_model)

    return lora_model, tokenizer, efficiency


if __name__ == "__main__":
    print("Setting up LoRA pipeline on base SLM...")
    model, tok, stats = setup_lora_pipeline()
    print("Base SLM + LoRA initialized successfully!")
    print(f"Trainable Parameters: {stats['trainable_parameters']:,} / {stats['total_parameters']:,}")
    print(f"Trainable %: {stats['trainable_percentage']}% (Budget < 1.0%: {stats['parameter_budget_satisfied']})")
