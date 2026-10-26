"""Mixed-Precision SFT Training Pipeline for LoRA Adaptation.

Trains low-rank adapters (r=8, alpha=16) on base SLM using bilingual instruction pairs
(Hindi & Telugu) with strict prompt-loss masking.

Evaluates validation loss and validation perplexity per epoch (AC-7).
Saves adapter weights to artifacts/lora_adapter/.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Dict, Any, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from src.sft.lora_model import setup_lora_pipeline, DEFAULT_BASE_MODEL
from src.sft.dataset import load_bilingual_datasets, InstructionDataCollator


def train_lora_sft(
    model_name: str = DEFAULT_BASE_MODEL,
    data_path: str = "data/instructions/bilingual_instructions.json",
    output_dir: str = "artifacts/lora_adapter",
    epochs: int = 2,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 3e-4,
    max_length: int = 384,
    max_samples: int = 40,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Execute end-to-end SFT training loop on base SLM with low-rank adapters."""
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
    print(f"Initializing SFT Pipeline on {device.upper()} for model '{model_name}'...", flush=True)

    # 1. Setup model, native tokenizer, and LoRA adapters
    model, tokenizer, param_stats = setup_lora_pipeline(model_name=model_name, device=device)
    print(f"Adapter parameter budget: {param_stats['trainable_parameters']:,} params ({param_stats['trainable_percentage']}%)", flush=True)

    # 2. Setup datasets and loaders with prompt masking
    train_ds, val_ds = load_bilingual_datasets(tokenizer, data_path=data_path, max_length=max_length, max_samples=max_samples)
    collator = InstructionDataCollator(pad_token_id=tokenizer.pad_token_id)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

    # 3. Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.01
    )
    total_steps = (len(train_loader) // gradient_accumulation_steps) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.1)),
        num_training_steps=max(1, total_steps)
    )

    history: List[Dict[str, Any]] = []
    print(f"Beginning SFT Training: {epochs} Epochs, {len(train_ds)} train examples, {len(val_ds)} val examples...", flush=True)

    # 4. Training loop
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()

            total_train_loss += loss.item() * gradient_accumulation_steps

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % 5 == 0 or (step + 1) == len(train_loader):
                print(f"    Epoch {epoch}/{epochs} | Step {step+1}/{len(train_loader)} | Batch Loss = {loss.item() * gradient_accumulation_steps:.4f}", flush=True)

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation evaluation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_val_loss += outputs.loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_ppl = math.exp(min(avg_val_loss, 20.0))

        epoch_stats = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_perplexity": round(val_ppl, 2)
        }
        history.append(epoch_stats)
        print(f"  Epoch {epoch}/{epochs}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f} | Val PPL = {val_ppl:.2f}", flush=True)

    total_training_time = round(time.time() - start_time, 2)
    print(f"SFT Training Completed in {total_training_time}s!")

    # 5. Save adapter weights and tokenizer configuration
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training metadata
    manifest = {
        "model_name": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "training_time_seconds": total_training_time,
        "parameter_efficiency": param_stats,
        "history": history,
        "output_dir": output_dir
    }
    history_file = os.path.join("artifacts", "training_history.json")
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


if __name__ == "__main__":
    train_lora_sft(epochs=2, batch_size=2)
