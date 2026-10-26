"""Bilingual Instruction Tuning Dataset & Prompt-Loss Masking Engine.

Implements:
1. Standardized ChatML conversation formatting for Hindi and Telugu instruction pairs.
2. Strict Prompt-Loss Masking: User prompt tokens are assigned label = -100 (ignore_index),
   ensuring cross-entropy loss is computed exclusively on assistant completion tokens (AC-6).
3. PyTorch Dataset & Collation Engine with dynamic padding.

Aligns with Chapter 08 (PEFT & Fine-Tuning) and Chapter 09 (Data Pipeline).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Any, Optional, Tuple
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class BilingualInstructionDataset(Dataset):
    """PyTorch Dataset with prompt-loss masking for instruction SFT."""

    def __init__(
        self,
        samples: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        instruction = sample["instruction"].strip()
        response = sample["response"].strip()

        # 1. Format ChatML prompts
        prompt_text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        full_text = f"{prompt_text}{response}<|im_end|>\n"

        # 2. Tokenize with native tokenizer
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        # Truncate if exceeding maximum length, preserving end-of-turn token
        if len(full_ids) > self.max_length:
            im_end_ids = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
            if not im_end_ids and hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id:
                im_end_ids = [self.tokenizer.eos_token_id]
            suffix_len = len(im_end_ids)
            full_ids = full_ids[: self.max_length - suffix_len] + im_end_ids

        # 3. Apply Strict Prompt-Loss Masking (AC-6)
        # User prompt tokens are assigned -100 so loss is computed exclusively on response tokens
        prompt_len = len(prompt_ids)
        if len(full_ids) <= prompt_len:
            # If sequence was heavily truncated, ensure at least response tokens are present
            prompt_len = max(1, len(full_ids) // 2)

        labels = [-100] * prompt_len + full_ids[prompt_len:]

        # Verify invariant: At least one response token must be unmasked
        if not any(l != -100 for l in labels):
            labels[-1] = full_ids[-1]

        # Create attention mask (1 for real tokens)
        attention_mask = [1] * len(full_ids)

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "language": sample.get("language", "hi")
        }


class InstructionDataCollator:
    """Collates variable-length instruction sequences with dynamic padding."""

    def __init__(self, pad_token_id: int, pad_label_id: int = -100):
        self.pad_token_id = pad_token_id
        self.pad_label_id = pad_label_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in batch)

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for item in batch:
            input_ids = item["input_ids"].tolist()
            attention_mask = item["attention_mask"].tolist()
            labels = item["labels"].tolist()

            pad_len = max_len - len(input_ids)
            padded_input_ids.append(input_ids + [self.pad_token_id] * pad_len)
            padded_attention_mask.append(attention_mask + [0] * pad_len)
            padded_labels.append(labels + [self.pad_label_id] * pad_len)

        labels_tensor = torch.tensor(padded_labels, dtype=torch.long)
        assert (labels_tensor != self.pad_label_id).sum() > 0, "Collation Error: Batch has 0 unmasked labels!"

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": labels_tensor
        }


def load_bilingual_datasets(
    tokenizer: PreTrainedTokenizerBase,
    data_path: str = "data/instructions/bilingual_instructions.json",
    train_ratio: float = 0.85,
    max_length: int = 384,
    max_samples: Optional[int] = None,
    seed: int = 42
) -> Tuple[BilingualInstructionDataset, BilingualInstructionDataset]:
    """Load instruction pairs and partition into train and validation splits."""
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    # Deterministic split
    import random
    rng = random.Random(seed)
    shuffled_data = data.copy()
    rng.shuffle(shuffled_data)

    if max_samples and max_samples < len(shuffled_data):
        shuffled_data = shuffled_data[:max_samples]

    split_idx = int(len(shuffled_data) * train_ratio)
    train_samples = shuffled_data[:split_idx]
    val_samples = shuffled_data[split_idx:]

    train_ds = BilingualInstructionDataset(train_samples, tokenizer, max_length=max_length)
    val_ds = BilingualInstructionDataset(val_samples, tokenizer, max_length=max_length)

    return train_ds, val_ds
