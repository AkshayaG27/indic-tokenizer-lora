"""Byte-Level BPE Tokenizer Training Pipeline.

Trains custom Grapheme-Aware Byte-Level BPE tokenizers across a geometric
vocabulary grid (|V| in {512, 1024, 2048, 4096}) using Hugging Face tokenizers.
Preserves Indic grapheme clusters via INDIC_GRAPHEME_REGEX.
"""

from __future__ import annotations

import json
import os
from typing import List, Dict, Any
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, Regex
from src.tokenizer.unicode_regex import INDIC_GRAPHEME_REGEX


# Geometric vocabulary grid spanning 512 -> 4096 to empirically measure the Pareto frontier
DEFAULT_VOCAB_SIZES = [512, 1024, 2048, 4096]
SPECIAL_TOKENS = ["<s>", "</s>", "<unk>", "<pad>"]


def build_grapheme_bpe_tokenizer() -> Tokenizer:
    """Instantiate a Byte-Level BPE Tokenizer with Indic grapheme pre-tokenization."""
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    
    # 1. Indic Grapheme Regex Split + ByteLevel mapping
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(INDIC_GRAPHEME_REGEX), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    ])
    
    # 2. ByteLevel decoder for 100% reversible roundtrip decoding
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def train_tokenizer(
    corpus_paths: List[str],
    vocab_size: int,
    output_path: str,
    min_frequency: int = 1
) -> Dict[str, Any]:
    """Train a single BPE tokenizer checkpoint with a target vocabulary size."""
    tokenizer = build_grapheme_bpe_tokenizer()
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False
    )
    
    tokenizer.train(corpus_paths, trainer)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tokenizer.save(output_path)
    
    achieved_vocab_size = tokenizer.get_vocab_size()
    return {
        "target_vocab_size": vocab_size,
        "achieved_vocab_size": achieved_vocab_size,
        "output_path": output_path
    }


def train_vocabulary_grid(
    corpus_path: str = "data/processed/train_corpus.txt",
    vocab_sizes: List[int] = DEFAULT_VOCAB_SIZES,
    output_dir: str = "artifacts"
) -> List[Dict[str, Any]]:
    """Train the full vocabulary grid across target sizes and save checkpoints."""
    os.makedirs(output_dir, exist_ok=True)
    manifest = []
    
    for size in vocab_sizes:
        label = f"{size // 1024}k" if size >= 1024 else str(size)
        out_file = os.path.join(output_dir, f"tokenizer_bpe_{label}.json")
        print(f"Training Grapheme BPE with target |V| = {size} -> {out_file}...")
        meta = train_tokenizer(
            corpus_paths=[corpus_path],
            vocab_size=size,
            output_path=out_file
        )
        manifest.append(meta)
        print(f"  Achieved vocab size: {meta['achieved_vocab_size']}")
        
    manifest_path = os.path.join(output_dir, "vocab_grid_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    return manifest


if __name__ == "__main__":
    train_vocabulary_grid()
