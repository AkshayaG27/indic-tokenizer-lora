"""Token Fertility and Compression Evaluation Engine.

Computes:
1. Token Fertility (Tokens / Word) across languages.
2. Parity Ratio relative to English: F(Lang) / F(English).
3. Byte Compression Ratio: UTF-8 Bytes / Token.

Aligns with Chapter 09, Section 9.3.3 (Multilingual Tokenization and Parity).
"""

from __future__ import annotations

import re
from typing import List, Dict, Any, Callable


def count_words(text: str) -> int:
    """Count lexical words in text (whitespace separated, ignoring bare punctuation)."""
    raw_tokens = text.strip().split()
    # Filter out empty or pure punctuation tokens across any Unicode script
    words = [w for w in raw_tokens if any(ch.isalnum() for ch in w)]
    return max(1, len(words))


def compute_fertility(
    texts: List[str],
    encode_fn: Callable[[str], List[int]]
) -> Dict[str, float]:
    """Calculate aggregate token fertility, token count, word count, and byte compression."""
    total_tokens = 0
    total_words = 0
    total_bytes = 0

    for text in texts:
        if not text.strip():
            continue
        token_ids = encode_fn(text)
        words = count_words(text)
        utf8_bytes = len(text.encode("utf-8"))

        total_tokens += len(token_ids)
        total_words += words
        total_bytes += utf8_bytes

    fertility = total_tokens / max(1, total_words)
    bytes_per_token = total_bytes / max(1, total_tokens)

    return {
        "fertility": round(fertility, 3),
        "total_tokens": total_tokens,
        "total_words": total_words,
        "total_bytes": total_bytes,
        "bytes_per_token": round(bytes_per_token, 3)
    }


def compute_stratified_fertility(
    dataset: List[Dict[str, str]],
    encode_fn: Callable[[str], List[int]]
) -> Dict[str, Any]:
    """Compute language-stratified fertility for English, Hindi, and Telugu.
    
    Also derives the Language Tax Parity Multiplier against English.
    """
    en_texts = [d["en"] for d in dataset if "en" in d]
    hi_texts = [d["hi"] for d in dataset if "hi" in d]
    te_texts = [d["te"] for d in dataset if "te" in d]

    en_stats = compute_fertility(en_texts, encode_fn)
    hi_stats = compute_fertility(hi_texts, encode_fn)
    te_stats = compute_fertility(te_texts, encode_fn)

    en_fert = en_stats["fertility"]
    hi_parity_ratio = round(hi_stats["fertility"] / max(0.01, en_fert), 2)
    te_parity_ratio = round(te_stats["fertility"] / max(0.01, en_fert), 2)

    return {
        "en": en_stats,
        "hi": hi_stats,
        "te": te_stats,
        "parity_ratios": {
            "hi_vs_en": hi_parity_ratio,
            "te_vs_en": te_parity_ratio
        }
    }
