"""Information-Theoretic Bits-Per-Byte (BPB) and Entropy Engine.

Computes:
1. Empirical unigram token frequency distributions P(t_i).
2. Shannon Entropy: H(X) = -sum P(t) log2 P(t).
3. Bits-Per-Byte (BPB):
   BPB(s) = [sum_{i=1}^N -log2 P(t_i)] / Total_UTF8_Bytes(s)
4. Character-level bits per byte and compression efficiency.

Aligns with Chapter 09, Section 9.3 and Worked Problem 9.6 of course textbook.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import List, Dict, Any, Callable


class BPBCalculator:
    """Computes information-theoretic Bits-Per-Byte under empirical unigram models."""

    def __init__(self, vocab_size: int, smoothing: float = 1.0):
        self.vocab_size = max(1, vocab_size)
        self.smoothing = smoothing
        self.token_counts: Counter[int] = Counter()
        self.total_tokens_observed = 0

    def fit_distribution(self, token_ids_list: List[List[int]]) -> None:
        """Fit empirical unigram token distribution from reference token sequences."""
        self.token_counts.clear()
        self.total_tokens_observed = 0
        for seq in token_ids_list:
            self.token_counts.update(seq)
            self.total_tokens_observed += len(seq)

    def get_token_prob(self, token_id: int) -> float:
        """Compute smoothed empirical probability of a token ID."""
        c = self.token_counts.get(token_id, 0)
        numerator = c + self.smoothing
        denominator = self.total_tokens_observed + (self.smoothing * self.vocab_size)
        return numerator / denominator

    def compute_sequence_bpb(self, text: str, token_ids: List[int]) -> Dict[str, float]:
        """Compute negative log-likelihood in bits and normalize by total UTF-8 bytes."""
        utf8_bytes = max(1, len(text.encode("utf-8")))
        if not token_ids:
            return {"bpb": 0.0, "total_bits": 0.0, "utf8_bytes": utf8_bytes}

        total_bits = 0.0
        for tid in token_ids:
            p = self.get_token_prob(tid)
            # Surprisal / self-information in bits: -log2(p)
            total_bits += -math.log2(p)

        bpb = total_bits / utf8_bytes
        return {
            "bpb": round(bpb, 4),
            "total_bits": round(total_bits, 2),
            "utf8_bytes": utf8_bytes,
            "bits_per_token": round(total_bits / len(token_ids), 3)
        }

    def compute_corpus_bpb(
        self,
        texts: List[str],
        encode_fn: Callable[[str], List[int]]
    ) -> Dict[str, Any]:
        """Compute aggregated corpus-level Bits-Per-Byte across a set of texts."""
        total_bits = 0.0
        total_bytes = 0
        total_tokens = 0

        for text in texts:
            if not text.strip():
                continue
            tids = encode_fn(text)
            b_info = self.compute_sequence_bpb(text, tids)
            total_bits += b_info["total_bits"]
            total_bytes += b_info["utf8_bytes"]
            total_tokens += len(tids)

        corpus_bpb = total_bits / max(1, total_bytes)
        bits_per_token = total_bits / max(1, total_tokens)

        return {
            "corpus_bpb": round(corpus_bpb, 4),
            "total_bits": round(total_bits, 2),
            "total_bytes": total_bytes,
            "total_tokens": total_tokens,
            "bits_per_token": round(bits_per_token, 3)
        }
