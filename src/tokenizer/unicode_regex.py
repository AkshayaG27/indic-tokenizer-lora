"""Unicode Grapheme-Aware Pre-Tokenization Engine.

Implements linguistically informed pre-tokenization regex for Indic scripts
(Devanagari, Telugu) to eliminate the 'Unicode Trap' where combining vowel marks
(\\p{M}) and viramas are severed from base consonants (\\p{L}).

Aligns directly with Chapter 09, Section 9.3.2 and Table 9.3 of course textbook.
"""

from __future__ import annotations

import unicodedata
from typing import List, Dict, Any
import regex


# Standard OpenAI cl100k_base regex (exhibits the Unicode Trap on Indic scripts)
CL100K_BASE_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

# Indic Grapheme-Aware Regex:
# Treats base letters (\p{L}) + combining marks (\p{M}) + ZWJ/ZWNJ (\u200c, \u200d)
# as atomic lexical chunks, preventing mid-syllable segmentation.
INDIC_GRAPHEME_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|(?:\p{L}[\p{M}\u200c\u200d]*)+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{M}\p{N}\u200c\u200d]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

_INDIC_COMPILED = regex.compile(INDIC_GRAPHEME_REGEX)
_CL100K_COMPILED = regex.compile(CL100K_BASE_REGEX)


def normalize_unicode(text: str, form: str = "NFC") -> str:
    """Apply standard Unicode normalization (default: NFC canonical composition)."""
    return unicodedata.normalize(form, text)


def pre_tokenize_indic(text: str) -> List[str]:
    """Segment text into pre-tokenization chunks using the Indic Grapheme Regex.
    
    Ensures that consonant + virama + matra clusters remain contiguous within chunks.
    """
    normalized = normalize_unicode(text, "NFC")
    return _INDIC_COMPILED.findall(normalized)


def pre_tokenize_standard(text: str) -> List[str]:
    """Segment text using standard cl100k_base regex (for baseline comparison)."""
    normalized = normalize_unicode(text, "NFC")
    return _CL100K_COMPILED.findall(normalized)


def check_syllable_fracture(text: str, chunks: List[str]) -> List[Dict[str, Any]]:
    """Detect if any combining marks or viramas were severed into independent chunks.
    
    Returns a list of violation dictionaries detailing the fractured chunks and characters.
    """
    violations = []
    for idx, chunk in enumerate(chunks):
        # A chunk that starts with a combining mark indicates a severed matra/virama
        if chunk and unicodedata.category(chunk[0]) == "Mn":
            violations.append({
                "chunk_index": idx,
                "chunk": chunk,
                "first_char": chunk[0],
                "char_name": unicodedata.name(chunk[0], "UNKNOWN"),
                "error_type": "ORPHANED_COMBINING_MARK",
                "reason": "Combining mark separated from preceding base consonant."
            })
    return violations


def verify_containment(text: str, tokens: List[str], chunks: List[str]) -> bool:
    """Verify Chapter 9 Listing 9.1 Containment Invariant.
    
    Every emitted subword token must be a substring of exactly one pre-tokenization chunk,
    and concatenation must reproduce the normalized text.
    """
    # 1. Exact roundtrip reconstruction
    if "".join(chunks) != text or "".join(tokens) != text:
        return False
    
    # 2. Check each token is fully contained inside a single chunk
    chunk_idx = 0
    chunk_offset = 0
    for tok in tokens:
        while chunk_idx < len(chunks):
            current_chunk = chunks[chunk_idx]
            rem_chunk = current_chunk[chunk_offset:]
            if rem_chunk.startswith(tok):
                chunk_offset += len(tok)
                if chunk_offset == len(current_chunk):
                    chunk_idx += 1
                    chunk_offset = 0
                break
            elif chunk_offset == 0 and not current_chunk:
                chunk_idx += 1
            else:
                return False
        else:
            return False
            
    return True
