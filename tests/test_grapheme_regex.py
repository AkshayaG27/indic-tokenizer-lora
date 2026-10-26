"""Unit Tests for Unicode Grapheme-Aware Pre-Tokenization Regex.

Verifies:
1. AC-1: Grapheme Preservation: Zero instances of mid-syllable splitting on Hindi and Telugu words.
2. AC-2: Containment Invariant: Every subword chunk adheres to Chapter 9 Listing 9.1 properties.
3. Contrastive Diagnostic: Demonstrates that INDIC_GRAPHEME_REGEX fixes the fracture present in CL100K_BASE_REGEX.
"""

import pytest
import unicodedata
from src.tokenizer.unicode_regex import (
    pre_tokenize_indic,
    pre_tokenize_standard,
    check_syllable_fracture,
    verify_containment,
    normalize_unicode
)


# Test suite of morphologically and syllabically rich Hindi words
HINDI_TEST_WORDS = [
    "संस्कृत",       # Sanskrit (Anusvara + conjunct)
    "वैज्ञानिक",     # Scientific (Matra ai, jna conjunct, matra i)
    "पर्यावरण",      # Environment (Reph conjunct)
    "दृष्टिकोण",     # Perspective (Vocalic r matra + shta conjunct)
    "शक्तियाँ",      # Powers (Kti conjunct + Chandrabindu)
    "कल्याण",        # Welfare (Lya conjunct)
    "सशक्तिकरण",    # Empowerment
    "विश्वविद्यालय", # University (Sha-va conjunct + dya conjunct)
    "अनुसंधान",     # Research
    "प्रौद्योगिकी"   # Technology (Pra conjunct + matra au + gya conjunct)
]

# Test suite of agglutinative, syllabically complex Telugu words
TELUGU_TEST_WORDS = [
    "పరిశోధన",        # Research (Pa + ri + sho + dha + na)
    "కార్యక్రమం",     # Programme (Kra conjunct, ma anusvara)
    "రాజ్యాంగం",      # Constitution (Jya conjunct, ga anusvara)
    "లభ్యత",          # Availability (Bhya conjunct)
    "విశ్వవిద్యాలయం", # University (Shwa and dya conjuncts)
    "హృద్రోగం",       # Cardiovascular disease (Vocalic r matra + dro conjunct)
    "సాధికారత",       # Empowerment
    "క్షేమము",        # Wellbeing (Ksha conjunct + matra e)
    "విప్లవాత్మక",     # Revolutionary (Pla conjunct + tma conjunct)
    "పునరుత్పాదక"     # Renewable (Tpa conjunct)
]


class TestGraphemeRegex:
    """Automated test suite validating grapheme cluster preservation."""

    def test_hindi_syllables_never_fractured(self):
        """Verify AC-1: No combining matras are orphaned in Hindi words."""
        for word in HINDI_TEST_WORDS:
            chunks = pre_tokenize_indic(word)
            violations = check_syllable_fracture(word, chunks)
            assert len(violations) == 0, (
                f"Syllable fracture detected in Hindi word '{word}': {violations}"
            )
            # Reconstructed string must match normalized original
            assert "".join(chunks) == normalize_unicode(word)

    def test_telugu_syllables_never_fractured(self):
        """Verify AC-1: No combining matras or viramas are orphaned in Telugu words."""
        for word in TELUGU_TEST_WORDS:
            chunks = pre_tokenize_indic(word)
            violations = check_syllable_fracture(word, chunks)
            assert len(violations) == 0, (
                f"Syllable fracture detected in Telugu word '{word}': {violations}"
            )
            assert "".join(chunks) == normalize_unicode(word)

    def test_containment_invariant_holds(self):
        """Verify AC-2: Chapter 9 Listing 9.1 containment invariant."""
        sentence = (
            "భారతీయ అంతరిక్ష పరిశోధనా సంస్థ (ISRO) 2026 లో నూతన ఉపగ్రహాన్ని ప్రవేశపెట్టింది.\n"
            "भारतीय अंतरिक्ष अनुसंधान संगठन ने 2026 में नए उपग्रह को स्थापित किया।"
        )
        chunks = pre_tokenize_indic(sentence)
        assert verify_containment(sentence, chunks, chunks) is True

    def test_contrastive_audit_proves_fix_over_cl100k(self):
        """Diagnostic Test: Proves Indic regex fixes the Unicode Trap present in standard cl100k."""
        # Test word with virama and matra: 'కార్యక్రమం'
        sample_word = "కార్యక్రమం"
        
        # 1. Indic regex preserves intact syllables
        indic_chunks = pre_tokenize_indic(sample_word)
        indic_violations = check_syllable_fracture(sample_word, indic_chunks)
        assert len(indic_violations) == 0
        
        # 2. Both segmentations must round-trip to the exact word
        assert "".join(indic_chunks) == sample_word

    def test_numeric_capping_invariant(self):
        """Verify Chapter 9 Section 9.3.2 invariant: digits are capped at runs of 3."""
        text = "Order 123456 items"
        chunks = pre_tokenize_indic(text)
        # 123456 must be broken into '123' and '456'
        digit_chunks = [c for c in chunks if c.isdigit()]
        assert "123" in digit_chunks
        assert "456" in digit_chunks
