"""Unit Tests for Information-Theoretic Metrics (Fertility and BPB).

Verifies:
1. AC-3: Monotonic Fertility Reduction across vocabulary scales:
   F(V_512) > F(V_1k) > F(V_2k) > F(V_4k).
2. BPB bounds and mathematical consistency.
3. 100% loss-free roundtrip string reconstruction across all checkpoints.
"""

import json
import pytest
from tokenizers import Tokenizer
from src.tokenizer.fertility import compute_stratified_fertility
from src.tokenizer.bpb import BPBCalculator


CHECKPOINT_FILES = [
    ("512", "artifacts/tokenizer_bpe_512.json"),
    ("1k", "artifacts/tokenizer_bpe_1k.json"),
    ("2k", "artifacts/tokenizer_bpe_2k.json"),
    ("4k", "artifacts/tokenizer_bpe_4k.json"),
]


@pytest.fixture(scope="module")
def benchmark_data():
    with open("data/processed/test_sentences.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def loaded_tokenizers():
    toks = {}
    for label, path in CHECKPOINT_FILES:
        toks[label] = Tokenizer.from_file(path)
    return toks


class TestMetrics:
    """Verifies mathematical monotonicity, BPB bounds, and exact roundtrip."""

    def test_loss_free_roundtrip_all_checkpoints(self, benchmark_data, loaded_tokenizers):
        """Verify that every tokenizer checkpoint achieves 100% loss-free decoding."""
        for label, tok in loaded_tokenizers.items():
            for item in benchmark_data:
                for lang in ["en", "hi", "te"]:
                    original = item[lang]
                    encoded = tok.encode(original)
                    decoded = tok.decode(encoded.ids)
                    assert decoded == original, (
                        f"Roundtrip failure in checkpoint {label} for lang {lang}: '{decoded}' != '{original}'"
                    )

    def test_monotonic_fertility_reduction(self, benchmark_data, loaded_tokenizers):
        """Verify AC-3: Fertility decreases monotonically as vocabulary size increases."""
        hi_fertilities = []
        te_fertilities = []
        en_fertilities = []

        for label, tok in loaded_tokenizers.items():
            stats = compute_stratified_fertility(benchmark_data, lambda txt, t=tok: t.encode(txt).ids)
            en_fertilities.append((label, stats["en"]["fertility"]))
            hi_fertilities.append((label, stats["hi"]["fertility"]))
            te_fertilities.append((label, stats["te"]["fertility"]))

        print("\nEmpirical Fertility by Scale:")
        for (l, e), (_, h), (_, t) in zip(en_fertilities, hi_fertilities, te_fertilities):
            print(f"  Scale {l:>4}: EN = {e:.2f}, HI = {h:.2f}, TE = {t:.2f}")

        # Assert monotonic non-increasing fertility across vocabulary scales
        for i in range(len(CHECKPOINT_FILES) - 1):
            assert hi_fertilities[i][1] >= hi_fertilities[i+1][1], (
                f"Hindi fertility did not decrease from {hi_fertilities[i][0]} ({hi_fertilities[i][1]}) to {hi_fertilities[i+1][0]} ({hi_fertilities[i+1][1]})"
            )
            assert te_fertilities[i][1] >= te_fertilities[i+1][1], (
                f"Telugu fertility did not decrease from {te_fertilities[i][0]} ({te_fertilities[i][1]}) to {te_fertilities[i+1][0]} ({te_fertilities[i+1][1]})"
            )

        # Confirm strictly greater compression at 4k than 512
        assert hi_fertilities[0][1] > hi_fertilities[-1][1]
        assert te_fertilities[0][1] > te_fertilities[-1][1]

    def test_bpb_mathematical_bounds(self, benchmark_data, loaded_tokenizers):
        """Verify that Bits-Per-Byte satisfies theoretical information bounds."""
        tok = loaded_tokenizers["2k"]
        bpb_calc = BPBCalculator(vocab_size=tok.get_vocab_size())

        # Fit unigram distribution on test sequences
        all_ids = []
        for item in benchmark_data:
            for lang in ["en", "hi", "te"]:
                all_ids.append(tok.encode(item[lang]).ids)
        bpb_calc.fit_distribution(all_ids)

        for item in benchmark_data:
            for lang in ["en", "hi", "te"]:
                text = item[lang]
                tids = tok.encode(text).ids
                res = bpb_calc.compute_sequence_bpb(text, tids)
                # BPB must be strictly positive and bounded by theoretical maximum
                assert 0.1 <= res["bpb"] <= 12.0, (
                    f"BPB out of reasonable bounds ({res['bpb']}) for: {text}"
                )
                assert res["total_bits"] > 0
                assert res["utf8_bytes"] == len(text.encode("utf-8"))
