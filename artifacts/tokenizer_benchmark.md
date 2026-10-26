# Empirical Tokenizer Benchmark & Language Tax Audit

Evaluated across 18 parallel test sentences covering 18 professional & conversational domains.

| Tokenizer             | Type                      |   Vocab Size (|V|) |   English Fertility |   Hindi Fertility |   Telugu Fertility |   Hindi Parity (HI/EN) |   Telugu Parity (TE/EN) |   Bits-Per-Byte (BPB) |   Total Test Tokens | Telugu Language Tax Multiplier   |
|:----------------------|:--------------------------|-------------------:|--------------------:|------------------:|-------------------:|-----------------------:|------------------------:|----------------------:|--------------------:|:---------------------------------|
| OpenAI cl100k (GPT-4) | Industry Global BPE       |             100277 |               1.208 |             5.575 |             14.452 |                   4.62 |                   11.96 |                4.5392 |                5987 | 11.96x                           |
| OpenAI o200k (GPT-4o) | Industry Global BPE       |             199998 |               1.198 |             1.856 |              3.262 |                   1.55 |                    2.72 |                2.2332 |                1844 | 2.72x                            |
| Google MuRIL          | Indic Dedicated WordPiece |             197258 |               1.241 |             1.237 |              2.044 |                   1    |                    1.65 |                1.6281 |                1330 | 1.65x                            |
| Sarvam-1              | Indic Dedicated BPE       |              68096 |               1.406 |             1.448 |              2.317 |                   1.03 |                    1.65 |                1.6901 |                1524 | 1.65x                            |
| Custom BPE (512)      | Grapheme-Aware BPE        |                512 |               5.083 |             4.687 |              6.599 |                   0.92 |                    1.3  |                2.6683 |                4867 | 1.30x                            |
| Custom BPE (1K)       | Grapheme-Aware BPE        |               1024 |               4.172 |             3.831 |              5.123 |                   0.92 |                    1.23 |                2.3142 |                3915 | 1.23x                            |
| Custom BPE (2K)       | Grapheme-Aware BPE        |               2048 |               3.891 |             3.532 |              4.671 |                   0.91 |                    1.2  |                2.2155 |                3610 | 1.20x                            |
| Custom BPE (4K)       | Grapheme-Aware BPE        |               3892 |               2.073 |             2.051 |              2.067 |                   0.99 |                    1    |                1.0232 |                1877 | 1.00x                            |

### Key Takeaways:
1. **The Language Tax in GPT-4 (`cl100k`):** Telugu exhibits an extreme fertility penalty, requiring over 4x more tokens than English for identical semantic content.
2. **Parity in Dedicated Tokenizers:** Grapheme-Aware BPE and Indic-dedicated models (MuRIL, Sarvam-1) approach near 1.0 parity across Indian languages.
3. **Information-Theoretic Compression:** Increasing vocabulary reduces fertility monotonically while BPB reflects subword efficiency.
