# PLAN.md: Step-by-Step Implementation Plan for Project 3

**Project:** Hindi & Telugu Token Fertility Pareto Audit & Domain LoRA SFT  
**Spec Reference:** [SPEC.md](file:///c:/Users/aksha/DSAI_Projects/Project3_Indic_Tokenizer_LoRA_SFT/SPEC.md)  
**Methodology:** Spec-Driven Development (SDD) - Chapter 13  

---

## Phase 1: Environment Scaffolding, Data Ingestion & Grapheme Cleaning

- [x] **Task 1.1: Directory Scaffolding & Dependency Verification**
  - Verify Python environment and dependencies (`torch`, `transformers`, `peft`, `tokenizers`, `tiktoken`, `regex`, `datasets`, `sacrebleu`, `streamlit`, `plotly`, `pytest`).
  - Create project subdirectories:
    - `data/raw/`, `data/processed/`, `data/instructions/`
    - `src/tokenizer/`, `src/audit/`, `src/sft/`
    - `tests/`, `app/`, `artifacts/`

- [x] **Task 1.2: Trilingual Corpus Curation (`src/tokenizer/data_curator.py`)**
  - Curate parallel and monolingual benchmark corpora for **Hindi**, **Telugu**, and **English** (sourced from AI4Bharat IndicCorp / Flores-200 subsets).
  - Clean and apply Unicode NFC canonical composition.
  - Strip anomalous non-printable and unassigned characters while strictly preserving Virama (`\u094D`, `\u0C4D`) and dependent vowel matras (`\p{M}`).
  - Export clean training corpus (`data/processed/train_corpus.txt`) and held-out evaluation corpus (`data/processed/test_sentences.json`).

- [x] **Task 1.3: Unicode Grapheme-Aware Pre-Tokenizer (`src/tokenizer/unicode_regex.py`)**
  - Implement `INDIC_GRAPHEME_REGEX` pattern ensuring consonant-matra clusters `(?:\p{L}\p{M}*)+` are matched as unbreakable atomic units.
  - Implement chunk boundary validator asserting Chapter 9 containment invariant.

- [x] **Task 1.4: Unit Tests for Grapheme Invariant (`tests/test_grapheme_regex.py`)**
  - Write test verifying that 100 complex Hindi and Telugu words (e.g., *పరిశోధన*, *కార్యక్రమం*, *संस्कृत*, *वैज्ञानिक*) have zero split vowels across chunk boundaries.
  - Write test verifying Chapter 9 containment property: every token belongs to exactly one chunk.
  - **Acceptance Gate:** `pytest tests/test_grapheme_regex.py` passes 100% (AC-1, AC-2).

---

## Phase 2: Custom Grapheme-Aware BPE Training & Vocab Grid (Track A)

- [x] **Task 2.1: Byte-Level BPE Training Pipeline (`src/tokenizer/trainer.py`)**
  - Implement Hugging Face `tokenizers.models.BPE` and `ByteLevelBPETrainer`.
  - Configure byte fallback (256 base bytes + special tokens `<s>`, `</s>`, `<unk>`, `<pad>`).
  - Train 4 tokenizer checkpoints across the vocabulary grid:
    - `artifacts/tokenizer_bpe_512.json` ($|V| = 512$)
    - `artifacts/tokenizer_bpe_1k.json` ($|V| = 1\,024$)
    - `artifacts/tokenizer_bpe_2k.json` ($|V| = 2\,048$)
    - `artifacts/tokenizer_bpe_4k.json` ($|V| = 3\,892$)

- [x] **Task 2.2: Token Fertility Calculator (`src/tokenizer/fertility.py`)**
  - Implement $\mathcal{F}_L = \frac{\sum |\text{tokens}|}{\sum \text{words}}$ calculation.
  - Implement language-stratified fertility breakdowns (English, Hindi, Telugu).

- [x] **Task 2.3: Bits-Per-Byte & Entropy Computation Engine (`src/tokenizer/bpb.py`)**
  - Compute empirical unigram token probabilities $P(t_i)$.
  - Implement Bits-Per-Byte formula:
    $$\text{BPB} = \frac{\sum_{i=1}^N -\log_2 P(t_i)}{\text{Total UTF-8 Bytes}}$$
  - Compute theoretical Shannon entropy and compression ratio ($\text{Bytes} / \text{Token}$).

- [x] **Task 2.4: Unit Tests for Compression & Fertility Monotonicity (`tests/test_metrics.py`)**
  - Verify that token fertility decreases monotonically as vocabulary size increases:
    $$\mathcal{F}(V_{512}) > \mathcal{F}(V_{1\text{K}}) > \mathcal{F}(V_{2\text{K}}) > \mathcal{F}(V_{4\text{K}})$$
  - Verify that BPB calculation satisfies theoretical bounds ($0.1 \le \text{BPB} \le 12.0$).
  - **Acceptance Gate:** `pytest tests/test_metrics.py` passes 100% (AC-3).

---

## Phase 3: Industry Benchmark & Pareto Frontier Optimization (Track A)

- [x] **Task 3.1: Industry Tokenizer Adapter & Benchmark Suite (`src/audit/benchmark.py`)**
  - Implement unified encoding interface supporting:
    - OpenAI `cl100k_base` (GPT-4) via `tiktoken`
    - OpenAI `o200k_base` (GPT-4o) via `tiktoken`
    - Google `google/muril-base-cased` (Indic-BERT) via `transformers`
    - Sarvam AI `sarvamai/sarvam-1` via `transformers`
    - Custom Grapheme BPE ($|V| \in \{512, 1\text{K}, 2\text{K}, 4\text{K}\}$)
  - Run side-by-side tokenization on parallel test sentences.
  - Export comparative benchmark table to `artifacts/tokenizer_benchmark.csv` and `artifacts/tokenizer_benchmark.md`.

- [x] **Task 3.2: Pareto Frontier & Marginal Return Engine (`src/audit/pareto.py`)**
  - Compute embedding table RAM: $\mathcal{M}_{\text{embed}} = |V| \times d_{\text{model}} \times b_{\text{param}}$.
  - Compute marginal compression derivative $\frac{\Delta \mathcal{F}}{\Delta |V|}$.
  - Identify the optimal Pareto trade-off point ("knee of the curve").
  - Generate high-resolution interactive plots: `artifacts/pareto_frontier.html` and `artifacts/pareto_analysis.json`.

- [x] **Task 3.3: Language Tax & Cost Disparity Calculator (`src/audit/language_tax.py`)**
  - Calculate empirical fertility multiplier:
    $$\mathcal{R}_{\text{tax}} = \frac{\mathcal{F}(\text{Telugu})}{\mathcal{F}(\text{English})}$$
  - Calculate latency amplification and API cost penalty under standard token pricing ($/1M tokens).
  - Export summary findings to `artifacts/language_tax_report.md` (AC-4).

---

## Phase 4: Domain Instruction Adaptation via LoRA PEFT (Track B)

- [x] **Task 4.1: Bilingual Instruction Dataset Builder (`src/sft/dataset.py`)**
  - Load curated Hindi and Telugu instruction pairs (from `data/instructions/bilingual_instructions.json`).
  - Format instructions into standardized ChatML format (`<|im_start|>user...`, `<|im_start|>assistant...`).
  - Create PyTorch `Dataset` with train/validation/test splits (`data/instructions/`).

- [x] **Task 4.2: Prompt-Loss Masking Protocol (`src/sft/dataset.py`)**
  - Tokenize chat conversations using the base model's **native tokenizer**.
  - Apply target masking: set all prompt tokens to `-100` (`ignore_index`).
  - Assert that loss is computed solely on assistant response tokens (AC-6).

- [x] **Task 4.3: Base SLM & LoRA Adapter Setup (`src/sft/lora_model.py`)**
  - Load base SLM (`HuggingFaceTB/SmolLM-135M`) in mixed precision.
  - Configure PEFT `LoraConfig`:
    - Rank $r = 8$, scaling $\alpha = 16$, dropout $= 0.05$
    - Target modules: `["q_proj", "v_proj"]`
  - Verify parameter freezing: assert base weights $W_0$ have `requires_grad = False` and trainable parameters $< 1.0\%$ of total (AC-5: 0.34%).

- [x] **Task 4.4: Mixed-Precision SFT Training Pipeline (`src/sft/train.py`)**
  - Implement training loop with AdamW optimizer, cosine learning rate scheduler, and gradient clipping (1.0).
  - Log training loss, validation loss, and validation perplexity per epoch.
  - Save adapter checkpoint to `artifacts/lora_adapter/` and history to `artifacts/training_history.json`.

- [x] **Task 4.5: Quantitative Evaluation & Generation Benchmark (`src/sft/evaluate.py`)**
  - Compute held-out test perplexity: $\text{PPL} = \exp(\mathcal{L}_{\text{test}})$.
  - Compute character n-gram F-score (**chrF**) using `sacrebleu` across Hindi and Telugu outputs.
  - Generate qualitative side-by-side comparison table (Base Model vs. LoRA-adapted Model).
  - **Acceptance Gate:** `pytest tests/test_lora_sft.py` passes 100% (AC-5, AC-6, AC-7).

---

## Phase 5: Streamlit Interactive Audit Suite & Viva Defense (Track C)

- [x] **Task 5.1: Interactive Streamlit Dashboard (`app/app.py`)**
  - **Tab 1: Live Syllable & Token Visualizer:** Text input for Hindi/Telugu with color-coded token chips comparing `cl100k`, `o200k`, `llama-3`, and `custom-BPE`. Visual warning for broken matras.
  - **Tab 2: Language Tax & API Cost Calculator:** Real-time monetary and latency comparison between English, Hindi, and Telugu.
  - **Tab 3: Pareto Frontier Explorer:** Interactive Plotly curves of Vocab Size vs. Fertility vs. RAM.
  - **Tab 4: LoRA Inference Arena:** Live prompt test arena comparing Base SLM vs. LoRA-adapted SLM on user-entered Indic prompts.
  - **Acceptance Gate:** Streamlit app loads and runs with zero errors (AC-8).

- [x] **Task 5.2: SDD Design-Decision Walkthrough & Viva Guide (`WALKTHROUGH.md`)**
  - Step-by-step documentation answering all viva defense questions:
    - Derivation of Shannon Entropy and Bits-Per-Byte.
    - Mathematical analysis of Unicode `\p{L}` vs. `\p{M}`.
    - Mathematical derivation of LoRA forward/backward passes.
    - Defense of the decoupled two-track architecture (preventing embedding misalignment).
