# SPEC.md: Hindi & Telugu Token Fertility Pareto Audit & Domain LoRA SFT
*(Information-Theoretic Grapheme-Aware BPE Audit, Pareto Vocabulary Optimization, and Native-Tokenizer LoRA PEFT)*

**Course:** UMC301 - Applied Data Science and Artificial Intelligence  
**Instructor:** Prof. Deepak Subramani, CDS, Indian Institute of Science (IISc) Bengaluru  
**Paradigm:** Spec-Driven Development (SDD) - Chapter 13  
**Textbook Alignment:** Chapter 09 (*Large Language Models: Pretraining*), Section 9.3 (*The Pretraining Data Pipeline, Multilingual Tokenization & Parity, Case Study: Indian Languages*), Table 9.3 (*Practical Pitfalls in Tokenization*), and Chapter 08 (*Transformer Architecture & PEFT*)  
**Target Milestone:** Project 3 (Pre/Post-Training Track: Token Economics + Instruction LoRA SFT)  

---

## 1. Goal

Build, mathematically audit, empirically benchmark, and deploy an end-to-end computational linguistics and parameter-efficient fine-tuning (PEFT) framework combining:

1. **Information-Theoretic Token Fertility Audit & Grapheme-Aware BPE (Track A):**
   * Implement a first-principles Byte-Pair Encoding (BPE) pre-tokenizer from scratch using a linguistically informed Unicode regular expression `(?:\p{L}\p{M}*)+` that treats Indic syllables (consonant + virama/matra clusters) as unbreakable atomic units, resolving the "Unicode Trap".
   * Train a grid of custom BPE tokenizers across vocabulary sizes $|V| \in \{4\,096, \; 8\,192, \; 16\,384, \; 32\,768\}$ on a curated trilingual corpus (Hindi, Telugu, English).
   * Map the **Pareto Frontier** balancing Token Fertility ($\mathcal{F}$), Bits-Per-Byte (BPB), and Embedding Table RAM ($\mathcal{M}_{\text{embed}} = |V| \times d_{\text{model}} \times b$), identifying the mathematical "knee of the curve".
   * Conduct an empirical benchmark comparing our custom tokenizer against industry tokenizers: `cl100k_base` (GPT-4), `o200k_base` (GPT-4o), `meta-llama/Meta-Llama-3-8B` (128k), `ai4bharat/indic-bert`, and `sarvamai/sarvam-1`, rigorously quantifying the **"Language Tax"** (latency and API cost disparity).

2. **Domain Instruction Adaptation via LoRA PEFT (Track B):**
   * Decouple the tokenizer audit from model fine-tuning to completely eliminate the **Embedding Misalignment Trap** (never pairing a custom tokenizer with frozen pre-trained embedding weights).
   * Implement Low-Rank Adaptation (LoRA, $r=8, \alpha=16$) on a modern open-weights Small Language Model (SLM: `Qwen/Qwen2.5-0.5B` or `HuggingFaceTB/SmolLM-135M`) using its **native pre-trained tokenizer**.
   * Curate a bilingual instruction dataset (Hindi and Telugu) with strict prompt-loss masking (`labels = -100` on user instructions).
   * Evaluate adaptation quality using validation cross-entropy loss, validation perplexity ($\text{PPL} = \exp(\mathcal{L})$), and character n-gram F-score (**chrF**).

3. **Interactive Visualizer & Viva Defense Suite (Track C):**
   * Deploy an interactive Streamlit application featuring live subword chunk visualizers, a Language Tax API cost and latency calculator, dynamic Pareto frontier exploration plots, and an interactive LoRA inference arena.
   * Provide complete whiteboard derivations for Shannon Entropy, Bits-Per-Byte, Unicode category mechanics, and LoRA gradient backpropagation.

---

## 2. Constraints

* **Language & Runtime:** Python 3.10+.
* **Core Libraries:** `torch`, `transformers`, `peft`, `tokenizers`, `tiktoken`, `regex`, `datasets`, `sacrebleu`, `streamlit`, `plotly`, `pandas`, `numpy`, `pytest`.
* **Architectural Decoupling Constraint (Anti-Misalignment Invariant):**
  * Under no circumstances shall a custom tokenizer trained from scratch in Track A be plugged into a frozen pre-trained base model in Track B.
  * Track A is an information-theoretic tokenizer audit and Pareto optimization.
  * Track B uses the base model's *native* tokenizer to preserve pre-trained embedding weights ($W_{\text{embed}}$).
* **Orthographic Invariant Constraint (Grapheme Preservation):**
  * The pre-tokenization regex must preserve consonant-matra clusters: dependent vowel signs (`\p{M}`) and viramas must never be split from their preceding consonant (`\p{L}`) across chunk boundaries.
  * The containment invariant from Chapter 9 (Listing 9.1) must hold: every emitted subword token must be a strict substring of exactly one pre-tokenization chunk.
* **Loss Masking Constraint:**
  * During instruction SFT, cross-entropy loss must be computed exclusively on assistant completion tokens. User prompt tokens must be masked with target ID `-100` (`ignore_index` in PyTorch `CrossEntropyLoss`).
* **Parameter Budget Constraint (LoRA Efficiency):**
  * Trainable adapter parameters must represent $< 1.0\%$ of total base model parameters. Base weights $W_0$ must remain strictly frozen (`requires_grad = False`).
* **Testing Constraint:**
  * 100% pass rate on automated `pytest` test suite covering:
    1. Grapheme cluster preservation (no mid-syllable splits on Hindi/Telugu test words).
    2. Containment property (all tokens lie within pre-tokenization chunks).
    3. Monotonic decrease of fertility with vocabulary expansion $|V|$.
    4. Mathematical correctness of Bits-Per-Byte (BPB) calculation.
    5. LoRA gradient propagation and parameter freezing.

---

## 3. Linguistic & Orthographic Foundations

### 3.1 The Akshara (Alphasyllabic) Structure
Devanagari (Hindi) and Telugu belong to the Brahmic family of **abugidas (alphasyllabaries)**. Unlike Latin alphabets where vowels and consonants are independent letters written in a linear sequence, Indian scripts organize text into **Aksharas** (orthographic syllables):

$$\text{Akshara} = C_1 + (\text{Virama} + C_2)^* + \text{Matra} + \text{Anusvara}$$

* **Base Consonants ($C$):** Independent consonants with an inherent default short vowel /a/ (e.g., Hindi क `\u0915`, Telugu క `\u0C15`).
* **Virama / Halant:** Pure consonant suppression character (Hindi ् `\u094D`, Telugu ్ `\u0C4D`) that mutes the inherent vowel to form consonant conjuncts (e.g., Telugu క్ష = క + ్ + ష).
* **Dependent Vowels / Matras ($M$):** Modifying vowel diacritics attached directly to consonants (e.g., Hindi िका `\u093F`, Telugu ి `\u0C3F` as in కి).

### 3.2 The "Unicode Trap" (`\p{L}` vs. `\p{M}`)
In the Unicode standard:
* Base consonants are classified as **`\p{L}`** (Letters).
* Dependent matras and viramas are classified as **`\p{M}`** (Combining Marks).

Standard industry tokenizers (e.g., GPT-4's `cl100k_base`) employ pre-tokenization regular expressions designed for Latin scripts:
$$\texttt{[^\textbackslash r\textbackslash n\textbackslash p\{L\}\textbackslash p\{N\}]?\textbackslash p\{L\}+}$$

**The Failure Mode:** The pattern matches runs of `\p{L}+` but excludes `\p{M}`. Consequently:
1. The regex matches the base consonant as one chunk, slicing the matra into an adjacent punctuation/symbol chunk.
2. Because BPE never permits merges across pre-tokenization chunk boundaries, the consonant and matra can **never** be merged into a single token, regardless of vocabulary size $|V|$.
3. Combining marks are forced into isolated single-character tokens or fall back to raw UTF-8 byte sequences (`<0xE0><0xA4>...`), artificially inflating token counts by $3\times - 8\times$.

### 3.3 Agglutinative Morphology of Telugu
Telugu is a Dravidian agglutinative language where grammatical relationships (case markers, plural suffixes, postpositions, tense markers) attach directly as suffixes to noun and verb stems. Without dedicated subword merge rules, a single meaningful word (e.g., *రామునితో* = "with Rama") is shattered into 4 to 8 fragmented byte pieces.

---

## 4. Track A: Information-Theoretic Tokenizer Audit Specifications

### 4.1 Grapheme-Preserving Pre-Tokenizer Specification
We define the Indic Grapheme Regex:
```python
INDIC_GRAPHEME_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|(?:\p{L}\p{M}*)+"
    r"|\p{N}{1,3}"
    r"|[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)
```
* **Invariant:** Any sequence of base letters `\p{L}` followed by combining vowel marks or viramas `\p{M}*` is matched as a single unbroken chunk.

### 4.2 Byte-Level BPE Training Pipeline
* Implemented via Hugging Face `tokenizers` using `ByteLevelBPETrainer`.
* Special tokens: `<s>` (BOS), `</s>` (EOS), `<unk>` (UNK), `<pad>` (PAD), `<mask_0>` (MASK).
* Base vocabulary: 256 individual byte values (guaranteeing 0% out-of-vocabulary rate).
* Target vocabulary sizes:
  $$|V| \in \{4\,096, \; 8\,192, \; 16\,384, \; 32\,768\}$$

### 4.3 Mathematical Metrics & Formulas

#### 1. Token Fertility ($\mathcal{F}$)
$$\mathcal{F}_L = \frac{\sum_{s \in \mathcal{S}_L} |\text{tokenize}(s)|}{\sum_{s \in \mathcal{S}_L} \text{word\_count}(s)}$$
* Measures average tokens generated per whitespace-delimited word in language $L$.
* Ideal Parity Ratio: $\frac{\mathcal{F}(\text{Telugu})}{\mathcal{F}(\text{English})} \approx 1.0$.

#### 2. Bits-Per-Byte (BPB)
$$\text{BPB}(s) = \frac{\sum_{i=1}^{N} -\log_2 P(t_i)}{\text{Total UTF-8 Bytes of } s}$$
Where $P(t_i)$ is the unigram empirical frequency of token $t_i$ over the evaluation corpus:
$$P(t_i) = \frac{\text{count}(t_i)}{\sum_j \text{count}(t_j)}$$
* Reflects the information-theoretic compression efficiency of the token vocabulary. Lower BPB indicates superior compression per raw byte.

#### 3. Embedding Table Memory Overhead ($\mathcal{M}_{\text{embed}}$)
$$\mathcal{M}_{\text{embed}} = |V| \times d_{\text{model}} \times b_{\text{param}}$$
For $d_{\text{model}} = 2048$ with FP16 weights ($b_{\text{param}} = 2$ bytes):
* $|V| = 4\text{K} \implies 16.38\text{ MB}$
* $|V| = 8\text{K} \implies 32.77\text{ MB}$
* $|V| = 16\text{K} \implies 65.54\text{ MB}$
* $|V| = 32\text{K} \implies 131.07\text{ MB}$
* $|V| = 128\text{K} \implies 524.29\text{ MB}$ (LLaMA-3)
* $|V| = 200\text{K} \implies 819.20\text{ MB}$ (GPT-4o)

#### 4. The Pareto Frontier & Marginal Return
We define the marginal compression gain per vocabulary expansion:
$$\Delta_{\text{marginal}}(V_k, V_{k+1}) = \frac{\mathcal{F}(V_k) - \mathcal{F}(V_{k+1})}{\mathcal{M}_{\text{embed}}(V_{k+1}) - \mathcal{M}_{\text{embed}}(V_k)}$$
The **"knee of the curve"** occurs where $|\Delta_{\text{marginal}}|$ collapses toward zero, proving that continuing to enlarge $|V|$ wastes GPU VRAM with negligible fertility improvement.

### 4.4 Industry Benchmarking Suite
Evaluate identical held-out test splits across 6 tokenizers:
1. `cl100k_base` (GPT-4, TikToken)
2. `o200k_base` (GPT-4o, TikToken)
3. `meta-llama/Meta-Llama-3-8B` (HuggingFace 128k BPE)
4. `ai4bharat/indic-bert` (WordPiece)
5. `sarvamai/sarvam-1` (Indic-native BPE)
6. **Our Custom Grapheme-Aware BPE** ($|V| = 32\text{K}$)

---

## 5. Track B: Domain Instruction SFT via LoRA Specifications

### 5.1 Base Model & Tokenizer Selection
* **Model:** `Qwen/Qwen2.5-0.5B` (or `HuggingFaceTB/SmolLM-135M`).
* **Tokenizer:** Base model's **native pretrained tokenizer** (strictly avoiding embedding misalignment).
* **Model Dimensions (Qwen2.5-0.5B):** $d_{\text{model}} = 896$, $n_{\text{heads}} = 14$, $n_{\text{layers}} = 24$, $|V| = 151\,936$.

### 5.2 Bilingual Instruction Dataset Specification
* Curated split of Hindi and Telugu instruction pairs covering translation, reasoning, summarization, and question answering.
* Split sizes: $N_{\text{train}} = 2\,000$ to $4\,000$ examples, $N_{\text{val}} = 300$, $N_{\text{test}} = 300$.
* Formatted using standard ChatML format:
  ```
  <|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>
  ```

### 5.3 Prompt-Loss Masking Specification
* PyTorch `Dataset` generates `input_ids`, `attention_mask`, and `labels`.
* `labels` is initialized as a copy of `input_ids`.
* All token positions belonging to `<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n` are replaced with `-100`.
* Only tokens in `{response}<|im_end|>` carry active label targets.

### 5.4 LoRA Formulation & Mathematical Backpropagation
Low-Rank Adaptation freezes base weights $W_0 \in \mathbb{R}^{d \times k}$ and injects low-rank matrices $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ ($r \ll \min(d, k)$):

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B A$$

* **Hyperparameters:**
  * Rank $r = 8$
  * Scaling factor $\alpha = 16$ ($\text{scale} = \frac{16}{8} = 2.0$)
  * Dropout = $0.05$
  * Target modules: `q_proj`, `v_proj` inside self-attention blocks.
* **Initialization:**
  * $A \sim \mathcal{N}(0, \sigma^2)$ (Gaussian initialization)
  * $B = 0$ (Zero initialization, ensuring $\Delta W = 0$ at step 0)

#### Forward Pass:
$$h = W x = W_0 x + \frac{\alpha}{r} B (A x)$$

#### Backward Pass (Derivation for Whiteboard Defense):
Let $G = \frac{\partial \mathcal{L}}{\partial h} \in \mathbb{R}^{B \times d}$ denote upstream gradients:
$$\frac{\partial \mathcal{L}}{\partial A} = \frac{\alpha}{r} B^T G x^T$$
$$\frac{\partial \mathcal{L}}{\partial B} = \frac{\alpha}{r} G (A x)^T$$
$$\frac{\partial \mathcal{L}}{\partial x} = \left( W_0^T + \frac{\alpha}{r} A^T B^T \right) G$$
$W_0$ remains strictly frozen, requiring zero gradient computation or optimizer momentum storage.

### 5.5 Quantitative Evaluation Protocol
1. **Validation Loss & Perplexity:**
   $$\mathcal{L}_{\text{val}} = -\frac{1}{|\mathcal{Y}|} \sum_{t \in \mathcal{Y}} \log P(y_t \mid x, y_{<t}), \quad \text{PPL} = \exp(\mathcal{L}_{\text{val}})$$
2. **chrF (Character n-gram F-score):**
   * Standard BLEU evaluates word-level n-gram overlap, severely penalizing agglutinative languages when suffixes or case markers vary slightly.
   * **chrF** evaluates character n-gram precision and recall ($n \in [1, 6]$):
     $$\text{chrF} = \frac{(1 + \beta^2) \cdot \text{chrP} \cdot \text{chrR}}{\beta^2 \cdot \text{chrP} + \text{chrR}}$$
   * Evaluated via `sacrebleu.corpus_chrf`.

---

## 6. Acceptance Criteria (Verifiable Pass/Fail Standards)

| ID | Specification Item | Target Metric / Acceptance Threshold | Verification Method |
| :--- | :--- | :--- | :--- |
| **AC-1** | Grapheme Preservation | 0 instances of mid-syllable splitting on 100 test words | `pytest tests/test_grapheme_regex.py` |
| **AC-2** | Containment Invariant | 100% of subword tokens are substrings of exactly 1 pre-token chunk | `pytest tests/test_grapheme_regex.py` |
| **AC-3** | Monotonic Fertility Reduction | $\mathcal{F}(V_{4\text{K}}) > \mathcal{F}(V_{8\text{K}}) > \mathcal{F}(V_{16\text{K}}) > \mathcal{F}(V_{32\text{K}})$ for Hindi and Telugu | `pytest tests/test_metrics.py` |
| **AC-4** | Language Tax Quantification | Document empirical fertility multiplier ($\mathcal{F}_{\text{Telugu}} / \mathcal{F}_{\text{English}} \ge 2.5\times$ in `cl100k`) | `pytest tests/test_metrics.py` |
| **AC-5** | LoRA Parameter Efficiency | Trainable parameters $< 1.0\%$ of base model parameters | `pytest tests/test_lora_sft.py` |
| **AC-6** | Prompt-Loss Masking | 100% of prompt token labels set to `-100` | `pytest tests/test_lora_sft.py` |
| **AC-7** | SFT Convergence | Validation loss decreases across training epochs; $\text{PPL}_{\text{SFT}} < \text{PPL}_{\text{base}}$ | `python src/sft/evaluate.py` |
| **AC-8** | Interactive Dashboard | Functional Streamlit UI running on `localhost:8501` | Automated app health check |

---

## 7. Diagnostic & Failure Remediation Protocol

1. **Diagnostic Test D-1: Syllable Fracture in Custom BPE**
   * *Trigger:* AC-1 failure (a dependent matra is isolated into its own token).
   * *Root Cause Analysis:* The pre-tokenizer regex permitted a chunk boundary between `\p{L}` and `\p{M}`.
   * *Remediation Procedure:* Verify that the regex uses possessive or greedy matching `(?:\p{L}\p{M}*)+` and that Unicode normalization form is NFC.
2. **Diagnostic Test D-2: LoRA Loss Plateauing / Divergence**
   * *Trigger:* Validation loss fails to decrease or exhibits NaN values.
   * *Root Cause Analysis:* Learning rate too aggressive for low-rank adapter matrices or unstable FP16 gradients.
   * *Remediation Procedure:* Lower learning rate from $2 \times 10^{-4}$ to $5 \times 10^{-5}$; switch to BF16 mixed precision or clip gradients at $1.0$.
3. **Diagnostic Test D-3: Out-of-Vocabulary Characters in Evaluation**
   * *Trigger:* Custom tokenizer encounters unknown character `<unk>`.
   * *Remediation Procedure:* Ensure Byte-Level BPE is enabled so any unmapped Unicode character falls back to its 256 fundamental UTF-8 byte tokens.
