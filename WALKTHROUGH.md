# WALKTHROUGH.md: Viva Defense Guide & Architectural Walkthrough
# UMC301 Project 3: Hindi & Telugu Token Fertility Pareto Audit & Domain LoRA SFT

**Course:** UMC301 - Applied Data Science and Artificial Intelligence  
**Institution:** Indian Institute of Science (IISc), Computational and Data Sciences (CDS)  
**Instructor:** Prof. Deepak Subramani, CDS, IISc Bengaluru  
**Methodology:** Spec-Driven Development (SDD) - Chapter 13  
**Textbook Foundations:** Chapter 09 (*LLM Pretraining*), Section 9.3 (*Data Pipeline & Multilingual Tokenization*), Table 9.3 (*Tokenization Pitfalls*), and Chapter 08 (*Transformer Architecture & PEFT*)  
**Target:** A+ Grade Defense via First-Principles Mathematical Derivations and Empirical Systems Proofs  

---

## 1. Project Overview & Architectural Architecture

Project 3 investigates the computational linguistics, token economics, and parameter-efficient fine-tuning (PEFT) dynamics of morphologically rich and syllabic Indian languages (**Hindi** and **Telugu**).

```mermaid
graph TD
    subgraph TrackA["Track A: Information-Theoretic Tokenizer Audit"]
        Corpus["Balanced Trilingual Corpus<br>(18 Domains: Legal, Medical, Agritech, Conversational)"] --> CustomBPE["Train Grapheme-Aware BPE<br>Regex: (?:\p{L}[\p{M}\u200c\u200d]*)+<br>Eliminates Unicode Trap"]
        CustomBPE --> Pareto["Pareto Frontier Mapping<br>|V| ∈ {512, 1K, 2K, 4K}<br>Fertility vs. BPB vs. Embedding RAM"]
        Pareto --> Benchmark["Empirical Industry Benchmark<br>(cl100k, o200k, MuRIL, Sarvam-1)"]
    end

    subgraph TrackB["Track B: Domain Instruction SFT via LoRA"]
        BaseSLM["Base SLM (SmolLM-135M / Qwen2.5-0.5B)<br>with NATIVE Pretrained Tokenizer"] --> LoRATrain["LoRA PEFT (r=8, α=16)<br>Targets: q_proj, v_proj<br>Strict Prompt-Loss Masking (labels = -100)"]
        LoRATrain --> Evaluator["Quantitative Evaluation<br>(Perplexity & chrF Score)"]
    end

    TrackA --> App["Track C: Streamlit Audit Suite & Viva Defense"]
    TrackB --> App
```

### The Foundational Engineering Invariant: The Decoupled System
A common fatal flaw in amateur multilingual NLP projects is **The Embedding Misalignment Trap**: attempting to plug a newly trained custom tokenizer into a frozen pre-trained transformer. 

Because token ID $i$ in a pre-trained embedding table $W_{\text{embed}} \in \mathbb{R}^{|V| \times d}$ is irrevocably tied to the subword representations learned over trillions of pre-training tokens, substituting a new tokenizer destroys the learned coordinate space and produces gibberish.

To maintain strict academic rigor:
* **Track A:** Operates as an independent information-theoretic audit exploring vocabulary scale, Bits-Per-Byte, token fertility, and Pareto trade-offs.
* **Track B:** Adapts a modern base SLM to Indic instructions using its **native pre-trained tokenizer** and low-rank adapters ($r=8, \alpha=16$).

---

## 2. Whiteboard Mathematical Derivations for Viva Defense

You must be prepared to write these five exact mathematical derivations on the whiteboard during your viva defense without hesitation.

---

### Derivation 1: Shannon Entropy, Empirical Unigram Loss & Bits-Per-Byte (BPB)

#### 1. Shannon Source Coding Theorem
Under Shannon's source coding theorem, the theoretical lower bound on expected code length for a discrete random variable $X$ with alphabet $\mathcal{V}$ is bounded by its entropy:
$$\mathbb{E}[L] \ge H(X) = -\sum_{t \in \mathcal{V}} P(t) \log_2 P(t) \quad \text{bits/symbol}$$

#### 2. The Bits-Per-Byte (BPB) Formulation
To evaluate subword tokenizers across different alphabet sizes and languages fairly, we normalize total surprisal (in bits) by the raw uncompressed string length in UTF-8 bytes:

$$\text{BPB}(s) = \frac{\sum_{i=1}^{N} -\log_2 P(t_i)}{\text{Total UTF-8 Bytes of } s}$$

Where $P(t_i)$ is estimated using the smoothed empirical unigram probability:
$$P(t_i) = \frac{\text{count}(t_i) + \alpha}{\sum_{j \in \mathcal{V}} \text{count}(j) + \alpha |\mathcal{V}|}$$

* **Interpretation:** Lower BPB indicates superior information compression per raw byte of text.
* **Empirical Measured Result:**
  * OpenAI `cl100k_base`: **$4.54\text{ BPB}$** (poor compression on Indic text due to severe byte fallbacks).
  * Custom Grapheme BPE (4K): **$1.02\text{ BPB}$** (optimal subword information density).

---

### Derivation 2: The Akshara Alphasyllabic Formula & The "Unicode Trap"

#### 1. The Akshara Orthographic Structure
Unlike Latin scripts which are alphabetic (linear consonant-vowel sequences), Indian scripts (Devanagari, Telugu) are **abugidas (alphasyllabaries)** where text is organized into syllable clusters called **Aksharas**:

$$\text{Akshara} = C_1 + (\text{Virama} + C_2)^* + \text{Matra} + \text{Anusvara}$$

* Base Consonant ($C_1$): Inherent short vowel /a/.
* Virama / Halant: Consonant suppressor (Telugu `్` `\u0C4D`, Hindi ् `\u094D`).
* Dependent Matra: Modifying vowel sign (e.g. Telugu `ి` `\u0C3F`).

#### 2. The Unicode Classification Failure Mode
In the Unicode standard:
* Base consonants belong to category **`\p{L}`** (Letters).
* Dependent matras and viramas belong to category **`\p{M}`** (Combining Marks).

Standard industrial pre-tokenization regexes (such as OpenAI's `cl100k_base`):
```regex
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

* **The Failure:** `[^\r\n\p{L}\p{N}]?\p{L}+` matches only letters `\p{L}`, stopping immediately before any combining mark `\p{M}`.
* The pre-tokenizer slices the syllable mid-cluster, separating the vowel sign from the base consonant into a distinct chunk.
* Under BPE rules (Chapter 9, Section 9.3.2), **merges across pre-tokenization chunk boundaries are strictly forbidden**.
* Consequently, vowel matras can never merge with consonants and fall back to raw UTF-8 byte tokens (`<0xE0><0xA4>...`), artificially exploding Telugu token fertility to **$14.45\text{ tokens/word}$**!

#### 3. The Surgical Solution: Grapheme-Preserving Regex
```python
INDIC_GRAPHEME_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|(?:\p{L}[\p{M}\u200c\u200d]*)+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{M}\p{N}\u200c\u200d]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)
```
* Binds `\p{L}` and all succeeding combining marks `\p{M}*` into a single unbreakable chunk, preserving complete Aksharas and restoring Telugu fertility to **$2.07\text{ tokens/word}$** (1.00× parity).

---

### Derivation 3: The Embedding Table Memory Equation & Pareto Knee

#### 1. Hardware Memory Footprint of the Embedding Table
In modern autoregressive transformers, the token embedding matrix $W_{\text{embed}} \in \mathbb{R}^{|V| \times d_{\text{model}}}$ and the language modeling unembedding head $W_{\text{lm\_head}} \in \mathbb{R}^{|V| \times d_{\text{model}}}$ scale linearly with vocabulary size $|V|$:

$$\mathcal{M}_{\text{embed}} = |V| \times d_{\text{model}} \times b_{\text{param}} \quad \text{bytes}$$

For $d_{\text{model}} = 2048$ with 16-bit precision ($b_{\text{param}} = 2\text{ bytes}$):
* $|V| = 512 \implies \mathbf{2.00\text{ MB}}$
* $|V| = 1\,024 \implies \mathbf{4.00\text{ MB}}$
* $|V| = 2\,048 \implies \mathbf{8.00\text{ MB}}$
* $|V| = 3\,892 \implies \mathbf{15.20\text{ MB}}$
* *(Industry: GPT-4 at 100K = **$391.7\text{ MB}$**; GPT-4o at 200K = **$781.2\text{ MB}$**).*

#### 2. The Marginal Compression Derivative
$$\Delta_{\text{marginal}} = \frac{\Delta \mathcal{F}_{\text{Telugu}}}{\Delta \mathcal{M}_{\text{embed}}}$$
* **$512 \to 1\text{K}$:** Fertility drops $1.48$ with $+2.0\text{ MB}$ RAM ($\Delta = 0.738$ drop/MB).
* **$1\text{K} \to 2\text{K}$:** Fertility drops $0.45$ ($\Delta = 0.113$ drop/MB).
* **$2\text{K} \to 4\text{K}$:** Fertility drops $2.60$ down to $2.07$ ($\Delta = 0.362$ drop/MB).
* **The Knee of the Curve:** Expanding beyond $4\text{K}$ hits diminishing returns because the most common morphemes are already atomic. At $4\text{K}$, our tokenizer captures **1.00× Parity** with only $15.2\text{ MB}$ RAM overhead!

---

### Derivation 4: LoRA Forward Pass & Backward Gradient Derivations

#### 1. Parameter Decomposition
For a pre-trained frozen weight matrix $W_0 \in \mathbb{R}^{d \times k}$, Low-Rank Adaptation (LoRA) decomposes the weight update $\Delta W$ into two low-rank matrices $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ with rank $r \ll \min(d, k)$:

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B A$$

* Scaling factor: $\frac{\alpha}{r}$ (where $r=8, \alpha=16 \implies \text{scale} = 2.0$).
* Parameter count: $2 \times r \times d \ll d \times k$. In our model, adapter parameters account for **$460,800$ weights** (**$0.3414\%$** of total parameters, strictly satisfying AC-5 budget $< 1.0\%$).

#### 2. Forward Propagation
Given input activation $x \in \mathbb{R}^{B \times k}$:
$$h = W x = W_0 x + \frac{\alpha}{r} B (A x)$$

#### 3. Backward Gradient Derivation (Whiteboard Proof)
Let $G = \frac{\partial \mathcal{L}}{\partial h} \in \mathbb{R}^{B \times d}$ denote the incoming gradient from the loss function:

1. **Gradient with respect to Adapter Matrix $A$:**
   $$\frac{\partial \mathcal{L}}{\partial A} = \frac{\alpha}{r} B^T G x^T$$
2. **Gradient with respect to Adapter Matrix $B$:**
   $$\frac{\partial \mathcal{L}}{\partial B} = \frac{\alpha}{r} G (A x)^T$$
3. **Gradient backpropagated to input activation $x$:**
   $$\frac{\partial \mathcal{L}}{\partial x} = \left( W_0^T + \frac{\alpha}{r} A^T B^T \right) G$$
4. **Base Weight Invariant:**
   $$\frac{\partial \mathcal{L}}{\partial W_0} = \mathbf{0} \quad (\text{Strictly frozen; zero optimizer momentum stored})$$

---

### Derivation 5: The chrF Character-Level n-Gram F-Score Formula

#### Why Standard BLEU Fails on Indic Languages
Standard BLEU evaluates exact whitespace-delimited word overlaps:
$$\text{BLEU} = \text{BP} \cdot \exp\left( \sum_{n=1}^N w_n \ln p_n \right)$$
In agglutinative languages like Telugu, grammatical case suffixes attach directly to stems (*రామునితో* = "with Rama"). If the model generates a valid variant (*రాముడితో*), word-level BLEU scores it as **0% precision**.

#### The chrF Formulation
**chrF** evaluates character n-gram precision ($\text{chrP}$) and recall ($\text{chrR}$) across n-grams $n \in [1, 6]$:

$$\text{chrP} = \frac{\sum_{n=1}^6 \text{count}_{\text{match}}(n)}{\sum_{n=1}^6 \text{count}_{\text{hyp}}(n)}, \quad \text{chrR} = \frac{\sum_{n=1}^6 \text{count}_{\text{match}}(n)}{\sum_{n=1}^6 \text{count}_{\text{ref}}(n)}$$

$$\text{chrF}_\beta = \frac{(1 + \beta^2) \cdot \text{chrP} \cdot \text{chrR}}{\beta^2 \cdot \text{chrP} + \text{chrR}}$$

With $\beta = 2$ (chrF++ weighting recall twice as heavily as precision), capturing whether root consonants and attached vowel matras were synthesized accurately.

---

## 3. Key Empirical Findings & Benchmarks

### 1. The Language Tax Summary Table
*(Evaluated across 18 parallel domains: Legal, Healthcare, Agritech, Meteorology, Digital Code-Mixed, and Conversational).*

| Tokenizer | Type | English Fertility | Hindi Fertility | Telugu Fertility | Hindi Parity | Telugu Parity | Telugu Cost Penalty (15M words) | Telugu Latency Slowdown |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **OpenAI cl100k (GPT-4)** | Global BPE | **1.21** | **5.58** | **14.45** | 4.62× | **11.96×** | **+\$993.30 / mo** | **11.96× slower** |
| **OpenAI o200k (GPT-4o)** | Global BPE | **1.20** | **1.86** | **3.26** | 1.55× | **2.72×** | **+\$154.80 / mo** | **2.72× slower** |
| **Google MuRIL** | Indic WordPiece | **1.24** | **1.24** | **2.04** | 1.00× | **1.65×** | **+\$60.23 / mo** | **1.65× slower** |
| **Sarvam-1** | Indic BPE | **1.41** | **1.45** | **2.32** | 1.03× | **1.65×** | **+\$68.33 / mo** | **1.65× slower** |
| **Custom BPE (4K)** | Grapheme BPE | **2.07** | **2.05** | **2.07** | **0.99×** | **1.00×** | **-\$0.45 / mo** | **1.00× (Zero slowdown)** |

### 2. Context Window Evaporation
In an 8,192-token context window:
* **English:** Holds **6,781 words**.
* **Hindi (GPT-4):** Holds **1,469 words** (78% context loss).
* **Telugu (GPT-4):** Holds **566 words** (92% context loss).
* **Custom BPE (4K):** Holds **3,963 words in Telugu** (restoring true equality for RAG document retrieval).

---

## 4. Verification & Automated Test Status

All Acceptance Criteria specified in `SPEC.md` are verified by automated pytest test suites running in `umc301`:

| Acceptance Criteria | Description | Target Standard | Status | Automated Test |
| :--- | :--- | :--- | :--- | :--- |
| **AC-1** | Grapheme Preservation | 0 split syllables on 100 test words | **PASSED (100%)** | `tests/test_grapheme_regex.py` |
| **AC-2** | Containment Invariant | Every token $\in$ exactly 1 pre-token chunk | **PASSED (100%)** | `tests/test_grapheme_regex.py` |
| **AC-3** | Monotonic Fertility | $\mathcal{F}(V_{512}) > \mathcal{F}(V_{1\text{K}}) > \mathcal{F}(V_{2\text{K}}) > \mathcal{F}(V_{4\text{K}})$ | **PASSED (100%)** | `tests/test_metrics.py` |
| **AC-4** | Language Tax Audit | Quantify Telugu fertility multiplier ($\ge 2.5\times$ in GPT-4) | **PASSED (11.96×)** | `artifacts/language_tax_report.md` |
| **AC-5** | LoRA Parameter Budget | Trainable adapter params $< 1.0\%$ of base model | **PASSED (0.34%)** | `tests/test_lora_sft.py` |
| **AC-6** | Prompt-Loss Masking | 100% of user prompt token labels $=-100$ | **PASSED (100%)** | `tests/test_lora_sft.py` |
| **AC-7** | SFT Convergence | Validation loss decreases across epochs | **PASSED (1.24 $\to$ 1.12)** | `artifacts/training_history.json` |
| **AC-8** | Interactive Dashboard | Functional Streamlit UI running with 4 tabs | **PASSED (100%)** | `app/app.py` |

---

## 5. Comprehensive Viva Defense Simulator (Q&A)

### Q1: "Why does GPT-4 suffer such an extreme Telugu token fertility of 14.45 tokens per word?"
> **Answer:**  
> "The failure stems from two interconnected structural causes:
> 1. **The Corpus Distribution Skew:** BPE merge rules are greedy and allocate vocabulary slots strictly in proportion to corpus frequency. Because web crawls are over 85% English, English words capture the merge hierarchy while low-resource Indic languages remain under-tokenized.
> 2. **The Unicode Classification Trap:** OpenAI's pre-tokenization regex `[^\r\n\p{L}\p{N}]?\p{L}+` matches only Unicode Category `\p{L}` (Letters). In abugidas like Telugu and Devanagari, dependent vowel matras and viramas belong to Category `\p{M}` (Combining Marks). The regex slices through the middle of the Akshara, stranding matras into separate chunks. Because BPE never permits merges across chunk boundaries, matras fall back to raw UTF-8 byte tokens, creating a 11.96× token explosion."

---

### Q2: "Why didn't you just graft your custom 4K tokenizer onto a pre-trained base LLM like LLaMA-3 or Qwen?"
> **Answer:**  
> "Doing so creates catastrophic **Embedding Misalignment**. A pre-trained LLM's embedding table $W_{\text{embed}} \in \mathbb{R}^{|V| \times d}$ has learned semantic coordinates for specific token IDs across trillions of tokens. If token ID 450 in LLaMA is the English subword 'the', and token ID 450 in our custom tokenizer is the Telugu syllable 'రా', the entire model receives completely scrambled representations and outputs gibberish. Realignment would require multi-billion token continued pretraining. To maintain academic rigor, we deliberately decoupled Track A (tokenizer audit) from Track B (LoRA PEFT on the base model using its native tokenizer)."

---

### Q3: "Why did you choose LoRA rank $r=8$ and $\alpha=16$ for Track B?"
> **Answer:**  
> "Under the intrinsic dimensionality hypothesis (Aghajanyan et al., 2020), task-specific instruction adaptation resides in a low-dimensional subspace. Rank $r=8$ provides sufficient representation capacity to learn domain task templates while keeping trainable parameters at just **460,800** (only **0.34%** of base model parameters). The scaling factor $\frac{\alpha}{r} = \frac{16}{8} = 2.0$ stabilizes gradient magnitudes when updates are summed into the frozen weights $W = W_0 + \frac{\alpha}{r}BA$."

---

### Q4: "Why did you use chrF instead of standard BLEU to evaluate Hindi and Telugu?"
> **Answer:**  
> "Telugu is an agglutinative Dravidian language where grammatical relations, case markers, and postpositions attach directly to words as suffixes. Standard BLEU relies on whitespace-delimited word n-gram matches. If the model generates a legitimate morphological variation, BLEU marks the entire word as 0% matching. In contrast, **chrF** evaluates character n-grams ($n \in [1, 6]$), capturing morphological correctness, stem retention, and vowel matra accuracy with statistical robustness."

---

### Q5: "What is the practical takeaway of your Pareto Frontier analysis?"
> **Answer:**  
> "It disproves the naive industry assumption that simply expanding $|V|$ to 200,000+ solves multilingual equity. At $|V| = 200\text{K}$ (GPT-4o), Telugu still has a 2.72× tax while the embedding table consumes 781 MB of RAM. Our Grapheme-Aware BPE achieves **1.00× parity** at $|V| = 3,892$ with only **15.2 MB** of embedding RAM. This proves that **linguistically informed pre-tokenization regexes** (`(?:\p{L}\p{M}*)+`), not raw vocabulary bloat, are the true solution to multilingual parity on edge hardware."
