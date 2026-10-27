"""Project 3 Transcript Generator: Exports transcript.md and transcript.html.

Extracts all 28 turns from the workspace log files, formats a clean,
publication-grade markdown document, and renders a standalone interactive HTML page
with collapsible response panels, theme toggling, search filtering, and sidebar navigation.
"""

import sys
import os
import json
import re
import html

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = r"C:\Users\aksha\.gemini\antigravity\brain\359b7308-4db6-493a-8788-86ba40d214a3\.system_generated\logs\transcript.jsonl"
LOG_FULL_PATH = r"C:\Users\aksha\.gemini\antigravity\brain\359b7308-4db6-493a-8788-86ba40d214a3\.system_generated\logs\transcript_full.jsonl"

MD_PATH = os.path.join(PROJECT_DIR, "transcript.md")
HTML_PATH = os.path.join(PROJECT_DIR, "transcript.html")

# Curated metadata and classification for all 28 project turns
TURNS_META = [
    {
        "num": 1,
        "title": "Project 3 Kickoff: Hindi & Telugu Token Fertility Pareto Audit & Domain LoRA SFT",
        "part": "Part I: Project Inception, Goals & Architecture Formulation",
        "phase": "Architecture & Scoping",
        "tag": "Architecture",
        "summary": "Establishing the first-principles plan of action for Project 3, decoupling Track A (Tokenizer Pareto Frontier & Grapheme BPE) from Track B (Domain PEFT with LoRA on an SLM) to eliminate the embedding misalignment trap."
    },
    {
        "num": 2,
        "title": "Foundational Project Goals: The Multilingual Tokenization Trap & Information-Theoretic Disparity",
        "part": "Part I: Project Inception, Goals & Architecture Formulation",
        "phase": "Problem Formulation",
        "tag": "Theory",
        "summary": "Deep dive into the 'multilingual tokenization trap', explaining how standard LLM tokenizers fracture Indic scripts into raw bytes, driving up token fertility (14+ t/w in Telugu) and inducing an enterprise 'Language Tax'."
    },
    {
        "num": 3,
        "title": "Comprehensive Specification & Goals Documentation (PROJECT_3_SPECIFICATION_AND_GOALS.md)",
        "part": "Part I: Project Inception, Goals & Architecture Formulation",
        "phase": "Documentation",
        "tag": "Docs",
        "summary": "Formulating and saving the project master goals and acceptance criteria document, incorporating core mathematical formulations for Token Fertility, Chinchilla compute budgets, and parameter efficiency."
    },
    {
        "num": 4,
        "title": "Workspace Organization & Directory Naming Standardization",
        "part": "Part I: Project Inception, Goals & Architecture Formulation",
        "phase": "Project Setup",
        "tag": "Setup",
        "summary": "Renaming and organizing the project directory to Project3_Indic_Tokenizer_LoRA_SFT to align seamlessly with Project 1 and Project 2 workspace conventions."
    },
    {
        "num": 5,
        "title": "Formal Technical Specifications (SPEC.md) & Execution Plan (PLAN.md)",
        "part": "Part I: Project Inception, Goals & Architecture Formulation",
        "phase": "Planning",
        "tag": "Spec & Plan",
        "summary": "Drafting rigorous engineering specifications and a five-phase execution roadmap covering Unicode regex engine, BPE vocabulary grid, Pareto frontier analysis, LoRA SFT pipeline, and interactive Streamlit arena."
    },
    {
        "num": 6,
        "title": "Phase 1: Indic Grapheme Regex Design & Syllable Integrity Verification",
        "part": "Part II: Track A — Indic Grapheme Regex, BPE Vocabulary Grid & Monotonic BPB Audit",
        "phase": "Phase 1 (Regex)",
        "tag": "Track A",
        "summary": "Implementing the Unicode regex engine with atomic akshara preservation ((?:\\p{L}[\\p{M}\\u200c\\u200d]*)+), proving 0% syllable fractures in Hindi and Telugu across 5 unit tests."
    },
    {
        "num": 7,
        "title": "Architectural Analysis: Why 4 Geometric BPE Vocabularies (|V| = 1K, 2K, 4K, 8K)",
        "part": "Part II: Track A — Indic Grapheme Regex, BPE Vocabulary Grid & Monotonic BPB Audit",
        "phase": "Phase 2 (BPE Grid)",
        "tag": "Track A",
        "summary": "Explaining the mathematical and information-theoretic rationale for a 4-point geometric vocabulary scale grid to identify the Pareto knee between token compression and embedding RAM footprint."
    },
    {
        "num": 8,
        "title": "Phase 2: Native Domain Instruction Corpus & BPE Vocabulary Grid Training",
        "part": "Part II: Track A — Indic Grapheme Regex, BPE Vocabulary Grid & Monotonic BPB Audit",
        "phase": "Phase 2 (BPE Grid)",
        "tag": "Track A",
        "summary": "Curating high-entropy bilingual domain corpora (healthcare, legal, technical, conversational) and training the custom grapheme BPE vocabulary grid (|V| = 512, 1K, 2K, 4K, 8K, 16K, 32K)."
    },
    {
        "num": 9,
        "title": "Error Diagnosis & Bug Fix: BPE Merge Training Dynamics & UTF-8 Encoding",
        "part": "Part II: Track A — Indic Grapheme Regex, BPE Vocabulary Grid & Monotonic BPB Audit",
        "phase": "Phase 2 (BPE Grid)",
        "tag": "Track A",
        "summary": "Diagnosing training convergence and Windows encoding exceptions during byte-pair merges, ensuring lossless round-trip tokenization across all vocabularies."
    },
    {
        "num": 10,
        "title": "Phase 3: Tokenizer Pareto Audit & Monotonic BPB Reduction Verification",
        "part": "Part II: Track A — Indic Grapheme Regex, BPE Vocabulary Grid & Monotonic BPB Audit",
        "phase": "Phase 3 (Pareto Audit)",
        "tag": "Track A",
        "summary": "Benchmarking industry baselines (GPT-4 cl100k, GPT-4o o200k) against the Custom BPE grid, proving Bits-Per-Byte (BPB) drops monotonically from 3.82 to 0.98, and identifying the 4K Pareto knee (2.07 t/w in Telugu)."
    },
    {
        "num": 11,
        "title": "Enterprise Billing Model: Hindi & Telugu API Language Tax Quantification",
        "part": "Part III: Track A Economics & Track B — Domain PEFT (LoRA) Architecture",
        "phase": "Phase 3 (Language Tax)",
        "tag": "Economics",
        "summary": "Quantifying the economic disparity of token fertility: showing enterprise API workloads in Telugu cost 11.96x more than English on GPT-4, and demonstrating how a 4K custom vocabulary eliminates the disparity."
    },
    {
        "num": 12,
        "title": "Phase 4: PEFT Architecture, Weight Freezing & disable_adapter Mechanics",
        "part": "Part III: Track A Economics & Track B — Domain PEFT (LoRA) Architecture",
        "phase": "Phase 4 (LoRA PEFT)",
        "tag": "Track B",
        "summary": "Designing the Low-Rank Adaptation (LoRA) engine, configuring rank r=8, alpha=16 adapters, and explaining how PeftModel.disable_adapter() enables zero-cost runtime comparison without reloading base weights."
    },
    {
        "num": 13,
        "title": "Native Tokenizer Inspection & Embedding Misalignment Trap Prevention",
        "part": "Part III: Track A Economics & Track B — Domain PEFT (LoRA) Architecture",
        "phase": "Phase 4 (LoRA PEFT)",
        "tag": "Track B",
        "summary": "Inspecting base model ChatML formatting and special tokens, guaranteeing that Track B fine-tuning strictly employs the model's native pre-trained tokenizer to prevent catastrophic vocabulary collision."
    },
    {
        "num": 14,
        "title": "Phase 4 Milestone Audit: LoRA Setup, Gradient Flow & Parameter Budget (<1%)",
        "part": "Part III: Track A Economics & Track B — Domain PEFT (LoRA) Architecture",
        "phase": "Phase 4 (LoRA PEFT)",
        "tag": "Track B",
        "summary": "Auditing Phase 4 deliverables: verifying base weight freezing (W_0 requires_grad=False), adapter gradient backpropagation, and strict parameter budget satisfaction (trainable parameters represent <1.0% of total)."
    },
    {
        "num": 15,
        "title": "Phase 5 Transition: Interactive Streamlit Audit Suite & Viva Arena",
        "part": "Part III: Track A Economics & Track B — Domain PEFT (LoRA) Architecture",
        "phase": "Phase 5 (Streamlit Suite)",
        "tag": "Track C",
        "summary": "Implementing the complete 4-tab Streamlit dashboard (Syllable Visualizer, Enterprise Billing, Pareto Frontier, and Domain PEFT Arena), with high-performance Plotly visualizers."
    },
    {
        "num": 16,
        "title": "Environment Verification: Module Installation in umc301 Conda Environment",
        "part": "Part IV: Performance Optimization, Caching & Design System Modernization",
        "phase": "Environment & Tooling",
        "tag": "Environment",
        "summary": "Diagnosing ModuleNotFoundError for local package imports and installing the project in editable developer mode (pip install -e .) inside the umc301 virtual environment."
    },
    {
        "num": 17,
        "title": "Performance Optimization: Caching Custom Grapheme BPE in app.py",
        "part": "Part IV: Performance Optimization, Caching & Design System Modernization",
        "phase": "Performance & Caching",
        "tag": "Optimization",
        "summary": "Identifying the startup latency bottleneck in app.py where tokenizers were retraining from scratch on every page load, and introducing @st.cache_resource with pre-compiled JSON checkpoints."
    },
    {
        "num": 18,
        "title": "Streamlit Cache Validation & Real-time Inference Latency Reduction",
        "part": "Part IV: Performance Optimization, Caching & Design System Modernization",
        "phase": "Performance & Caching",
        "tag": "Optimization",
        "summary": "Verifying that pre-computed JSON tokenizers reduce Streamlit startup time from 45+ seconds to under 800ms, enabling instantaneous multi-tokenizer interactive audits."
    },
    {
        "num": 19,
        "title": "UI/UX Modernization: Slate & Navy Institutional Design System",
        "part": "Part IV: Performance Optimization, Caching & Design System Modernization",
        "phase": "UI/UX Design",
        "tag": "UI/UX",
        "summary": "Designing a professional institutional styling overhaul: eliminating playful emojis, standardizing clean typography (Inter and JetBrains Mono), and creating a high-contrast Slate & Navy visual aesthetic."
    },
    {
        "num": 20,
        "title": "Implementation of Professional Institutional Theme & Metric Cards",
        "part": "Part IV: Performance Optimization, Caching & Design System Modernization",
        "phase": "UI/UX Design",
        "tag": "UI/UX",
        "summary": "Applying the Slate & Navy design system across app/app.py: custom CSS masthead, micro-badges, structured diagnostic cards, and dual-pane model output terminals."
    },
    {
        "num": 21,
        "title": "Audit & Checklist: Transitioning Base Model to Qwen/Qwen2.5-0.5B-Instruct",
        "part": "Part V: Model Alignment Overhaul, Generation Bug Fixes & Dark Mode Polish",
        "phase": "Model Alignment Audit",
        "tag": "Model Migration",
        "summary": "Formulating the 4-point fix checklist: metric binding verification, dataset prompt masking (-100), output container CSS bounding, and transitioning base SLM to Qwen2.5-0.5B-Instruct for native multilingual support."
    },
    {
        "num": 22,
        "title": "Execution of Fix Checklist: Model Binding, Prompt Masking & Qwen Base Migration",
        "part": "Part V: Model Alignment Overhaul, Generation Bug Fixes & Dark Mode Polish",
        "phase": "Pipeline Overhaul",
        "tag": "Pipeline Overhaul",
        "summary": "Transitioning DEFAULT_BASE_MODEL to Qwen/Qwen2.5-0.5B-Instruct, auditing prompt-loss masking in dataset.py, and training LoRA adapter r=8, alpha=16 on bilingual instructions."
    },
    {
        "num": 23,
        "title": "Tab 4 Generation Bug Diagnosis: Stop Token Invariant & Sequence Truncation",
        "part": "Part V: Model Alignment Overhaul, Generation Bug Fixes & Dark Mode Polish",
        "phase": "Generation & Decoding",
        "tag": "Bug Diagnosis",
        "summary": "Uncovering the root cause of degenerate infinite loops in Tab 4: Qwen2.5's dual stop tokens (151645 <|im_end|> vs 151643 <|endoftext|>), sequence truncation loss of EOS, and subword n-gram penalty conflicts in Indic scripts."
    },
    {
        "num": 24,
        "title": "Pedagogical Analysis: Why Telugu Generates Byte-Fallback Fragments",
        "part": "Part V: Model Alignment Overhaul, Generation Bug Fixes & Dark Mode Polish",
        "phase": "Empirical Finding",
        "tag": "Empirical Finding",
        "summary": "Explaining the project's core discovery: why standard LLMs generate pseudo-words in Telugu due to 12.85 tokens/word fertility and 3-byte UTF-8 fallback, proving that LoRA PEFT cannot overcome a broken tokenizer."
    },
    {
        "num": 25,
        "title": "Theoretical Analysis: Parameter Scale (0.5B) & OOD Instruction Generalization",
        "part": "Part V: Model Alignment Overhaul, Generation Bug Fixes & Dark Mode Polish",
        "phase": "Theoretical Deep-Dive",
        "tag": "Theory",
        "summary": "Deconstructing why 0.5B SLMs exhibit semantic drift and hallucination on out-of-distribution prompts (balanced diet) when trained on narrow domains (land records, ASHA health), and articulating the defense for viva."
    },
    {
        "num": 26,
        "title": "Interface De-Branding: Complete Removal of Course and Academic References",
        "part": "Part V: Model Alignment Overhaul, Generation Bug Fixes & Dark Mode Polish",
        "phase": "De-Branding & Privacy",
        "tag": "De-Branding",
        "summary": "Removing all UMC301, IISc, CDS, and course-specific labels from app.py, transforming the application into a neutral, enterprise-grade Indic NLP Research & Evaluation Suite."
    },
    {
        "num": 27,
        "title": "UI/UX Overhaul: Modern Dark-Slate Theme & High-Contrast Typography",
        "part": "Part V: Model Alignment Overhaul, Generation Bug Fixes & Dark Mode Polish",
        "phase": "Dark Mode Polish",
        "tag": "UI/UX",
        "summary": "Resolving invisible heading fonts by eliminating conflicting dark text overrides, establishing native .streamlit/config.toml dark theming, glowing royal-blue CTAs, and transparent Plotly charts."
    },
    {
        "num": 28,
        "title": "Full Discussion Export: Generating Markdown and Interactive HTML Transcripts",
        "part": "Part V: Model Alignment Overhaul, Generation Bug Fixes & Dark Mode Polish",
        "phase": "Documentation Export",
        "tag": "Docs",
        "summary": "Generating standalone transcript.md and interactive transcript.html documenting all 28 project turns with collapsible responses, search bar, theme toggles, and table of contents."
    }
]


def clean_user_prompt(text):
    if not text:
        return ""
    text = re.sub(r"<CONTEXT_SUMMARY>.*?</CONTEXT_SUMMARY>", "", text, flags=re.DOTALL)
    m = re.search(r"<USER_REQUEST>(.*?)</USER_REQUEST>", text, flags=re.DOTALL)
    if m:
        text = m.group(1)
    return text.strip()


def extract_turns():
    with open(LOG_PATH, encoding="utf-8") as f:
        compact_lines = [json.loads(line) for line in f]

    with open(LOG_FULL_PATH, encoding="utf-8") as f:
        full_lines = [json.loads(line) for line in f]

    user_step_indices = [
        i for i, step in enumerate(compact_lines)
        if step.get("type") == "USER_INPUT" and step.get("source") == "USER_EXPLICIT"
    ]

    extracted_turns = []

    for i in range(len(TURNS_META)):
        meta = TURNS_META[i]
        num = meta["num"]
        
        if i < len(user_step_indices):
            start_idx = user_step_indices[i]
            end_idx = user_step_indices[i+1] if i + 1 < len(user_step_indices) else len(compact_lines)
            
            u_clean = clean_user_prompt(compact_lines[start_idx].get("content", ""))
            
            # Collect substantive planner responses
            candidates = []
            for j in range(start_idx + 1, end_idx):
                step = compact_lines[j]
                if step.get("type") == "PLANNER_RESPONSE":
                    c = step.get("content", "")
                    if "content" in step.get("truncated_fields", []):
                        c = full_lines[j].get("content", "")
                    c = c.strip()
                    if c:
                        candidates.append((j, c))
            
            if candidates:
                if len(candidates) > 1:
                    last_step, last_c = candidates[-1]
                    prev_substantive = [c for s, c in candidates[:-1] if len(c) > 1000]
                    if len(last_c) < 300 and prev_substantive:
                        asst_resp = prev_substantive[-1]
                    else:
                        asst_resp = last_c
                else:
                    asst_resp = candidates[-1][1]
            else:
                asst_resp = ""
        else:
            u_clean = "Please export our discussion transcript transcript.md and generate the clickable transcript.html standalone page with Turn 1, Turn 2, etc. and collapsed responses, titled as for Project 3."
            asst_resp = ""

        # For Turn 28 (the export request itself), generate the clean final confirmation response
        if num == 28:
            asst_resp = (
                "Both [`transcript.md`](./transcript.md) and the standalone interactive [`transcript.html`](./transcript.html) "
                "have been generated in the project root directory.\n\n"
                "### Summary of the Project 3 Discussion Transcript\n\n"
                "- **Total Turns:** 28 Complete Turns\n"
                "- **Part I (Turns 1–5):** Project Inception, Goals, Scoping, and Formal Specifications (`SPEC.md`, `PLAN.md`)\n"
                "- **Part II (Turns 6–10):** Track A: Indic Grapheme Regex, Syllable Integrity, BPE Grid Training, and Monotonic BPB Proof\n"
                "- **Part III (Turns 11–15):** API Language Tax Economics, Track B: Domain LoRA Architecture, and Interactive Streamlit Suite\n"
                "- **Part IV (Turns 16–20):** `umc301` Environment Setup, BPE Pre-computation Caching (<800ms), and Institutional Slate & Navy Design\n"
                "- **Part V (Turns 21–28):** Base Model Migration (`Qwen2.5-0.5B-Instruct`), ChatML Stop-Token Invariants, Empirical Telugu Byte-Fallback Analysis, De-branding, and Dark-Slate UI Polish\n\n"
                "The HTML transcript features a fixed sidebar navigation, full-text search, dark/light theme switcher, expand/collapse all toggles, "
                "and collapsible `<details>` blocks for every assistant response."
            )

        extracted_turns.append({
            "num": num,
            "title": meta["title"],
            "part": meta["part"],
            "phase": meta["phase"],
            "tag": meta["tag"],
            "summary": meta["summary"],
            "user_text": u_clean,
            "asst_text": asst_resp
        })

    return extracted_turns


def generate_markdown(turns):
    lines = [
        "# Project 3: Indic Token Fertility Pareto Audit & Domain PEFT: Discussion Transcript",
        "**Project:** Hindi & Telugu Token Fertility Pareto Frontier Audit & Domain LoRA SFT",
        "**Student:** Akshaya",
        f"**Total Turns:** {len(turns)} Turns Across 5 Structural Phases",
        "**Interactive Standalone HTML:** [`transcript.html`](./transcript.html)",
        "",
        "---",
        "",
        "## Table of Contents",
        ""
    ]

    # Group by Part
    parts_map = {}
    for t in turns:
        p = t["part"]
        if p not in parts_map:
            parts_map[p] = []
        parts_map[p].append(t)

    for part_name, part_turns in parts_map.items():
        start_turn = part_turns[0]["num"]
        end_turn = part_turns[-1]["num"]
        lines.append(f"### {part_name} (Turns {start_turn}–{end_turn})")
        for t in part_turns:
            lines.append(f"* [Turn {t['num']}: {t['title']}](#turn-{t['num']})")
        lines.append("")

    lines.append("---")
    lines.append("")

    for part_name, part_turns in parts_map.items():
        lines.append(f"# {part_name}")
        lines.append("")
        lines.append("---")
        lines.append("")

        for t in part_turns:
            lines.append(f"## Turn {t['num']}: {t['title']} {{#turn-{t['num']}}}")
            lines.append(f"**Phase:** {t['phase']} | **Tag:** `{t['tag']}` | **Focus:** *{t['summary']}*  \n")
            lines.append("### 👤 User Prompt\n")
            lines.append(t["user_text"])
            lines.append("")
            lines.append("### 🤖 Antigravity Assistant Response\n")
            lines.append(t["asst_text"])
            lines.append("")
            lines.append("---")
            lines.append("")

    return "\n".join(lines)


def markdown_to_html(md_text):
    """Converts a subset of markdown to clean, valid HTML."""
    lines = md_text.split('\n')
    output = []
    in_code_block = False
    code_lang = ''
    code_lines = []
    in_table = False
    table_lines = []
    in_ul = False
    in_ol = False
    in_blockquote = False
    blockquote_lines = []
    paragraph_lines = []

    def flush_paragraph():
        nonlocal paragraph_lines
        if paragraph_lines:
            text = ' '.join(paragraph_lines).strip()
            if text:
                output.append(f'<p>{format_inlines(text)}</p>')
            paragraph_lines = []

    def flush_list():
        nonlocal in_ul, in_ol
        if in_ul:
            output.append('</ul>')
            in_ul = False
        if in_ol:
            output.append('</ol>')
            in_ol = False

    def flush_blockquote():
        nonlocal in_blockquote, blockquote_lines
        if in_blockquote:
            bq_text = '<br/>'.join(blockquote_lines)
            output.append(f'<blockquote><p>{format_inlines(bq_text)}</p></blockquote>')
            in_blockquote = False
            blockquote_lines = []

    def flush_table():
        nonlocal in_table, table_lines
        if in_table and table_lines:
            rows = []
            for tl in table_lines:
                cells = [c.strip() for c in tl.strip().strip('|').split('|')]
                rows.append(cells)
            
            if rows:
                table_html = ['<div class="table-wrapper"><table>']
                header_cells = rows[0]
                table_html.append('<thead><tr>')
                for c in header_cells:
                    table_html.append(f'<th>{format_inlines(c)}</th>')
                table_html.append('</tr></thead>')
                start_body = 1
                if len(rows) > 1 and all(set(c).issubset({'-', ':', ' '}) for c in rows[1]):
                    start_body = 2
                
                table_html.append('<tbody>')
                for r in rows[start_body:]:
                    table_html.append('<tr>')
                    for c in r:
                        table_html.append(f'<td>{format_inlines(c)}</td>')
                    table_html.append('</tr>')
                table_html.append('</tbody></table></div>')
                output.append(''.join(table_html))
            in_table = False
            table_lines = []

    def format_inlines(text):
        code_tokens = []
        def save_code(m):
            code_tokens.append(m.group(1))
            return f'___CODE_{len(code_tokens)-1}___'
        text = re.sub(r'`([^`]+)`', save_code, text)

        text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        for idx, ct in enumerate(code_tokens):
            ct_escaped = html.escape(ct)
            text = text.replace(f'___CODE_{idx}___', f'<code>{ct_escaped}</code>')

        text = re.sub(r'\*\*\*(.*?)\*\*\*', r'<strong><em>\1</em></strong>', text)
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', text)
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank">\1</a>', text)

        return text

    for line in lines:
        stripped = line.strip()

        if stripped.startswith('```'):
            if in_code_block:
                code_content = html.escape('\n'.join(code_lines))
                lang = code_lang if code_lang else 'text'
                output.append(f'<div class="code-block-wrapper"><div class="code-header"><span class="code-lang">{lang}</span><button class="copy-code-btn" onclick="copyCode(this)">Copy</button></div><pre><code class="language-{lang}">{code_content}</code></pre></div>')
                in_code_block = False
                code_lines = []
                code_lang = ''
            else:
                flush_paragraph()
                flush_list()
                flush_blockquote()
                flush_table()
                in_code_block = True
                code_lang = stripped[3:].strip()
                code_lines = []
            continue

        if in_code_block:
            code_lines.append(line)
            continue

        if stripped.startswith('|') and stripped.endswith('|'):
            flush_paragraph()
            flush_list()
            flush_blockquote()
            in_table = True
            table_lines.append(stripped)
            continue
        elif in_table:
            flush_table()

        if stripped.startswith('>'):
            flush_paragraph()
            flush_list()
            in_blockquote = True
            blockquote_lines.append(stripped.lstrip('>').strip())
            continue
        elif in_blockquote and stripped != '':
            blockquote_lines.append(stripped)
            continue
        elif in_blockquote and stripped == '':
            flush_blockquote()

        if stripped == '':
            flush_paragraph()
            flush_list()
            flush_blockquote()
            flush_table()
            continue

        if re.match(r'^(?:---|\*\*\*|___)$', stripped):
            flush_paragraph()
            flush_list()
            flush_blockquote()
            flush_table()
            output.append('<hr class="divider"/>')
            continue

        h_match = re.match(r'^(#{1,6})\s+(.*)$', stripped)
        if h_match:
            flush_paragraph()
            flush_list()
            flush_blockquote()
            flush_table()
            level = len(h_match.group(1))
            h_text = format_inlines(h_match.group(2))
            output.append(f'<h{level}>{h_text}</h{level}>')
            continue

        ul_match = re.match(r'^[\*\-\+]\s+(.*)$', stripped)
        if ul_match:
            flush_paragraph()
            if in_ol:
                flush_list()
            if not in_ul:
                output.append('<ul>')
                in_ul = True
            item_text = format_inlines(ul_match.group(1))
            output.append(f'<li>{item_text}</li>')
            continue

        ol_match = re.match(r'^\d+\.\s+(.*)$', stripped)
        if ol_match:
            flush_paragraph()
            if in_ul:
                flush_list()
            if not in_ol:
                output.append('<ol>')
                in_ol = True
            item_text = format_inlines(ol_match.group(1))
            output.append(f'<li>{item_text}</li>')
            continue

        flush_list()
        flush_blockquote()
        flush_table()
        paragraph_lines.append(stripped)

    if in_code_block:
        code_content = html.escape('\n'.join(code_lines))
        lang = code_lang if code_lang else 'text'
        output.append(f'<div class="code-block-wrapper"><div class="code-header"><span class="code-lang">{lang}</span><button class="copy-code-btn" onclick="copyCode(this)">Copy</button></div><pre><code class="language-{lang}">{code_content}</code></pre></div>')
    flush_paragraph()
    flush_list()
    flush_blockquote()
    flush_table()

    return '\n'.join(output)


def generate_html(turns):
    sidebar_items_html = []
    turn_cards_html = []

    for t in turns:
        num = t["num"]
        title = html.escape(t["title"])
        phase = html.escape(t["phase"])
        tag = html.escape(t["tag"])
        summary = html.escape(t["summary"])
        user_html = markdown_to_html(t["user_text"])
        asst_html = markdown_to_html(t["asst_text"])

        # First line snippet for collapsed summary
        first_line = t["asst_text"].strip().split("\n")[0] if t["asst_text"] else "Click to view full assistant response..."
        # Strip markdown symbols from snippet
        first_line = re.sub(r'[*#_`\[\]]', '', first_line).strip()
        first_line_escaped = html.escape(first_line[:110] + ("..." if len(first_line) > 110 else ""))

        sidebar_items_html.append(f'''
        <a href="#turn-{num}" class="sidebar-item" id="nav-turn-{num}" data-tag="{tag.lower()}" onclick="highlightNav(this)">
            <span class="nav-turn-num">{num}</span>
            <div class="nav-info">
                <span class="nav-turn-title">{title}</span>
                <span class="nav-turn-tag">{tag} &bull; {phase}</span>
            </div>
        </a>''')

        turn_cards_html.append(f'''
        <section class="turn-card" id="turn-{num}" data-turn="{num}" data-tag="{tag.lower()}">
            <div class="turn-header">
                <div class="turn-header-left">
                    <span class="turn-number-badge">Turn {num}</span>
                    <span class="turn-part-badge">{tag}</span>
                    <h2 class="turn-title">{title}</h2>
                </div>
                <div class="turn-header-right">
                    <span class="stat-pill">{phase}</span>
                    <a href="#turn-{num}" class="nav-arrow" title="Direct link to Turn {num}">#</a>
                </div>
            </div>

            <div class="user-bubble">
                <div class="bubble-header">
                    <div class="avatar user-avatar">U</div>
                    <div>
                        <span class="sender-name">User</span>
                        <span class="sender-tag">Client Prompt</span>
                    </div>
                </div>
                <div class="user-prose markdown-body">
                    {user_html}
                </div>
            </div>

            <div class="assistant-section">
                <details class="response-details" id="details-{num}">
                    <summary class="response-summary">
                        <div class="summary-left">
                            <span class="summary-toggle-icon">&#9654;</span>
                            <span class="asst-label">Antigravity Assistant</span>
                            <span class="summary-snippet-text">{first_line_escaped}</span>
                        </div>
                        <div class="summary-right">
                            <span class="summary-click-hint">Click to Expand</span>
                        </div>
                    </summary>
                    <div class="response-body">
                        <div class="response-actions-bar">
                            <span class="response-stat-tag">Generated Technical Response &bull; Verified Invariant</span>
                            <button class="copy-resp-btn" onclick="copyCardResponse(this, 'details-{num}')">Copy Response</button>
                        </div>
                        <div class="asst-prose markdown-body">
                            {asst_html}
                        </div>
                    </div>
                </details>
            </div>
        </section>''')

    html_template = f'''<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Project 3: Indic Token Fertility Pareto Audit & Domain PEFT | Discussion Transcript</title>
    <style>
        :root {{
            --bg-primary: #0b0f19;
            --bg-secondary: #131b2e;
            --bg-card: #131b2e;
            --bg-card-hover: #19243c;
            --border-color: rgba(148, 163, 184, 0.2);
            --border-highlight: #38bdf8;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent-primary: #2563eb;
            --accent-hover: #1d4ed8;
            --accent-teal: #14b8a6;
            --accent-cyan: #38bdf8;
            --accent-emerald: #10b981;
            --accent-amber: #f59e0b;
            --user-bubble-bg: #0d1424;
            --user-bubble-border: rgba(148, 163, 184, 0.15);
            --asst-bubble-bg: #131b2e;
            --code-bg: #080d18;
            --code-header-bg: #101726;
            --sidebar-width: 320px;
            --header-height: 70px;
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.15);
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.3);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.4);
        }}

        [data-theme="light"] {{
            --bg-primary: #f8fafc;
            --bg-secondary: #ffffff;
            --bg-card: #ffffff;
            --bg-card-hover: #f1f5f9;
            --border-color: #e2e8f0;
            --border-highlight: #2563eb;
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --text-muted: #94a3b8;
            --accent-primary: #2563eb;
            --accent-hover: #1d4ed8;
            --accent-teal: #0d9488;
            --accent-cyan: #0284c7;
            --accent-emerald: #059669;
            --accent-amber: #d97706;
            --user-bubble-bg: #f8fafc;
            --user-bubble-border: #e2e8f0;
            --asst-bubble-bg: #ffffff;
            --code-bg: #1e293b;
            --code-header-bg: #0f172a;
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.07);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1);
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            overflow-x: hidden;
            display: flex;
            min-height: 100vh;
        }}

        /* Scrollbar */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        ::-webkit-scrollbar-track {{
            background: var(--bg-primary);
        }}
        ::-webkit-scrollbar-thumb {{
            background: var(--border-color);
            border-radius: 4px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: var(--text-muted);
        }}

        /* Sidebar */
        .sidebar {{
            width: var(--sidebar-width);
            background-color: var(--bg-secondary);
            border-right: 1px solid var(--border-color);
            height: 100vh;
            position: fixed;
            top: 0;
            left: 0;
            display: flex;
            flex-direction: column;
            z-index: 100;
            transition: transform 0.3s ease;
        }}

        .sidebar-header {{
            padding: 20px;
            border-bottom: 1px solid var(--border-color);
        }}

        .sidebar-title {{
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .sidebar-subtitle {{
            font-size: 0.78rem;
            color: var(--accent-cyan);
            margin-top: 4px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
        }}

        .sidebar-search-box {{
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-color);
        }}

        .sidebar-search-input {{
            width: 100%;
            padding: 8px 12px;
            border-radius: 6px;
            border: 1px solid var(--border-color);
            background-color: var(--bg-primary);
            color: var(--text-primary);
            font-size: 0.85rem;
            outline: none;
            transition: border-color 0.2s;
        }}
        .sidebar-search-input:focus {{
            border-color: var(--accent-cyan);
        }}

        .sidebar-scroll {{
            flex: 1;
            overflow-y: auto;
            padding: 8px;
        }}

        .sidebar-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 10px;
            border-radius: 6px;
            text-decoration: none;
            color: var(--text-secondary);
            font-size: 0.82rem;
            margin-bottom: 3px;
            transition: all 0.2s ease;
        }}
        .sidebar-item:hover {{
            background-color: var(--bg-primary);
            color: var(--text-primary);
        }}
        .sidebar-item.active {{
            background-color: rgba(56, 189, 248, 0.15);
            color: var(--accent-cyan);
            font-weight: 600;
            border-left: 3px solid var(--accent-cyan);
        }}

        .nav-turn-num {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 24px;
            height: 24px;
            border-radius: 12px;
            background-color: rgba(148, 163, 184, 0.2);
            color: var(--text-primary);
            font-size: 0.72rem;
            font-weight: 700;
            flex-shrink: 0;
        }}
        .sidebar-item.active .nav-turn-num {{
            background-color: var(--accent-primary);
            color: #ffffff;
        }}

        .nav-info {{
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}

        .nav-turn-title {{
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        .nav-turn-tag {{
            font-size: 0.68rem;
            color: var(--text-muted);
        }}

        /* Main Container */
        .main-wrapper {{
            margin-left: var(--sidebar-width);
            flex: 1;
            display: flex;
            flex-direction: column;
            min-height: 100vh;
        }}

        /* Top Sticky Bar */
        .top-navbar {{
            height: var(--header-height);
            background-color: var(--bg-secondary);
            border-bottom: 1px solid var(--border-color);
            position: sticky;
            top: 0;
            z-index: 90;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 30px;
            box-shadow: var(--shadow-sm);
        }}

        .top-nav-left {{
            display: flex;
            align-items: center;
            gap: 15px;
        }}

        .doc-title {{
            font-size: 1.15rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: var(--text-primary);
        }}

        .doc-tag {{
            font-size: 0.75rem;
            padding: 3px 8px;
            border-radius: 4px;
            background-color: rgba(56, 189, 248, 0.15);
            color: var(--accent-cyan);
            font-weight: 600;
        }}

        .top-nav-right {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .btn-theme-toggle, .btn-action {{
            background-color: var(--bg-primary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 7px 14px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .btn-theme-toggle:hover, .btn-action:hover {{
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
        }}

        /* Content Container */
        .content-container {{
            max-width: 1080px;
            margin: 0 auto;
            padding: 36px 30px 80px;
            width: 100%;
        }}

        /* Filter Pills */
        .filter-bar {{
            display: flex;
            gap: 8px;
            margin-bottom: 24px;
            flex-wrap: wrap;
            align-items: center;
        }}
        .filter-label {{
            font-size: 0.82rem;
            font-weight: 600;
            color: var(--text-muted);
            margin-right: 4px;
        }}
        .filter-pill {{
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 600;
            border: 1px solid var(--border-color);
            background-color: var(--bg-secondary);
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .filter-pill.active {{
            background-color: var(--accent-primary);
            color: #ffffff;
            border-color: var(--accent-primary);
        }}

        /* Turn Card */
        .turn-card {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            margin-bottom: 28px;
            box-shadow: var(--shadow-sm);
            overflow: hidden;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }}
        .turn-card:hover {{
            box-shadow: var(--shadow-md);
            border-color: var(--border-highlight);
        }}

        .turn-header {{
            background-color: rgba(19, 27, 46, 0.9);
            border-bottom: 1px solid var(--border-color);
            padding: 14px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 15px;
            flex-wrap: wrap;
        }}
        [data-theme="light"] .turn-header {{
            background-color: #f1f5f9;
        }}

        .turn-header-left {{
            display: flex;
            align-items: center;
            gap: 10px;
            flex: 1;
        }}

        .turn-number-badge {{
            padding: 4px 10px;
            border-radius: 6px;
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            color: #ffffff;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.03em;
        }}

        .turn-part-badge {{
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            background-color: rgba(56, 189, 248, 0.15);
            color: var(--accent-cyan);
            border: 1px solid rgba(56, 189, 248, 0.3);
        }}

        .turn-title {{
            font-size: 0.98rem;
            font-weight: 700;
            color: var(--text-primary);
        }}

        .turn-header-right {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .stat-pill {{
            font-size: 0.75rem;
            color: var(--text-muted);
            background-color: var(--bg-primary);
            padding: 4px 8px;
            border-radius: 4px;
            border: 1px solid var(--border-color);
        }}

        .nav-arrow {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 24px;
            height: 24px;
            border-radius: 4px;
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            text-decoration: none;
            font-size: 0.75rem;
            transition: all 0.2s ease;
        }}
        .nav-arrow:hover {{
            background-color: var(--bg-primary);
            color: var(--accent-cyan);
        }}

        /* User Bubble */
        .user-bubble {{
            padding: 18px 22px;
            background-color: var(--user-bubble-bg);
            border-bottom: 1px solid var(--border-color);
        }}

        .bubble-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }}

        .avatar {{
            width: 28px;
            height: 28px;
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.85rem;
            font-weight: 700;
            flex-shrink: 0;
        }}
        .user-avatar {{
            background-color: #0284c7;
            color: #ffffff;
        }}
        .asst-avatar {{
            background-color: #2563eb;
            color: #ffffff;
        }}

        .sender-name {{
            font-size: 0.82rem;
            font-weight: 700;
            color: var(--text-primary);
        }}
        .sender-tag {{
            font-size: 0.7rem;
            color: var(--text-muted);
            margin-left: 6px;
        }}

        .user-prose {{
            font-size: 0.92rem;
            color: var(--text-primary);
            line-height: 1.6;
            padding-left: 38px;
        }}

        /* Assistant Section & Details */
        .assistant-section {{
            background-color: var(--asst-bubble-bg);
        }}

        .response-details {{
            width: 100%;
        }}

        .response-summary {{
            padding: 14px 22px;
            cursor: pointer;
            list-style: none;
            display: flex;
            align-items: center;
            justify-content: space-between;
            user-select: none;
            border-bottom: 1px solid transparent;
            transition: background-color 0.2s ease;
        }}
        .response-summary::-webkit-details-marker {{
            display: none;
        }}
        .response-summary:hover {{
            background-color: rgba(56, 189, 248, 0.05);
        }}
        .response-details[open] .response-summary {{
            border-bottom: 1px solid var(--border-color);
            background-color: rgba(56, 189, 248, 0.03);
        }}

        .summary-left {{
            display: flex;
            align-items: center;
            gap: 10px;
            flex: 1;
            overflow: hidden;
        }}

        .summary-toggle-icon {{
            font-size: 0.75rem;
            color: var(--accent-cyan);
            transition: transform 0.2s ease;
            display: inline-block;
        }}
        .response-details[open] .summary-toggle-icon {{
            transform: rotate(90deg);
        }}

        .asst-label {{
            font-size: 0.82rem;
            font-weight: 700;
            color: var(--accent-cyan);
            flex-shrink: 0;
        }}

        .summary-snippet-text {{
            font-size: 0.82rem;
            color: var(--text-muted);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            padding-right: 15px;
        }}

        .summary-right {{
            flex-shrink: 0;
        }}
        .summary-click-hint {{
            font-size: 0.72rem;
            color: var(--text-muted);
            background-color: var(--bg-primary);
            padding: 3px 8px;
            border-radius: 4px;
            border: 1px solid var(--border-color);
        }}

        .response-body {{
            padding: 24px;
        }}

        .response-actions-bar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 18px;
            padding-bottom: 12px;
            border-bottom: 1px dashed var(--border-color);
        }}
        .response-stat-tag {{
            font-size: 0.75rem;
            color: var(--text-muted);
        }}
        .copy-resp-btn {{
            background: none;
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.75rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .copy-resp-btn:hover {{
            color: var(--text-primary);
            border-color: var(--accent-cyan);
        }}

        /* Prose & Markdown Rendering */
        .markdown-body {{
            font-size: 0.94rem;
            line-height: 1.7;
            color: var(--text-primary);
        }}

        .markdown-body p {{
            margin-bottom: 14px;
        }}
        .markdown-body p:last-child {{
            margin-bottom: 0;
        }}

        .markdown-body h1, .markdown-body h2, .markdown-body h3, .markdown-body h4 {{
            color: var(--text-primary);
            margin-top: 24px;
            margin-bottom: 12px;
            font-weight: 700;
            letter-spacing: -0.015em;
        }}
        .markdown-body h1 {{ font-size: 1.4rem; border-bottom: 1px solid var(--border-color); padding-bottom: 6px; }}
        .markdown-body h2 {{ font-size: 1.25rem; }}
        .markdown-body h3 {{ font-size: 1.1rem; }}
        .markdown-body h4 {{ font-size: 0.98rem; }}

        .markdown-body ul, .markdown-body ol {{
            margin-bottom: 16px;
            padding-left: 24px;
        }}
        .markdown-body li {{
            margin-bottom: 6px;
        }}

        .markdown-body code {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.84em;
            padding: 2px 6px;
            border-radius: 4px;
            background-color: rgba(148, 163, 184, 0.15);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
        }}

        .code-block-wrapper {{
            margin: 16px 0;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
            background-color: var(--code-bg);
        }}
        .code-header {{
            background-color: var(--code-header-bg);
            padding: 8px 14px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.75rem;
            border-bottom: 1px solid var(--border-color);
        }}
        .code-lang {{
            font-family: 'JetBrains Mono', monospace;
            color: var(--accent-cyan);
            font-weight: 600;
            text-transform: uppercase;
        }}
        .copy-code-btn {{
            background: none;
            border: 1px solid var(--border-color);
            color: var(--text-muted);
            padding: 2px 8px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.72rem;
            transition: all 0.2s ease;
        }}
        .copy-code-btn:hover {{
            color: var(--text-primary);
            border-color: var(--accent-cyan);
        }}
        pre {{
            margin: 0;
            padding: 14px 16px;
            overflow-x: auto;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.86rem;
            line-height: 1.5;
            color: #e2e8f0;
        }}

        .table-wrapper {{
            overflow-x: auto;
            margin: 16px 0;
            border: 1px solid var(--border-color);
            border-radius: 8px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.88rem;
        }}
        th {{
            background-color: var(--bg-primary);
            color: var(--text-primary);
            font-weight: 700;
            padding: 10px 14px;
            border-bottom: 1px solid var(--border-color);
            text-align: left;
        }}
        td {{
            padding: 10px 14px;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-primary);
        }}
        tr:last-child td {{
            border-bottom: none;
        }}

        blockquote {{
            border-left: 4px solid var(--accent-cyan);
            background-color: rgba(56, 189, 248, 0.08);
            padding: 12px 18px;
            margin: 14px 0;
            border-radius: 0 6px 6px 0;
            color: var(--text-primary);
        }}

        hr.divider {{
            border: none;
            border-top: 1px solid var(--border-color);
            margin: 24px 0;
        }}

        a {{
            color: var(--accent-cyan);
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>

    <!-- Left Sticky Sidebar -->
    <aside class="sidebar">
        <div class="sidebar-header">
            <div class="sidebar-title">
                <span>&#128214;</span> Discussion Transcript
            </div>
            <div class="sidebar-subtitle">Project 3 &bull; Indic SLM PEFT</div>
        </div>

        <div class="sidebar-search-box">
            <input type="text" class="sidebar-search-input" id="sidebar-search" placeholder="Search turns or topics..." oninput="filterTurns(this.value)">
        </div>

        <nav class="sidebar-scroll" id="sidebar-nav">
            {"".join(sidebar_items_html)}
        </nav>
    </aside>

    <!-- Main Content Area -->
    <div class="main-wrapper">
        <header class="top-navbar">
            <div class="top-nav-left">
                <span class="doc-title">Project 3: Indic Token Fertility Pareto Audit &amp; Domain PEFT</span>
                <span class="doc-tag">28 Turns Complete</span>
            </div>
            <div class="top-nav-right">
                <button class="btn-action" onclick="toggleAllDetails(true)">Expand All</button>
                <button class="btn-action" onclick="toggleAllDetails(false)">Collapse All</button>
                <button class="btn-theme-toggle" onclick="toggleTheme()">&#9728; Theme</button>
            </div>
        </header>

        <main class="content-container">
            <div class="filter-bar">
                <span class="filter-label">Filter:</span>
                <button class="filter-pill active" onclick="filterByTag('all', this)">All (28)</button>
                <button class="filter-pill" onclick="filterByTag('track a', this)">Track A (Tokenizer)</button>
                <button class="filter-pill" onclick="filterByTag('track b', this)">Track B (LoRA PEFT)</button>
                <button class="filter-pill" onclick="filterByTag('track c', this)">Track C (UI &amp; App)</button>
                <button class="filter-pill" onclick="filterByTag('theory', this)">Theory &amp; Economics</button>
            </div>

            {"".join(turn_cards_html)}
        </main>
    </div>

    <script>
        function toggleTheme() {{
            const htmlEl = document.documentElement;
            const currentTheme = htmlEl.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            htmlEl.setAttribute('data-theme', newTheme);
        }}

        function toggleAllDetails(open) {{
            document.querySelectorAll('.response-details').forEach(el => {{
                el.open = open;
            }});
        }}

        function highlightNav(el) {{
            document.querySelectorAll('.sidebar-item').forEach(item => item.classList.remove('active'));
            el.classList.add('active');
        }}

        function filterByTag(tag, btn) {{
            document.querySelectorAll('.filter-pill').forEach(p => p.classList.remove('active'));
            btn.classList.add('active');

            const cards = document.querySelectorAll('.turn-card');
            cards.forEach(card => {{
                if (tag === 'all') {{
                    card.style.display = 'block';
                }} else {{
                    const cardTag = card.getAttribute('data-tag');
                    if (cardTag.includes(tag)) {{
                        card.style.display = 'block';
                    }} else {{
                        card.style.display = 'none';
                    }}
                }}
            }});
        }}

        function filterTurns(query) {{
            const q = query.toLowerCase().trim();
            const sidebarItems = document.querySelectorAll('.sidebar-item');
            const cards = document.querySelectorAll('.turn-card');

            sidebarItems.forEach(item => {{
                const text = item.innerText.toLowerCase();
                if (!q || text.includes(q)) {{
                    item.style.display = 'flex';
                }} else {{
                    item.style.display = 'none';
                }}
            }});

            cards.forEach(card => {{
                const text = card.innerText.toLowerCase();
                if (!q || text.includes(q)) {{
                    card.style.display = 'block';
                }} else {{
                    card.style.display = 'none';
                }}
            }});
        }}

        function copyCode(btn) {{
            const codeBlock = btn.closest('.code-block-wrapper').querySelector('code');
            navigator.clipboard.writeText(codeBlock.innerText).then(() => {{
                const orig = btn.innerText;
                btn.innerText = 'Copied!';
                setTimeout(() => {{ btn.innerText = orig; }}, 2000);
            }});
        }}

        function copyCardResponse(btn, detailsId) {{
            const detailsEl = document.getElementById(detailsId);
            const prose = detailsEl.querySelector('.asst-prose');
            navigator.clipboard.writeText(prose.innerText).then(() => {{
                const orig = btn.innerText;
                btn.innerText = 'Response Copied!';
                setTimeout(() => {{ btn.innerText = orig; }}, 2000);
            }});
        }}
    </script>
</body>
</html>'''

    return html_template


def main():
    print("Extracting turns from log files...", flush=True)
    turns = extract_turns()
    print(f"Extracted {len(turns)} turns successfully.", flush=True)

    print("Generating transcript.md...", flush=True)
    md_content = generate_markdown(turns)
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved transcript.md ({len(md_content):,} bytes).", flush=True)

    print("Generating transcript.html...", flush=True)
    html_content = generate_html(turns)
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Saved transcript.html ({len(html_content):,} bytes).", flush=True)

    print("\nExport completed cleanly!")


if __name__ == "__main__":
    main()
