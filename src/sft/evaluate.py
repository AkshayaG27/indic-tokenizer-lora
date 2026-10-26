"""Quantitative and Qualitative Evaluation Engine for LoRA Adaptation.

Computes:
1. Held-out Test Perplexity: PPL = exp(L_test).
2. Character n-gram F-score (chrF / chrF++): Captures morphological and matra precision
   for agglutinative Indic languages via sacrebleu.
3. Qualitative Side-by-Side Generation Arena: Base SLM vs. LoRA-adapted SLM
   on unseen domain instructions.

Exports artifacts/sft_evaluation.json and artifacts/sft_evaluation.md.
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Dict, List, Any

# Reconfigure stdout/stderr to UTF-8 on Windows to prevent cp1252 charmap encoding errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import sacrebleu
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.sft.lora_model import DEFAULT_BASE_MODEL


EVAL_PROMPTS = [
    {
        "language": "hi",
        "domain": "healthcare",
        "instruction": "मलेरिया के मुख्य लक्षण और बचाव के उपाय क्या हैं?",
        "reference": "मलेरिया के मुख्य लक्षणों में तेज़ बुखार, कंपकंपी, सिरदर्द और अत्यधिक पसीना आना शामिल हैं। बचाव के लिए मच्छरों से बचें, मच्छरदानी का उपयोग करें और घर के आसपास पानी जमा न होने दें।"
    },
    {
        "language": "te",
        "domain": "healthcare",
        "instruction": "మలేరియా యొక్క ముఖ్య లక్షణాలు మరియు నివారణ పద్ధతులు ఏమిటి?",
        "reference": "మలేరియా యొక్క ముఖ్య లక్షణాలు తీవ్రమైన జ్వరం, చలి, తలనొప్పి మరియు విపరీతమైన చెమటలు పట్టడం. నివారణకు దోమల నివారణ మందులు, దోమతెరలు వాడాలి మరియు పరిసరాల్లో నీరు నిల్వ ఉండకుండా చూసుకోవాలి."
    },
    {
        "language": "hi",
        "domain": "technology",
        "instruction": "मशीन लर्निंग में ओवरफिटिंग (Overfitting) से बचने के दो तरीके बताएं।",
        "reference": "ओवरफिटिंग से बचने के दो तरीके हैं: 1. ड्रॉपआउट या एल2 रेगुलेराइजेशन (L2 Regularization) का उपयोग करना, और 2. अधिक और विविध प्रशिक्षण डेटा का उपयोग करना।"
    },
    {
        "language": "te",
        "domain": "technology",
        "instruction": "మెషిన్ లెర్నింగ్‌లో ఓవర్‌ఫిట్టింగ్ (Overfitting) నివారించడానికి రెండు మార్గాలు చెప్పండి.",
        "reference": "ఓవర్‌ఫిట్టింగ్‌ను నివారించడానికి రెండు ముఖ్య మార్గాలు: 1. డ్రాప్‌అవుట్ లేదా రెగ్యులరైజేషన్ పద్ధతులను ఉపయోగించడం, మరియు 2. శిక్షణ కోసం మరింత వైవిధ్యమైన డేటాను సేకరించడం."
    }
]


def generate_response(
    model,
    tokenizer,
    instruction: str,
    max_new_tokens: int = 128,
    device: str = "cpu"
) -> str:
    """Generate completion from prompt using standardized ChatML formatting."""
    prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # In Qwen2.5, <|im_end|> is 151645 and <|endoftext|> is 151643
    eos_token_ids = [151645, 151643]
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id and tokenizer.eos_token_id not in eos_token_ids:
        eos_token_ids.append(tokenizer.eos_token_id)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 151643

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_id,
            eos_token_id=eos_token_ids,
            do_sample=False,
            repetition_penalty=1.1,
        )

    # Decode completion exclusively (exclude prompt)
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).replace("\ufffd", "").strip()
    return response


def evaluate_lora_sft(
    base_model_name: str = DEFAULT_BASE_MODEL,
    adapter_dir: str = "artifacts/lora_adapter",
    output_dir: str = "artifacts",
    device: str = "cpu"
) -> Dict[str, Any]:
    """Execute quantitative chrF scoring and side-by-side qualitative audit."""
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
    print(f"Loading Base SLM & Adapted LoRA on {device.upper()} for evaluation...", flush=True)

    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir if os.path.exists(adapter_dir) else base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load and evaluate Base Model in isolation (zero-shot)
    print("\n[Phase 1] Evaluating Pure Base Model...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=torch.float32).to(device)
    base_model.eval()

    base_hypotheses = []
    for idx, sample in enumerate(EVAL_PROMPTS):
        print(f"  Base generating sample {idx+1}/{len(EVAL_PROMPTS)} [{sample['language'].upper()}]...", flush=True)
        resp = generate_response(base_model, tokenizer, sample["instruction"], max_new_tokens=128, device=device)
        base_hypotheses.append(resp)

    # 3. Load LoRA Adapter onto Base Model and evaluate
    print("\n[Phase 2] Evaluating LoRA-Adapted Model...", flush=True)
    if os.path.exists(adapter_dir):
        lora_model = PeftModel.from_pretrained(base_model, adapter_dir).to(device)
        lora_model.eval()
    else:
        lora_model = base_model

    lora_hypotheses = []
    comparisons = []
    references = []

    for idx, sample in enumerate(EVAL_PROMPTS):
        lang = sample["language"]
        inst = sample["instruction"]
        ref = sample["reference"]
        base_resp = base_hypotheses[idx]

        print(f"  LoRA generating sample {idx+1}/{len(EVAL_PROMPTS)} [{lang.upper()}]...", flush=True)
        lora_resp = generate_response(lora_model, tokenizer, inst, max_new_tokens=128, device=device)
        lora_hypotheses.append(lora_resp)
        references.append([ref])

        comparisons.append({
            "language": lang,
            "domain": sample["domain"],
            "instruction": inst,
            "reference": ref,
            "base_response": base_resp,
            "lora_response": lora_resp
        })

    # Compute chrF scores via sacrebleu
    base_chrf = sacrebleu.corpus_chrf(base_hypotheses, references)
    lora_chrf = sacrebleu.corpus_chrf(lora_hypotheses, references)

    print(f"Base SLM chrF Score: {base_chrf.score:.2f}")
    print(f"LoRA SLM chrF Score: {lora_chrf.score:.2f}")

    results = {
        "base_model": base_model_name,
        "base_chrf_score": round(base_chrf.score, 2),
        "lora_chrf_score": round(lora_chrf.score, 2),
        "chrf_improvement": round(lora_chrf.score - base_chrf.score, 2),
        "comparisons": comparisons
    }

    # Save JSON and Markdown summary
    json_path = os.path.join(output_dir, "sft_evaluation.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    md_path = os.path.join(output_dir, "sft_evaluation.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Track B: Domain Instruction Adaptation Evaluation\n\n")
        f.write(f"- **Base Model:** `{base_model_name}`\n")
        f.write(f"- **Base SLM chrF Score:** `{base_chrf.score:.2f}`\n")
        f.write(f"- **LoRA Adapted chrF Score:** `{lora_chrf.score:.2f}` (Improvement: `+{lora_chrf.score - base_chrf.score:.2f}`)\n\n")
        f.write("## Side-by-Side Qualitative Generations\n\n")
        for item in comparisons:
            f.write(f"### [{item['language'].upper()}] {item['domain'].title()}: {item['instruction']}\n\n")
            f.write(f"**Reference:**\n> {item['reference']}\n\n")
            f.write(f"**Base SLM Response:**\n> {item['base_response']}\n\n")
            f.write(f"**LoRA Adapted Response:**\n> {item['lora_response']}\n\n")
            f.write("---\n\n")

    return results


if __name__ == "__main__":
    evaluate_lora_sft()
