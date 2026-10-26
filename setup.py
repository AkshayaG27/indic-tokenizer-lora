from setuptools import setup, find_packages

setup(
    name="indic_tokenizer_lora_sft",
    version="0.1.0",
    description="Hindi & Telugu Token Fertility Pareto Audit & Domain LoRA SFT",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
)
