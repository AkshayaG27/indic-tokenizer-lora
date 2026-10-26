"""Unit Tests for LoRA SFT, Parameter Freezing, and Prompt-Loss Masking.

Verifies:
1. AC-5: Parameter Budget Constraint: Trainable adapter parameters < 1.0% of total weights.
2. AC-5: Weight Freezing: All base model weights W_0 have requires_grad = False.
3. AC-6: Prompt-Loss Masking: User prompt tokens have label = -100, completions have valid IDs.
4. Adapter Forward/Backward Pass: Verifies gradient flow through adapter matrices A and B.
"""

import pytest
import torch
from transformers import AutoTokenizer
from src.sft.lora_model import setup_lora_pipeline, audit_parameter_efficiency, DEFAULT_BASE_MODEL
from src.sft.dataset import BilingualInstructionDataset, InstructionDataCollator


@pytest.fixture(scope="module")
def initialized_pipeline():
    model, tokenizer, efficiency = setup_lora_pipeline(model_name=DEFAULT_BASE_MODEL, device="cpu")
    return model, tokenizer, efficiency


class TestLoRASFT:
    """Automated unit test suite verifying PEFT LoRA architecture and loss masking."""

    def test_parameter_budget_and_freezing(self, initialized_pipeline):
        """Verify AC-5: Adapter parameters < 1.0% and base weights are frozen."""
        model, tokenizer, efficiency = initialized_pipeline

        # 1. Parameter budget check
        assert efficiency["trainable_percentage"] < 1.0, (
            f"Trainable parameters ({efficiency['trainable_percentage']}%) exceed 1.0% budget!"
        )
        assert efficiency["parameter_budget_satisfied"] is True

        # 2. Assert base model weights are strictly frozen
        for name, param in model.named_parameters():
            if "lora_" in name:
                assert param.requires_grad is True, f"LoRA parameter '{name}' should require grad!"
            else:
                assert param.requires_grad is False, f"Base parameter '{name}' must be frozen!"

    def test_prompt_loss_masking_protocol(self, initialized_pipeline):
        """Verify AC-6: All prompt tokens have label = -100."""
        model, tokenizer, _ = initialized_pipeline

        sample = {
            "language": "te",
            "instruction": "ఈ వాక్యాన్ని అనువదించండి.",
            "response": "This is the translation."
        }
        dataset = BilingualInstructionDataset([sample], tokenizer)
        item = dataset[0]

        input_ids = item["input_ids"]
        labels = item["labels"]

        # Number of -100 labels
        masked_tokens = (labels == -100).sum().item()
        unmasked_tokens = (labels != -100).sum().item()

        # Prompt must be masked, response must be active
        assert masked_tokens > 0, "No prompt tokens were masked with -100!"
        assert unmasked_tokens > 0, "No completion tokens were preserved for loss computation!"

        # Unmasked labels must exactly match the corresponding input_ids
        unmasked_indices = (labels != -100).nonzero(as_tuple=True)[0]
        for idx in unmasked_indices:
            assert labels[idx].item() == input_ids[idx].item()

    def test_adapter_gradient_flow(self, initialized_pipeline):
        """Verify gradient backpropagation updates adapter matrices A and B."""
        model, tokenizer, _ = initialized_pipeline

        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long),
                "attention_mask": torch.tensor([1, 1, 1, 1], dtype=torch.long),
                "labels": torch.tensor([-100, -100, 3, 4], dtype=torch.long)
            }
        ]
        collator = InstructionDataCollator(pad_token_id=tokenizer.pad_token_id)
        collated = collator(batch)

        model.train()
        outputs = model(
            input_ids=collated["input_ids"],
            attention_mask=collated["attention_mask"],
            labels=collated["labels"]
        )
        loss = outputs.loss
        assert torch.isfinite(loss), "Loss must be a finite numerical value!"

        loss.backward()

        # Verify adapter gradients exist and base gradients are None
        for name, param in model.named_parameters():
            if "lora_" in name:
                assert param.grad is not None, f"Gradient missing for LoRA parameter '{name}'"
            else:
                assert param.grad is None, f"Gradient erroneously computed for frozen base parameter '{name}'"
