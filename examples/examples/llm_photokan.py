#!/usr/bin/env python3
"""
Replace GPT-2 FFN layers 0-2 with PhotoKAN, measure perplexity delta.
Requires: pip install transformers
"""
import torch
from torch.utils.data import DataLoader, TensorDataset
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("transformers not installed — skipping GPT-2 example.")
    exit(0)

import photokan.llm as pkl
import photokan as pk


def perplexity(model, dataloader, device):
    model.eval()
    total_loss = total_tokens = 0
    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            out = model(input_ids=input_ids, labels=labels)
            total_loss   += out.loss.item() * labels.numel()
            total_tokens += labels.numel()
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading GPT-2...")
    model     = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Tiny eval set
    texts = ["The quick brown fox jumps over the lazy dog. " * 4] * 16
    enc   = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    ids   = enc["input_ids"]
    dl    = DataLoader(TensorDataset(ids, ids), batch_size=4)

    ppl_base = perplexity(model, dl, device)
    print(f"Baseline GPT-2 perplexity: {ppl_base:.2f}")

    # Replace FFN layers 0–2
    photo_model = pkl.replace_mlp_with_photokan(
        model,
        activation="sine",
        backend="auto",
        layers_to_replace=[0, 1, 2],
        preserve_residuals=True,
        noise_sim=False,
    )

    orig_params  = sum(p.numel() for p in model.parameters())
    photo_params = sum(p.numel() for p in photo_model.parameters())
    print(f"Param reduction: {orig_params:,} → {photo_params:,} "
          f"({100*(orig_params-photo_params)/orig_params:.1f}% fewer)")

    ppl_photo = perplexity(photo_model, dl, device)
    print(f"PhotoKAN perplexity (before fine-tune): {ppl_photo:.2f}")
    print(f"Delta: {ppl_photo - ppl_base:+.2f}")

    # Compile PhotoKAN layers
    bundles = pkl.compile_photokan_layers(photo_model, "./gpt2_photokan")
    print(f"Compiled {len(bundles)} PhotoKAN bundle(s).")


if __name__ == "__main__":
    main()
