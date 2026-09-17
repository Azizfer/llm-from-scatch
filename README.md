# GPT from Scratch

A from-scratch implementation of a GPT-style language model, 
pre-trained on WebText and fine-tuned on Databricks-Dolly-15k.

## Overview

This project implements a decoder-only transformer language model 
from scratch in PyTorch, trained in three stages:

1. **Pre-training** on WebText (~13,000 steps)
2. **Fine-tuning** on Databricks-Dolly-15k (~5,000 steps)
3. **Evaluation** on WikiText-2

## Model Architecture

- ~50M parameters
- 12 layers, 12 attention heads, 768 embedding dimensions
- Block size: 1024 tokens
- Tokenizer: GPT-2 BPE (50,257 vocab)

## Results

| Stage | Loss | Perplexity |
|-------|------|------------|
| Pre-training (WebText) | 4.22 | ~68 |
| Fine-tuning (Dolly) | 1.02 
| WikiText-2 (zero-shot) | 5.68 | 292.45 |

### Sample Generation

**Before fine-tuning:**
> "When it comes to uckams and explosives, and the American leadership..."

**After fine-tuning:**
> "Question: What is the capital of Japan?
> Answer: The capital city of Japan is Tokyo..."

## What I Learned

- Built a transformer from scratch (attention, blocks, layer norm)
- Implemented cosine LR scheduling with linear warmup
- Implemented Top-K + Top-P (nucleus) sampling
- Handled special tokens (EOS) correctly during training and generation
- Understood the difference between format learning and factual recall

