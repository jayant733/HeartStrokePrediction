# AI Concepts Glossary

## Transformer
A Transformer is a deep learning architecture based on self-attention, which allows the model to understand relationships between tokens in a sequence. In self-attention, the model uses Query, Key, and Value vectors to determine how much focus to place on other tokens when processing a given token.

## LoRA and QLoRA
- **LoRA (Low-Rank Adaptation):** A parameter-efficient fine-tuning method that injects trainable low-rank matrices into transformer layers while keeping the original weights frozen.
- **QLoRA:** An extension of LoRA that combines quantization (like 4-bit precision) with LoRA fine-tuning, enabling fine-tuning of large models with drastically reduced memory requirements.

