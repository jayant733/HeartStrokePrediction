# AI Concepts Glossary

## Transformer
A Transformer is a deep learning architecture based on self-attention, which allows the model to understand relationships between tokens in a sequence. In self-attention, the model uses Query, Key, and Value vectors to determine how much focus to place on other tokens when processing a given token.

## LoRA and QLoRA
- **LoRA (Low-Rank Adaptation):** A parameter-efficient fine-tuning method that injects trainable low-rank matrices into transformer layers while keeping the original weights frozen.
- **QLoRA:** An extension of LoRA that combines quantization (like 4-bit precision) with LoRA fine-tuning, enabling fine-tuning of large models with drastically reduced memory requirements.

## RAG (Retrieval-Augmented Generation)
RAG is a technique that enhances large language models by retrieving relevant information from an external knowledge base before generating a response, thereby grounding the output in factual and up-to-date information.

## Agentic AI and ReAct
- **Agentic AI:** AI systems designed to act autonomously (or semi-autonomously) to achieve specific goals. They can perceive their environment, make decisions, and execute actions using tools.
- **ReAct (Reason and Act):** A prompting framework for agentic AI that interleaves reasoning (thinking about what to do) and acting (using tools or interacting with the environment) to solve complex tasks step by step.

