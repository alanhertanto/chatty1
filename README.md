# LLM R&D Workspace 🧪

This repository is my personal sandbox for **Large Language Model (LLM) experiments**.  
Main focus is on **inference methods** and **optimizations** when running LLMs in limited hardware setups (e.g., RTX 3060 Laptop, Google Colab T4, MacBook Air M1).

## 🔥 Goals
- Test different **quantization formats** (AWQ, GPTQ, etc.)
- Benchmark **inference speed** (tok/s) across devices (CPU, GPU, MPS, Colab)
- Explore **resource-efficient techniques** (layer fusion, caching, device mapping)
- Experiment with **retrieval-augmented generation (RAG)** for regulatory/legal text (e.g., OJK rules)

## 🛠️ Tech Stack
- [Transformers](https://huggingface.co/docs/transformers/index)
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)
- [PyTorch](https://pytorch.org/)
- Google Colab (Free/Pro)

## 📊 Example Benchmark
| Device              | Model              | Quantization | Speed (tok/s) |
|---------------------|--------------------|--------------|---------------|
| Colab T4 GPU        | Mistral-7B         | AWQ          | 6–9           |
| MacBook Air M1 16GB | Mistral-7B         | AWQ (MPS)    | 2–3           |

## 🚀 Notes
- This repo is not a production framework, but rather an **R&D notebook**.  
- Expect messy experiments, quick hacks, and trial-and-error approaches ⚡  
- Use at your own risk 😅

---
_“Trying inference methods for the LLM, one token at a time.”_
