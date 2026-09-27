---
title: Jev-Style 2B v3
emoji: ⚖️
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
pinned: true
license: apache-2.0
short_description: 2B decisions, 25,600 tokens, no option cap
models:
  - chaoliangUNSW/Jev-Style-2B-Decision-v3
  - chaoliangUNSW/Jev-Style-2B-Decision-v3-GGUF
  - chaoliangUNSW/Jev-Style-2B-Decision-v3-MLX
preload_from_hub:
  - chaoliangUNSW/Jev-Style-2B-Decision-v3 LICENSE,NOTICE,chat_template.jinja,config.json,generation_config.json,jev_style_decision.py,manifest.json,model-00001-of-00002.safetensors,model-00002-of-00002.safetensors,model.safetensors.index.json,readout_config.json,release_config.json,requirements.txt,tokenizer.json,tokenizer_config.json 1b7b03951a21b363d8fbc949207d0fa195a13eff
tags:
  - text-classification
  - llm-routing
  - guardrails
  - calibration
---

# Jev-Style 2B v3

Try [Jev-Style-2B-Decision-v3](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v3): give it a text or a
JSON state and a typed question, get a calibrated probability for every option. Choice, yes/no or score; up to
25,600 tokens of input and no cap on the number of options. Ask several questions about one text and the text is
computed once.

Runs the model repo's own PyTorch runtime (float32) on ZeroGPU. Other builds:
[GGUF](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v3-GGUF) ·
[MLX](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v3-MLX). The smaller model:
[Jev-Style v3 0.8B Space](https://huggingface.co/spaces/chaoliangUNSW/jev-style-v3). Website: [jevstyle.com](https://jevstyle.com).

**Use it locally** (Python with `torch`, `transformers`, `tokenizers`, `numpy`; versions in the model repo's
`requirements.txt`):

```python
import sys
from huggingface_hub import snapshot_download

path = snapshot_download("chaoliangUNSW/Jev-Style-2B-Decision-v3")
sys.path.insert(0, path)
from jev_style_decision import JevStyleDecision

m = JevStyleDecision(path)                     # float32; CUDA, Apple MPS or CPU
r = m.decide("I was charged twice for my subscription this month.",
             "Which team should handle this ticket?",
             options={"billing": "payments, invoices, refunds", "technical": "bugs and outages",
                      "sales": "new purchases"})
print(r["answer"], r["probabilities"])
```

`m.decide_many(text, [q1, q2, ...])` scores several questions about one text; each question is
`{"t": "choice" | "noul" | "score", "ins": ..., "crit": ...}`. The "Use it locally" box in the app shows the code
for whatever you ran.

Scope: the decision scores need the model's own runtimes (PyTorch, GGUF or MLX, shipped in the model repos); stock
llama.cpp, Ollama, LM Studio or `mlx_lm.generate` can load the weights but cannot produce them. Inputs over 25,600
tokens are refused, never truncated. The model was trained on a reduced data pool (60M tokens).

The 19K-token example is public-domain text (U.S. founding documents, Project Gutenberg eBooks 1, 5, 2 and 1404).
Code: Apache-2.0. The model is Apache-2.0, but some of its training data has restrictive or unclear terms and some
training rows are outputs of OpenAI GPT and Anthropic Claude models; see
[Training data and licences](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v3#training-data-and-licences)
on the model card. Not affiliated with TypeSafe or Jev. AI disclosure: code written with AI coding assistants
(Claude Code) under my direction; I designed the project, trained the models and verified the results.
