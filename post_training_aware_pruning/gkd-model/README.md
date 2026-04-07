---
base_model: Qwen/Qwen2-0.5B-Instruct
library_name: transformers
model_name: gkd-model
tags:
- generated_from_trainer
- gkd
- trl
licence: license
---

# Model Card for gkd-model

This model is a fine-tuned version of [Qwen/Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 



This model was trained with GKD, a method introduced in [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://huggingface.co/papers/2306.13649).

### Framework versions

- TRL: 0.29.0
- Transformers: 5.2.0
- Pytorch: 2.10.0
- Datasets: 4.6.1
- Tokenizers: 0.22.2

## Citations

Cite GKD as:

```bibtex
@inproceedings{agarwal2024on-policy,
    title        = {{On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes}},
    author       = {Rishabh Agarwal and Nino Vieillard and Yongchao Zhou and Piotr Stanczyk and Sabela Ramos Garea and Matthieu Geist and Olivier Bachem},
    year         = 2024,
    booktitle    = {The Twelfth International Conference on Learning Representations, {ICLR} 2024, Vienna, Austria, May 7-11, 2024},
    publisher    = {OpenReview.net},
    url          = {https://openreview.net/forum?id=3zKtaqxLhW},
}
```

Cite TRL as:
    
```bibtex
@software{vonwerra2020trl,
  title   = {{TRL: Transformers Reinforcement Learning}},
  author  = {von Werra, Leandro and Belkada, Younes and Tunstall, Lewis and Beeching, Edward and Thrush, Tristan and Lambert, Nathan and Huang, Shengyi and Rasul, Kashif and Gallouédec, Quentin},
  license = {Apache-2.0},
  url     = {https://github.com/huggingface/trl},
  year    = {2020}
}
```