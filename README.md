<div align="center">
  <h2>Making Large Language Models Efficient Dense Retrievers </h2>
</div>

**EffiR** is a framework for building efficient LLM-based **dense retrievers** through large-scale MLP compression with a coarse-to-fine strategy:

- **Analyzes layer redundancy** in LLM-based dense retrievers, showing that **MLP layers are substantially more prunable** than attention layers for retrieval.
- **Compresses LLMs with a two-stage strategy**, followed by retrieval-specific fine-tuning:
  1) **Coarse-grained depth reduction** (dropping redundant MLP layers)  
  2) **Fine-grained width reduction** (slimming MLP intermediate dimensions)
 
EffiR achieves substantial reductions in **model size** and **inference latency** while largely preserving effectiveness of full-size retrievers.
 
---
## Models

Models on Hugging Face:

- [EffiR-Mistral-Drop-8-MLP](https://huggingface.co/yibinlei/effir-mistral-drop-8-mlp)
- [EffiR-Mistral-Drop-16-MLP](https://huggingface.co/yibinlei/effir-mistral-drop-16-mlp)
- [EffiR-Mistral-Drop-8-Attn](https://huggingface.co/yibinlei/effir-mistral-drop-8-attn)
- [EffiR-Mistral-Drop-16-Attn](https://huggingface.co/yibinlei/effir-mistral-drop-16-attn)

## Evaluation

```bash
MODEL_NAME_OR_PATH=/path/to/mistral-base \
LORA_DIR=path/effir-mistral-drop-8-mlp \
TASK=NFCorpus \
BATCH_SIZE=32 \
bash scripts/eval_mteb.sh
```

---
## 📑 Release Plan

- Model checkpoints
- [ ] Training code
- Evaluation code

---
