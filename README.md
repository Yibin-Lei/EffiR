<div align="center">
  <h1>Making Large Language Models Efficient Dense Retrievers </h1>
</div>

**EffiR** is a framework for building efficient LLM-based **dense retrievers** through large-scale MLP compression with a coarse-to-fine strategy:

- **Analyzes layer redundancy** in LLM-based dense retrievers, showing that **MLP layers are substantially more prunable** than attention layers for retrieval.
- **Compresses LLMs with a two-stage strategy**, followed by retrieval-specific fine-tuning:
  1) **Coarse-grained depth reduction** (dropping redundant MLP layers)  
  2) **Fine-grained width reduction** (slimming MLP hidden dimensions)
 
EffiR achieves substantial reductions in **model size** and **inference latency** while largely preserving effectiveness of full-size retrievers.
 
---

## 📑 Release Plan

- [ ] Model checkpoints
- [ ] Training code
- [ ] Evaluation code

---
