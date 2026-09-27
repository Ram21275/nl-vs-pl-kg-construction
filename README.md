# Natural Language vs Programming Language for Knowledge Graph Construction

This repository provides a clean, Kaggle-ready reproduction scaffold for the core comparison in Gajo and Barrón-Cedeño, **Natural vs programming language in LLM knowledge graph construction** (Information Processing and Management, 2025, DOI: 10.1016/j.ipm.2025.104195).

The experiment holds the task, model, data split, demonstrations, LoRA settings, decoding, parser strictness, and metric constant while changing only the prompt/output representation:

- **Natural language condition:** JSON arrays of `[head, relation, tail]` triples.
- **Programming language condition:** Python-style `Triple(head, relation, tail)` values returned by an `extract` function.

The default configuration uses the paper's strongest model family, `mistralai/Mistral-7B-Instruct-v0.3`, and the ADE dataset because ADE is the smallest and least structurally complex of the paper's three datasets. Four-bit QLoRA makes the experiment practical on a Kaggle GPU.

## What is reproduced

- ADE, CoNLL04, and SciERC-compatible JSON loader
- Matched natural-language and Python-style prompts
- Three in-context demonstrations by default
- LoRA on attention and MLP projection layers with rank and alpha 16
- 200 training steps, learning rate `2e-4`, warm-up, AdamW-style 8-bit optimizer
- Greedy decoding
- Strict exact-match micro precision, recall, and F1 over complete triples
- Separate predictions and metrics for each prompt condition

The notebook focuses on the no-rationale setting because the paper found rationale training usually reduced performance.

## Comparative research report

The [`report/`](report/) directory contains the reviewed LaTeX source and its compiled PDF. The report compares five NLP papers across their objectives, motivations, methodologies, datasets, evaluation metrics, findings, limitations, and reproducibility considerations. Direct paper evidence is distinguished from interpretive synthesis and independent critical analysis.

## Repository provenance

The paper authors publish a reference repository at [TinfFoil/natcode-llm-kgc](https://github.com/TinfFoil/natcode-llm-kgc). It was consulted to confirm file formats and experimental details. That repository did not expose a software license when this implementation was prepared, so its source code is not redistributed here. This repository is an independent implementation based on the paper and documented interfaces.

## Kaggle quick start

1. Create a Kaggle notebook with a GPU accelerator and enable Internet access for the initial model download.
2. Add this repository as a notebook input or clone your private repository using a Kaggle secret.
3. Add the ADE split files described in [`data/README.md`](data/README.md).
4. Run [`notebooks/nl_vs_pl_kgc_ade_kaggle.ipynb`](notebooks/nl_vs_pl_kgc_ade_kaggle.ipynb).

For a command-line run:

```bash
pip install -r requirements-kaggle.txt
pip install -e .
nl-pl-kgc train --config configs/ade_mistral7b.yaml --format both
nl-pl-kgc evaluate --config configs/ade_mistral7b.yaml --format natural --adapter outputs/ade-mistral7b/natural/adapter
nl-pl-kgc evaluate --config configs/ade_mistral7b.yaml --format code --adapter outputs/ade-mistral7b/code/adapter
```

## Fast validation without a GPU

The smoke test exercises loading, prompt generation, output parsing, result serialization, and the strict metric without claiming model quality:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
nl-pl-kgc smoke-test --dataset tests/fixtures.json --output-dir outputs/smoke-test
```

## Interpreting results

Compare `micro_f1` in the two condition directories. Use at least three seeds for a substantive conclusion. A small difference supports the paper's finding that prompt representation has limited effect after supervised fine-tuning. A single Kaggle run is a reproducibility check, not a statistically reliable replication of all 175 model conditions in the paper.

## Practical constraints

- Kaggle GPU availability varies. A T4 should use 4-bit loading, batch size 1, and gradient accumulation.
- Mistral model weights and the datasets retain their own licenses and terms.
- Exact scores can differ because of GPU kernels, library versions, seed choices, and dataset preparation.
- This implementation intentionally never executes model-generated Python; the code-output parser extracts literal `Triple(...)` calls only.

## References

- Paolo Gajo and Alberto Barrón-Cedeño. Natural vs programming language in LLM knowledge graph construction. Information Processing and Management 62(5), 104195, 2025.
- Gurulingappa et al. Development of a benchmark corpus to support the automatic extraction of drug-related adverse effects from medical case reports, 2012.
- Bi et al. CodeKGC: Code Language Model for Generative Knowledge Graph Construction, 2024.
