# Scientific QA LLM Fine-Tuning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="Hugging Face Transformers">
  <img src="https://img.shields.io/badge/Datasets-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="Hugging Face Datasets">
  <img src="https://img.shields.io/badge/PEFT-FF9D00?style=flat-square&logo=huggingface&logoColor=black" alt="PEFT">
  <img src="https://img.shields.io/badge/Accelerate-5C4EE5?style=flat-square&logo=huggingface&logoColor=white" alt="Accelerate">
  <img src="https://img.shields.io/badge/bitsandbytes-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="bitsandbytes">
</p>
<p align="center">
  <img src="https://img.shields.io/badge/Evaluate-4B8BBE?style=flat-square" alt="Hugging Face Evaluate">
  <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="pandas">
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=flat-square" alt="Matplotlib">
  <img src="https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square" alt="Seaborn">
</p>

For one part of my final-year project, I fine-tuned and evaluated 13 open language models on the QASPER scientific question-answering dataset. I used the same main data preparation and evaluation steps for every model so that the resulting artifacts could be compared consistently.

## Final-year project context

This repository documents the multi-model fine-tuning task from my final-year project. I kept each experiment in its own directory, together with the notebook, aggregate metrics, and per-example evaluation records from the completed run.

The methodology, preprocessing choices, metric definitions, model comparison, and limitations are documented in the [evaluation report](evaluation.md).

## What I worked on

- Cleaning and validating the QASPER train, validation, and test splits
- Removing unusable and duplicate samples
- Packing long paper contexts within the model's token budget
- Prioritizing annotated evidence during context selection
- Formatting model-specific chat and instruction prompts
- Fine-tuning with QLoRA or LoRA, gradient checkpointing, and mixed precision
- Supporting Kaggle environments with two Tesla T4 GPUs
- Evaluating generated answers on the validation and test sets
- Saving aggregate metrics and per-example evaluation records

## Exploratory data analysis

The EDA is available in `eda/qasper_eda.ipynb`. It downloads Qasper, converts the nested dataset into tabular files, checks missing values and noisy records, measures text and token lengths, and estimates context limits for training. The figures saved under `eda/reports/figures/` show the main distributions and data-quality checks used during preparation.

## Models

| Directory | Hugging Face model |
| --- | --- |
| `arcee-maestro-7b-preview` | `arcee-ai/Arcee-Maestro-7B-Preview` |
| `bitnet-b1.58-2b-4t-bf16` | `microsoft/bitnet-b1.58-2B-4T-bf16` |
| `DeepSeek-R1-Distill-Qwen-7B` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` |
| `gemma-3-4b` | `google/gemma-3-4b-it` |
| `glm-edge-4b` | `zai-org/glm-edge-4b-chat` |
| `llama-3-2-3b-instruct` | `meta-llama/Llama-3.2-3B-Instruct` |
| `mistral-7b-instruct` | `mistralai/Mistral-7B-Instruct-v0.3` |
| `olmo3-7b-instruct` | `allenai/Olmo-3-7B-Instruct` |
| `phi-4-mini-instruct` | `microsoft/Phi-4-mini-instruct` |
| `qwen2-5-instruct` | `Qwen/Qwen2.5-7B-Instruct` |
| `qwen3-4b` | `Qwen/Qwen3-4B` |
| `smollm3-3b` | `HuggingFaceTB/SmolLM3-3B` |
| `synlogic7b` | `MiniMaxAI/SynLogic-7B` |

## Experiment workflow

Each model directory contains a notebook built around the same main stages:

1. Configure the runtime, model, data paths, and training parameters.
2. Load the QASPER splits and the evidence statistics produced during exploratory analysis.
3. Normalize, filter, deduplicate, and cap the data when required.
4. Split long paper contexts into chunks and rank them using question and evidence overlap.
5. Build model-specific prompts and tokenized training datasets.
6. Load the base model and attach QLoRA or LoRA adapters.
7. Train with Hugging Face Trainer and save the adapter checkpoint.
8. Generate answers for the validation and test sets.
9. Export metrics and per-example evaluation records.

## Evaluation

The saved metrics cover several aspects of model behavior:

- Exact match and token-level precision, recall, F1, and accuracy
- BLEU, ROUGE-L, and BERTScore
- Grounding and hallucination rates
- Abstention accuracy for unanswerable questions
- Confidence calibration, including ECE and Brier score
- Confidence-aware error rates
- Evidence recall after context packing
- Training loss by step

The `metrics.json` files contain the run configuration, dataset statistics, training summary, validation metrics, test metrics, and training curve. The `evaluation.json` files contain the corresponding per-example predictions and supporting context.

## Repository structure

```text
scientific-qa-llm-finetuning/
|-- eda/
|   |-- qasper_eda.ipynb
|   `-- reports/figures/
|-- <model-directory>/
|   |-- qasper-<model>-train.ipynb
|   |-- metrics.json
|   `-- evaluation.json
|-- scripts/
|   `-- summarize_metrics.py
|-- evaluation.md
`-- README.md
```

The Qwen2.5 directory uses model-specific names for its metric and evaluation files, but their contents follow the same artifact structure.

## Running a notebook

The notebooks were prepared for Kaggle with the QASPER data available at:

```text
/kaggle/input/datasets/yassireyassir/qasper/data
```

To reproduce an experiment:

1. Add the QASPER dataset to the Kaggle notebook.
2. Enable the required GPU accelerator.
3. Accept the model license on Hugging Face when the selected model is gated.
4. Provide authentication through Kaggle Secrets or the `HF_TOKEN` environment variable.
5. Open the notebook for the selected model and run its cells in order.

Do not commit access tokens to the repository.

## Outputs

The repository includes the saved notebook outputs and evaluation artifacts from the completed runs. Model weights and adapter checkpoints are not stored here.
