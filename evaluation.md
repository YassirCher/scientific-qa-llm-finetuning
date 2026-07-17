# Evaluation report

This report summarizes the saved EDA, training, and generation artifacts in this repository. It is based on the notebook outputs, every model's `metrics.json` file, and the 600 validation plus 800 test records stored in each `evaluation.json` file. No notebook was rerun to produce this report.

## Dataset and EDA

Qasper is organized around research papers, questions about those papers, and one or more annotated answers. The EDA notebook flattened the nested dataset into paper-, question-, and answer-level tables.

| Split | Papers | Questions | Answers |
| --- | ---: | ---: | ---: |
| Train | 888 | 2,593 | 2,675 |
| Validation | 281 | 1,005 | 1,764 |
| Test | 416 | 1,451 | 3,554 |
| **Total** | **1,585** | **5,049** | **7,993** |

The main observations were:

- Questions are short: the median is 8 words and the 95th percentile is 15 words.
- Answers are usually concise: the median is 6 words, while the 95th percentile is 42 words.
- Papers are much longer: the median context is 3,711 words, or about 5,120 reference-tokenizer tokens. The 95th percentile is about 9,688 tokens.
- More than 99.7% of full contexts exceed 1,024 reference-tokenizer tokens, so sending the complete paper to the model was not practical for these runs.
- The heuristic answer-type analysis found 3,539 abstractive, 2,421 extractive, 113 hybrid, 1,110 other, and 810 unanswerable answer records.
- The EDA marked about 70% of answer records as difficult because of long context, anomaly flags, or abstractive/hybrid answers.

The plots supporting these checks are stored in [`eda/reports/figures/`](eda/reports/figures/).

## Preprocessing used for fine-tuning

The model notebooks use the same main preparation steps, with small model-specific differences:

1. Normalize Unicode with NFKC, remove control characters, trim the text, and collapse repeated whitespace.
2. Resolve the answer type and convert unanswerable targets to `INSUFFICIENT_CONTEXT`.
3. Remove empty questions, empty non-unanswerable answers, unusually short or long questions, oversized answers, suspicious Unicode rows, and duplicate question-answer pairs.
4. Keep the original test split separate. The optional train/validation resplit is disabled in the saved configurations.
5. Cap the prepared datasets when needed. Most runs contain 2,246 training, 800 validation, and 1,200 test rows before generation sampling.
6. Split each paper into 384-token chunks with a 96-token overlap.
7. Rank chunks using question overlap and a small lead-position bonus. When evidence is available, evidence overlap contributes 40% of the chunk score, with an extra weight for sentence-level highlights.
8. Pack the highest-ranked chunks until the prompt context budget is reached.
9. Duplicate the rare `hybrid`, `unanswerable`, and `boolean` training rows once, then shuffle the training data.
10. Format each example with the model's chat template and supervise only the assistant answer tokens.

The training artifacts report 87.2% evidence coverage. Mean evidence recall after context packing ranges from 84.3% to 91.5% on training data and from 82.2% to 90.6% on validation data, depending on the model's prompt budget.

## Why the main sequence length is 1,024

Ten of the thirteen runs use a maximum sequence length of 1,024. For those runs, approximately 824 tokens are reserved for packed paper context and about 200 tokens are left for the system message, question, answer prefix, and special tokens. Gemma uses 768, Llama 3.2 uses 2,048, and Mistral uses 4,096 because of model-specific memory and runtime choices. Mistral's saved prompt-context cap is still 1,024 tokens despite the larger maximum sequence length.

Within this project, 1,024 was the best operational default for the two-T4 setup when combined with evidence-aware context packing. The purpose was not to truncate the first 1,024 tokens of every paper. The notebook first selects the chunks that are most related to the question and annotated evidence, then places only those chunks in the prompt.

Feeding the full paper would add many tokens that are unrelated to the question. The more precise term for the resulting problem is **attention dilution**, sometimes discussed together with the **lost-in-the-middle** effect. Every additional token competes for attention probability, so irrelevant sections can weaken the signal from the useful evidence. Very long sequences also increase activation memory and attention computation, reducing the batch size and the number of experiments that fit in a Kaggle session.

This does not prove that 1,024 is universally optimal. It is the best practical setting used by most runs in this repository. A controlled sequence-length ablation with the same model and training seed would be required to make a causal claim. The poor 4,096-token Mistral run, for example, also has a generation-collapse issue and cannot be used as proof that longer context alone caused the failure.

## Evaluation protocol

Generation evaluation is capped at 600 validation and 800 test examples for every model. The report below uses the test split.

- **Exact match** requires the normalized prediction and reference to be identical.
- **Token F1** measures token overlap and is more tolerant of wording differences.
- **BLEU** is the corpus-level score returned by the saved evaluation pipeline.
- **ROUGE-L** measures longest-common-subsequence overlap.
- **BERTScore F1** measures semantic similarity with `distilbert-base-uncased`.
- **Grounding rate** checks whether the answer appears in, or has strong lexical overlap with, the packed context.
- **Abstention accuracy** measures whether unanswerable questions produce `INSUFFICIENT_CONTEXT`.
- **Critical error rate** is the fraction of confident predictions that are wrong.

The tables can be regenerated from the saved JSON files with:

```bash
python scripts/summarize_metrics.py
```

## Test-set quality metrics

The rows are ordered by token F1. BLEU is shown on its saved scale; the other quality values are percentages.

| Model | Seq. | Exact match | Token F1 | BLEU | ROUGE-L | BERTScore F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3-4B` | 1024 | 19.62% | 36.87% | 2.92 | 36.87% | 79.24% |
| `Qwen/Qwen2.5-7B-Instruct` | 1024 | 20.50% | 36.64% | 2.27 | 36.80% | 79.67% |
| `microsoft/Phi-4-mini-instruct` | 1024 | 19.62% | 36.60% | 2.59 | 36.44% | 79.86% |
| `HuggingFaceTB/SmolLM3-3B` | 1024 | 20.50% | 34.86% | 1.55 | 35.09% | 78.80% |
| `allenai/Olmo-3-7B-Instruct` | 1024 | 20.25% | 34.52% | 1.59 | 34.33% | 79.29% |
| `meta-llama/Llama-3.2-3B-Instruct` | 2048 | 4.62% | 28.12% | 3.14 | 28.04% | 78.67% |
| `MiniMaxAI/SynLogic-7B` | 1024 | 16.50% | 24.94% | 2.51 | 23.87% | 75.24% |
| `google/gemma-3-4b-it` | 768 | 19.75% | 19.81% | 0.00 | 19.84% | 72.75% |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 1024 | 3.38% | 19.55% | 3.09 | 19.35% | 75.00% |
| `arcee-ai/Arcee-Maestro-7B-Preview` | 1024 | 2.38% | 18.69% | 2.24 | 18.06% | 74.75% |
| `zai-org/glm-edge-4b-chat` | 1024 | 4.25% | 15.87% | 1.51 | 15.27% | 74.14% |
| `mistralai/Mistral-7B-Instruct-v0.3` | 4096 | 0.00% | 0.38% | 0.00 | 0.38% | 66.74% |
| `microsoft/bitnet-b1.58-2B-4T-bf16` | 1024 | 0.00% | 0.00% | 0.00 | 0.00% | 54.55% |

## Grounding, abstention, and failure indicators

| Model | Grounding | Abstention accuracy | Critical error | Insufficient outputs | Unique answers |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3-4B` | 69.25% | 82.28% | 49.12% | 38.88% | 376 |
| `Qwen/Qwen2.5-7B-Instruct` | 67.38% | 84.18% | 42.65% | 40.62% | 371 |
| `microsoft/Phi-4-mini-instruct` | 67.75% | 84.81% | 45.78% | 41.12% | 365 |
| `HuggingFaceTB/SmolLM3-3B` | 64.62% | 87.34% | 49.10% | 47.25% | 322 |
| `allenai/Olmo-3-7B-Instruct` | 61.25% | 86.71% | 53.50% | 49.25% | 323 |
| `meta-llama/Llama-3.2-3B-Instruct` | 89.00% | 0.00% | 86.79% | 0.00% | 578 |
| `MiniMaxAI/SynLogic-7B` | 53.00% | 83.54% | 79.10% | 42.50% | 385 |
| `google/gemma-3-4b-it` | 26.25% | 100.00% | 80.25% | 100.00% | 1 |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 89.12% | 1.90% | 93.15% | 0.62% | 610 |
| `arcee-ai/Arcee-Maestro-7B-Preview` | 86.62% | 1.27% | 94.92% | 1.62% | 626 |
| `zai-org/glm-edge-4b-chat` | 56.25% | 21.52% | 87.33% | 9.50% | 618 |
| `mistralai/Mistral-7B-Instruct-v0.3` | 29.50% | 0.00% | 100.00% | 96.50% | 14 |
| `microsoft/bitnet-b1.58-2B-4T-bf16` | 100.00% | 0.00% | 100.00% | 0.00% | 1 |

## Reading the results

### Strongest balanced group

Qwen3-4B, Qwen2.5-7B-Instruct, Phi-4-mini, SmolLM3, and OLMo form the strongest group across exact match, token F1, ROUGE-L, BERTScore, and unanswerable-question handling.

- Qwen3-4B has the highest token F1 and ROUGE-L.
- Qwen2.5 and SmolLM3 share the highest exact-match score. Qwen2.5 also has the lowest critical-error rate in this group.
- Phi-4-mini has the highest BERTScore F1.
- SmolLM3 and OLMo have the best abstention accuracy among the non-collapsed runs.

The critical-error rates remain high, even for the strongest models. Their confidence scores should therefore not be used as calibrated probabilities without a separate calibration stage.

### Llama 3.2

Llama 3.2 has the highest BLEU score and a high grounding rate, but its exact match and abstention accuracy are low. It often produces context-related text without matching the expected short-answer format or abstaining when the question is unanswerable.

### DeepSeek-R1-Distill-Qwen-7B

DeepSeek produces some good individual indicators: its BLEU score is 3.09, its grounding rate is 89.12%, and it generates 610 unique answers. However, exact match is only 3.38%, token F1 is 19.55%, abstention accuracy is 1.90%, and 93.15% of its confident answers are wrong under the saved correctness rule.

This pattern suggests a task-format mismatch. The checkpoint was distilled for explicit reasoning behavior, while Qasper supervision mostly contains short final answers without reasoning traces. A better use of this model would be fine-tuning on reasoning-oriented datasets, or on a mixed dataset that contains scientific questions together with supervised rationales and a clearly separated short final answer. The current metrics support this interpretation, but they do not prove that the architecture itself is unsuitable for scientific QA.

### Collapsed or unreliable runs

Three runs should be debugged before they are compared as normal model results:

- BitNet generates the same 99-character exclamation-mark sequence for all 800 test records. Its reported 100% grounding rate is therefore not meaningful.
- Gemma returns `INSUFFICIENT_CONTEXT` for all 800 test records. Its 100% abstention accuracy only reflects that collapse and does not indicate general QA quality.
- Mistral returns `INSUFFICIENT_CONTEXT` for 772 of 800 records. Its metrics mainly measure this generation failure.

These cases point to tokenizer, padding, EOS handling, chat-template, or generation-configuration issues rather than a clean comparison of model capability.

## Limitations

- The runs use different sequence lengths, epoch counts, and some model-specific settings, so the table is a practical experiment comparison rather than a controlled architecture benchmark.
- Evaluation uses capped samples: 600 validation and 800 test records rather than every prepared row.
- The grounding metric is lexical and can reward copied or degenerate text; it is not a factuality judge.
- Exact match is strict for questions with several acceptable phrasings, while BERTScore can be generous to semantically related but incomplete answers. The metrics should be read together.
- The saved EDA output reports empty `evidence_text` lengths, while the training artifacts report 87.2% evidence coverage and high packed-evidence recall. This mismatch should be resolved by regenerating the EDA and training CSV files from one shared evidence-extraction path.
- A controlled ablation is still needed for sequence length, context-ranking weight, LoRA rank, and confidence calibration.
