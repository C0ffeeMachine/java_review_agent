# Java Code Review Agent 🔍

A fine-tuned LLM that reviews Java code, identifies bugs, and suggests fixes — trained on real-world bug-fix commits from open-source projects.

## Demo

### Base Model vs Fine-Tuned Model

**Input:**
```java
private Map<String, Set<String>> getServiceAppMap(List<Provider> providers) {
    Map<String, Set<String>> serviceAppMap = new HashMap<String, Set<String>>();
    if (providers != null && providers.size() >= 0) {
        for (Provider provider : providers) {
```

**Base Model Output:**
```
The code is wrong because it does not handle the case where the provider 
list is null or empty.

The corrected version is:
The corrected version is:
The corrected version is:
The corrected version is:
... (repeats 20 times)
```

**Fine-Tuned Model Output:**
```
Bug Type: CHANGE_OPERATOR

Explanation: In `ProviderService.java`, the expression `providers.size() >= 0` 
is incorrect. It should be `providers.size() > 0`. This is a CHANGE_OPERATOR error.

Fixed Code:
if (providers != null && providers.size() > 0) {
```

## Architecture

```
ManySStuBs4J Dataset (Java bug-fix commits)
        │
        ▼
prepare_dataset.py  →  Instruction pairs (buggy code / explanation / fix)
        │
        ▼
train.py  →  QLoRA fine-tuning on CodeLlama-7b-Instruct
        │
        ▼
LoRA Adapter (saved to adapter/final)
        │
        ▼
inference.py  →  Structured review output
        │
        ▼
Spring Boot REST API  →  POST /review
```

## Training Details

| Parameter | Value |
|---|---|
| Base Model | CodeLlama-7b-Instruct-hf |
| Method | QLoRA (4-bit NF4 + LoRA rank-16) |
| Dataset | ManySStuBs4J (~2,200 samples) |
| Epochs | 3 |
| Hardware | NVIDIA RTX 5060 Ti 16GB |
| Training Time | 2h 12m |
| Train Loss | 0.348 |
| Eval Loss | 0.371 |
| Eval Token Accuracy | 91.3% |

## Setup

```bash
# Clone
git clone https://github.com/C0ffeeMachine/java_review_agent
cd java-review-agent

# Install dependencies
pip install -r requirements.txt

# Prepare dataset
python src/prepare_dataset.py

# Fine-tune
python src/train.py

# Run inference
python src/inference.py
```

## Requirements

```
torch
transformers
peft
trl
bitsandbytes
datasets
accelerate
pyyaml
```

## Adapter Weights

Fine-tuned LoRA adapter available on HuggingFace Hub:  
👉 `https://huggingface.co/C0ffeeMachine/java-review-agent`

## Bug Types Covered

Trained on 9 real-world Java bug categories from ManySStuBs4J:
- `CHANGE_OPERATOR` — wrong comparison/arithmetic operator
- `CHANGE_IDENTIFIER` — wrong variable or method name
- `DIFFERENT_METHOD_SAME_ARGS` — wrong method called with same args
- `LESS_SPECIFIC_IF` — overly broad conditional
- `CHANGE_UNARY_OPERATOR` — wrong unary operator applied
- `CHANGE_NUMERAL` — wrong numeric literal
- `OVERLOADED_METHOD_DELETED_ARGS` — wrong method overload
- `WRONG_FUNCTION_NAME` — incorrect function referenced
- `SAME_FUNCTION_MORE_ARGS` — extra args passed to function


## Evaluation (50 held-out samples)

| Metric | Base Model | Fine-Tuned |
|---|---|---|
| Structure Accuracy | 0.0% | 94.0% |
| Bug Type Accuracy | 0.0% | 56.0% |

Structure Accuracy: % of outputs correctly following Bug Type / Explanation / Fixed Code format  
Bug Type Accuracy: % of outputs correctly classifying the bug category