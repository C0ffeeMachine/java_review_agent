import os
os.environ["PYTHONUTF8"] = "1"

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --- Config ---
BASE_MODEL_ID = "codellama/CodeLlama-7b-Instruct-hf"
ADAPTER_PATH = "adapter/final"
TEST_SPLIT_PATH = "data/splits/test.json"
SAMPLE_SIZE = 50  # evaluate on 50 held-out samples — enough to be meaningful

# --- Load tokenizer ---
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# --- 4-bit quantization ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

def build_prompt(text: str) -> str:
    """Extract instruction+input from the formatted text, strip the response."""
    if "### Response:" in text:
        return text.split("### Response:")[0] + "### Response:\n"
    return text

def generate(model, prompt: str, max_new_tokens: int = 200) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)

def has_correct_structure(output: str) -> bool:
    """Check if output follows trained format."""
    return (
        "Bug Type:" in output and
        "Explanation:" in output and
        "Fixed Code:" in output
    )

def extract_bug_type(text: str) -> str | None:
    """Extract bug type from output."""
    for line in text.split("\n"):
        if line.startswith("Bug Type:"):
            return line.replace("Bug Type:", "").strip()
    return None

def evaluate(model, test_samples: list, label: str) -> dict:
    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"{'='*60}")

    structure_hits = 0
    bug_type_hits = 0
    total = len(test_samples)

    for i, sample in enumerate(test_samples):
        prompt = build_prompt(sample["text"])
        expected_response = sample["text"].split("### Response:")[-1].strip()
        expected_bug_type = extract_bug_type(expected_response)

        output = generate(model, prompt)

        # Metric 1: structure correctness
        if has_correct_structure(output):
            structure_hits += 1

        # Metric 2: bug type classification accuracy
        predicted_bug_type = extract_bug_type(output)
        if expected_bug_type and predicted_bug_type:
            if expected_bug_type.upper() == predicted_bug_type.upper():
                bug_type_hits += 1

        if i % 10 == 0:
            print(f"  Progress: {i+1}/{total}")
            print(f"  Expected Bug Type : {expected_bug_type}")
            print(f"  Predicted Bug Type: {predicted_bug_type}")
            print(f"  Structure OK      : {has_correct_structure(output)}")
            print()

    results = {
        "model": label,
        "total_samples": total,
        "structure_accuracy": round(structure_hits / total * 100, 2),
        "bug_type_accuracy": round(bug_type_hits / total * 100, 2),
    }

    print(f"Results for {label}:")
    print(f"  Structure Accuracy : {results['structure_accuracy']}%")
    print(f"  Bug Type Accuracy  : {results['bug_type_accuracy']}%")

    return results


# --- Load test data ---
with open(TEST_SPLIT_PATH) as f:
    test_data = json.load(f)

test_samples = test_data[:SAMPLE_SIZE]
print(f"Evaluating on {len(test_samples)} held-out samples")

# --- Load base model ---
print("\nLoading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

base_results = evaluate(base_model, test_samples, "BASE MODEL")

# --- Load fine-tuned model ---
print("\nLoading fine-tuned adapter...")
ft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
ft_model.eval()

ft_results = evaluate(ft_model, test_samples, "FINE-TUNED MODEL")

# --- Final comparison ---
print(f"\n{'='*60}")
print("FINAL COMPARISON")
print(f"{'='*60}")
print(f"{'Metric':<25} {'Base':>10} {'Fine-Tuned':>12}")
print(f"{'-'*50}")
print(f"{'Structure Accuracy':<25} {base_results['structure_accuracy']:>9}% {ft_results['structure_accuracy']:>11}%")
print(f"{'Bug Type Accuracy':<25} {base_results['bug_type_accuracy']:>9}% {ft_results['bug_type_accuracy']:>11}%")

# --- Save results ---
all_results = {
    "base": base_results,
    "fine_tuned": ft_results
}

with open("data/eval_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\nResults saved to data/eval_results.json")