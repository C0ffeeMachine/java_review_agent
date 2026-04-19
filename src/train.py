import torch
import yaml
from datasets import load_from_disk, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import json

# --- Load config ---
with open("config/train_config.yaml") as f:
    cfg = yaml.safe_load(f)

# --- Load dataset from splits ---
with open("data/splits/train.json") as f:
    train_data = json.load(f)

with open("data/splits/test.json") as f:
    test_data = json.load(f)

train_dataset = Dataset.from_list(train_data)
test_dataset = Dataset.from_list(test_data)

print(f"Train: {len(train_dataset)} | Test: {len(test_dataset)}")

# --- 4-bit quantization config ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# --- Load base model ---
model_id = cfg["model_id"]  # e.g. "codellama/CodeLlama-7b-Instruct-hf"

print(f"Loading model: {model_id}")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)

# --- LoRA config ---
lora_config = LoraConfig(
    r=cfg["lora_rank"],
    lora_alpha=cfg["lora_alpha"],
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected: ~0.5% of total params — confirms LoRA is working

# --- Training config ---
sft_config = SFTConfig(
    output_dir=cfg["output_dir"],
    num_train_epochs=cfg["epochs"],
    per_device_train_batch_size=cfg["batch_size"],
    gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
    learning_rate=float(cfg["learning_rate"]),
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    max_length=512,
    dataset_text_field="text",
    report_to="none"
)

# --- Train ---
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    args=sft_config,
    processing_class=tokenizer,
)

print("Starting training...")
trainer.train()

# --- Save LoRA adapter ---
model.save_pretrained(cfg["adapter_output"])
tokenizer.save_pretrained(cfg["adapter_output"])
print(f"Adapter saved to {cfg['adapter_output']}")