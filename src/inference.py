import os
os.environ["PYTHONUTF8"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --- Config ---
BASE_MODEL_ID = "codellama/CodeLlama-7b-Instruct-hf"
ADAPTER_PATH = "adapter/final"

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

def build_prompt(code: str) -> str:
    return f"""### Instruction:
You are a Java code reviewer. Review the following Java code, identify the bug, explain why it is wrong, and provide the corrected version.

### Input:
{code}

### Response:
"""

def generate(model, prompt: str, max_new_tokens: int = 300) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    # Decode only the newly generated tokens
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# --- Test samples (take from your test split) ---
test_samples = [
    # Sample 1 — wrong operator
    """if (submittedNode == null || submittedNode.get("values") != null) {
    continue;
}""",
    # Sample 2 — wrong variable
    """public int getMax(int a, int b) {
    if (a > b) return b;
    return a;
}""",
    # Sample 3 — add one from your test.json for a real held-out sample
    """protected void invalidateExpiredEntries() {
        long now = System.nanoTime();
        synchronized (messages) {

            Set<String> inactiveClients = new HashSet<String>();
            logger.debug("Active clients {}", activeClients());
        }

        long now = System.nanoTime();
        String messageId = UUID.randomUUID().toString();
        CacheMessage cacheMessage = new CacheMessage(messageId, e);
        synchronized (messages) {
    @Override
    public List<Object> retrieveFromCache(String broadcasterId, AtmosphereResource r) {
        String clientId = r.uuid();
        long now = System.nanoTime();

        List<Object> result = new ArrayList<Object>();
}""",
"""{
    private Map<String, Set<String>> getServiceAppMap(List<Provider> providers) {
        Map<String, Set<String>> serviceAppMap = new HashMap<String, Set<String>>();
        if (providers != null && providers.size() >= 0) {
            for (Provider provider : providers) {
                Set<String> appSet;
                String service = provider.getService();
}""",
]

# --- Load BASE model only ---
print("\nLoading BASE model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

print("\n" + "="*60)
print("BASE MODEL OUTPUT")
print("="*60)
for i, code in enumerate(test_samples):
    prompt = build_prompt(code)
    print(f"\n--- Sample {i+1} ---")
    print("INPUT:\n", code)
    print("OUTPUT:\n", generate(base_model, prompt))

# --- Swap in LoRA adapter ---
print("\nLoading fine-tuned adapter...")
ft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
ft_model.eval()

print("\n" + "="*60)
print("FINE-TUNED MODEL OUTPUT")
print("="*60)
for i, code in enumerate(test_samples):
    prompt = build_prompt(code)
    print(f"\n--- Sample {i+1} ---")
    print("INPUT:\n", code)
    print("OUTPUT:\n", generate(ft_model, prompt))


# To pick first 3 samples from test data and use it in test_samples list

# import json

# with open("data/splits/test.json") as f:
#     test_data = json.load(f)

# # Print first 3 raw inputs to use as test samples
# for entry in test_data[:3]:
#     print(entry["text"])
#     print("---")