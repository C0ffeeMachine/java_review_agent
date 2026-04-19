import json
import re

def extract_hunk_context(patch: str) -> list[str]:
    """Extract only the actual code lines from diff hunks, ignoring headers."""
    lines = patch.split('\n')
    code_lines = []
    in_hunk = False

    for line in lines:
        if line.startswith('@@'):
            in_hunk = True
            continue
        if not in_hunk:
            continue
        # These are actual code lines
        if line.startswith('-') or line.startswith('+') or line.startswith(' '):
            code_lines.append(line)

    return code_lines


def build_buggy_and_fixed(patch: str) -> tuple[str, str]:
    hunk_lines = extract_hunk_context(patch)

    buggy = []
    fixed = []

    for line in hunk_lines:
        if line.startswith('-'):
            buggy.append(line[1:])   # removed line → only in buggy
        elif line.startswith('+'):
            fixed.append(line[1:])   # added line → only in fixed
        else:
            # context line → appears in both
            buggy.append(line[1:])
            fixed.append(line[1:])

    return '\n'.join(buggy).strip(), '\n'.join(fixed).strip()


def convert_entry(entry: dict) -> dict | None:
    try:
        before = entry["before"]
        after = entry["after"]
        bug_type = entry["bugType"]
        file_name = entry["commitFile"].split("/")[-1]

        buggy_code, fixed_code = build_buggy_and_fixed(entry["patch"])

        # Fallback: if patch parsing still gives garbage, use before/after directly
        if not buggy_code or len(buggy_code) < 5:
            buggy_code = before
            fixed_code = after

        output = (
            f"Bug Type: {bug_type}\n\n"
            f"Explanation: In `{file_name}`, the expression `{before}` is incorrect. "
            f"It should be `{after}`. This is a {bug_type} error.\n\n"
            f"Fixed Code:\n{fixed_code}"
        )

        return {
            "instruction": (
                "You are a Java code reviewer. Review the following Java code, "
                "identify the bug, explain why it is wrong, and provide the corrected version."
            ),
            "input": buggy_code,
            "output": output
        }

    except Exception as e:
        print(f"Skipping: {e}")
        return None


# --- Run ---
with open("data\\raw\\sstubs.json","r",encoding="utf-8") as f:
    raw_data = json.load(f)

converted = [convert_entry(e) for e in raw_data]
converted = [e for e in converted if e is not None]
converted = [e for e in converted if len(e["input"]) < 1500]

# Deduplicate
seen = set()
unique = []
for e in converted:
    if e["input"] not in seen:
        seen.add(e["input"])
        unique.append(e)

print(f"Final dataset size: {len(unique)}")

with open("data\\processed\\java_review_dataset.json", "w", encoding="utf-8") as f:
    json.dump(unique, f, indent=2)


# with open("data\\raw\\sstubs.json","r",encoding="utf-8") as f:
#     raw = json.load(f)

# # Test on first 5 entries
# for entry in raw[:5]:
#     buggy, fixed = build_buggy_and_fixed(entry["patch"])
#     print("BEFORE field:", entry["before"])
#     print("AFTER field :", entry["after"])
#     print("=== BUGGY (from patch) ===")
#     print(buggy[:400])
#     print("=== FIXED (from patch) ===")
#     print(fixed[:400])
#     print("---")



from datasets import Dataset
import json

def format_for_sft(entry: dict) -> dict:
    return {
        "text": f"""### Instruction:
{entry['instruction']}

### Input:
{entry['input']}

### Response:
{entry['output']}"""
    }

def load_and_split(path: str, test_size: float = 0.1):
    with open(path) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples from {path}")

    formatted = [format_for_sft(e) for e in data]

    hf_dataset = Dataset.from_list(formatted)
    hf_dataset = hf_dataset.train_test_split(test_size=test_size, seed=42)

    return hf_dataset


if __name__ == "__main__":
    # Step 1: Convert raw bugs.json → instruction pairs (your existing code)
    # Assumes unique[] is already built from your conversion logic above
    with open("data/processed/java_review_dataset.json") as f:
        unique = json.load(f)

    # Step 2: Save splits to disk
    dataset = load_and_split("data/processed/java_review_dataset.json")

    train_data = [entry for entry in dataset["train"]]
    test_data = [entry for entry in dataset["test"]]

    with open("data/splits/train.json", "w") as f:
        json.dump(train_data, f, indent=2)

    with open("data/splits/test.json", "w") as f:
        json.dump(test_data, f, indent=2)

    print(f"Train size: {len(train_data)}")
    print(f"Test size : {len(test_data)}")

    # Step 3: Sanity check — print one sample
    print("\n=== Sample Training Entry ===")
    print(dataset["train"][0]["text"])
