import os
os.environ["PYTHONUTF8"] = "1"

from huggingface_hub import HfApi

api = HfApi()

# Create repo
api.create_repo(
    repo_id="C0ffeeMachine/java-review-agent",
    exist_ok=True
)

# Upload adapter
api.upload_folder(
    folder_path="adapter/final",
    repo_id="C0ffeeMachine/java-review-agent",
    repo_type="model"
)

print("Done! https://huggingface.co/C0ffeeMachine/java-review-agent")