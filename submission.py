import os
import sys
import time
from pathlib import Path
import requests
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50

"""
Submission script for the Robustness task.

Submission Requirements (read carefully to avoid automatic rejection):

1. FILE FORMAT
----------------
- The file must be a PyTorch state dict saved as a .pt file.
- Save only the state dict, not the full model instance:
      torch.save(model.state_dict(), "model.pt")  # correct
      torch.save(model, "model.pt")               # wrong

2. MODEL ARCHITECTURE
----------------------
- You must specify the model architecture using the model-name field.
- Allowed values: resnet18, resnet34, resnet50
- The architecture must match the state dict you are submitting.

3. MODEL REQUIREMENTS
----------------------
- Input shape must be (1, 3, 32, 32)
- Output shape must be (1, 9)
- The final fc layer must be replaced to output 9 classes

4. EVALUATION
--------------
- Your model must achieve clean accuracy greater than 50% to be accepted.
- Submissions below this threshold will be automatically rejected.
- Score = 0.5 * clean accuracy + 0.5 * robustness accuracy

5. TECHNICAL LIMITS
--------------------
- Only one submission per group every 60 minutes.
- If your submission fails due to an error on your side, the cooldown is 2 minutes.

Your submission will fail if:
- The file is not a valid .pt state dict
- The model-name does not match the submitted state dict
- The output shape is not (1, 9)
- The input shape is not (1, 3, 32, 32)
- Clean accuracy is below 50%
"""

BASE_URL = "http://34.63.153.158"
API_KEY = os.environ.get("TML_API_KEY", "")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", str(Path(SCRIPT_DIR) / "model.pt"))
MODEL_NAME = os.environ.get("MODEL_NAME", "resnet18")

SUBMIT = os.environ.get("SUBMIT", "0") == "1"

TASK_ID = "03-robustness" # donot change
MAX_UPLOAD_ATTEMPTS = 3
CONNECT_TIMEOUT_SECONDS = 120
RESPONSE_TIMEOUT_SECONDS = 1800
MODEL_BUILDERS = {"resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50}


def die(msg):
    print(f"{msg}", file=sys.stderr)
    sys.exit(1)


def validate_model():
    """Fail locally before consuming a submission attempt."""
    try:
        state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not state:
            die("model.pt is not a non-empty state dict.")
        if not all(isinstance(k, str) and torch.is_tensor(v) for k, v in state.items()):
            die("model.pt contains values that are not state-dict tensors.")

        if MODEL_NAME not in MODEL_BUILDERS:
            die(f"Unsupported MODEL_NAME={MODEL_NAME!r}. Use one of {sorted(MODEL_BUILDERS)}.")
        model = MODEL_BUILDERS[MODEL_NAME](weights=None)
        model.fc = nn.Linear(model.fc.in_features, 9)
        model.load_state_dict(state, strict=True)
        model.eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 3, 32, 32))
        if output.shape != (1, 9):
            die(f"Wrong output shape: {tuple(output.shape)}")
        print(f"Local validation passed: strict {MODEL_NAME} load and output shape (1, 9).")
    except Exception as exc:
        die(f"Local model validation failed: {exc}")


if SUBMIT:
    if not API_KEY:
        die("Set TML_API_KEY before submitting.")

    if not os.path.isfile(MODEL_PATH):
        die(f"File not found: {MODEL_PATH}")

    validate_model()

    try:
        resp = None
        for attempt in range(1, MAX_UPLOAD_ATTEMPTS + 1):
            try:
                size_mb = os.path.getsize(MODEL_PATH) / (1024 ** 2)
                print(
                    f"Uploading {size_mb:.1f} MiB "
                    f"(attempt {attempt}/{MAX_UPLOAD_ATTEMPTS})..."
                )
                with open(MODEL_PATH, "rb") as f:
                    files = {
                        "file": (
                            os.path.basename(MODEL_PATH),
                            f,
                            "application/x-pytorch",
                        )
                    }
                    resp = requests.post(
                        f"{BASE_URL}/submit/{TASK_ID}",
                        headers={"X-API-Key": API_KEY},
                        files=files,
                        data={"model_name": MODEL_NAME},
                        timeout=(CONNECT_TIMEOUT_SECONDS, RESPONSE_TIMEOUT_SECONDS),
                    )
                break
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if attempt == MAX_UPLOAD_ATTEMPTS:
                    raise
                wait_seconds = 15 * attempt
                print(f"Upload connection failed: {e}")
                print(f"Retrying in {wait_seconds} seconds...")
                time.sleep(wait_seconds)

        if resp is None:
            die("Upload failed before receiving a server response.")

        try:
            body = resp.json()
        except Exception:
            body = {"raw_text": resp.text}

        if resp.status_code == 413:
            die("Upload rejected: file too large (HTTP 413). Reduce size and try again.")

        resp.raise_for_status()

        print("Successfully submitted.")
        print("Server response:", body)

    except requests.exceptions.RequestException as e:
        detail = getattr(e, "response", None)
        print(f"Submission error: {e}")
        if detail is not None:
            try:
                print("Server response:", detail.json())
            except Exception:
                print("Server response (text):", detail.text)
        sys.exit(1)
