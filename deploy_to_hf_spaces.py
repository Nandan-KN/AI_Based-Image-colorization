"""
deploy_to_hf_spaces.py
======================
One-shot deployment of the AI Colorization app to Hugging Face Spaces.
No checkpoint is uploaded — the Space runs with random model weights.

Steps:
  1. Login check (HF token from env or cached login)
  2. Create (or reuse) a Space repo (Gradio SDK)
  3. Upload: app.py, model.py, dataset.py, requirements.txt, README.md
  4. Print the public URL

Usage:
    python deploy_to_hf_spaces.py --username YOUR_HF_USERNAME

Authentication (choose one):
    huggingface-cli login          # interactive, stores token
    $env:HF_TOKEN = "hf_xxx..."   # PowerShell env var
    HF_TOKEN=hf_xxx python ...    # bash env var
"""

import os
import sys
import io
import textwrap
import argparse
from pathlib import Path

try:
    from huggingface_hub import HfApi, login, whoami, create_repo, upload_file
except ImportError:
    print("[ERROR] huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

PROJECT_DIR = Path(__file__).parent.resolve()

SPACE_FILES = [
    "app.py",
    "model.py",
    "dataset.py",
    "requirements.txt",
]


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def ensure_logged_in() -> str:
    """Returns the logged-in HF username."""
    token = os.environ.get("HF_TOKEN", "")
    if token:
        login(token=token, add_to_git_credential=False)
    try:
        info = whoami()
        print(f"[AUTH] Logged in as: {info['name']}")
        return info["name"]
    except Exception:
        print("[AUTH] Not authenticated. Run one of:")
        print("       huggingface-cli login")
        print("       $env:HF_TOKEN = 'hf_...'   (PowerShell)")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Repo helpers
# ---------------------------------------------------------------------------

def create_or_get_repo(repo_id: str, repo_type: str) -> str:
    kwargs = dict(repo_id=repo_id, repo_type=repo_type,
                  exist_ok=True, private=False)
    if repo_type == "space":
        kwargs["space_sdk"] = "gradio"
    url = create_repo(**kwargs)
    label = "Space" if repo_type == "space" else "Model"
    print(f"[REPO] {label}: {url}")
    return url


def build_readme(space_name: str) -> str:
    title = space_name.replace("-", " ").title()
    return textwrap.dedent(f"""\
        ---
        title: {title}
        emoji: 🎨
        colorFrom: gray
        colorTo: indigo
        sdk: gradio
        sdk_version: 6.29.1
        app_file: app.py
        pinned: false
        license: apache-2.0
        ---

        # AI Image Colorization

        Upload a grayscale image — the **U-Net** model predicts colors using the **LAB color space**.

        > **Note:** This Space runs with **randomly initialized weights** — it demonstrates the
        > colorization pipeline without a pre-trained checkpoint.
        > To colorize with learned colors, train on real photos using `train.py` and upload weights.

        ### Architecture
        - 4-level U-Net encoder-decoder (31M parameters)
        - Input: L channel (grayscale) → Output: predicted ab channels
        - Reconstruction: L + ab → LAB → RGB
        """)


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def upload_space(space_repo_id: str):
    print("[SPACE] Uploading source files ...")
    for fname in SPACE_FILES:
        fpath = PROJECT_DIR / fname
        if not fpath.exists():
            print(f"  [SKIP] {fname} not found")
            continue
        upload_file(
            path_or_fileobj=str(fpath),
            path_in_repo=fname,
            repo_id=space_repo_id,
            repo_type="space",
            commit_message=f"Add {fname}",
        )
        print(f"  [OK]   {fname}")

    # Space card README
    space_name = space_repo_id.split("/")[-1]
    readme     = build_readme(space_name)
    upload_file(
        path_or_fileobj=io.BytesIO(readme.encode()),
        path_in_repo="README.md",
        repo_id=space_repo_id,
        repo_type="space",
        commit_message="Add Space card README",
    )
    print("  [OK]   README.md")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Deploy Colorizer to Hugging Face Spaces")
    parser.add_argument("--username",   type=str, default="",
                        help="HF username (auto-detected if logged in)")
    parser.add_argument("--space-name", type=str, default="ai-image-colorizer",
                        help="Space repo name (default: ai-image-colorizer)")
    args = parser.parse_args()

    detected = ensure_logged_in()
    username  = args.username or detected

    space_repo_id = f"{username}/{args.space_name}"

    print()
    print("=" * 55)
    print(f"  Space  : {space_repo_id}")
    print(f"  Weights: NONE (random init — no checkpoint)")
    print("=" * 55)
    print()

    create_or_get_repo(space_repo_id, repo_type="space")
    upload_space(space_repo_id)

    print()
    print("=" * 55)
    print("[DONE] Deployment complete!")
    print(f"  URL: https://huggingface.co/spaces/{space_repo_id}")
    print("  The Space will build in ~2 min then go live.")
    print("=" * 55)


if __name__ == "__main__":
    main()
