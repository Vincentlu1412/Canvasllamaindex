"""Pre-download the embedding model into the local HuggingFace cache.

Run this once, before the first `streamlit run`. Doing the download here
rather than on first app start keeps it out of Streamlit's rerun/threading
model, where a failed download surfaces as the confusing "Cannot send a
request, as the client has been closed" rather than as a timeout. After
this succeeds the app loads the model with local_files_only=True and needs
no network at all.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --mirror   # if huggingface.co is blocked/slow
"""

from __future__ import annotations

import argparse
import os
import sys

MODEL_NAME = os.getenv("LOCAL_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Community mirror of huggingface.co, used where the main host is slow or
# unreachable. HF_ENDPOINT has to be set before huggingface_hub is imported,
# which is why the import below is deferred into main().
MIRROR_ENDPOINT = "https://hf-mirror.com"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mirror",
        action="store_true",
        help=f"download via {MIRROR_ENDPOINT} instead of huggingface.co",
    )
    args = parser.parse_args()

    if args.mirror and not os.getenv("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = MIRROR_ENDPOINT

    endpoint = os.getenv("HF_ENDPOINT", "https://huggingface.co")
    print(f"Endpoint: {endpoint}")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers is not installed. Run: pip install -r requirements.txt", file=sys.stderr)
        return 1

    print(f"Downloading {MODEL_NAME} into the local HuggingFace cache...")
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as exc:
        print(f"\nDownload failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        if endpoint == "https://huggingface.co":
            print(
                f"\nIf huggingface.co is slow or blocked on your network, retry via the mirror:\n"
                f"    python scripts/download_models.py --mirror",
                file=sys.stderr,
            )
        print(
            "\nNote: 'Cannot send a request, as the client has been closed' is "
            "usually a connection timeout surfacing late, not a bug in this "
            "project — check connectivity to the endpoint above first.",
            file=sys.stderr,
        )
        return 1

    dim = model.get_sentence_embedding_dimension()
    print(f"Done. {MODEL_NAME} is cached locally ({dim}-dimensional embeddings).")
    print("The app will now load it offline via local_files_only=True.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
