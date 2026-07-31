"""Pre-download the embedding model into the local HuggingFace cache.

Run this once, before the first `streamlit run`. Downloading here rather
than on first app start keeps the network call out of Streamlit's
rerun/threading model, where a shared huggingface_hub httpx client can be
closed and then reused ("Cannot send a request, as the client has been
closed"). After this succeeds the app loads the model with
local_files_only=True and needs no network at all.

Usage:
    python scripts/download_models.py
"""

from __future__ import annotations

import os
import sys

MODEL_NAME = os.getenv("LOCAL_EMBED_MODEL", "BAAI/bge-small-en-v1.5")


def main() -> int:
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
        print(
            "\nIf this is 'Cannot send a request, as the client has been closed', "
            "the installed huggingface_hub is hitting a known shared-httpx-client "
            "issue. Try:  pip install 'huggingface_hub<1.0'",
            file=sys.stderr,
        )
        return 1

    dim = model.get_sentence_embedding_dimension()
    print(f"Done. {MODEL_NAME} is cached locally ({dim}-dimensional embeddings).")
    print("The app will now load it offline via local_files_only=True.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
