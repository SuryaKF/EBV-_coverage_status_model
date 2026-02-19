from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from urllib.parse import urlparse


def save_artifact(local_file: str, model_version: str, artifact_root_uri: str) -> str:
    parsed = urlparse(artifact_root_uri)

    if parsed.scheme in ("", "file"):
        base = Path(parsed.path or ".").resolve()
        target_dir = base / "models" / model_version
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / Path(local_file).name
        shutil.copy2(local_file, target)
        return f"file://{target.as_posix()}"

    if parsed.scheme == "s3":
        raise NotImplementedError("S3 upload is not implemented yet. Use file:// artifact root for now.")

    raise ValueError(f"Unsupported artifact_root_uri scheme: {parsed.scheme}")


def load_artifact_path(artifact_uri: str) -> str:
    parsed = urlparse(artifact_uri)
    if parsed.scheme in ("", "file"):
        return parsed.path if parsed.scheme else artifact_uri
    raise ValueError(json.dumps({"error": "unsupported_artifact_uri", "artifact_uri": artifact_uri}))
