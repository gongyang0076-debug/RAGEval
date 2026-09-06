"""Audit the Git publication candidate set without printing secret values.

Includes tracked files and non-ignored untracked files. Local .env values are
used only for exact-match detection; they are never written to the report.
"""

import json
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True)
    return result.stdout


def main():
    files = sorted(set(git("ls-files", "--cached", "--others", "--exclude-standard", "-z").decode().split("\0")) - {""})
    problems = []
    secrets = []
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip().endswith(("API_KEY", "TOKEN", "SECRET")):
                value = value.strip().strip('"').strip("'")
                if len(value) >= 10:
                    secrets.append(value.encode())
    personal = re.compile(rb"(?<![\w])(?:[A-Za-z]:[\\/]|/(?:Users|home)/)")
    token = re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{32,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)")
    total = 0
    for name in files:
        path = ROOT / name
        if not path.is_file():
            problems.append(f"{name}: file unavailable")
            continue
        size = path.stat().st_size
        total += size
        if size > 25 * 1024 * 1024:
            problems.append(f"{name}: exceeds 25 MiB publication limit")
            continue
        parts = Path(name).parts
        if any(p in {".cache", "__pycache__", ".pytest_cache", "models", "chroma", "chroma_db"} or p.startswith(".venv") for p in parts):
            problems.append(f"{name}: local cache/environment in publication")
        if path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example"):
            problems.append(f"{name}: secret configuration in publication")
            continue
        if path.suffix.lower() in {".safetensors", ".gguf", ".onnx", ".pt", ".pth", ".pyc", ".sqlite3", ".pem", ".key"}:
            problems.append(f"{name}: model, database or credential file")
        if path.suffix.lower() == ".xlsx":
            with zipfile.ZipFile(path) as archive:
                content = b"\n".join(archive.read(item) for item in archive.namelist() if item.endswith(".xml"))
        elif path.suffix.lower() == ".png":
            # Screenshots are visually reviewed; do not mistake compressed bytes for text.
            continue
        else:
            content = path.read_bytes()
        if any(secret in content for secret in secrets) or token.search(content):
            problems.append(f"{name}: possible credential (value suppressed)")
        if personal.search(content):
            problems.append(f"{name}: personal absolute path")
    for name in [".env", ".cache/probe.bin", ".venv-ci/pyvenv.cfg", "models/probe.safetensors",
                 ".streamlit/secrets.toml", "artifacts/new-run/latest_report.json"]:
        result = subprocess.run(["git", "check-ignore", "-q", "--no-index", name], cwd=ROOT, capture_output=True)
        if result.returncode != 0:
            problems.append(f"{name}: required ignore rule missing")
    for name in ["data/corpus/ecommerce_v1.json", "data/datasets/ecommerce_eval_v1.json"]:
        if git("hash-object", "--path=" + name, name) != git("hash-object", "--no-filters", name):
            problems.append(f"{name}: Git filters would change dataset version bytes")
    print(json.dumps({"status": "FAIL" if problems else "PASS", "candidate_files": len(files),
                      "total_bytes": total, "findings": problems}, ensure_ascii=False, indent=2))
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
