"""Collect continuous public text and tokenize with the existing CPU container."""

import array
import hashlib
import json
import re
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREP = ROOT.parents[1]
MODEL = Path("/home/intel/LLM/MiniCPM-V-4.7-35B-A3B-1008")
CONTAINER = "codex-minicpm47-bf16-build-20261008"
SOURCE = "https://www.gutenberg.org/cache/epub/2600/pg2600.txt"
IMAGE = "sha256:c1ae7c6509f6cda9341c15fd947da0000faa4a5d437af651d7f5018689818566"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    inspected = json.loads(
        subprocess.check_output(["docker", "inspect", CONTAINER], text=True)
    )[0]
    assert inspected["State"]["Running"]
    assert inspected["Image"] == IMAGE
    assert not inspected["HostConfig"]["Devices"]
    raw = ROOT / "pg2600.raw.txt"
    response_metadata = ROOT / "download.json"
    if not raw.exists():
        request = urllib.request.Request(
            SOURCE, headers={"User-Agent": "MiniCPM-long-text-input-preparation"}
        )
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = response.read()
            metadata = {
                "url": SOURCE,
                "resolved_url": response.url,
                "http_status": response.status,
                "headers": dict(response.headers),
                "downloaded_utc": datetime.now(timezone.utc).isoformat(),
            }
        raw.write_bytes(payload)
        response_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    text = raw.read_bytes().decode("utf-8-sig").replace("\r\n", "\n")
    start = re.search(r"(?m)^\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*\*\*\*", text)
    end = re.search(r"(?m)^\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*\*\*\*", text)
    assert start and end and end.start() > start.end()
    licensed_content = text[start.end() : end.start()]
    narrative = re.search(
        r"(?m)^BOOK ONE: 1805[ \t]*\n\s*CHAPTER I[ \t]*\n", licensed_content
    )
    assert narrative, "Cannot locate the actual first chapter after the contents"
    body = licensed_content[narrative.start() :].strip() + "\n"
    body_path = ROOT / "war-and-peace.body.txt"
    body_path.write_text(body, encoding="utf-8")
    container_path = "/workspace/preparation/" + str(body_path.relative_to(PREP))
    code = """
import array, importlib.metadata, json, sys
from pathlib import Path
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_str(sys.stdin.read())
tokenizer.no_truncation()
tokenizer.no_padding()
text = Path(sys.argv[1]).read_text()
encoding = tokenizer.encode(text, add_special_tokens=False)
sys.stderr.write(json.dumps({
    'tokenizers_version': importlib.metadata.version('tokenizers'),
    'tokens': len(encoding.ids), 'characters': len(text),
    'first_token_offsets': encoding.offsets[:3],
    'last_token_offsets': encoding.offsets[-3:],
    'tensor_device': 'none; CPU tokenizer only',
}) + '\\n')
ids = array.array('I', encoding.ids)
assert ids.itemsize == 4
if sys.byteorder != 'little':
    ids.byteswap()
sys.stdout.buffer.write(ids.tobytes())
"""
    token_output = subprocess.run(
        [
            "docker",
            "exec",
            "-i",
            CONTAINER,
            "/opt/venv/bin/python",
            "-c",
            code,
            container_path,
        ],
        input=(MODEL / "tokenizer.json").read_bytes(),
        capture_output=True,
        check=True,
    )
    counts = json.loads(token_output.stderr)
    ids = array.array("I")
    ids.frombytes(token_output.stdout)
    if sys.byteorder != "little":
        ids.byteswap()
    assert counts["tokens"] == len(ids) > 262144
    tokens_path = ROOT / "war-and-peace.tokens.u32le"
    tokens_path.write_bytes(token_output.stdout)
    manifest = {
        "title": "War and Peace",
        "author": "Leo Tolstoy",
        "translators": ["Aylmer Maude", "Louise Maude"],
        "language": "English",
        "source_page": "https://www.gutenberg.org/ebooks/2600",
        "text_url": SOURCE,
        "source_copyright_label": "Public domain in the USA",
        "raw": {"path": raw.name, "bytes": raw.stat().st_size, "sha256": digest(raw)},
        "body": {
            "path": body_path.name,
            "bytes": body_path.stat().st_size,
            "characters": len(body),
            "sha256": digest(body_path),
            "normalization": "UTF-8 BOM removed; CRLF converted to LF; source header, contents and license footer excluded; continuous narrative retained",
        },
        "tokenization": {
            **counts,
            "add_special_tokens": False,
            "padding": False,
            "truncation": False,
            "includes_chat_template": False,
            "token_ids_path": tokens_path.name,
            "token_ids_encoding": "uint32 little-endian",
            "token_ids_sha256": digest(tokens_path),
            "tokenizer_source": str(MODEL / "tokenizer.json"),
            "tokenizer_sha256": digest(MODEL / "tokenizer.json"),
            "tokenizer_config_sha256": digest(MODEL / "tokenizer_config.json"),
            "cpu_container_id": inspected["Id"],
            "image_id": inspected["Image"],
        },
        "input_length_requirement_verified": {
            "characters_exceed_262144": len(body) > 262144,
            "model_tokens_exceed_262144": len(ids) > 262144,
        },
        "benchmark_execution": "not started",
        "pending_choices": [
            "cutoff unit",
            "prefix/window and continuation/question-answering",
            "physical GPU permission for trace collection",
        ],
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "body_characters": len(body),
                "body_model_tokens": len(ids),
                "manifest": str(ROOT / "manifest.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
