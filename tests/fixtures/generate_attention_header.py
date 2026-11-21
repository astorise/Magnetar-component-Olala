"""Extract layer-4 attention metadata from the pinned Safetensors header.

Usage: python generate_attention_header.py /path/to/header-with-length-prefix
"""
import hashlib
import json
from pathlib import Path
import sys

SOURCE_HASH = "81df2fdc92cc4faa6c66b9786d2103045c6b466a575b31f97151d64f653882f5"


def main(path):
    source = Path(path).read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_HASH:
        raise ValueError("unexpected Safetensors header digest")
    header = json.loads(source[8:])
    metadata = {name: {"shape": entry["shape"], "dtype": entry["dtype"]}
                for name, entry in header.items() if name.startswith("model.layers.4.mixer.")}
    destination = Path(__file__).with_name("olala-attention-header.json")
    destination.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main(sys.argv[1])
