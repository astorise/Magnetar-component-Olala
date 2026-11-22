"""Extract all Mamba-3 tensor metadata from the pinned checkpoint header.

Usage: python tests/fixtures/generate_mamba3_header.py /path/to/header
The input includes the eight-byte Safetensors length prefix; no weights are read.
"""
import hashlib
import json
from pathlib import Path
import sys

SOURCE_HASH = "81df2fdc92cc4faa6c66b9786d2103045c6b466a575b31f97151d64f653882f5"
LAYOUT = "MMMMVMMMMVMMMMVMMMMVMMMMVMMMMVMMMMVM"


def main(path):
    source = Path(path).read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_HASH:
        raise ValueError("unexpected Safetensors header digest")
    header = json.loads(source[8:])
    metadata = {}
    for layer, kind in enumerate(LAYOUT):
        if kind != "M":
            continue
        prefix = f"model.layers.{layer}.mixer."
        tensors = {name: {"shape": entry["shape"], "dtype": entry["dtype"]}
                   for name, entry in header.items() if name.startswith(prefix)}
        if len(tensors) != 11:
            raise ValueError(f"expected eleven mixer tensors for layer {layer}")
        metadata.update(tensors)
    destination = Path(__file__).with_name("olala-mamba3-header.json")
    entries = (f"  {json.dumps(name)}: {json.dumps(metadata[name], sort_keys=True)}"
               for name in sorted(metadata))
    destination.write_text("{\n" + ",\n".join(entries) + "\n}\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main(sys.argv[1])
