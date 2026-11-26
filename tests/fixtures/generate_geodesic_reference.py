"""Run the pinned upstream OlalaGeodesicNorm with synthetic vectors.

Usage: python generate_geodesic_reference.py path/to/modeling_olala.py
Requires PyTorch 2.15.0. The source SHA-256 is checked before executing the
upstream class definition. No model weights are included.
"""

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import torch

SOURCE_HASH = "357c01b34490ee8650a42a8acdd38bc186bd35863edbbeb4e2bf0746e9b31bcf"


def load_upstream_class(path):
    source = Path(path).read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_HASH:
        raise ValueError("upstream source digest differs from the pinned version")
    tree = ast.parse(source)
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OlalaGeodesicNorm")
    namespace = {"torch": torch, "nn": torch.nn, "OlalaConfig": object}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), namespace)
    return namespace["OlalaGeodesicNorm"]


def main():
    upstream_class = load_upstream_class(sys.argv[1])
    cases = [
        ("ordinary", [1.0, -2.0, 0.5, 4.0], [0.7, 1.2, -1.0, 0.3], 1.2, 0.04, 0),
        ("zero_residual", [0.0, 0.0, 0.0], [1.0, -2.0, 3.0], 1.0, 0.0, 0),
        ("near_zero_residual", [1e-9, -2e-9, 3e-9], [-0.2, 0.4, 0.1], 1.0, 0.0, 3),
        ("collinear", [1.0, -2.0, 3.0], [2.0, -4.0, 6.0], 1.0, 0.0, 2),
        ("zero_branch_negative_bias", [1.0, 2.0, -3.0], [0.0, 0.0, 0.0], 1.0, -0.3, 0),
        ("upper_angle_clamp", [1.0, 0.0, 0.0], [0.0, 100.0, 0.0], 4.0, 3.0, 0),
        ("negative_angle", [1.0, -0.5, 0.25], [0.1, 0.7, 0.2], 0.1, -2.0, 1),
        ("later_layer", [1.0, -2.0, 0.5, 4.0], [0.7, 1.2, -1.0, 0.3], 1.2, 0.04, 11),
    ]
    results = []
    for name, residual, branch, scale, bias, layer_index in cases:
        module = upstream_class(SimpleNamespace(), layer_index).eval()
        with torch.no_grad():
            module.scale.copy_(torch.tensor(scale))
            module.bias.copy_(torch.tensor(bias))
            result = module(torch.tensor(residual).reshape(1, 1, -1),
                            torch.tensor(branch).reshape(1, 1, -1))
        results.append({"name": name, "residual": residual, "branch": branch,
                        "scale": scale, "bias": bias, "layer_index": layer_index,
                        "output": result.reshape(-1).tolist()})
    destination = Path(__file__).with_name("olala-geodesic-reference.json")
    destination.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
