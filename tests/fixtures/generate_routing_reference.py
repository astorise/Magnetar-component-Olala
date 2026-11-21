"""Run the pinned upstream OlalaMoE.forward with synthetic inputs.

Usage: python generate_routing_reference.py path/to/modeling_olala.py
Requires PyTorch 2.15.0. The source SHA-256 is verified before executing the
upstream class definition. Only routing inputs and outputs are written; no
model weights are included.
"""

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import torch

SOURCE_HASH = "357c01b34490ee8650a42a8acdd38bc186bd35863edbbeb4e2bf0746e9b31bcf"


class CaptureExperts(torch.nn.Module):
    def forward(self, x, probabilities, indices):
        self.probabilities = probabilities.detach().clone()
        self.indices = indices.detach().clone()
        return torch.zeros_like(x)


def load_upstream_class(path):
    source = Path(path).read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_HASH:
        raise ValueError("upstream source digest differs from the pinned version")
    tree = ast.parse(source)
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OlalaMoE")
    namespace = {"torch": torch, "nn": torch.nn, "OlalaConfig": object}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), namespace)
    return namespace["OlalaMoE"]


def main():
    upstream_class = load_upstream_class(sys.argv[1])
    logits_256 = [((i * 37) % 257 - 128) / 37.0 for i in range(256)]
    bias_256 = [((i * 17) % 47 - 23) / 113.0 for i in range(256)]
    cases = [
        ("bias_changes_selection", [0.0, 2.0, -1.0, 0.4, -2.0], [0.0, -0.9, 0.7, 0.15, 0.1], 2, 2.45),
        ("top_one_is_not_normalized", [0.3, -0.5, 2.0], [0.0, 1.0, -1.0], 1, 2.45),
        ("published_expert_count_and_top_k", logits_256, bias_256, 6, 2.45),
        ("saturated_zero_scores", [-100.0, -101.0, -102.0], [0.0, 0.1, 0.2], 2, 1.0),
    ]
    results = []
    for name, logits, bias, top_k, scale in cases:
        count = len(logits)
        instance = upstream_class.__new__(upstream_class)
        torch.nn.Module.__init__(instance)
        instance.config = SimpleNamespace(moe_num_active_experts=top_k, moe_routed_scaling_factor=scale,
                                          moe_routed_input_dim=0, moe_shared_expert_gate=False)
        instance.moe_gate = torch.nn.Linear(count, count, bias=False)
        with torch.no_grad():
            instance.moe_gate.weight.copy_(torch.eye(count))
        instance.expert_bias = torch.tensor(bias, dtype=torch.float32)
        instance.experts = CaptureExperts()
        instance.shared_experts = None
        instance.eval()
        with torch.no_grad():
            instance(torch.tensor(logits, dtype=torch.float32).reshape(1, 1, count))
        results.append({"name": name, "logits": logits, "selection_bias": bias,
                        "top_k": top_k, "routed_scaling": scale,
                        "indices": instance.experts.indices[0].tolist(),
                        "weights": instance.experts.probabilities[0].tolist()})
    destination = Path(__file__).with_name("olala-routing-reference.json")
    destination.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
