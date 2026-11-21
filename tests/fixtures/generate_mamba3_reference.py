"""Generate a preprocessed Mamba-3 recurrence fixture from pinned upstream.

Usage: python generate_mamba3_reference.py /path/to/modeling_olala.py
The source must match the recorded SHA-256. No checkpoint weights are used.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

SOURCE_HASH = "357c01b34490ee8650a42a8acdd38bc186bd35863edbbeb4e2bf0746e9b31bcf"


def upstream(path):
    source = Path(path).read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_HASH:
        raise ValueError("upstream source digest differs from the pinned version")
    names = {"_mamba3_rotate", "_mamba3_mimo_torch_segment",
             "_mamba3_rotary_step_torch", "_mamba3_step_torch"}
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError("missing upstream Mamba-3 functions")
    namespace = {"torch": torch, "F": F, "math": math}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(path), "exec"), namespace)
    return namespace


def flat(value):
    return value.detach().reshape(-1).tolist()


def case(functions, heads, groups, rank, width, state_dim, angles):
    torch.manual_seed(9041 + groups)
    t = 5
    random = lambda *shape: torch.randn(*shape, dtype=torch.float32) * 0.4
    q = random(1, t, rank, groups, state_dim)
    k = random(1, t, rank, groups, state_dim)
    x = random(1, t, heads, width)
    z = random(1, t, heads, width)
    a = -torch.rand(1, t, heads) - 0.1
    dt = torch.rand(1, t, heads) * 0.4 + 0.05
    trap_logits = random(1, t, heads)
    angle_projection = random(1, t, heads, angles)
    q_bias = random(heads, rank, state_dim)
    k_bias = random(heads, rank, state_dim)
    x_proj = random(heads, rank, width)
    z_proj = random(heads, rank, width)
    out_proj = random(heads, rank, width)
    skip = random(heads)
    y_full, final = functions["_mamba3_mimo_torch_segment"](
        q, k, x, (a * dt).transpose(1, 2), dt.transpose(1, 2),
        trap_logits.transpose(1, 2), q_bias, k_bias, x_proj, z_proj,
        out_proj, angle_projection, skip, z,
    )
    angle_state = torch.zeros(1, heads, angles)
    ssm_state = torch.zeros(1, heads, width, state_dim)
    k_state = torch.zeros(1, rank, heads, state_dim)
    v_state = torch.zeros(1, heads, width)
    steps = []
    tokens = []
    for i in range(t):
        expanded_q = q[:, i].repeat_interleave(heads // groups, dim=2)
        expanded_k = k[:, i].repeat_interleave(heads // groups, dim=2)
        q_rot, k_rot, angle_state = functions["_mamba3_rotary_step_torch"](
            expanded_q, expanded_k, angle_state, angle_projection[:, i], dt[:, i],
            q_bias.transpose(0, 1), k_bias.transpose(0, 1),
        )
        trap = torch.sigmoid(trap_logits[:, i])
        output, ssm_state = functions["_mamba3_step_torch"](
            ssm_state, k_state, v_state, a[:, i], k_rot, q_rot, skip, x[:, i],
            dt[:, i], trap, x_proj.transpose(0, 1), out_proj.transpose(0, 1),
            z[:, i], z_proj.transpose(0, 1),
        )
        k_state = k_rot
        v_state = x[:, i]
        tokens.append({"query": flat(q[:, i]), "key": flat(k[:, i]),
                       "input": flat(x[:, i]), "gate": flat(z[:, i]),
                       "decay": flat(a[:, i]), "dt": flat(dt[:, i]),
                       "trap": flat(trap), "angle_projection": flat(angle_projection[:, i])})
        steps.append({"output": flat(output), "angle": flat(angle_state),
                      "ssm": flat(ssm_state), "key": flat(k_state), "input": flat(v_state)})
    return {
        "geometry": {"heads": heads, "groups": groups, "rank": rank,
                     "head_dim": width, "state_dim": state_dim, "angles": angles},
        "parameters": {"query_bias": flat(q_bias.transpose(0, 1)),
                       "key_bias": flat(k_bias.transpose(0, 1)),
                       "input_projection": flat(x_proj.transpose(0, 1)),
                       "gate_projection": flat(z_proj.transpose(0, 1)),
                       "output_projection": flat(out_proj.transpose(0, 1)),
                       "skip": flat(skip)},
        "tokens": tokens,
        "steps": steps,
        "prefill_output": flat(y_full),
        "prefill_angle": flat(final[0]), "prefill_ssm": flat(final[1]),
        "prefill_key": flat(final[2]), "prefill_input": flat(final[3]),
    }


if __name__ == "__main__":
    functions = upstream(sys.argv[1])
    cases = [case(functions, 2, 1, 2, 3, 4, 2),
             case(functions, 4, 2, 2, 2, 6, 2)]
    destination = Path(__file__).with_name("olala-mamba3-reference.json")
    destination.write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
    print(destination)
