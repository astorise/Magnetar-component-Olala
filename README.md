# magnetar-component-olala

Olala model component for [Magnetar](https://github.com/astorise/Magnetar).

## Status

The initial Rust library reads Olala's ordered Mamba-3 and differential TPA
attention layer layout from a Hugging Face configuration. It also contains
independent CPU references for the sigmoid expert router, with biased
selection and unbiased combination weights, and for learned geodesic residual
updates. Integration tests exercise the published Olala mini configuration and
the actual upstream PyTorch routing and geodesic code at a pinned revision,
plus malformed inputs. A separate CPU reference evaluates the preprocessed
Mamba-3 MIMO recurrence with explicit angle, SSM, previous-key and previous-input
states; its synthetic prefill, continuation and reset cases match pinned
upstream PyTorch. The library does not load weights or implement the complete
Mamba-3 layer, graph production, Runtime-managed recurrent state, Component
Model exports, or inference; Olala is not yet runnable in Magnetar.

The library can also validate the checkpoint's no-RoPE differential attention
settings and derive head grouping, score scale, window and the portable CPU
reference's query/history size limits. It lists the expected checkpoint tensors
in operator input order, including the matrix transpositions required on load;
their names, shapes and storage dtypes were checked against the pinned header.
This validates a configuration profile;
it does not make the checkpoint executable. In particular, the CPU graph's
synthetic f32 projection fixture is not a qualification of real bfloat16
checkpoint inference.

## Work required

Mamba-3 recurrent state, hybrid attention, mixture-of-experts routing and
checkpoint-specific loading remain to be implemented. The necessary Magnetar
state and graph work is tracked in [Magnetar #213](https://github.com/astorise/Magnetar/issues/213),
Provider-side routing in [Magnetar #216](https://github.com/astorise/Magnetar/issues/216),
geodesic updates in [Magnetar #220](https://github.com/astorise/Magnetar/issues/220),
and differential tensor-product attention in
[Magnetar #222](https://github.com/astorise/Magnetar/issues/222).

A real implementation must identify a specific checkpoint and its license, integrate the relevant Magnetar component and runtime contracts, and validate outputs against an independent reference. See the [upstream model or project](https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B). Model weights are not included.

This repository contains a model component, not a hardware Provider. It is intended to be checked out by Magnetar at `components/olala`.
