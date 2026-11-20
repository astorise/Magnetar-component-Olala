# magnetar-component-olala

Olala model component for [Magnetar](https://github.com/astorise/Magnetar).

## Status

The initial Rust library reads Olala's ordered Mamba-3 and differential TPA
attention layer layout from a Hugging Face configuration. Integration tests
exercise the published Olala mini configuration at a pinned revision and
malformed inputs. The library does not load weights or implement graph
production, recurrent execution, Component Model exports, or inference; Olala
is not yet runnable in Magnetar.

## Work required

Mamba-3 recurrent state, hybrid attention, mixture-of-experts routing and
checkpoint-specific loading remain to be implemented. The necessary Magnetar
state and graph work is tracked in [Magnetar #213](https://github.com/astorise/Magnetar/issues/213).

A real implementation must identify a specific checkpoint and its license, integrate the relevant Magnetar component and runtime contracts, and validate outputs against an independent reference. See the [upstream model or project](https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B). Model weights are not included.

This repository contains a model component, not a hardware Provider. It is intended to be checked out by Magnetar at `components/olala`.
