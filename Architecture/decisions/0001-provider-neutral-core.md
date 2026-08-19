# ADR 0001: Provider-neutral core

- **Status:** Accepted
- **Date:** 2026-08-19

## Context

Apple Foundation Models, MLX, and DwarfStar expose different lifecycle,
capability, cache, tokenizer, and streaming semantics. A closed backend enum or
engine-specific chat interface would force every consumer to change with each
provider.

## Decision

`AFMKitCore` defines provider/model descriptors, requests, structured generation
events, errors, lifecycle protocols, type erasure, and an open provider registry.
It does not import an inference engine or Apple Foundation Models. Optional
capabilities are separate protocols.

## Consequences

- Consumers can render one event model across providers.
- New providers register without changing a core enum.
- Not every provider capability can be reduced to a universal field; descriptors,
  optional protocols, and namespaced metadata remain necessary.
- Provider adapters carry translation complexity.

## Alternatives rejected

- A single enum of known backends: closed to third-party providers.
- OpenAI DTOs as the core domain: couples an in-process SDK to one transport.
- Lowest-common-denominator strings: loses reasoning, tool stages, usage, and
  custom segments.
