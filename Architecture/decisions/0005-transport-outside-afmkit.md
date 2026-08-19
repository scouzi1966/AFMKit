# ADR 0005: Transport and server concerns stay outside AFMKit

- **Status:** Accepted
- **Date:** 2026-08-19

## Context

maclocal-api exposes HTTP, OpenAI-compatible routes, WebUI, CLI, metrics, and
request orchestration. Embedding those concerns in AFMKit would couple an
in-process Swift SDK to Vapor and one deployment model.

## Decision

AFMKit provides neutral models/events plus `AFMOpenAICompat` serializable DTOs.
Consumers own HTTP routing, SSE framing, status mapping, Prometheus/vLLM/GuideLLM
metrics, files, authentication, rate limits, and request IDs.

## Consequences

- GUI apps and services can use the same provider SDK.
- Transport adapters must map cancellation and streaming carefully.
- Engine-neutral telemetry belongs in core; wire-format metrics belong in the
  server layer.
- OpenAI compatibility can evolve without making HTTP a core dependency.
