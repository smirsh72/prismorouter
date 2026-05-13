# PrismoRouter

PrismoRouter is a small Python library for cost-aware LLM model routing. Give it
a prompt and a list of model candidates, and it selects the cheapest capable
model based on task complexity, safety/risk signals, model tier, context needs,
latency, and optional Elo feedback.

This repository contains only the routing engine. It does not include the Prismo
hosted dashboard, billing, auth, provider credential storage, FinOps workflows,
or frontend.

## Install

```bash
pip install prismorouter
```

For local development:

```bash
pip install -e .
```

## Quick Start

```python
from prismo_router import ModelCandidate, route_request

candidates = [
    ModelCandidate(
        name="gpt-5-nano",
        provider="openai",
        family="gpt-5",
        tier="nano",
        input_price=0.0001,
        output_price=0.0004,
        context_window=128_000,
    ),
    ModelCandidate(
        name="gpt-5-mini",
        provider="openai",
        family="gpt-5",
        tier="mini",
        input_price=0.0003,
        output_price=0.0012,
        context_window=128_000,
        supports_tools=True,
    ),
    ModelCandidate(
        name="gpt-5",
        provider="openai",
        family="gpt-5",
        tier="base",
        input_price=0.010,
        output_price=0.030,
        context_window=128_000,
        supports_tools=True,
        supports_vision=True,
    ),
]

decision = route_request(
    prompt="Write a Python function that validates an email address.",
    requested_model="gpt-5",
    candidates=candidates,
)

print(decision.selected_model)
print(decision.reason)
```

## What It Does

- Extracts task, structure, domain-risk, token, and safety signals from a prompt.
- Builds a routing profile with a minimum recommended model tier.
- Filters candidates by provider/family scope, capabilities, context window, and
  quality preference.
- Ranks candidates by reliability, cost, optional latency, and optional Elo
  feedback.
- Returns a transparent routing decision with a score breakdown.

## What It Does Not Do

- It does not call OpenAI, Anthropic, or any other model provider.
- It does not proxy requests.
- It does not store API keys.
- It does not include a database, dashboard, auth system, billing, or FinOps
  enforcement.

## License And Attribution

PrismoRouter is licensed under the Apache License, Version 2.0.

Portions of the routing, scoring, complexity detection, prompt-injection
detection, latency, and Elo approaches are adapted from vLLM Semantic Router.
See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

