# PrismoRouter

PrismoRouter is the standalone routing engine extracted from Prismo. Give it a
prompt and a list of model candidates, and it selects the cheapest capable model
based on task complexity, safety/risk signals, model tier, context needs,
latency, and optional Elo feedback.

This repository contains only the router. It does not include the Prismo hosted
dashboard, billing, auth, provider credential storage, FinOps workflows,
provider proxy, semantic cache service, or frontend.

## Install

```bash
pip install prismorouter
```

For local development:

```bash
pip install -e ".[dev]"
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

- Uses Prismo's production-derived feature extraction, jailbreak detection,
  complexity detection, heuristic scoring, hybrid scoring, cost-aware ranking,
  latency-aware ranking, and Elo constants.
- Builds a routing profile with a minimum recommended model tier.
- Filters candidates by provider/family scope, capabilities, context window,
  quality preference, and text-chat compatibility.
- Ranks candidates using the same cost/quality formulas as the hosted router,
  with standalone in-memory adapters instead of Prismo's database.
- Returns a transparent routing decision with a score breakdown.

## What It Does Not Do

- It does not call OpenAI, Anthropic, or any other model provider.
- It does not proxy requests.
- It does not store API keys.
- It does not include a database, dashboard, auth system, billing, or FinOps
  enforcement.

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

## License And Attribution

PrismoRouter is licensed under the Apache License, Version 2.0.

Portions of the routing, scoring, complexity detection, prompt-injection
detection, latency, and Elo approaches are adapted from vLLM Semantic Router.
See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
