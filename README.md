# PrismoRouter

PrismoRouter is the routing engine behind Prismo, extracted into a standalone
Python package.

It takes a prompt and a list of model candidates, then selects the cheapest
model that can handle the request based on things like complexity, latency,
capabilities, safety signals, and context size.

This repo only contains the router itself. No hosted API, dashboard, auth,
billing, proxy infrastructure, or provider key management.

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

- Analyzes prompts for complexity and risk.
- Filters models by capabilities and context window.
- Ranks candidates using cost, latency, and quality signals.
- Returns a routing decision with scoring details.

## What It Does Not Do

- Make provider API calls.
- Proxy requests.
- Store API keys.
- Handle billing, auth, or dashboard logic.

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
