# prismorouter

prismorouter is the routing engine behind prismo, extracted into a standalone
python package.

it takes a prompt and a list of model candidates, then picks the cheapest
model that can handle the request based on complexity, latency, capabilities,
safety checks, and context size.

this repo only includes the router itself. no hosted api, dashboard, auth,
billing, proxy infrastructure, or provider key management.

## why

most apps send every prompt to the same model, which gets expensive fast.

prismorouter routes simple prompts to cheaper models and keeps harder requests
on stronger ones.

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

## what it does

- analyzes prompts for complexity and risk
- filters models by capabilities and context window
- ranks candidates using cost, latency, and quality signals
- returns a routing decision with scoring details

## what it does not do

- make provider api calls
- proxy requests
- store api keys
- handle billing, auth, or dashboard logic

## development

```bash
pip install -e ".[dev]"
pytest -q
```

## license and attribution

prismorouter is licensed under the apache license, version 2.0.

parts of the routing, scoring, complexity detection, prompt injection
detection, latency, and elo logic were adapted from vllm semantic router.

see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
