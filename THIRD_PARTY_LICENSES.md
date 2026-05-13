# Third-Party Notices

This project includes code, algorithms, and implementation ideas adapted from
third-party open-source projects.

## vLLM Semantic Router

Portions of PrismoRouter's request feature extraction, jailbreak/prompt-injection
detection, complexity detection, heuristic scoring, hybrid scoring,
cost-aware ranking, latency-aware metrics, and pairwise Elo rating approach are
adapted from vLLM Semantic Router.

- Source: https://github.com/vllm-project/semantic-router
- License: Apache License, Version 2.0
- Upstream copyright: vLLM Semantic Router contributors

The adapted portions have been modified into a standalone Python library API
using plain dataclasses and in-memory state instead of Prismo's hosted backend,
database models, dashboard, billing, auth, provider credential storage, or
FinOps modules.

