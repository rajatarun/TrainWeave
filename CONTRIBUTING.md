# Contributing to TrainWeave

Thanks for your interest. This project orchestrates **real GPU training on AWS**;
please keep changes small, tested, and honest about what requires cloud resources.

## Development setup

```bash
git clone https://github.com/YOUR_ORG/TrainWeave.git
cd TrainWeave
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt pytest
```

Optional: install SAM CLI when working on `template.yaml` (not required for unit tests).

## Running tests

```bash
python3 -m pytest -q
```

Tests use `pytest.ini` to put `src/orchestrator` and `src/training` on `PYTHONPATH`.
No AWS credentials are required.

Offline DPO conversion smoke test:

```bash
python3 scripts/convert_dpo_local.py --input examples/dpo-records --output /tmp/tw-out
```

Synthetic conversion benchmark (optional):

```bash
python3 scripts/bench_dpo_conversion.py
```

## Style

- Python 3.12+ syntax in orchestrator code (matches Lambda runtime).
- Prefer explicit errors in `app.handler` for misconfigured events.
- Match existing logging patterns (`logger.info` with structured fields).
- Shell in `bootstrap.sh`: `set -euo pipefail`.

## Pull requests

1. Fork and branch from `main`.
2. Add or update tests for Python changes.
3. Run `pytest` locally.
4. Update docs when behavior or parameters change.
5. Do not commit secrets, account IDs, or personal emails in **docs** (existing
   workflow defaults for the owner's CI may remain unchanged).

Use the PR template checklist.

## Deploying your fork (optional)

Training costs money. Use `sam deploy` in a dev account and set `skip_training: true`
in the GitHub workflow when testing infrastructure only.

See `README.md` for parameters and `docs/design/architecture.md` for topology.

## Code of conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Security

See [SECURITY.md](SECURITY.md).
