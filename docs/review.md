# Code and Documentation Review

## Code issues

1. **Event log preview helper ignores its own truncation logic.**
   The `_preview_for_log` helper claims to return a shortened string for
   logging, yet it currently returns the original `payload` object instead of
   the trimmed `text` that it computes. This regressed behaviour (note the
   commented-out truncation helpers) means large request/response payloads are
   inserted into the event log verbatim, making the log noisy and defeating the
   purpose of the helper. The docstring and return type annotation also become
   incorrect. Restoring the intended `return text` (optionally with
   `_shorten_middle`) would fix this.【F:homeai_app.py†L308-L315】

2. **HTTP client does not reuse connections despite documentation claims.**
   `LocalModelEngine` issues bare `requests.post(...)` calls for every chat or
   generate request. Without a shared `requests.Session`, TLS handshakes and TCP
   setup are repeated on each call, which the README explicitly claims was
   avoided. Introducing a session object (and updating the README) would reduce
   latency when talking to local model servers.【F:homeai/model_engine.py†L20-L104】

## Documentation issues

1. **Editable install instructions cannot work with the current packaging
   config.** The Quick Start tells users to run `pip install -e .`, but the
   project’s `pyproject.toml` explicitly sets `packages = []`. As a result the
   `homeai` package and scripts will not be installed, breaking the documented
   workflow. Configure setuptools to include the package directory (e.g. via
   `packages = ["homeai"]` or `packages = {"find": {}}`).【F:README.md†L33-L40】【F:pyproject.toml†L1-L25】

2. **Optional PostgreSQL dependencies are marked as mandatory.** The README
   stresses that PostgreSQL support (and `psycopg`) is optional, yet the base
   dependency list always installs both `psycopg[binary]` and `psycopg_pool`.
   This increases install size and fails on systems without libpq even when the
   database backend is unused. Moving those requirements into an optional extra
   would align the packaging with the documentation.【F:README.md†L11-L27】【F:pyproject.toml†L10-L25】

3. **Troubleshooting section refers to an optimisation that is not present.**
   The README claims the engine already reuses “a single HTTP session”, but the
   implementation sends ad-hoc `requests.post` calls. Update the text once the
   session reuse is implemented (or revise the claim).【F:README.md†L227-L232】【F:homeai/model_engine.py†L20-L104】

