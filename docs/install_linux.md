# Linux Installation Guide

This guide describes how to install and run HomeAI on Ubuntu 24.04 LTS. The
steps also apply to most Debian-based distributions with small changes to the
package manager commands. For Fedora, Arch, and other Linux flavours, use their
respective package managers but keep the same Python and PostgreSQL versions.

## 1. System preparation

1. Update package metadata and install the Python build tooling used by the
   project:

   ```bash
   sudo apt update
   sudo apt install -y python3 python3-venv python3-pip build-essential
   ```

2. (Optional) Install Git if you plan to clone the repository:

   ```bash
   sudo apt install -y git
   ```

3. Install GPU drivers and CUDA tooling if you intend to run large local models.
   Refer to your GPU vendor's documentation for the exact steps.

## 2. Obtain the source code

Clone the repository (or download a release archive) into your preferred
workspace:

```bash
git clone https://github.com/your-org/HomeAI.git
cd HomeAI
```

## 3. Create a Python environment

HomeAI targets Python 3.10 or newer. Create an isolated virtual environment so
that system packages stay untouched:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

To enable the optional PostgreSQL backend, install the extra dependencies
instead:

```bash
pip install -e .[postgres]
```

For development or contributing, install all tooling with:

```bash
pip install -e .[dev]
```

## 4. Configure model access

HomeAI talks to a local model host that exposes `/api/chat` (and optionally
`/api/generate`). Popular choices on Linux include [Ollama](https://ollama.ai/)
and [lmstudio.ai](https://lmstudio.ai/). Install and configure your preferred
runtime, then set these environment variables:

```bash
export HOMEAI_MODEL_HOST=http://127.0.0.1:11434
export HOMEAI_MODEL_NAME=gpt-oss:20b
export HOMEAI_ALLOWLIST_BASE="$HOME"
```

Adjust the host URL and model name to match the backend you installed.

## 5. Optional: PostgreSQL backend

The application ships with a JSON file store by default. To enable durable
memory, install PostgreSQL 16+ with the `pgvector` and `pg_trgm` extensions.

```bash
sudo apt install -y postgresql postgresql-contrib postgresql-16-pgvector
```

Use the helper script to create the database, role, and tables:

```bash
source .venv/bin/activate
POSTGRES_SUPERUSER=postgres \
POSTGRES_SUPERUSER_PASSWORD='your-postgres-password' \
HOMEAI_DB_PASSWORD='your-app-password' \
python scripts/bootstrap_postgres.py
```

Alternatively, follow the detailed walkthrough in
[`docs/postgresql_setup.md`](postgresql_setup.md) for manual steps. After
provisioning, configure the DSN and storage backend:

```bash
export HOMEAI_PG_DSN=postgresql://homeai:your-app-password@127.0.0.1:5432/homeai
export HOMEAI_STORAGE=pg
```

Apply the optional semantic-search migration if you plan to use embeddings:

```bash
psql "$HOMEAI_PG_DSN" -f migrations/001_create_vector_store.sql
```

## 6. Launch HomeAI

Run the application from the project root with the virtual environment active:

```bash
python homeai_app.py
```

The Gradio UI becomes available at http://127.0.0.1:7860. When using the
PostgreSQL backend, ensure `HOMEAI_STORAGE=pg` and `HOMEAI_PG_DSN` are exported
before starting the app.

## 7. Next steps and automation

- Add the environment exports to your shell profile (e.g., `~/.bashrc`) so they
  persist between sessions.
- Use `systemd` user services or tools like `tmux` to keep the server running in
  the background.
- For scripted deployments, wrap the steps above in an internal shell script.
  Publishing a fully automated installer is possible but requires maintaining
  OS-specific branches for drivers and package names; the documented manual flow
  keeps the setup transparent and debuggable.
