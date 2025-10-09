# macOS Installation Guide

The following steps target macOS 13 Ventura or newer on Apple Silicon hardware.
Intel-based Macs can follow the same flow but may require Rosetta 2 for some
third-party binaries. Use Terminal.app or your preferred shell.

## 1. Install command-line tools and Homebrew

1. Install Apple's Command Line Tools (provides Git, compilers, and SDKs):

   ```bash
   xcode-select --install
   ```

2. Install [Homebrew](https://brew.sh/) if it is not already present:

   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

   Follow the on-screen prompts, then add Homebrew to your shell profile. For
   `zsh` (the default shell on macOS):

   ```bash
   echo 'eval "$($(brew --prefix)/bin/brew shellenv)"' >> ~/.zprofile
   eval "$($(brew --prefix)/bin/brew shellenv)"
   ```

## 2. Install prerequisites

Use Homebrew to install Python 3.11 (or newer), Git, and PostgreSQL if you plan
on using the optional database backend:

```bash
brew update
brew install python@3.11 git
```

If you need PostgreSQL with the `vector` extension:

```bash
brew install postgresql@16 pgvector
brew services start postgresql@16
```

Homebrew adds the `pgvector` files automatically. To keep `psql` and
`pg_config` on your path, add the PostgreSQL bin directory to your shell
profile:

```bash
echo 'export PATH="/opt/homebrew/opt/postgresql@16/bin:$PATH"' >> ~/.zprofile
source ~/.zprofile
```

## 3. Obtain the source code

Clone the repository and enter it:

```bash
git clone https://github.com/your-org/HomeAI.git
cd HomeAI
```

## 4. Create a Python virtual environment

Use the Homebrew-installed Python to create and activate a virtual environment:

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Install extras depending on your use case:

- PostgreSQL backend: `pip install -e .[postgres]`
- Development tooling: `pip install -e .[dev]`

## 5. Configure the model backend

macOS users commonly run [Ollama](https://ollama.ai/) or
[LM Studio](https://lmstudio.ai/). Install your preferred runtime and confirm it
serves `http://127.0.0.1:11434` (Ollama's default). Export the required
variables in your shell profile or session:

```bash
export HOMEAI_MODEL_HOST=http://127.0.0.1:11434
export HOMEAI_MODEL_NAME=qwen2.5vl:32b-q4_K_M
export HOMEAI_ALLOWLIST_BASE="$HOME"
```

For embedding support, pull the `mxbai-embed-large` model in Ollama:

```bash
ollama pull mxbai-embed-large
```

## 6. Optional: PostgreSQL configuration

If you installed PostgreSQL via Homebrew, run the bootstrap helper to create the
application role and database:

```bash
source .venv/bin/activate
POSTGRES_SUPERUSER=postgres \
HOMEAI_DB_PASSWORD='your-app-password' \
HOMEAI_PG_DSN=postgresql://homeai:your-app-password@127.0.0.1:5432/homeai \
python scripts/bootstrap_postgres.py
```

Homebrew's PostgreSQL service uses password-less local connections by default,
so you usually do not need to set `POSTGRES_SUPERUSER_PASSWORD`. After running
the script, enable the backend by exporting:

```bash
export HOMEAI_PG_DSN=postgresql://homeai:your-app-password@127.0.0.1:5432/homeai
export HOMEAI_STORAGE=pg
```

Apply the optional vector-store migration if desired:

```bash
psql "$HOMEAI_PG_DSN" -f migrations/001_create_vector_store.sql
```

## 7. Launch HomeAI

From the project root with the virtual environment activated:

```bash
python homeai_app.py
```

The Gradio UI opens at http://127.0.0.1:7860. Stop the server with `Ctrl+C`.

## 8. Notes on automation and packaging

- macOS users can create an Automator or LaunchAgent job that activates the
  virtual environment and starts `homeai_app.py` at login.
- A future standalone installer could bundle Python and dependencies via
  [`pyapp`](https://ofek.dev/pyapp/) or [`shiv`](https://shiv.readthedocs.io/),
  but keeping the Homebrew-based workflow provides transparency and easier
  debugging during active development.
