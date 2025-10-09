# Windows Installation Guide

These instructions target Windows 11 with administrator access. Windows 10 works
with the same steps as long as you install Python 3.10+ and recent Visual C++
build tools. Commands can be run inside **Windows Terminal** using PowerShell.

## 1. Install prerequisites

1. **Python 3.11+** – install from the Microsoft Store or from
   [python.org](https://www.python.org/downloads/windows/). When using the MSI
   installer, ensure that you check **"Add Python to PATH"**.

2. **Git** – install via [https://git-scm.com/download/win](https://git-scm.com/download/win)
   or with Winget:

   ```powershell
   winget install --id Git.Git -e --source winget
   ```

3. **Microsoft Visual C++ Build Tools** (for compiling Python wheels that ship
   as source). Install from
   [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   and include the "Desktop development with C++" workload.

4. (Optional) **PostgreSQL 16+ with pgvector** – download the EnterpriseDB
   installer from [https://www.postgresql.org/download/windows/](https://www.postgresql.org/download/windows/)
   or install via Winget:

   ```powershell
   winget install --id PostgreSQL.PostgreSQL -e --source winget
   ```

   During installation:

   - Set a memorable superuser password for the `postgres` account.
   - Enable the `pgvector` extension (available in the StackBuilder component
     list). If StackBuilder is not installed, download the Windows pgvector
     binaries from the official release page and copy them into the PostgreSQL
     `lib` and `share/extension` directories.

5. (Optional) **Ollama** – run the official Windows installer from
   [https://ollama.ai/download](https://ollama.ai/download) to host the default
   chat and embedding models locally.

## 2. Clone the repository

Open a new PowerShell session and run:

```powershell
git clone https://github.com/your-org/HomeAI.git
cd HomeAI
```

## 3. Create a virtual environment

Use the Python launcher to create and activate a virtual environment inside the
project:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

To enable PostgreSQL support, install the extras after activation:

```powershell
pip install -e .[postgres]
```

For development tooling, run `pip install -e .[dev]`.

## 4. Configure the model backend

Ensure your local model host is running and reachable. When using Ollama's
default port, export the required environment variables in PowerShell:

```powershell
$env:HOMEAI_MODEL_HOST = "http://127.0.0.1:11434"
$env:HOMEAI_MODEL_NAME = "qwen2.5vl:32b-q4_K_M"
$env:HOMEAI_ALLOWLIST_BASE = "$env:USERPROFILE"
```

If you plan to use embeddings, download the model once:

```powershell
ollama pull mxbai-embed-large
```

## 5. Optional: PostgreSQL configuration

After installing PostgreSQL, open a PowerShell session with the virtual
environment activated and run the bootstrap helper (adjust the passwords to your
values):

```powershell
$env:POSTGRES_SUPERUSER = "postgres"
$env:POSTGRES_SUPERUSER_PASSWORD = "your-postgres-password"
$env:HOMEAI_DB_PASSWORD = "your-app-password"
python scripts/bootstrap_postgres.py
```

The script creates the `homeai` role, database, and the `homeai_memory` table
with indexes. When it finishes, set the DSN and enable the PostgreSQL backend:

```powershell
$env:HOMEAI_PG_DSN = "postgresql://homeai:your-app-password@127.0.0.1:5432/homeai"
$env:HOMEAI_STORAGE = "pg"
```

To install the semantic-search schema:

```powershell
psql "$env:HOMEAI_PG_DSN" -f migrations/001_create_vector_store.sql
```

If `psql` is not on your PATH, add the PostgreSQL `bin` directory (for example,
`C:\\Program Files\\PostgreSQL\\16\\bin`) to the system PATH via System
Properties → Environment Variables.

## 6. Launch HomeAI

Start the application from the project root with the virtual environment active:

```powershell
python homeai_app.py
```

The Gradio UI runs at http://127.0.0.1:7860. Use `Ctrl+C` in the terminal to
stop the server.

## 7. Tips for Windows automation

- Create a PowerShell script that activates the environment, exports the needed
  variables, and launches the app. Pin it to Start or schedule it with Task
  Scheduler for one-click startup.
- For services that should persist in the background, run HomeAI from a Windows
  Subsystem for Linux (WSL) distribution or use third-party tools like
  [nssm](https://nssm.cc/) to wrap the script as a Windows service.
- Publishing a single "downloadable installer" similar to Ollama is feasible but
  requires bundling Python, managing antivirus false positives, and ensuring the
  embedded PostgreSQL binaries stay updated. The documented manual steps keep the
  setup maintainable while the project evolves.
