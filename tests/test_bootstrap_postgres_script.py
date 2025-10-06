import os
import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "bootstrap_postgres.py"


def run_script(env):
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        capture_output=True,
        text=True,
        env=env,
    )


def test_bootstrap_requires_psycopg(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path)

    psycopg_pkg = tmp_path / "psycopg"
    psycopg_pkg.mkdir()
    (psycopg_pkg / "__init__.py").write_text("raise ImportError('psycopg missing')\n")

    result = run_script(env)

    assert result.returncode != 0
    assert "psycopg package is required" in result.stderr


def test_bootstrap_dry_run_reports_configuration(tmp_path):
    schema_file = tmp_path / "schema.sql"
    schema_file.write_text("SELECT 1;\n")

    env = os.environ.copy()
    env["HOMEAI_BOOTSTRAP_DRY_RUN"] = "1"
    env["HOMEAI_DB_NAME"] = "homeai_dev"
    env["HOMEAI_DB_USER"] = "homeai_app"
    env["HOMEAI_SCHEMA_FILE"] = str(schema_file)

    result = run_script(env)

    assert result.returncode == 0, result.stderr
    stdout = result.stdout
    assert "DRY RUN" in stdout
    assert "homeai_dev" in stdout
    assert "homeai_app" in stdout
    assert str(schema_file) in stdout
