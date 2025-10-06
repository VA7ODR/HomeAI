import os
import subprocess
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "bootstrap_postgres.sh"


def run_script(env):
    result = subprocess.run(
        ["/bin/bash", str(SCRIPT_PATH)],
        capture_output=True,
        text=True,
        env=env,
    )
    return result


def test_bootstrap_requires_psql(tmp_path):
    env = os.environ.copy()
    env["PATH"] = str(tmp_path)

    result = run_script(env)

    assert result.returncode != 0
    assert "psql command not found" in result.stderr


def test_bootstrap_invokes_psql_with_defaults(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_file = tmp_path / "psql.log"

    stub_script = bin_dir / "psql"
    stub_script.write_text(
        "#!/usr/bin/env bash\n"
        f"echo \"$0 $@\" >> '{log_file}'\n"
        "cat >/dev/null\n"
    )
    stub_script.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env.pop("PGPASSWORD", None)

    result = run_script(env)

    assert result.returncode == 0, result.stderr
    log_contents = log_file.read_text().strip().splitlines()

    assert len(log_contents) == 2
    for line in log_contents:
        assert "--username=postgres" in line
        assert "--host=localhost" in line
        assert "--port=5432" in line
        assert "--dbname=postgres" in line

