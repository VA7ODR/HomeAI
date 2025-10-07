#!/usr/bin/env python3
"""Bootstrap a PostgreSQL database for the HomeAI application."""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass
class BootstrapConfig:
    superuser: str
    superuser_db: str
    host: str
    port: str
    superuser_password: Optional[str]
    db_name: str
    db_user: str
    db_password: str
    db_schema: Optional[str]
    memory_table: str
    schema_file: Optional[Path]
    dry_run: bool

    @classmethod
    def from_env(cls, args: argparse.Namespace) -> "BootstrapConfig":
        schema_path: Optional[Path] = None
        schema_env = args.schema or os.environ.get("HOMEAI_SCHEMA_FILE", "").strip()
        if schema_env:
            schema_path = Path(schema_env).expanduser().resolve()

        dry_run = args.dry_run or _env_bool("HOMEAI_BOOTSTRAP_DRY_RUN", False)

        schema_env = os.environ.get("HOMEAI_PG_SCHEMA") or os.environ.get("HOMEAI_DB_SCHEMA")
        if schema_env:
            schema = schema_env.strip() or None
        else:
            schema = None

        return cls(
            superuser=os.environ.get("POSTGRES_SUPERUSER", "postgres"),
            superuser_db=os.environ.get("POSTGRES_SUPERUSER_DB", "postgres"),
            host=os.environ.get("POSTGRES_SUPERUSER_HOST", "localhost"),
            port=os.environ.get("POSTGRES_SUPERUSER_PORT", "5432"),
            superuser_password=(
                os.environ.get("POSTGRES_SUPERUSER_PASSWORD")
                or os.environ.get("PGPASSWORD")
                or os.environ.get("POSTGRES_PASSWORD")
                or None
            ),
            db_name=os.environ.get("HOMEAI_DB_NAME", "homeai"),
            db_user=os.environ.get("HOMEAI_DB_USER", "homeai"),
            db_password=os.environ.get("HOMEAI_DB_PASSWORD", "homeai_password"),
            db_schema=schema,
            memory_table="homeai_memory",
            schema_file=schema_path,
            dry_run=dry_run,
        )


class BootstrapError(RuntimeError):
    """Raised when bootstrapping fails."""


def _log(message: str) -> None:
    print(f"[homeai-bootstrap] {message}")


def _log_error(message: str) -> None:
    print(f"[homeai-bootstrap] ERROR: {message}", file=sys.stderr)


def _ensure_psycopg():
    try:
        import psycopg  # type: ignore
        from psycopg import sql  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised via subprocess test
        _log_error(
            "The psycopg package is required to run this script. Install it with `pip install psycopg[binary]`."
        )
        raise SystemExit(1) from exc

    return psycopg, sql


def _assert_schema_exists(path: Optional[Path]) -> None:
    if path is not None and not path.exists():
        raise BootstrapError(f"Schema file not found: {path}")


def _bootstrap_database(config: BootstrapConfig) -> None:
    psycopg, sql = _ensure_psycopg()

    conninfo = {
        "user": config.superuser,
        "password": config.superuser_password,
        "host": config.host,
        "port": config.port,
        "dbname": config.superuser_db,
    }

    _log(
        "Connecting as superuser '%s' to database '%s' on %s:%s..."
        % (config.superuser, config.superuser_db, config.host, config.port)
    )

    try:
        with psycopg.connect(**conninfo) as conn:  # type: ignore[arg-type]
            conn.autocommit = True
            with conn.cursor() as cur:
                _ensure_role(cur, sql, config)
                _ensure_database(cur, sql, config)
                _grant_privileges(cur, sql, config)
    except Exception as exc:  # pragma: no cover - exercised via integration usage
        message = str(exc)
        if (
            config.superuser_password is None
            and "password" in message.lower()
        ):
            message += (
                "\nHint: provide the superuser password via POSTGRES_SUPERUSER_PASSWORD, "
                "PGPASSWORD, or POSTGRES_PASSWORD."
            )
        raise BootstrapError(message) from exc

    _ensure_application_objects(psycopg, sql, config)

    if config.schema_file is not None:
        _apply_schema(psycopg, config)


def _table_identifier(sql, config: BootstrapConfig):
    if config.db_schema:
        return sql.SQL("{}.{}").format(
            sql.Identifier(config.db_schema), sql.Identifier(config.memory_table)
        )
    return sql.Identifier(config.memory_table)


def _ensure_application_objects(psycopg, sql, config: BootstrapConfig) -> None:
    conninfo = {
        "user": config.superuser,
        "password": config.superuser_password,
        "host": config.host,
        "port": config.port,
        "dbname": config.db_name,
    }

    table_path = (
        f"{config.db_schema}.{config.memory_table}"
        if config.db_schema
        else config.memory_table
    )

    try:
        with psycopg.connect(**conninfo) as conn:  # type: ignore[arg-type]
            conn.autocommit = True
            with conn.cursor() as cur:
                if config.db_schema:
                    _log(f"Ensuring schema '{config.db_schema}' exists...")
                    cur.execute(
                        sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                            sql.Identifier(config.db_schema)
                        )
                    )
                    cur.execute(
                        sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(
                            sql.Identifier(config.db_schema),
                            sql.Identifier(config.db_user),
                        )
                    )

                _log(f"Ensuring memory table '{table_path}' exists...")
                table_identifier = _table_identifier(sql, config)
                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {} (
                            id TEXT PRIMARY KEY,
                            kind TEXT NOT NULL,
                            source TEXT NOT NULL,
                            user_id TEXT NULL,
                            session_id TEXT NOT NULL,
                            tags JSONB NOT NULL DEFAULT '[]'::jsonb,
                            content JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                            plain_text TEXT NOT NULL DEFAULT '',
                            created_at TIMESTAMPTZ NOT NULL,
                            updated_at TIMESTAMPTZ NOT NULL
                        )
                        """
                    ).format(table_identifier)
                )

                cur.execute(
                    sql.SQL("GRANT ALL PRIVILEGES ON TABLE {} TO {}").format(
                        table_identifier, sql.Identifier(config.db_user)
                    )
                )

                session_index = sql.Identifier(
                    f"{config.memory_table}_session_created_idx"
                )
                cur.execute(
                    sql.SQL(
                        """
                        CREATE INDEX IF NOT EXISTS {} ON {} (session_id, created_at)
                        """
                    ).format(session_index, table_identifier)
                )

                fts_index = sql.Identifier(
                    f"{config.memory_table}_plain_text_fts_idx"
                )
                try:
                    cur.execute(
                        sql.SQL(
                            """
                            CREATE INDEX IF NOT EXISTS {} ON {} USING GIN (
                                to_tsvector('simple', coalesce(plain_text, ''))
                            )
                            """
                        ).format(fts_index, table_identifier)
                    )
                except Exception as exc:
                    _log(f"Skipping FTS index creation: {exc}")
    except Exception as exc:  # pragma: no cover - exercised via integration usage
        raise BootstrapError(f"Failed to ensure application schema: {exc}") from exc


def _ensure_role(cur, sql, config: BootstrapConfig) -> None:
    cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (config.db_user,))
    exists = cur.fetchone() is not None

    if not exists:
        _log(f"Creating role '{config.db_user}'...")
        cur.execute(
            sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                sql.Identifier(config.db_user),
                sql.Literal(config.db_password),
            )
        )
    else:
        _log(f"Updating password for role '{config.db_user}'...")
        cur.execute(
            sql.SQL("ALTER ROLE {} WITH LOGIN PASSWORD {}").format(
                sql.Identifier(config.db_user),
                sql.Literal(config.db_password),
            )
        )


def _ensure_database(cur, sql, config: BootstrapConfig) -> None:
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (config.db_name,))
    exists = cur.fetchone() is not None

    if not exists:
        _log(f"Creating database '{config.db_name}' owned by '{config.db_user}'...")
        cur.execute(
            sql.SQL(
                "CREATE DATABASE {} WITH OWNER {} TEMPLATE template0 "
                "ENCODING 'UTF8' LC_COLLATE 'en_US.UTF-8' LC_CTYPE 'en_US.UTF-8' CONNECTION LIMIT -1"
            ).format(sql.Identifier(config.db_name), sql.Identifier(config.db_user))
        )
    else:
        _log(f"Database '{config.db_name}' already exists; skipping creation.")


def _grant_privileges(cur, sql, config: BootstrapConfig) -> None:
    _log(f"Granting privileges on database '{config.db_name}' to '{config.db_user}'...")
    cur.execute(
        sql.SQL("GRANT ALL PRIVILEGES ON DATABASE {} TO {}").format(
            sql.Identifier(config.db_name), sql.Identifier(config.db_user)
        )
    )


def _apply_schema(psycopg, config: BootstrapConfig) -> None:
    schema_file = config.schema_file
    assert schema_file is not None

    _log(f"Applying schema from {schema_file}...")
    sql_text = schema_file.read_text(encoding="utf-8")

    if not sql_text.strip():
        _log("Schema file is empty; nothing to apply.")
        return

    conninfo = {
        "user": config.db_user,
        "password": config.db_password,
        "host": config.host,
        "port": config.port,
        "dbname": config.db_name,
    }

    try:
        with psycopg.connect(**conninfo) as conn:  # type: ignore[arg-type]
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(sql_text, prepare=False)
    except Exception as exc:  # pragma: no cover - exercised via integration usage
        raise BootstrapError(f"Failed to apply schema: {exc}") from exc


def _dry_run(config: BootstrapConfig) -> None:
    _log("DRY RUN: no changes will be applied.")
    _log(
        "Would connect as '%s' to '%s' on %s:%s."
        % (config.superuser, config.superuser_db, config.host, config.port)
    )
    _log(
        "Would ensure role '%s' exists with the configured password." % config.db_user
    )
    _log(
        "Would ensure database '%s' exists owned by '%s'."
        % (config.db_name, config.db_user)
    )
    if config.db_schema:
        _log(f"Would ensure schema '{config.db_schema}' exists and is accessible to '{config.db_user}'.")
    _log(
        "Would ensure memory table '%s' exists with indexes."
        % (
            f"{config.db_schema}.{config.memory_table}" if config.db_schema else config.memory_table
        )
    )
    if config.schema_file is not None:
        _log(f"Would apply schema from {config.schema_file}.")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the operations without executing them (can also be enabled via HOMEAI_BOOTSTRAP_DRY_RUN).",
    )
    parser.add_argument(
        "--schema",
        type=str,
        help="Path to a schema file to apply (overrides HOMEAI_SCHEMA_FILE).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    config = BootstrapConfig.from_env(args)

    try:
        _assert_schema_exists(config.schema_file)

        if config.dry_run:
            _dry_run(config)
        else:
            _bootstrap_database(config)

    except BootstrapError as exc:
        _log_error(str(exc))
        return 1

    _log("PostgreSQL bootstrap complete.")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    sys.exit(main())
