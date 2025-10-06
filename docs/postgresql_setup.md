# PostgreSQL Setup Guide

This guide explains how to prepare a local PostgreSQL instance on Ubuntu Linux for the HomeAI application. The steps assume that PostgreSQL is already installed and the server is running on the same machine.

## 1. Install client tools

If you only have the PostgreSQL server packages, ensure the command‐line tools are available:

```bash
sudo apt update
sudo apt install postgresql-client
```

## 2. Confirm service status

Check that the service is running and reachable:

```bash
sudo systemctl status postgresql
psql --version
```

For a default local installation the instance listens on `localhost:5432` and the default superuser is `postgres`.

## 3. Configure authentication (optional)

If you plan to connect over TCP using passwords, confirm that the `pg_hba.conf` file allows it:

1. Locate the file (for Ubuntu packages it is usually `/etc/postgresql/<version>/main/pg_hba.conf`).
2. Ensure there is a line similar to the following near the top:

   ```text
   host    all             all             127.0.0.1/32            scram-sha-256
   ```

3. Reload PostgreSQL to apply changes:

   ```bash
   sudo systemctl reload postgresql
   ```

## 4. Create the application role and database

You can create everything manually or by using the provided bootstrap script.

### Manual steps

```bash
sudo -u postgres psql <<'SQL'
CREATE ROLE homeai LOGIN PASSWORD 'homeai_password';
CREATE DATABASE homeai WITH OWNER = homeai ENCODING = 'UTF8';
GRANT ALL PRIVILEGES ON DATABASE homeai TO homeai;
SQL
```

Adjust the role name, password, and database name as required. If the role or database already exists, change `CREATE` to `ALTER` or use `psql` commands to inspect them before running the statements.

### Automated bootstrap

Use the `scripts/bootstrap_postgres.py` helper to apply the configuration idempotently:

```bash
# Run as a user that can connect as the PostgreSQL superuser
python scripts/bootstrap_postgres.py
```

Add `--dry-run` (or set `HOMEAI_BOOTSTRAP_DRY_RUN=1`) to preview the actions without connecting to the database. The script accepts several environment variables to customise the setup:

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_SUPERUSER` | Superuser role used to perform setup | `postgres` |
| `POSTGRES_SUPERUSER_DB` | Database used for the initial connection | `postgres` |
| `POSTGRES_SUPERUSER_HOST` | Host where PostgreSQL is running | `localhost` |
| `POSTGRES_SUPERUSER_PORT` | TCP port | `5432` |
| `POSTGRES_SUPERUSER_PASSWORD` | Password for the superuser (optional) | *(empty)* |
| `HOMEAI_DB_NAME` | Application database name | `homeai` |
| `HOMEAI_DB_USER` | Application role | `homeai` |
| `HOMEAI_DB_PASSWORD` | Application role password | `homeai_password` |
| `HOMEAI_SCHEMA_FILE` | Path to an SQL file that should be applied after creation | *(empty)* |
| `HOMEAI_BOOTSTRAP_DRY_RUN` | When set to a truthy value, only log the planned actions | `false` |

If `POSTGRES_SUPERUSER_PASSWORD` is not set the script will fall back to the standard
`PGPASSWORD` or `POSTGRES_PASSWORD` environment variables when available. When none of
those are provided and the server requires authentication, the script now emits a hint
describing how to supply the password.

Example with custom credentials:

```bash
POSTGRES_SUPERUSER=postgres \
POSTGRES_SUPERUSER_PASSWORD='supersecret' \
HOMEAI_DB_NAME=homeai_dev \
HOMEAI_DB_USER=homeai_app \
HOMEAI_DB_PASSWORD='supersecret' \
python scripts/bootstrap_postgres.py
```

If you want to seed the database with a schema file:

```bash
HOMEAI_SCHEMA_FILE=sql/schema.sql python scripts/bootstrap_postgres.py
```

## 5. Test the connection

After creating the database and role, verify that the application role can log in:

```bash
PGPASSWORD=homeai_password psql \
  --username=homeai \
  --dbname=homeai \
  --host=localhost \
  --port=5432 \
  --command='SELECT current_user, current_database();'
```

You should see the `homeai` user and database in the output.

## 6. Next steps

Once the database is prepared you can point the application to it by configuring the appropriate environment variables (for example, `DATABASE_URL=postgresql://homeai:homeai_password@localhost:5432/homeai`).
