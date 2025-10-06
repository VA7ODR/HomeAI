#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a PostgreSQL database for the HomeAI application.
# The script is idempotent and safe to run multiple times.  It requires
# connectivity to a PostgreSQL superuser (defaults to the local 'postgres'
# role on localhost).

# Configurable environment variables:
#   POSTGRES_SUPERUSER          - superuser role used to execute setup commands (default: postgres)
#   POSTGRES_SUPERUSER_DB       - database that the superuser connects to (default: postgres)
#   POSTGRES_SUPERUSER_HOST     - host of the PostgreSQL instance (default: localhost)
#   POSTGRES_SUPERUSER_PORT     - port of the PostgreSQL instance (default: 5432)
#   POSTGRES_SUPERUSER_PASSWORD - password for the superuser role (optional; use PGPASSWORD env var instead)
#   HOMEAI_DB_NAME              - name of the application database (default: homeai)
#   HOMEAI_DB_USER              - name of the application role (default: homeai)
#   HOMEAI_DB_PASSWORD          - password for the application role (default: homeai_password)
#   HOMEAI_SCHEMA_FILE          - optional SQL file to seed the database schema/data after creation

POSTGRES_SUPERUSER=${POSTGRES_SUPERUSER:-postgres}
POSTGRES_SUPERUSER_DB=${POSTGRES_SUPERUSER_DB:-postgres}
POSTGRES_SUPERUSER_HOST=${POSTGRES_SUPERUSER_HOST:-localhost}
POSTGRES_SUPERUSER_PORT=${POSTGRES_SUPERUSER_PORT:-5432}
POSTGRES_SUPERUSER_PASSWORD=${POSTGRES_SUPERUSER_PASSWORD:-}
HOMEAI_DB_NAME=${HOMEAI_DB_NAME:-homeai}
HOMEAI_DB_USER=${HOMEAI_DB_USER:-homeai}
HOMEAI_DB_PASSWORD=${HOMEAI_DB_PASSWORD:-homeai_password}
HOMEAI_SCHEMA_FILE=${HOMEAI_SCHEMA_FILE:-}

if ! command -v psql >/dev/null 2>&1; then
  echo "psql command not found. Please install PostgreSQL client tools." >&2
  exit 1
fi

# Configure connection arguments.  We avoid putting the password directly on the
# command line to prevent leaking it via process listings.
PSQL_ARGS=(
  "--username=${POSTGRES_SUPERUSER}"
  "--host=${POSTGRES_SUPERUSER_HOST}"
  "--port=${POSTGRES_SUPERUSER_PORT}"
  "--dbname=${POSTGRES_SUPERUSER_DB}"
  "--set=ON_ERROR_STOP=1"
)

if [[ -n "${POSTGRES_SUPERUSER_PASSWORD}" ]]; then
  export PGPASSWORD="${POSTGRES_SUPERUSER_PASSWORD}"
fi

echo "Configuring PostgreSQL for HomeAI..."

psql "${PSQL_ARGS[@]}" <<SQL
DO
$$
BEGIN
    IF NOT EXISTS (
        SELECT FROM pg_roles WHERE rolname = '${HOMEAI_DB_USER}'
    ) THEN
        EXECUTE format('CREATE ROLE %I LOGIN PASSWORD %L;', '${HOMEAI_DB_USER}', '${HOMEAI_DB_PASSWORD}');
    ELSE
        EXECUTE format('ALTER ROLE %I WITH LOGIN PASSWORD %L;', '${HOMEAI_DB_USER}', '${HOMEAI_DB_PASSWORD}');
    END IF;
END
$$;

DO
$$
BEGIN
    IF NOT EXISTS (
        SELECT FROM pg_database WHERE datname = '${HOMEAI_DB_NAME}'
    ) THEN
        EXECUTE format('CREATE DATABASE %I WITH OWNER %I TEMPLATE template0 ENCODING ''UTF8'' LC_COLLATE ''en_US.UTF-8'' LC_CTYPE ''en_US.UTF-8'' CONNECTION LIMIT -1;', '${HOMEAI_DB_NAME}', '${HOMEAI_DB_USER}');
    END IF;
END
$$;
SQL

psql "${PSQL_ARGS[@]}" <<SQL
DO
$$
BEGIN
    EXECUTE format('GRANT ALL PRIVILEGES ON DATABASE %I TO %I;', '${HOMEAI_DB_NAME}', '${HOMEAI_DB_USER}');
END
$$;
SQL

# Apply optional schema if provided
if [[ -n "${HOMEAI_SCHEMA_FILE}" ]]; then
  if [[ ! -f "${HOMEAI_SCHEMA_FILE}" ]]; then
    echo "Schema file not found: ${HOMEAI_SCHEMA_FILE}" >&2
    exit 1
  fi
  echo "Applying schema from ${HOMEAI_SCHEMA_FILE}..."
  PGPASSWORD="${HOMEAI_DB_PASSWORD}" psql \
    --username="${HOMEAI_DB_USER}" \
    --host="${POSTGRES_SUPERUSER_HOST}" \
    --port="${POSTGRES_SUPERUSER_PORT}" \
    --dbname="${HOMEAI_DB_NAME}" \
    --set=ON_ERROR_STOP=1 \
    --file="${HOMEAI_SCHEMA_FILE}"
fi

echo "PostgreSQL bootstrap complete."
