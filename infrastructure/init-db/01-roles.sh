#!/bin/sh
# Solace-AI init-db: the two platform roles and the single application database (IF-03; MVP-249, MVP-041).
#
# The postgres image runs this once, from /docker-entrypoint-initdb.d, on an empty data directory. It
# executes the file when it is executable and sources it into its own bash entrypoint otherwise, so this
# is plain POSIX sh that changes no shell options, defines only _solace_* names, checks every step
# explicitly and unsets the passwords once they have been used.
#
# Required environment (give each one directly or as NAME_FILE naming a file, never both):
#   SOLACE_APP_PASSWORD       solace_app: services, DML only, NOSUPERUSER NOBYPASSRLS
#   SOLACE_MIGRATOR_PASSWORD  solace_migrator: the migrate job only, owns the database, BYPASSRLS
# POSTGRES_PASSWORD stays the image superuser's (ops only). POSTGRES_DB names the application database
# (solace); it is handed to solace_migrator and closed to PUBLIC.
#
# No password is interpolated by the shell, echoed or put on a command line. Each reaches psql in its
# environment, is read with \getenv and quoted by psql (:'name'), and the server is told not to log the
# statements that carry them.

_solace_fail() {
    printf >&2 '01-roles.sh: error: %s\n' "$1"
    exit 1
}

# _solace_file_env NAME: set NAME from $NAME or from the file named by $NAME_FILE (exactly one of them).
_solace_file_env() {
    _solace_var=$1
    eval "_solace_val=\${$_solace_var:-}"
    eval "_solace_file=\${${_solace_var}_FILE:-}"
    if [ -n "$_solace_val" ] && [ -n "$_solace_file" ]; then
        _solace_fail "both $_solace_var and ${_solace_var}_FILE are set (but are exclusive)"
    fi
    if [ -z "$_solace_val" ] && [ -z "$_solace_file" ]; then
        _solace_fail "$_solace_var or ${_solace_var}_FILE must be set"
    fi
    if [ -n "$_solace_file" ]; then
        if [ ! -f "$_solace_file" ] || [ ! -r "$_solace_file" ]; then
            _solace_fail "${_solace_var}_FILE does not name a readable file"
        fi
        _solace_val=$(cat "$_solace_file") || _solace_fail "cannot read ${_solace_var}_FILE"
        if [ -z "$_solace_val" ]; then
            _solace_fail "${_solace_var}_FILE names an empty file"
        fi
    fi
    eval "$_solace_var=\$_solace_val"
    unset _solace_var _solace_val _solace_file
}

_solace_file_env SOLACE_APP_PASSWORD
_solace_file_env SOLACE_MIGRATOR_PASSWORD

case "${POSTGRES_DB:-}" in
    '' | postgres | template0 | template1)
        _solace_fail "POSTGRES_DB must name the application database, not '${POSTGRES_DB:-}'"
        ;;
esac

# The passwords reach psql through its environment only (the prefix also exports _FILE-loaded values),
# never on its argv, which any process can read from /proc/<pid>/cmdline.
SOLACE_APP_PASSWORD="$SOLACE_APP_PASSWORD" SOLACE_MIGRATOR_PASSWORD="$SOLACE_MIGRATOR_PASSWORD" \
    psql -v ON_ERROR_STOP=1 -v VERBOSITY=terse --no-psqlrc --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
    <<'EOSQL' || _solace_fail "creating the platform roles failed (psql exited non-zero)"
\getenv app_password SOLACE_APP_PASSWORD
\getenv migrator_password SOLACE_MIGRATOR_PASSWORD
-- Keep the password-bearing statements out of the server log, even if one of them fails.
SET log_statement = 'none';
SET log_min_duration_statement = -1;
SET log_min_error_statement = panic;

CREATE ROLE solace_migrator WITH LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION BYPASSRLS
    PASSWORD :'migrator_password';
CREATE ROLE solace_app WITH LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS
    PASSWORD :'app_password';

-- The migrator owns the database, and through pg_database_owner the public schema.
SELECT format('ALTER DATABASE %I OWNER TO solace_migrator', current_database()) \gexec
SELECT format('REVOKE ALL ON DATABASE %I FROM PUBLIC', current_database()) \gexec
SELECT format('GRANT CONNECT, TEMPORARY ON DATABASE %I TO solace_migrator, solace_app', current_database()) \gexec
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
EOSQL

unset SOLACE_APP_PASSWORD SOLACE_MIGRATOR_PASSWORD
unset -f _solace_file_env _solace_fail
