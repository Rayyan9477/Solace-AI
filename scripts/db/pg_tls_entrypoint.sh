#!/bin/sh
# Solace-AI Postgres TLS wrapper around the official image's docker-entrypoint.sh (D-15; MVP-010, MVP-271).
#
#   entrypoint: ["sh", "/usr/local/bin/solace-pg-entrypoint.sh"]
#   command:    ["postgres"]
#
# Environment (neither is a secret):
#   SOLACE_PG_TLS      optional (default) or required
#   SOLACE_PG_TLS_DIR  the directory holding server.crt and server.key (default /run/secrets), as written by
#                      scripts/db/gen_certs.py; it may be read-only, because the files are copied
#
# When the directory holds server.crt and server.key, both are copied to /var/lib/postgresql/tls (outside PGDATA,
# so initdb still finds an empty data directory), the copies and the directory are given to the postgres user
# with the key at mode 0600 (the server refuses a key others can read), and the server is started with
# "-c ssl=on -c ssl_cert_file=... -c ssl_key_file=...". With SOLACE_PG_TLS=required it also gets this hba file,
# so TCP clients must use TLS (the unix socket stays open to the init scripts, pg_isready and docker exec psql):
#     local   all  all       trust
#     hostssl all  all  all  scram-sha-256
#     host    all  all  all  reject
# and the container stops with exit 1 when the files are missing. In either mode it also stops when only one of
# the two files is present, when they cannot be read or installed, or when SOLACE_PG_TLS has another value.
# With SOLACE_PG_TLS=optional and neither file, the server starts exactly as the plain image would.
#
# Every setting is appended after the caller's arguments, so it wins over any earlier -c of the same name. Nothing
# here prints certificate or key material. Plain POSIX sh: the image's /bin/sh is busybox ash.

_solace_fail() {
    printf >&2 'solace-pg-entrypoint: error: %s\n' "$1"
    exit 1
}

_solace_mode=${SOLACE_PG_TLS:-optional}
_solace_src=${SOLACE_PG_TLS_DIR:-/run/secrets}
_solace_dst=/var/lib/postgresql/tls

case "$_solace_mode" in
    required | optional) ;;
    *) _solace_fail "SOLACE_PG_TLS must be 'required' or 'optional'" ;;
esac

# The image's own rule: a first argument starting with '-' means "postgres <those flags>".
case "${1:-}" in
    -*) set -- postgres "$@" ;;
esac

# Anything other than the server (psql, bash, ...) runs unchanged.
if [ "${1:-}" != postgres ]; then
    exec docker-entrypoint.sh "$@"
fi

_solace_have=
_solace_missing=
for _solace_name in server.crt server.key; do
    if [ -e "$_solace_src/$_solace_name" ]; then
        _solace_have="$_solace_have $_solace_name"
    else
        _solace_missing="$_solace_missing $_solace_name"
    fi
done

if [ -z "$_solace_have" ]; then
    if [ "$_solace_mode" = required ]; then
        _solace_fail "SOLACE_PG_TLS=required but $_solace_src has neither server.crt nor server.key (missing:$_solace_missing); generate them with scripts/db/gen_certs.py"
    fi
    exec docker-entrypoint.sh "$@"
fi
if [ -n "$_solace_missing" ]; then
    _solace_fail "$_solace_src has$_solace_have but not$_solace_missing; mount server.crt and server.key together"
fi

for _solace_name in server.crt server.key; do
    if [ ! -f "$_solace_src/$_solace_name" ] || [ ! -r "$_solace_src/$_solace_name" ]; then
        _solace_fail "$_solace_src/$_solace_name is not a readable file"
    fi
done

_solace_umask=$(umask)
umask 077
mkdir -p "$_solace_dst" || _solace_fail "cannot create $_solace_dst"
for _solace_name in server.crt server.key; do
    rm -f "$_solace_dst/$_solace_name" &&
        cp "$_solace_src/$_solace_name" "$_solace_dst/$_solace_name" ||
        _solace_fail "cannot copy $_solace_name to $_solace_dst"
done
chmod 0700 "$_solace_dst" &&
    chmod 0644 "$_solace_dst/server.crt" &&
    chmod 0600 "$_solace_dst/server.key" ||
    _solace_fail "cannot set the modes of $_solace_dst"
set -- "$@" -c ssl=on -c "ssl_cert_file=$_solace_dst/server.crt" -c "ssl_key_file=$_solace_dst/server.key"

if [ "$_solace_mode" = required ]; then
    rm -f "$_solace_dst/pg_hba.conf" || _solace_fail "cannot replace $_solace_dst/pg_hba.conf"
    printf '%s\n' \
        '# Written by solace-pg-entrypoint (SOLACE_PG_TLS=required): TCP clients must use TLS.' \
        'local   all  all       trust' \
        'hostssl all  all  all  scram-sha-256' \
        'host    all  all  all  reject' >"$_solace_dst/pg_hba.conf" &&
        chmod 0600 "$_solace_dst/pg_hba.conf" ||
        _solace_fail "cannot write $_solace_dst/pg_hba.conf"
    set -- "$@" -c "hba_file=$_solace_dst/pg_hba.conf"
fi

# As root the image's entrypoint steps down to the postgres user, which must own the copies.
if [ "$(id -u)" = 0 ]; then
    chown -R postgres:postgres "$_solace_dst" || _solace_fail "cannot give $_solace_dst to the postgres user"
fi
umask "$_solace_umask"

unset _solace_mode _solace_src _solace_dst _solace_have _solace_missing _solace_name _solace_umask
unset -f _solace_fail
exec docker-entrypoint.sh "$@"
