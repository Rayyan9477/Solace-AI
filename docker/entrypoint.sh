#!/bin/sh
# Container entry contract (IF-21; NAME_FILE semantics IF-22; MVP-211; D-07).
#
# Strict secrets arrive as one root-only file per secret under /run/secrets (rendered from SOPS + age by
# deploy/render_secrets.py) and are referenced as NAME_FILE=/run/secrets/<name>. For every environment variable
# NAME_FILE this script exports NAME with the file's content minus exactly one trailing newline, then execs the
# command so it runs as PID 1 with the loaded environment.
#
#   - The file is missing or unreadable  -> exit 1, naming the variable (never a path's content).
#   - NAME and NAME_FILE are both set     -> exit 1, naming both variables (the file is never read).
#   - NAME_FILE stays set in the exec'd environment, so the service can check provenance (WS-04.11).
#   - SSL_CERT_FILE is OpenSSL's CA-bundle path, not a secret reference, and is left alone.
#
# This script handles secret values: never add `set -x`, never echo or print a value, and keep messages to
# variable names only. POSIX sh, LF line endings (the image build also strips any CR).
set -eu

nl='
'

_fail() {
    printf 'entrypoint: %s\n' "$1" >&2
    exit 1
}

# Names are taken from `env` once, before anything is exported. A line that only looks like NAME_FILE=... (inside
# another variable's multi-line value) is skipped by the "is it really set" check below.
for _name in $(env | sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)_FILE=.*$/\1/p' | sort -u); do
    case "$_name" in
        SSL_CERT) continue ;;
    esac
    eval "_file_set=\${${_name}_FILE+set} _name_set=\${${_name}+set}"
    [ "$_file_set" = set ] || continue
    if [ "$_name_set" = set ]; then
        _fail "${_name} and ${_name}_FILE are both set; set only one"
    fi
    eval "_path=\${${_name}_FILE}"
    if [ ! -f "$_path" ] || [ ! -r "$_path" ]; then
        _fail "${_name}_FILE does not name a readable file"
    fi
    # A NUL byte cannot live in a shell variable (it would be silently dropped), so refuse the file outright.
    # The count goes through `cat`, the same reader as the value below, so procfs-style files (reported size 0) are
    # judged by what is actually read. An unreadable file yields 0 here and fails at the read just after.
    _nuls=$(cat -- "$_path" 2>/dev/null | tr -cd '\000' | wc -c) || _fail "${_name}_FILE could not be read"
    [ "$_nuls" -eq 0 ] || _fail "${_name}_FILE contains a NUL byte"
    _value=$(cat -- "$_path" 2>/dev/null && printf x) || _fail "${_name}_FILE could not be read"
    _value=${_value%x}
    _value=${_value%"$nl"}
    eval "export ${_name}=\"\$_value\""
done
unset _name _file_set _name_set _path _value _nuls

[ "$#" -gt 0 ] || _fail "no command to run"
exec "$@"
