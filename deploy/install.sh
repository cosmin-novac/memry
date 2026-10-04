#!/usr/bin/env bash
#
# Memry VPS installer - one command on a fresh Ubuntu/Debian server.
#
#   curl -fsSL https://raw.githubusercontent.com/cosmin-novac/memry/main/deploy/install.sh | bash
#
# To run Memry you need a text model and a decision model; pass their keys on the
# first run (they are kept in /opt/memry/.env). With a domain (DNS A record already
# pointing at this server) for automatic HTTPS:
#
#   curl -fsSL https://raw.githubusercontent.com/cosmin-novac/memry/main/deploy/install.sh \
#     | MEMRY_DOMAIN=memory.example.com OPENAI_API_KEY=sk-... \
#       MEMRY_DECISION_PROVIDER=jev MEMRY_DECISION_API_KEY=... bash
#
# With MEMRY_DECISION_PROVIDER=llm, Memry sends the decision questions to the
# text model, merges entities only by fixed rules, and you confirm the other
# merges yourself.
#
# Re-running the same command updates Memry and keeps your configuration.
# Layout: code in /opt/memry/app (disposable), config in /opt/memry/.env,
# data in Docker volumes (memry_memry-data, memry_caddy-data), the nightly
# snapshot in /var/backups/memry (MEMRY_SNAPSHOT_HOST_DIR) on the host.

set -euo pipefail

REPO="${MEMRY_REPO:-cosmin-novac/memry}"
REF="${MEMRY_REF:-main}"
BASE_DIR="/opt/memry"
APP_DIR="$BASE_DIR/app"
ENV_FILE="$BASE_DIR/.env"

say()  { printf '\033[1;36m[memry]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[memry]\033[0m %s\n' "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || fail "run as root (or: curl ... | sudo bash)"
command -v curl >/dev/null 2>&1 || fail "curl is required"

# --- Docker ------------------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  say "installing Docker (get.docker.com)..."
  curl -fsSL https://get.docker.com | sh
fi
docker compose version >/dev/null 2>&1 \
  || fail "the 'docker compose' plugin is missing; install docker-compose-plugin"

# --- Source ------------------------------------------------------------------
mkdir -p "$BASE_DIR"
if [ -n "${MEMRY_LOCAL_SOURCE:-}" ]; then
  say "using local source at $MEMRY_LOCAL_SOURCE"
  rm -rf "$APP_DIR"
  cp -a "$MEMRY_LOCAL_SOURCE" "$APP_DIR"
else
  say "downloading $REPO@$REF..."
  rm -rf "$APP_DIR.new"
  mkdir -p "$APP_DIR.new"
  curl -fsSL "https://codeload.github.com/$REPO/tar.gz/refs/heads/$REF" \
    | tar -xz --strip-components=1 -C "$APP_DIR.new"
  rm -rf "$APP_DIR"
  mv "$APP_DIR.new" "$APP_DIR"
fi

# --- Configuration -----------------------------------------------------------
set_kv() {
  local key="$1" val="$2"
  if grep -q "^${key}=" "$ENV_FILE" 2>/dev/null; then
    sed -i "s|^${key}=.*|${key}=${val}|" "$ENV_FILE"
  else
    printf '%s=%s\n' "$key" "$val" >>"$ENV_FILE"
  fi
}

touch "$ENV_FILE"
chmod 600 "$ENV_FILE"

if ! grep -q '^MEMRY_API_KEY=' "$ENV_FILE"; then
  say "generating MEMRY_API_KEY"
  if command -v openssl >/dev/null 2>&1; then
    set_kv MEMRY_API_KEY "$(openssl rand -hex 24)"
  else
    set_kv MEMRY_API_KEY "$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')"
  fi
fi
for key in MEMRY_DOMAIN ANTHROPIC_API_KEY OPENAI_API_KEY MEMRY_LLM_PROVIDER MEMRY_LLM_MODEL \
           MEMRY_TENANTS MEMRY_DECISION_PROVIDER MEMRY_DECISION_API_KEY MEMRY_DECISION_MODEL; do
  val="${!key:-}"
  [ -n "$val" ] && set_kv "$key" "$val"
done

# --- Models ------------------------------------------------------------------
# The installer checks this before building: if you haven't set a model, the
# running server stays as it is.
env_val() { grep "^$1=" "$ENV_FILE" 2>/dev/null | tail -1 | cut -d= -f2- || true; }
missing=""
if [ -z "$(env_val OPENAI_API_KEY)$(env_val ANTHROPIC_API_KEY)$(env_val MEMRY_LLM_PROVIDER)" ]; then
  missing="$missing\n  - a text model: OPENAI_API_KEY (or ANTHROPIC_API_KEY)"
fi
decision="$(env_val MEMRY_DECISION_PROVIDER)"
if [ -z "$decision" ]; then
  missing="$missing\n  - a decision model: MEMRY_DECISION_PROVIDER=jev and MEMRY_DECISION_API_KEY"
  missing="$missing (a TypeSafe key), or MEMRY_DECISION_PROVIDER=llm to send the decision"
  missing="$missing questions to the text model (Memry then merges entities only by fixed rules, and you confirm the other merges yourself)"
elif [ "$decision" = "jev" ] && [ -z "$(env_val MEMRY_DECISION_API_KEY)" ]; then
  missing="$missing\n  - MEMRY_DECISION_API_KEY: the TypeSafe key for MEMRY_DECISION_PROVIDER=jev"
fi
if [ -n "$missing" ]; then
  fail "$(printf '%b' "Set a text model and a decision model to run Memry. Not set yet:$missing\nAdd them to $ENV_FILE, or pass them to this installer:\n  curl ... | OPENAI_API_KEY=sk-... MEMRY_DECISION_PROVIDER=jev MEMRY_DECISION_API_KEY=... bash")"
fi

# --- Snapshot directory ------------------------------------------------------
# The nightly copy of the database lives on the host, outside the Docker data
# volume, so a fault in the live files cannot reach it. Root-only: it holds every
# memory. A local copy does not survive losing the disk; see
# docs/self-hosting.md#nightly-snapshot for the optional offsite copy.
snapshot_dir="$(env_val MEMRY_SNAPSHOT_HOST_DIR)"
snapshot_dir="${snapshot_dir:-/var/backups/memry}"
mkdir -p "$snapshot_dir"
chmod 700 "$snapshot_dir"

# --- Launch ------------------------------------------------------------------
compose() {
  docker compose --env-file "$ENV_FILE" -f "$APP_DIR/deploy/vps/docker-compose.yml" "$@"
}

say "building and starting (first build takes a minute or two)..."
compose up -d --build

say "waiting for the server to become healthy..."
cid="$(compose ps -q memry)"
status=starting
for _ in $(seq 1 40); do
  status="$(docker inspect -f '{{.State.Health.Status}}' "$cid" 2>/dev/null || echo unknown)"
  [ "$status" = "healthy" ] && break
  sleep 3
done
[ "$status" = "healthy" ] || {
  compose logs --tail 50 memry || true
  fail "server did not become healthy; see logs above"
}

# --- Summary -----------------------------------------------------------------
domain="$(grep '^MEMRY_DOMAIN=' "$ENV_FILE" | cut -d= -f2- || true)"
api_key="$(grep '^MEMRY_API_KEY=' "$ENV_FILE" | cut -d= -f2-)"
if [ -n "$domain" ]; then
  url="https://$domain"
else
  ip="$(curl -fsS --max-time 5 https://api.ipify.org 2>/dev/null || hostname -I | awk '{print $1}')"
  url="http://$ip"
fi

say ""
say "Memry is up."
say ""
say "  Dashboard:   $url/"
say "  REST API:    $url/api/v1/..."
say "  MCP (HTTP):  $url/mcp"
say "  API key:     $api_key"
say ""
say "  Config:      $ENV_FILE   (edit, then re-run this script or:"
say "               docker compose --env-file $ENV_FILE -f $APP_DIR/deploy/vps/docker-compose.yml up -d)"
say "  Update:      re-run this installer"
say "  Logs:        docker compose --env-file $ENV_FILE -f $APP_DIR/deploy/vps/docker-compose.yml logs -f"
say "  Snapshot:    $snapshot_dir   (nightly verified copy of the database; check it with"
say "               docker compose --env-file $ENV_FILE -f $APP_DIR/deploy/vps/docker-compose.yml exec memry memry snapshot --check)"
say ""
if [ -z "$domain" ]; then
  say "  NOTE: no MEMRY_DOMAIN set - serving plain HTTP. Point a DNS A record at"
  say "  this server and re-run with MEMRY_DOMAIN=your.domain for automatic HTTPS."
fi
