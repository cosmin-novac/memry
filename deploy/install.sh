#!/usr/bin/env bash
#
# Memry VPS installer - one command on a fresh Ubuntu/Debian server.
#
#   curl -fsSL https://raw.githubusercontent.com/cosmin-novac/memry/main/deploy/install.sh | bash
#
# With a domain (DNS A record already pointing at this server) for automatic HTTPS,
# and an optional LLM key for extraction:
#
#   curl -fsSL https://raw.githubusercontent.com/cosmin-novac/memry/main/deploy/install.sh \
#     | MEMRY_DOMAIN=memory.example.com ANTHROPIC_API_KEY=sk-ant-... bash
#
# Re-running the same command updates Memry and keeps your configuration.
# Layout: code in /opt/memry/app (disposable), config in /opt/memry/.env,
# data in Docker volumes (memry_memry-data, memry_caddy-data).

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
for key in MEMRY_DOMAIN ANTHROPIC_API_KEY OPENAI_API_KEY MEMRY_LLM_MODEL MEMRY_TENANTS; do
  val="${!key:-}"
  [ -n "$val" ] && set_kv "$key" "$val"
done

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
say ""
if [ -z "$domain" ]; then
  say "  NOTE: no MEMRY_DOMAIN set - serving plain HTTP. Point a DNS A record at"
  say "  this server and re-run with MEMRY_DOMAIN=your.domain for automatic HTTPS."
fi
