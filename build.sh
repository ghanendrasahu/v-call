#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# build.sh — reproducible image build with Git lineage tracking.
# Tags images with the short SHA + branch so every running container can
# be traced back to an exact commit.
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

REGISTRY="${REGISTRY:-vcall}"
IMAGE_NAME="${IMAGE_NAME:-}"
EXTRAS="${INSTALL_EXTRAS:-ai}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"

# ── Git lineage ───────────────────────────────────────────────────────
GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo 'nogit')"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'main')"
GIT_DIRTY=""
if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
  GIT_DIRTY="-dirty"
fi
BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
VERSION="${GIT_SHA}${GIT_DIRTY}"

echo "▶ Git lineage: branch=${GIT_BRANCH} sha=${VERSION} built=${BUILD_DATE}"

build_target() {
  local target="$1"
  local image="${REGISTRY}/${target}:${VERSION}"
  local latest="${REGISTRY}/${target}:latest"

  echo "▶ Building ${image} (extras=${EXTRAS})"
  docker build \
    --build-arg INSTALL_EXTRAS="${EXTRAS}" \
    --label "org.opencontainers.image.revision=$(git rev-parse HEAD 2>/dev/null || echo 'unknown')" \
    --label "org.opencontainers.image.created=${BUILD_DATE}" \
    --label "com.vcall.git-branch=${GIT_BRANCH}" \
    -t "${image}" \
    -t "${latest}" \
    -f Dockerfile .

  echo "✓ Built ${image}"
}

# ── Entry point ───────────────────────────────────────────────────────
if [[ "${1:-}" == "compose" ]]; then
  echo "▶ Booting full stack via docker-compose"
  GIT_SHA="${VERSION}" docker compose -f "${COMPOSE_FILE}" up -d --build
  docker compose -f "${COMPOSE_FILE}" ps
  exit 0
fi

if [[ -n "${IMAGE_NAME}" ]]; then
  build_target "${IMAGE_NAME}"
else
  build_target "web"
  build_target "worker"
  build_target "beat"
  build_target "flower"
fi

echo "✓ All images tagged with ${VERSION}"
