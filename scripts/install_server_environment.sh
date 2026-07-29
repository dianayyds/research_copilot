#!/usr/bin/env bash
set -euo pipefail

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    docker --version
    docker compose version
    exit 0
fi

if [[ ! -r /etc/os-release ]]; then
    echo "Cannot identify the operating system. Install Docker Engine and the Compose plugin manually." >&2
    exit 1
fi

# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" ]]; then
    echo "This installer supports Ubuntu only. Follow https://docs.docker.com/engine/install/ for ${ID:-this OS}." >&2
    exit 1
fi

if [[ "$(id -u)" -eq 0 ]]; then
    SUDO_CMD=()
else
    if ! command -v sudo >/dev/null 2>&1; then
        echo "sudo is required to install Docker packages." >&2
        exit 1
    fi
    SUDO_CMD=(sudo)
fi

"${SUDO_CMD[@]}" apt-get update
"${SUDO_CMD[@]}" apt-get install -y ca-certificates curl
"${SUDO_CMD[@]}" install -m 0755 -d /etc/apt/keyrings

DOCKER_KEY_TMP="$(mktemp)"
DOCKER_SOURCE_TMP="$(mktemp)"
trap 'rm -f "$DOCKER_KEY_TMP" "$DOCKER_SOURCE_TMP"' EXIT

curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o "$DOCKER_KEY_TMP"
"${SUDO_CMD[@]}" install -m 0644 "$DOCKER_KEY_TMP" /etc/apt/keyrings/docker.asc

UBUNTU_CODENAME="${UBUNTU_CODENAME:-${VERSION_CODENAME:-}}"
if [[ -z "$UBUNTU_CODENAME" ]]; then
    echo "Unable to determine the Ubuntu codename." >&2
    exit 1
fi

DOCKER_ARCH="$(dpkg --print-architecture)"
{
    echo "Types: deb"
    echo "URIs: https://download.docker.com/linux/ubuntu"
    echo "Suites: $UBUNTU_CODENAME"
    echo "Components: stable"
    echo "Architectures: $DOCKER_ARCH"
    echo "Signed-By: /etc/apt/keyrings/docker.asc"
} >"$DOCKER_SOURCE_TMP"
"${SUDO_CMD[@]}" install -m 0644 "$DOCKER_SOURCE_TMP" /etc/apt/sources.list.d/docker.sources

"${SUDO_CMD[@]}" apt-get update
"${SUDO_CMD[@]}" apt-get install -y \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin

if command -v systemctl >/dev/null 2>&1; then
    "${SUDO_CMD[@]}" systemctl enable --now docker
fi

TARGET_USER="${SUDO_USER:-${USER:-}}"
if [[ -n "$TARGET_USER" && "$TARGET_USER" != "root" ]]; then
    "${SUDO_CMD[@]}" usermod -aG docker "$TARGET_USER"
    echo "Added $TARGET_USER to the docker group. Log out and back in before running Docker without sudo."
fi

"${SUDO_CMD[@]}" docker --version
"${SUDO_CMD[@]}" docker compose version
