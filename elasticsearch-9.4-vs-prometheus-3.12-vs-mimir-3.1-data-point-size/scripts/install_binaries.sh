#!/usr/bin/env bash
# Download a binary to .bin/ if it is not already present.
# Usage: install_binaries.sh <binary> <version>
set -euo pipefail

BINARY=$1
VERSION=$2
BIN_DIR="$(cd "$(dirname "$0")/.." && pwd)/.bin"
mkdir -p "$BIN_DIR"

OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)
case "$ARCH" in
  x86_64)        ARCH=amd64 ;;
  arm64|aarch64) ARCH=arm64 ;;
  *) echo "Unsupported arch: $ARCH" >&2; exit 1 ;;
esac

case "$BINARY" in
  metricsgenreceiver)
    DEST="$BIN_DIR/metricsgenreceiver"
    [[ -f "$DEST" ]] && echo "✓ metricsgenreceiver already installed" && exit 0
    URL="https://github.com/elastic/metricsgenreceiver/releases/download/v${VERSION}/metricsgenreceiver_${OS}_${ARCH}.tar.gz"
    echo "Downloading metricsgenreceiver v${VERSION} (${OS}/${ARCH})..."
    TMP=$(mktemp)
    trap "rm -f $TMP" EXIT
    curl -fsSL "$URL" -o "$TMP"
    # The archive entry is named metricsgenreceiver_<os>_<arch>; extract and rename.
    tar -xzf "$TMP" -C "$BIN_DIR"
    mv "$BIN_DIR/metricsgenreceiver_${OS}_${ARCH}" "$DEST"
    chmod +x "$DEST"
    echo "✓ metricsgenreceiver → $DEST"
    ;;
  vegeta)
    DEST="$BIN_DIR/vegeta"
    [[ -f "$DEST" ]] && echo "✓ vegeta already installed" && exit 0
    URL="https://github.com/tsenart/vegeta/releases/download/v${VERSION}/vegeta_${VERSION}_${OS}_${ARCH}.tar.gz"
    echo "Downloading vegeta v${VERSION} (${OS}/${ARCH})..."
    TMP=$(mktemp)
    trap "rm -f $TMP" EXIT
    curl -fsSL "$URL" -o "$TMP"
    tar -xzf "$TMP" -C "$BIN_DIR" vegeta
    chmod +x "$DEST"
    echo "✓ vegeta → $DEST"
    ;;
  *)
    echo "Unknown binary: $BINARY" >&2
    exit 1
    ;;
esac
