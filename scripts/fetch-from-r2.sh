#!/usr/bin/env bash
# Pull the staged corpus onto a training box and verify it against the manifest.
#
#   DEST=/scratch/slm-data R2_ACCOUNT_ID=... ... scripts/fetch-from-r2.sh
#
# Point training at whatever DEST is with `--data-dir` and `--tokenizer`; the corpus hash is
# computed from the spec and the tokenizer fingerprint, never from a path, so the files may
# live anywhere. Their *names* must survive intact — that is what `locate` looks for.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/r2.sh

DEST="${DEST:-$PWD/data}"
mkdir -p "$DEST"

avail=$(df -PBG "$DEST" | awk 'NR==2 {print $4+0}')
(( avail >= 60 )) || { echo "only ${avail}G free at $DEST; need ~60G" >&2; exit 1; }

echo "=== downloading to $DEST ==="
r2 copy "${REMOTE}/data" "$DEST"

echo "=== verifying ==="
( cd "$DEST" && sha256sum -c SHA256SUMS )
echo "ok — corpus verified at $DEST"
