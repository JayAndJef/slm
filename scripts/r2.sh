#!/usr/bin/env bash
# Shared R2 wiring for stage-to-r2.sh and fetch-from-r2.sh.
#
# Credentials come from the environment, never a config file: an rclone.conf holding an R2
# secret is the kind of thing that gets committed by accident, and a rented box is thrown
# away with whatever is on it.
#
# Required:  R2_ACCOUNT_ID  R2_ACCESS_KEY_ID  R2_SECRET_ACCESS_KEY  R2_BUCKET
# Optional:  R2_PREFIX (default "slm")
set -euo pipefail

: "${R2_ACCOUNT_ID:?set R2_ACCOUNT_ID}"
: "${R2_ACCESS_KEY_ID:?set R2_ACCESS_KEY_ID}"
: "${R2_SECRET_ACCESS_KEY:?set R2_SECRET_ACCESS_KEY}"
: "${R2_BUCKET:?set R2_BUCKET}"
R2_PREFIX="${R2_PREFIX:-slm}"

RCLONE="${RCLONE:-$(command -v rclone || true)}"
if [[ -z "$RCLONE" ]]; then
  # No root needed; the release is a single static binary.
  echo "rclone not found — installing to ~/.local/bin" >&2
  tmp=$(mktemp -d)
  curl -fsSL -o "$tmp/r.zip" https://downloads.rclone.org/rclone-current-linux-amd64.zip
  unzip -q -j "$tmp/r.zip" '*/rclone' -d "$tmp"
  mkdir -p "$HOME/.local/bin" && install -m 755 "$tmp/rclone" "$HOME/.local/bin/rclone"
  rm -rf "$tmp"
  RCLONE="$HOME/.local/bin/rclone"
fi

# chunk-size 128M is not tuning, it is a correctness bound: S3 multipart caps at 10,000 parts,
# and 56 GB at rclone's 5M default needs 11,468.
r2() {
  "$RCLONE" \
    --s3-provider=Cloudflare \
    --s3-access-key-id="$R2_ACCESS_KEY_ID" \
    --s3-secret-access-key="$R2_SECRET_ACCESS_KEY" \
    --s3-endpoint="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
    --s3-no-check-bucket \
    --s3-chunk-size=128M \
    --s3-upload-concurrency=8 \
    --multi-thread-streams=8 \
    --transfers=4 \
    --stats=20s --stats-one-line --progress \
    "$@"
}

REMOTE=":s3:${R2_BUCKET}/${R2_PREFIX}"

# The corpus a 500M pretrain needs, and nothing else. The sorted twin (52bdf831, another
# 56 GB) is only for length-band context extension; smoltalk2 is only for SFT.
CORPUS_FILES=(
  "train_smollm_5000+all_9d1f6853.npy"
  "train_smollm_5000+all_9d1f6853.meta"
  "val_smollm_5000_9d1f6853.npy"
  "val_smollm_5000_9d1f6853.meta"
  "tokenizer-32k.json"
)
