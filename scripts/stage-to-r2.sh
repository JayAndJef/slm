#!/usr/bin/env bash
# Upload the smollm corpus + tokenizer to R2, with a checksum manifest.
#
# The manifest is the point. rclone verifies each multipart *part*, but a whole-file digest
# is what catches a transfer that silently lost a chunk — and a corrupt token stream trains
# without complaint, since every uint16 is a valid id.
#
#   R2_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... R2_BUCKET=... \
#     scripts/stage-to-r2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/r2.sh

echo "=== files to stage ==="
for f in "${CORPUS_FILES[@]}"; do
  [[ -f "data/$f" ]] || { echo "missing: data/$f" >&2; exit 1; }
  printf '  %10s  %s\n' "$(du -h "data/$f" | cut -f1)" "$f"
done
printf '  %10s  total\n' "$(du -ch "${CORPUS_FILES[@]/#/data/}" | tail -1 | cut -f1)"

echo "=== hashing (~1 min) ==="
( cd data && sha256sum "${CORPUS_FILES[@]}" ) > data/SHA256SUMS
cat data/SHA256SUMS

echo "=== uploading to ${REMOTE}/data/ ==="
list=$(mktemp); trap 'rm -f "$list"' EXIT
printf '%s\n' "${CORPUS_FILES[@]}" SHA256SUMS > "$list"
r2 copy data "${REMOTE}/data" --files-from "$list"

echo "=== remote listing ==="
r2 ls "${REMOTE}/data"
echo "done — fetch on the box with scripts/fetch-from-r2.sh"
