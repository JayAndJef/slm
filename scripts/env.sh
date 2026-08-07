# Caches that `slm.paths` cannot own, kept off the two nearly-full filesystems here.
#
#   source scripts/env.sh
#
# /tmp is on / (70 G) and /home is at 99%; only /data has room. The inductor and triton
# caches are handled automatically by slm/paths.py — everything below is what no import of
# `slm` can reach, because it is read before Python starts or by a different tool.
CACHE="${SLM_CACHE_DIR:-/data/zejiaqi/cache}"
export SLM_CACHE_DIR="$CACHE"   # slm/paths.py reads this for the compile caches

export TMPDIR="$CACHE/tmp"                         # backstop for anything not named here
export WANDB_CACHE_DIR="$CACHE/wandb"
export WANDB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # this clone, not a fixed path
export UV_CACHE_DIR="$CACHE/uv"

mkdir -p "$TMPDIR" "$WANDB_CACHE_DIR" "$UV_CACHE_DIR"
