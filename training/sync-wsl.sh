#!/usr/bin/env bash
# Mirror a Windows working tree into the WSL2 training copy, for iterating on uncommitted
# code. Runs inside WSL:
#
#   bash training/sync-wsl.sh [SRC] [DEST]
#
#   SRC   the Windows checkout, as WSL sees it (default /mnt/c/Users/$WINUSER/Documents/projects/hobson-bidi)
#   DEST  the Linux-side copy (default ~/hobson-bidi)
#
# Code and committed data files only. The built corpora, checkpoints, reports and the HF cache
# live on the Linux filesystem and are never touched: reads across /mnt/c are slow enough to
# bottleneck loading (training/README.md#setup). Without --delete, rsync overwrites only files
# the source has, so files built under DEST/data/ survive. The excludes are anchored at the
# root: an unanchored `data/` would also skip src/strands_decider/data/.
#
# For a run that a preregistration records, check the commit out in DEST instead
# (`git fetch && git checkout <commit>`), so the run maps to a hash.
set -euo pipefail
SRC="${1:-/mnt/c/Users/${WINUSER:-marcb}/Documents/projects/hobson-bidi}"
DEST="${2:-$HOME/hobson-bidi}"
rsync -a --exclude .git --exclude /data/ --exclude /checkpoints/ --exclude /reports/ \
  --exclude __pycache__ "$SRC/" "$DEST/"
rsync -a --exclude raw/ "$SRC/data/" "$DEST/data/"
echo "synced $SRC -> $DEST"
