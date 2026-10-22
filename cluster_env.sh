# Sourced by the SLURM job scripts. Keeps multi-GB files out of $HOME, as the
# MS cluster handbook (Wang, 2025, sec. 3.2) asks: big data goes in
# /gluster/<username>/ or /datasets/<username>/. On the MS cluster
# /gluster does not exist (checked 10 Oct 2026), so /datasets/$USER is used.
USER="${USER:-$(id -un)}"
for d in "/gluster/$USER" "/datasets/$USER"; do
  if [ -d "$d" ] && [ -w "$d" ]; then BIG_STORE="$d"; break; fi
done
if [ -n "${BIG_STORE:-}" ]; then
  export HF_HOME="${HF_HOME:-$BIG_STORE/hf_cache}"
  mkdir -p "$HF_HOME" "$BIG_STORE/RESEARCH-models"
  # models/ in the repo is a symlink to the big store
  [ -e models ] || ln -s "$BIG_STORE/RESEARCH-models" models
  echo "Large files -> $BIG_STORE (HF_HOME=$HF_HOME)"
else
  echo "WARNING: no writable /gluster/$USER or /datasets/$USER; using \$HOME for model files" >&2
fi
# Plain HTTP downloads with timeouts: a COMET download sat at
# "Fetching 5 files: 0%" for 8 h (Oct 2026). Disabling the Xet transfer
# backend and setting timeouts makes a stalled download fail instead.
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_HUB_ETAG_TIMEOUT=60
