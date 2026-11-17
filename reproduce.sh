#!/usr/bin/env bash
# Reproduce results/reference.md: baseline (24 layers) and the 14-layer detector, scored in one encoder pass,
# on In-the-Wild and ASVspoof 5 dev, window 4 s and full length, through the simulated channels; then the
# paired-bootstrap EER difference for every cell.
#
#   ITW=/data/in_the_wild ASV5=/data/asvspoof5/flac_D ASV5_PROTOCOL=/data/.../ASVspoof5.dev.track_1.tsv \
#       bash reproduce.sh
# Optional: ITW_CHANNELS / ASV5_CHANNELS (window 4 s), ITW_FULL_CHANNELS / ASV5_FULL_CHANNELS (full length),
# THREADS (default 8), OUT (default results). On an RTX 5060 one In-the-Wild channel takes about 30 min at full
# length for both models; the defaults match the cells in results/reference.md.
# An interrupted run resumes: cells already scored are skipped clip by clip.
set -euo pipefail
cd "$(dirname "$0")"

BASE=weights/nes2netx_itw_vg.pt
L14=weights/voiceguard_L14.pt
OUT=${OUT:-results}
THREADS=${THREADS:-8}
ITW_CHANNELS=${ITW_CHANNELS:-"none g711u nb8k gsm amrnb amrwb g722 opus12 mp3_32"}
ITW_FULL_CHANNELS=${ITW_FULL_CHANNELS:-"none gsm amrnb opus12 mp3_32"}
ASV5_CHANNELS=${ASV5_CHANNELS:-"none amrnb opus12"}
ASV5_FULL_CHANNELS=${ASV5_FULL_CHANNELS:-"none amrnb opus12"}

for f in "$BASE" "$L14"; do
  [ -f "$f" ] || { echo "missing $f: see weights/README.md"; exit 1; }
done
python -m voiceguard.channel || echo "warning: some channels are unavailable in this ffmpeg (see codecs/README.md)"

durations() {  # channel window_list full_list -> "4", "full" or "4 full"
  local out=""
  [[ " $2 " == *" $1 "* ]] && out="4"
  [[ " $3 " == *" $1 "* ]] && out="$out full"
  echo $out
}

score() {  # name data_dir labels channel durations extra_args...
  local name=$1 dir=$2 labels=$3 ch=$4 durs=$5; shift 5
  [ -n "$durs" ] || return 0
  python -m voiceguard.evaluate --ckpt "$BASE" --ckpt_shared "$L14" --data_dir "$dir" --labels "$labels" \
      --channel "$ch" --durations $durs --fp16 --threads "$THREADS" --out_dir "$OUT" \
      --prefix "${name}_base_" --prefix_shared "${name}_L14_" "$@"
  local c=$([ "$ch" = none ] && echo clean || echo "$ch")
  for d in $durs; do
    d=$([ "$d" = full ] && echo full || echo "${d}s")
    python -m voiceguard.compare --labels "$labels" --ref "$OUT/scores/${name}_base_${c}_${d}.txt" \
        --model "$OUT/scores/${name}_L14_${c}_${d}.txt" --out "$OUT/delta_${name}_${c}_${d}.json"
  done
}

if [ -n "${ITW:-}" ]; then
  python -m voiceguard.datasets check --data_dir "$ITW" --labels "$ITW/meta.csv"
  for ch in $(echo $ITW_CHANNELS $ITW_FULL_CHANNELS | tr ' ' '\n' | awk '!s[$0]++'); do
    score itw "$ITW" "$ITW/meta.csv" "$ch" "$(durations "$ch" "$ITW_CHANNELS" "$ITW_FULL_CHANNELS")"
  done
fi

if [ -n "${ASV5:-}" ]; then
  LABELS=datasets/asv5_dev_labels.txt
  [ -f "$LABELS" ] || python -m voiceguard.datasets asv5 --protocol "${ASV5_PROTOCOL:?set ASV5_PROTOCOL}" --out "$LABELS"
  python -m voiceguard.datasets check --data_dir "$ASV5" --labels "$LABELS"
  for ch in $(echo $ASV5_CHANNELS $ASV5_FULL_CHANNELS | tr ' ' '\n' | awk '!s[$0]++'); do
    score asv5dev "$ASV5" "$LABELS" "$ch" "$(durations "$ch" "$ASV5_CHANNELS" "$ASV5_FULL_CHANNELS")" --subset 20000 --seed 2026
  done
fi
