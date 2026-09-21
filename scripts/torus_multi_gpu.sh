#!/usr/bin/env bash
# One process per GPU, each takes every NUM_GPUS-th (prompt, seed) job; then the gallery.
#   bash scripts/torus_multi_gpu.sh                                   prompts.txt, seeds 0-3, 8 GPUs
#   RUN=big SIZE=512 BATCH=3 bash scripts/torus_multi_gpu.sh
#   PROMPTS=prompts.txt,prompts3.txt bash scripts/torus_multi_gpu.sh  several sets (names may repeat across sets)
# Anything after the script name goes to torus_cli.py, e.g.  --unanchor_text=True --geo_guidance=4
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-8}
RUN=${RUN:-klein9b_$(date +%m%d_%H%M)}
PROMPTS=${PROMPTS:-prompts.txt}
SEEDS=${SEEDS:-0,1,2,3}
SIZE=${SIZE:-256}
BATCH=${BATCH:-6}

mkdir -p "output/$RUN/logs"
# Fill the HF cache once, so eight processes do not download the same files at the same time.
uv run python -c "
from huggingface_hub import hf_hub_download, snapshot_download
snapshot_download('Qwen/Qwen3-8B-FP8')
hf_hub_download('black-forest-labs/FLUX.2-klein-base-9B', 'flux-2-klein-base-9b.safetensors')
hf_hub_download('black-forest-labs/FLUX.2-dev', 'ae.safetensors')
"

for ((i = 0; i < NUM_GPUS; i++)); do
  CUDA_VISIBLE_DEVICES=$i PYTHONPATH=src uv run python scripts/torus_cli.py \
    --prompts_file "$PROMPTS" --seeds "$SEEDS" --run_name "$RUN" --shard "$i" --num_shards "$NUM_GPUS" \
    --width "$SIZE" --height "$SIZE" --batch_size "$BATCH" "$@" \
    > "output/$RUN/logs/gpu$i.log" 2>&1 &
done
echo "running on $NUM_GPUS GPUs; follow with:  tail -f output/$RUN/logs/gpu0.log"

failed=0
for job in $(jobs -p); do wait "$job" || failed=1; done
uv run python scripts/torus_gallery.py "output/$RUN"
[ $failed -eq 0 ] || echo "a shard failed: see output/$RUN/logs/ (gallery holds what finished)"
