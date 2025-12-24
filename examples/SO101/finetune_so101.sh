set -x -e

"""
from huggingface_hub import snapshot_download
from huggingface_hub import login

login()

snapshot_download(
    repo_id="5hadytru/so101_grasp_2",
    repo_type="dataset",
    revision="v3.0",
    local_dir="/workspace/",
    local_dir_use_symlinks=False,
)

d = "so101_grasp_2"
snapshot_download(
    repo_id="5hadytru/{d}",
    repo_type="dataset",
    local_dir=f"{d}",
    local_dir_use_symlinks=False,
)
"""

export NUM_GPUS=1
source /workspace/export_vars.sh
export HUGGINGFACE_HUB_CACHE="/workspace/hf_cache"
CUDA_VISIBLE_DEVICES=0 uv run python \
    gr00t/experiment/launch_finetune.py \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path  examples/SO101/so101_grasp_2 examples/SO101/so101_IF_1_v2.1 \
    --modality_config_path examples/SO101/so101_config.py \
    --embodiment_tag NEW_EMBODIMENT \
    --num_gpus $NUM_GPUS \
    --output_dir /workspace/so101_GR00T-N1.6-3B_v5_1 \
    --save_steps 1000 \
    --save_total_limit 1 \
    --max_steps 60000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --use_wandb \
    --global_batch_size 128 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader_num_workers 4

"""
uv run python scripts/lerobot_conversion/convert_v3_to_v2.py --repo-id 5hadytru/so101_grasp_3 --root examples/SO101/so101_grasp_3
cp modality.json examples/SO101/so101_grasp_3/meta/modality.json
"""
