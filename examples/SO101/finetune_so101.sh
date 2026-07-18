set -x -e

export NUM_GPUS=1
source /workspace/export_vars.sh
export HUGGINGFACE_HUB_CACHE="/workspace/hf_cache"
CUDA_VISIBLE_DEVICES=0 uv run python \
    gr00t/experiment/launch_finetune.py \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path  examples/SO101/so101_bench_sim_4_WM \
    --modality_config_path examples/SO101/so101_config.py \
    --embodiment_tag NEW_EMBODIMENT \
    --num_gpus $NUM_GPUS \
    --output_dir /workspace/so101_GR00T-N1.6-3B_WM_sim_v1 \
    --save_steps 2500 \
    --save_total_limit 8 \
    --max_steps 21000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --use_wandb \
    --global_batch_size 112 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader_num_workers 4
