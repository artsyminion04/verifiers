#!/bin/bash

# gantry commands 
# verifiers test :gantry run --show-logs --cluster=ai2/saturn --workspace=ai2/general-tool-use  --priority=normal --gpus=1  -- python -c 'print("Checking gantry in verifiers")'
# sft: gantry run --show-logs --cluster=ai2/saturn --workspace=ai2/general-tool-use --priority=high --gpus=1  -- python -c --model="Qwen/Qwen2-1.5B" examples/sft_appworld.py
# eval + sft: gantry run --show-logs --cluster=ai2/saturn --workspace=ai2/general-tool-use --priority=high --gpus=1  --python -c -n 20 -r 3 -m Qwen/Qwen3-4B -mt qwen3-4b -lt True -e appworld_env

# local python commands
# python3 examples/sft_appworld.py --name-to-save Qwen3-4B --model Qwen/Qwen3-4B --port 8001 --save-dataset  True #--name-to-save=Qwen2-1.5B-AppWorld-S

# Qwen8B eval
python mason.py \
    --cluster ai2/saturn \
    --workspace ai2/general-tool-use \
    --image shaktis/verifiers \
    --priority high \
    --budget ai2/oe-adapt \
    --gpus 1 -- python verifiers/scripts/eval.py appworld-env \
    --model-tag "qwen3-8b" \
    --model Qwen/Qwen3-8B \
    --rollouts-per-example 3 \
    --num-examples 20 \
    --local-serve True \
    --env-args '{"eval_set": "test_normal"}' \
    --save-dataset 

# Qwen8B train
# python mason.py \
#     --cluster ai2/saturn \
#     --workspace ai2/general-tool-use \
#     --image shaktis/verifiers \
#     --priority high \
#     --budget ai2/oe-adapt \
#     --gpus 2 -- accelerate launch --num-processes 2 --config-file configs/zero3.yaml examples/grpo/train_appworld.py 

# python mason.py \
#     --cluster ai2/saturn \
#     --workspace ai2/general-tool-use \
#     --image shaktis/verifiers \
#     --priority high \
#     --budget ai2/oe-adapt \
#     --gpus 2 -- vf-vllm --model willcb/Qwen3-8B-Wiki-Search-SFT \
#     --data-parallel-size 6 --enforce-eager --disable-log-requests \&\& accelerate launch --num-processes 2 --config-file configs/zero3.yaml examples/grpo/train_appworld.py 

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vf-vllm --model willcb/Qwen3-8B-Wiki-Search-SFT \
#     --data-parallel-size 6 --enforce-eager --disable-log-requests

# training:
# CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num-processes 2 \
#     --config-file configs/zero3.yaml examples/grpo/train_wiki_search.py

# python mason.py \
#     --cluster ai2/jupiter-cirrascale-2 \
#     --workspace ai2/tulu-3-dev \
#     --priority high \
#     --image nathanl/open_instruct_auto --pure_docker_mode \
#     --preemptible \
#     --num_nodes 4 \
#     --budget ai2/oe-adapt \
#     --gpus 8 -- source configs/beaker_configs/ray_node_setup.sh \&\& source configs/beaker_configs/code_api_setup.sh \&\& python open_instruct/grpo_fast.py \
#     --exp_name qwen2.5_7b_grpo_fast_zero_orz \
#     --beta 0.0 \
#     --num_unique_prompts_rollout 128 \
#     --num_samples_per_prompt_rollout 64 \
#     --kl_estimator kl3 \
#     --learning_rate 5e-7 \
#     --dataset_mixer_list vwxyzjn/rlvr_acecoder 1.0 \
#     --dataset_mixer_list_splits train \
#     --dataset_mixer_eval_list vwxyzjn/rlvr_acecoder 16 \
#     --dataset_mixer_eval_list_splits train \
#     --max_token_length 8192 \
#     --max_prompt_token_length 2048 \
#     --response_length 8192 \
#     --pack_length 16384 \
#     --model_name_or_path Qwen/Qwen2.5-7B \
#     --stop_strings '"</answer>"' \
#     --apply_r1_style_format_reward True \
#     --apply_verifiable_reward true \
#     --code_api_url \$CODE_API_URL/test_program \
#     --non_stop_penalty True \
#     --non_stop_penalty_value 0.0 \
#     --chat_template_name r1_simple_chat_postpend_think \
#     --oe_eval_tasks gsm8k::tulu,bbh:cot-v1::tulu,codex_humaneval::tulu,codex_humanevalplus::tulu,mbppplus::openinstruct \
#     --oe_eval_max_length 8192 \
#     --temperature 1.0 \
#     --masked_mean_axis 1 \
#     --total_episodes 10000000 \
#     --deepspeed_stage 2 \
#     --per_device_train_batch_size 1 \
#     --num_mini_batches 1 \
#     --num_learners_per_node 8 8 \
#     --num_epochs 1 \
#     --vllm_tensor_parallel_size 1 \
#     --vllm_num_engines 16 \
#     --lr_scheduler_type constant \
#     --seed 3 \
#     --num_evals 200 \
#     --save_freq 40 \
#     --try_launch_beaker_eval_jobs_on_weka \
#     --gradient_checkpointing \
#     --with_tracking