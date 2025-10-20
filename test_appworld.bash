#!/bin/bash

# verifiers test
# gantry run --show-logs --cluster=ai2/saturn --workspace=ai2/general-tool-use  --priority=normal --gpus=1  -- python -c 'print("Checking gantry in verifiers")'


# sft 
# gantry run --show-logs --cluster=ai2/saturn --workspace=ai2/general-tool-use --priority=high --gpus=1  -- python -c --model="Qwen/Qwen2-1.5B" examples/sft_appworld.py


# Eval for gantry
# PORT=8001

# echo "Starting vLLM server..."

# vllm serve "Qwen/Qwen3-4B" --port $PORT --enable-auto-tool-choice --tool-call-parser hermes
# VLLM_PID=$!

# until curl -s http://0.0.0.0:$PORT/v1/models > /dev/null; do
#   echo "Waiting for vLLM API to become available..."
#   sleep 2
# done

# # eval command
# python3 examples/sft_appworld.py --name-to-save Qwen3-4B --model Qwen/Qwen3-4B --port 8001 --save-dataset  True #--name-to-save=Qwen2-1.5B-AppWorld-S

# echo "Stopping vLLM server..."
# kill $VLLM_PID

# eval + launch vllm in same script
# gantry run --show-logs --cluster=ai2/saturn --workspace=ai2/general-tool-use --priority=high --gpus=1  --python -c -n 20 -r 3 -m Qwen/Qwen3-4B -mt qwen3-4b -lt True -e appworld_env
