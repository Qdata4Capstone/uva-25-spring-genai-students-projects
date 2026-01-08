#!/bin/bash
conda activate dl
MAX_JOBS_PER_GPU=5
NUM_GPUS=2
declare -A GPU_JOBS

# 初始化每个 GPU 的任务数为 0
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    GPU_JOBS[$gpu]=0
done

# 用于记录后台运行的进程 PID，方便 wait 控制
PIDS=()

# 遍历所有任务组合
for topic in {0..4}; do
    for example in {0..9}; do
        # 找到第一个空闲的 GPU（任务数 < 6）
        while true; do
            for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
                if [ ${GPU_JOBS[$gpu]} -lt $MAX_JOBS_PER_GPU ]; then
                    echo "Assigning GPU $gpu -> topic_idx=$topic, example_idx=$example"
                    # 启动任务
                    /home/mpp7ez/miniconda3/envs/dl/bin/python parallel_inference.py --gpu_num $gpu --topic_idx $topic --example_idx $example &
                    pid=$!
                    PIDS+=($pid)
                    GPU_JOBS[$gpu]=$((GPU_JOBS[$gpu] + 1))
                    break 2
                fi
            done

            # 如果所有 GPU 都满了，等待任一任务完成
            if [ ${#PIDS[@]} -gt 0 ]; then
                wait -n
                # 清理完成的任务
                for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
                    GPU_JOBS[$gpu]=$(ps -eo pid,args | grep -v grep | grep -c "/home/mpp7ez/miniconda3/envs/dl/bin/python parallel_inference.py --gpu_num $gpu")
                done
            fi
        done
    done
done

# 等待所有剩余任务完成
wait
echo "All tasks completed."
