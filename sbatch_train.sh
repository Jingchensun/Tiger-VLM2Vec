#!/bin/bash

# SLURM job directives
#SBATCH --job-name=vlm_ori                     # 作业名称
#SBATCH --output=vlm_ori.out                   # 输出日志文件
#SBATCH --error=vlm_ori.err                    # 错误日志文件
#SBATCH --ntasks=1                             # 启动 1 个任务
#SBATCH --cpus-per-task=8                      # 每个任务 8 个 CPU 核心
#SBATCH --mem=128GB                            # 每个 CPU 核心分配 16GB 内存
#SBATCH --time=23:00:00                        # 运行时间（1小时）
#SBATCH --partition=prod                       # 分区设置（prod 分区）
#SBATCH --gres=gpu:4


export WANDB_API_KEY="595cc8071abc681aa346ae6017f73fc16a9b2033"  # 替换为你的API Key
export WANDB_MODE=online  # 确保 wandb 处于在线模式


# 加载 Conda 环境
export PATH=/usr/local/cuda-11.8/bin:$PATH   #
export CUDA_HOME=/usr/local/cuda-11.8   # 
export CUDA=1
export PATH=/home/onsi/jsun/miniconda3/envs/vlm/bin:$PATH  # Conda 环境路径
source /home/onsi/jsun/miniconda3/bin/activate vlm        # 激活 Conda 环境

 torchrun --nproc_per_node=4 --master_port=22449 --max_restarts=0 /home/onsi/jsun/VLM2Vec/train.py \
 --model_name Alibaba-NLP/gme-Qwen2-VL-2B-Instruct --bf16 --pooling last \
 --model_backbone qwen \
 --dataset_name TIGER-Lab/MMEB-train \
 --dataset_split original \
 --subset_name MSCOCO_t2i \
 --num_sample_per_subset 50000 \
 --image_dir MMEB-train \
 --max_len 4096 --num_crops 4 --output_dir checkpoint_mscoco_t2i_qwengme --logging_steps 1 \
 --lr_scheduler_type linear --learning_rate 2e-5 --max_steps 502 \
 --warmup_steps 200 --save_steps 100 --normalize True \
 --temperature 0.02 --per_device_train_batch_size 16 \
 --grad_cache True --gc_q_chunk_size 2 --gc_p_chunk_size 2 --lora --lora_r 8

# ImageNet-1K N24News HatefulMemes VOC2007 SUN397 OK-VQA A-OKVQA DocVQA InfographicsVQA 
# ChartQA Visual7W VisDial CIRR VisualNews_t2i VisualNews_i2t MSCOCO_t2i MSCOCO_i2t NIGHTS WebQA MSCOCO 