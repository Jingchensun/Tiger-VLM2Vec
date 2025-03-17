#!/bin/bash

# SLURM job directives
#SBATCH --job-name=vlmeval                     # 作业名称
#SBATCH --output=vlmeval.out                   # 输出日志文件
#SBATCH --error=vlmeval.err                    # 错误日志文件
#SBATCH --ntasks=1                             # 启动 1 个任务
#SBATCH --cpus-per-task=8                      # 每个任务 8 个 CPU 核心
#SBATCH --mem=128GB                            # 每个 CPU 核心分配 16GB 内存
#SBATCH --time=20:00:00                        # 运行时间（1小时）
#SBATCH --partition=prod                       # 分区设置（prod 分区）
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPUMODEL_A100-SXM4|GPUMODEL_A100-PCIE|GPUMODEL_H100-SXM5

# export WANDB_API_KEY="595cc8071abc681aa346ae6017f73fc16a9b2033"  # 替换为你的API Key
# export WANDB_MODE=online  # 确保 wandb 处于在线模式


# 加载 Conda 环境
export PATH=/usr/local/cuda-11.8/bin:$PATH   #
export CUDA_HOME=/usr/local/cuda-11.8   # 
export CUDA=1
export PATH=/home/onsi/jsun/miniconda3/envs/vlm/bin:$PATH  # Conda 环境路径
source /home/onsi/jsun/miniconda3/bin/activate vlm        # 激活 Conda 环境


python /home/onsi/jsun/VLM2Vec/eval_gme.py --subset VisualNews_t2i

# N24News CIFAR-100 HatefulMemes VOC2007 SUN397 ImageNet-A ImageNet-R ObjectNet Country211
 
# ImageNet-1K N24News HatefulMemes VOC2007 SUN397 Place365 ImageNet-A ImageNet-R ObjectNet Country211
# OK-VQA A-OKVQA DocVQA InfographicsVQA ChartQA Visual7W ScienceQA VizWiz GQA TextVQA
# VisDial CIRR VisualNews_t2i VisualNews_i2t MSCOCO_t2i MSCOCO_i2t NIGHTS WebQA FashionIQ Wiki-SS-NQ OVEN EDIS
# MSCOCO RefCOCO RefCOCO-Matching Visual7W-Pointing