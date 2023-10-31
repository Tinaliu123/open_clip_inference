#!/bin/bash


MODEL_NAME="ViT-B-32"
#模型名称

MODEL_PATH="/remote-home/liuyi/open_clip-main/models/open_clip_ViT-B-32.pt"
#权重路径

DATASET_PATH="/remote-home/liuyi/open_clip-main/data/clsf/EuroSAT_RGB"
#数据集路径


# 解析命令行参数，如果用户提供了新的参数，则使用新的参数值
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_name)
            MODEL_NAME=$2
            shift 2
            ;;
        --model_path)
            MODEL_PATH=$2
            shift 2
            ;;
        --dataset_path)
            DATASET_PATH=$2
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done


# 运行 inference.py
python scripts/inference.py --model_name "$MODEL_NAME" --model_path "$MODEL_PATH" --dataset_path "$DATASET_PATH" 


# 运行evaluate.py

python scripts/evaluate.py --model_name "$MODEL_NAME" --model_path "$MODEL_PATH" --dataset_path "$DATASET_PATH" 

# 运行 compare.py
python scripts/compare.py --model_name "$MODEL_NAME" --model_path "$MODEL_PATH" --dataset_path "$DATASET_PATH" 