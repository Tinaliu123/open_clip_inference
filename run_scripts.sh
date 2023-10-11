#!/bin/bash


MODEL_NAME="ViT-B-32"
#inference输入参数，模型名称

MODEL_PATH="/root/home/open_clip-main/RemoteCLIP-ViT-B-32.pt"
#inference输入参数，权重路径

DATASET_PATH="/root/home/open_clip-main/EuroSAT_RGB"
#inference输入参数，数据集路径

TRUE_LABELS_OUTPUT_PATH="true_labels.csv"
#inference输出/evaluate输入，ground truth，csv格式

PREDICTED_LABELS_OUTPUT_PATH="predicted_labels.csv" 
#inference输出/evaluate输入，预测标签，csv格式

UNIQUE_LABELS_OUTPUT_PATH="unique_labels.csv"
#inference输出/evaluate输入，类别名称，csv格式 

CLASS_METRICS_OUTPUT_PATH="class_metrics.json"
#evaluate输出/compare输入，（各类别）recall，precision，F1中间结果，json格式

AVERAGE_F1_TOP1_ACCURACY_OUTPUT_PATH="average_f1_top1_accuracy.json"
#evaluate输出/compare输入，（所有类）平均F1、top1准确率中间结果，json格式

CLASS_RESULT_PATH="class_metrics.csv"
#compare输出，（各类别）recall，precision，F1最终结果，csv保存

OVERALL_RESULT_PATH="overall_metrics.csv"
#compare输出，（所有类）平均F1、top1准确率最终结果，csv格式

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
python inference.py --model_name "$MODEL_NAME" --model_path "$MODEL_PATH" --dataset_path "$DATASET_PATH" --true_labels_output_path "$TRUE_LABELS_OUTPUT_PATH" --predicted_labels_output_path "$PREDICTED_LABELS_OUTPUT_PATH" --unique_labels_output_path "$UNIQUE_LABELS_OUTPUT_PATH"


# 运行evaluate.py

python evaluate.py --true_labels_output_path "$TRUE_LABELS_OUTPUT_PATH" --predicted_labels_output_path "$PREDICTED_LABELS_OUTPUT_PATH" --unique_labels_output_path "$UNIQUE_LABELS_OUTPUT_PATH" --class_metrics_output_path "$CLASS_METRICS_OUTPUT_PATH" --average_f1_top1_accuracy_output_path "$AVERAGE_F1_TOP1_ACCURACY_OUTPUT_PATH"

# 运行 compare.py
python compare.py --class_metrics_output_path "$CLASS_METRICS_OUTPUT_PATH" --average_f1_top1_accuracy_output_path "$AVERAGE_F1_TOP1_ACCURACY_OUTPUT_PATH" --class_result_path "$CLASS_RESULT_PATH" --overall_result_path "$OVERALL_RESULT_PATH"