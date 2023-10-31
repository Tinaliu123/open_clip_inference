# evaluate.py
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from typing import List, Dict,Tuple
import json
import pandas as pd
import argparse
import os
def evaluate(true_labels, predicted_labels, unique_labels) -> Tuple[np.ndarray, List[Dict[str, float]]]:
  
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    class_metrics1 = []
    class_metrics2 = []
    for idx, label in enumerate(unique_labels):
      true_positive = conf_matrix[idx, idx]
      false_positive = conf_matrix[:, idx].sum() - true_positive
      false_negative = conf_matrix[idx, :].sum() - true_positive

      precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
      recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
      f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
      
      class_metrics1.append({
        "Label": label,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
      })
      # 将 Precision、Recall、F1 扩大一百倍并保留两位小数
      precision *= 100
      recall *= 100
      f1 *= 100
      precision = round(precision, 2)
      recall = round(recall, 2)
      f1 = round(f1, 2)

      class_metrics2.append({
        "Label": label,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
      })

# 计算所有类别的平均F1、precision、recall和top-1准确率
    average_precision = np.mean([metric["Precision"] for metric in class_metrics1])
    average_recall = np.mean([metric["Recall"] for metric in class_metrics1])
    average_f1 = np.mean([metric["F1"] for metric in class_metrics1])
    top1_accuracy = accuracy_score(true_labels, predicted_labels)

    average_precision*= 100
    average_recall *= 100
    average_f1 *= 100
    top1_accuracy *= 100
    average_precision = round(average_precision, 2)
    average_recall = round(average_recall, 2)
    average_f1 = round(average_f1, 2)
    top1_accuracy = round(top1_accuracy, 2)
    return class_metrics2,average_precision, average_recall,average_f1,top1_accuracy
def main(true_labels_path, predicted_labels_path, unique_labels_path,class_metrics_output_path,average_f1_top1_accuracy_output_path):
    # 读取 true labels、predicted labels 和 unique labels 的内容
    true_labels_df = pd.read_csv(true_labels_path)
    predicted_labels_df = pd.read_csv(predicted_labels_path)
    unique_labels_df = pd.read_csv(unique_labels_path)


    true_labels = true_labels_df["true_labels"].tolist()
    predicted_labels = predicted_labels_df["predicted_labels"].tolist()
    unique_labels = unique_labels_df["unique_labels"].tolist()

    # 调用 evaluate 函数获取结果
    class_metrics, average_precision, average_recall, average_f1, top1_accuracy = evaluate(true_labels, predicted_labels, unique_labels)

    class_metrics_json_path = class_metrics_output_path # Specify the path and filename
    average_f1_top1_accuracy_json_path = average_f1_top1_accuracy_output_path  # Specify the path and filename

    class_metrics_dict = {"class_metrics": class_metrics}
    with open(class_metrics_json_path, "w") as class_metrics_json_file:
      json.dump(class_metrics_dict, class_metrics_json_file)

    average_f1_top1_accuracy_dict = {
      "average_precision": average_precision,
      "average_recall": average_recall,
      "average_f1": average_f1,
      "top1_accuracy": top1_accuracy
    }
    with open(average_f1_top1_accuracy_json_path, "w") as af1_top1_acc_json_file:
      json.dump(average_f1_top1_accuracy_dict, af1_top1_acc_json_file)
    print(f"Evaluation results saved to {class_metrics_json_path}, {average_f1_top1_accuracy_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate script")
    parser.add_argument("--model_name", type=str, default="ViT-B-32", help="Model name")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model weights")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to the dataset")
    args = parser.parse_args()

    # 提取 model_path 和 dataset_path 的最后一部分
    if args.model_path is not None:
      model_name_with_extension = os.path.basename(args.model_path)
      model_name, extension = os.path.splitext(model_name_with_extension)
    else:
      model_name = "default_model"

    if args.dataset_path is not None:
      dataset_name = os.path.basename(args.dataset_path)
    else:
      dataset_name = "default_dataset"
  
     # 获取当前脚本所在文件夹的路径
    current_script_path = os.path.dirname(os.path.abspath(__file__))
     # 获取上一级的路径
    parent_path = os.path.dirname(current_script_path)
    intermediate = os.path.join(parent_path,"intermediate")
    intermediate_path = os.path.join(intermediate, f"{model_name}_{dataset_name}")
    true_labels_path = os.path.join(intermediate_path,"true_labels.csv") 
    predicted_labels_path = os.path.join(intermediate_path,"predicted_labels.csv")
    unique_labels_path = os.path.join(intermediate_path,"unique_labels.csv")
    class_metrics_output_path = os.path.join(intermediate_path,"recall_precision_f1.json")
    average_f1_top1_accuracy_output_path = os.path.join(intermediate_path,"avgf1_top1acc.json")
    main(true_labels_path, predicted_labels_path, unique_labels_path,class_metrics_output_path,average_f1_top1_accuracy_output_path)

    
