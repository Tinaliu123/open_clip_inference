# evaluate.py
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from typing import List, Dict,Tuple
import json
import pandas as pd
import argparse
def evaluate(true_labels, predicted_labels, unique_labels) -> Tuple[np.ndarray, List[Dict[str, float]]]:
  
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    class_metrics = []
    for idx, label in enumerate(unique_labels):
      true_positive = conf_matrix[idx, idx]
      false_positive = conf_matrix[:, idx].sum() - true_positive
      false_negative = conf_matrix[idx, :].sum() - true_positive

      precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
      recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
      f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

      class_metrics.append({
        "Label": label,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
      })

# 计算所有类别的平均F1和top-1准确率
    average_f1 = np.mean([metric["F1"] for metric in class_metrics])
    top1_accuracy = accuracy_score(true_labels, predicted_labels)
    return class_metrics,average_f1,top1_accuracy
def main(true_labels_path, predicted_labels_path, unique_labels_path,class_metrics_output_path,average_f1_top1_accuracy_output_path):
    # 读取 true labels、predicted labels 和 unique labels 的内容
    true_labels_df = pd.read_csv(true_labels_path)
    predicted_labels_df = pd.read_csv(predicted_labels_path)
    unique_labels_df = pd.read_csv(unique_labels_path)


    true_labels = true_labels_df["true_labels"].tolist()
    predicted_labels = predicted_labels_df["predicted_labels"].tolist()
    unique_labels = unique_labels_df["unique_labels"].tolist()

    # 调用 evaluate 函数获取结果
    class_metrics, average_f1, top1_accuracy = evaluate(true_labels, predicted_labels, unique_labels)

    class_metrics_json_path = class_metrics_output_path # Specify the path and filename
    average_f1_top1_accuracy_json_path = average_f1_top1_accuracy_output_path  # Specify the path and filename

    class_metrics_dict = {"class_metrics": class_metrics}
    with open(class_metrics_json_path, "w") as class_metrics_json_file:
      json.dump(class_metrics_dict, class_metrics_json_file)

    average_f1_top1_accuracy_dict = {
      "average_f1": average_f1,
      "top1_accuracy": top1_accuracy
    }
    with open(average_f1_top1_accuracy_json_path, "w") as af1_top1_acc_json_file:
      json.dump(average_f1_top1_accuracy_dict, af1_top1_acc_json_file)
    print(f"Evaluation results saved to {class_metrics_json_path}, {average_f1_top1_accuracy_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate script")
    parser.add_argument("--true_labels_output_path", type=str, default=None, help="Path to the true_labels_output")
    parser.add_argument("--predicted_labels_output_path", type=str, default=None, help="Path to the predicted_labels_output")
    parser.add_argument("--unique_labels_output_path", type=str, default=None, help="Path to the unique_labels_output")
    parser.add_argument("--class_metrics_output_path", type=str, default=None, help="Path to the class_metrics_output")
    parser.add_argument("--average_f1_top1_accuracy_output_path", type=str, default=None, help="Path to the average_f1_top1_accuracy_output")
    args = parser.parse_args()
    true_labels_path = args.true_labels_output_path
    predicted_labels_path = args.predicted_labels_output_path
    unique_labels_path = args.unique_labels_output_path
    class_metrics_output_path = args.class_metrics_output_path
    average_f1_top1_accuracy_output_path = args.average_f1_top1_accuracy_output_path
    main(true_labels_path, predicted_labels_path, unique_labels_path,class_metrics_output_path,average_f1_top1_accuracy_output_path)

    
