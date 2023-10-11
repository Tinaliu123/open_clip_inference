# compare.py
import pandas as pd
import json
import argparse
def save_to_csv(metrics: pd.DataFrame, overall_metrics: pd.DataFrame,class_result_path,overall_result_path) -> None:
 
# 保存为CSV文件
   metrics_filename = class_result_path
   overall_metrics_filename = overall_result_path
   metrics.to_csv(metrics_filename, index=False)
   overall_metrics.to_csv(overall_metrics_filename, index=False)
   print(f"Class Metrics saved to {metrics_filename}")
   print(f"Overall Metrics saved to {overall_metrics_filename}")

def main(class_metrics_path, average_f1_top1_accuracy_path,class_result_path,overall_result_path) -> None:
   with open(class_metrics_path, "r") as class_metrics_file:
        class_metrics = json.load(class_metrics_file)

   with open(average_f1_top1_accuracy_path, "r") as average_f1_top1_accuracy_file:
        result_dict = json.load(average_f1_top1_accuracy_file)
   average_f1 = result_dict["average_f1"]
   top1_accuracy = result_dict["top1_accuracy"]
   metrics_df = pd.DataFrame(class_metrics)
   overall_metrics_df = pd.DataFrame({
    "Average_F1": [average_f1],
    "Top1_Accuracy": [top1_accuracy]
   })
   save_to_csv(metrics_df,overall_metrics_df,class_result_path,overall_result_path)
   
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compare script")
    parser.add_argument("--class_metrics_output_path", type=str, default=None, help="Path to the class_metrics_output")
    parser.add_argument("--average_f1_top1_accuracy_output_path", type=str, default=None, help="Path to the average_f1_top1_accuracy_output")
    parser.add_argument("--class_result_path", type=str, default=None, help="Path to the class_metrics_output")
    parser.add_argument("--overall_result_path", type=str, default=None, help="Path to the average_f1_top1_accuracy_output")
    args = parser.parse_args()
    class_metrics_path = args.class_metrics_output_path
    average_f1_top1_accuracy_path = args.average_f1_top1_accuracy_output_path
    class_result_path = args.class_result_path
    overall_result_path = args.overall_result_path
    main(class_metrics_path, average_f1_top1_accuracy_path,class_result_path,overall_result_path)
