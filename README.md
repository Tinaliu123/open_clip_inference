使用说明  
========  
##1. 存放路径          
run_scripts.sh、inference.py、evaluate.py、compare.py存放至open_clip-main的**一级目录下**  
##2. 参数修改
打开run_scripts.sh  
```python  
MODEL_NAME="ViT-B-32" #inference输入参数，模型名称
MODEL_PATH="/root/home/open_clip-main/RemoteCLIP-ViT-B-32.pt" #inference输入参数，权重路径  
DATASET_PATH="/root/home/open_clip-main/EuroSAT_RGB" #inference输入参数，数据集路径 
TRUE_LABELS_OUTPUT_PATH="true_labels.csv" #inference输出/evaluate输入，ground truth，csv格式  
PREDICTED_LABELS_OUTPUT_PATH="predicted_labels.csv" #inference输出/evaluate输入，预测标签，csv格式  
UNIQUE_LABELS_OUTPUT_PATH="unique_labels.csv" #inference输出/evaluate输入，类别名称，csv格式  
CLASS_METRICS_OUTPUT_PATH="class_metrics.json" #evaluate输出/compare输入，（各类别）recall，precision，F1中间结果，json格式  
AVERAGE_F1_TOP1_ACCURACY_OUTPUT_PATH="average_f1_top1_accuracy.json" #evaluate输出/compare输入，（所有类）平均F1、top1准确率中间结果，json格式 
CLASS_RESULT_PATH="class_metrics.csv"#compare输出，（各类别）recall，precision，F1最终结果，csv保存
OVERALL_RESULT_PATH="overall_metrics.csv" #compare输出，（所有类）平均F1、top1准确率最终结果，csv格式
```
##3. 执行bash文件  
命令行输入**run_scripts.sh存放路径**，回车运行 




