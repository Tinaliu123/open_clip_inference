使用说明  
========  
说明
-------------          
scripts下包含所有的脚本文件       
模型参数保存在models文件夹   
分类任务的数据集保存在data/clsf  
测试集会自动生成在test_set目录下  
中间结果保存在intermediate下  
最终结果保存在results下  

参数修改
-------
打开run.sh  
```python

MODEL_NAME="ViT-B-32"
#指定模型名称

MODEL_PATH="/root/home/open_clip-main/RemoteCLIP-ViT-B-32.pt"
#指定模型参数路径

DATASET_PATH="/root/home/open_clip-main/EuroSAT_RGB"
#指定数据集路径


```
执行bash文件  
-------------
命令行输入**run.sh路径**，回车运行 




