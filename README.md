使用说明  
========  
存放路径
-------------          
run.sh、inference.py、evaluate.py、compare.py存放至open_clip-main的**一级目录下**  

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




