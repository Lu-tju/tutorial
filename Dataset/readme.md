环境：pytorch+win10  

1、将图片文件夹（class）存到在当前工程(project)目录下，并按类别分开（1、2、3）：  

```
-project  
   -class  
     -1  
     -2  
     -3  
     -...  
   -label.py
   -build_dataset.py
```

2、运行label.py为数据集生成.txt格式的标签文件，包含目录和类别（适用于win10下,目录为\\\，Ubuntu下自行修改文件）   

3、build_dataset为数据集的使用方法  
