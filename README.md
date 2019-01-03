# DeepFaceProper

## 0. 效果展示  
![image](https://github.com/shen1994/DeepFaceProper/raw/master/show/DeepFaceProper-male.jpg)
![image](https://github.com/shen1994/DeepFaceProper/raw/master/show/DeepFaceProper-female.jpg)

## 1. 数据集及工具  
> * [imdb face](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)

## 2. 运行命令  
> * 2.0 执行`python data_wash.py`筛选有效数据 
> * 2.1 执行`python train.py`训练数据  
> * 2.2 执行`Python convert.py`转换模型  
> * 2.3 执行`python test.py`测试单张图片数据  

## 3. 参考链接  
> * [Wide Residual Networks 论文](https://arxiv.org/pdf/1605.07146.pdf)
> * [age-gender-estimation 代码](https://github.com/yu4u/age-gender-estimation)   

## 4. 问题   
> * 年龄是一个比较感性的问题，对于老人小孩区分可能比较明显，但是对于中年人效果不是很好，数据会抖动厉害
> * 对于同一张人脸，人靠近摄像头程度不同，光照影响的不同，数据都会剧烈变化  

## 5. 更新  

