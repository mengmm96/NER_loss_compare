# NER_loss_compare

 基于CLUENER 细粒度命名实体识别的数据集使用不同损失函数下的效果对比。



## 项目描述

本项目主要是使用了Bert + two_pointer的方式在CLUENER的任务上对四种常用的损失函数的表现进行了对比。

涉及的损失函数有CE,Label Smoothing,Focal loss和GHMC。

效果的对比如下：

![1598236997310](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\1598236997310.png)

各损失函数的变化趋势：

![1598237047652](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\1598237047652.png)

![1598237066432](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\1598237066432.png)

从loss曲线来看，四个损失函数下模型已经收敛，从F1效果看整体的效果是：

lsr的效果最好，其次是focal，然后是ghmc最后是ce。

也可以看出四个损失函数下其实模型的差距并不显著，不同的损失函数的表现和数据集也是有很大的关系的。



## 运行环境

python 3.6,pytorch == 1.5.0，tensorboardx



## 项目结构

- datasets：CLUENER 的数据集
- display：模型训练过程中的Train loss,Eval loss和各标签F1的趋势图
- models：模型代码
- output：模型训练生成文件
- pretrained_model:预训练bert模型
- common.py：通用方法代码
- data_process.py ： 数据处理代码
- train_model.py ： 模型训练代码
- eval_model.py：模型验证代码
- run_ner_span.py：main文件，相关参数，路径等配置

模型参数(详见run_ner_span.py中的parser)



## 运行

```python
python run_ner_span.py
```





