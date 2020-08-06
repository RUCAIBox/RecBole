# 内部开发手册

## 开发规范

### 1. 代码文件header注明
在新建或者更改代码文件时，需在文件头部注明时间，作者和邮件信息。具体格式如下：

````
\# @Time   : xxxx/x/x
\# @Author : Xxx Xx
\# @Email  : xxx@xx

# UPDATE:
\# @Time   : xxxx/x/x, xxxx/x/x
\# @Author : Xxx Xx, Xxx Xx
\# @Email  : xxx@xx, xxx@xx
````
最上方为创建者的信息，下方为更新者的信息，如果有多次更新，只需要保持最新的时间在上面即可。
其中`@` 和`:`之间为7个字符的间距.

## 如何开发一个新模型？

### 1.确定要开发的模型

在下表中登记模型信息，包括模型名称，类别，模型状态，以及现负责人。

   模型名称：该模型常用的简称或全称

   模型类别：general （传统top-k推荐），context-aware （FM类等利用较多特征信息方法），
   sequential（序列推荐方法，包括next-item，next-session）。如有新的模型类别，请加入到此处。

   模型状态：按照开发程度由低到高分为：未认领（没有人认领），开发中（正在编写代码，模型没有办法在现有框架下跑通），
   调试中（编码基本完成，也可以顺利跑通，但未经过最后的测试），已上线（编写模型通过最后测试，上线成功）

   现负责人：保证每一个模型都有专门的负责人，负责开源之后的模型维护工作。当不处于这个项目后，需要把自己负责的模型进行转交，更新最新的负责人。

| 序号 | 模型名称   | 模型类别      | 模型状态 | 现负责人 |
| ---- | ---------- | ------------- | -------- | -------- |
| 1    | Popularity | general       | 开发中   | 林子涵   |
| 2    | ItemKNN    | general       | 开发中   | 林子涵   |
| 3    | BPRMF      | general       | 测试中   | 林子涵   |
| 4    | NCF        | general       | 测试中   | 林子涵   |
| 5    | NGCF       | general       | 开发中   | 林子涵   |
| 6    | LightGCN   | general       | 未认领   |          |
| 7    | FM         | context-aware | 开发中   | 牟善磊   |
| 8    | DeepFM     | context-aware | 开发中   | 牟善磊   |
| 9    | NFM        | context-aware | 开发中   | 林子涵   |
| 10   | Wide&Deep  | context-aware | 开发中   | 牟善磊   |
| 11   | GRU4Rec    | sequential    | 开发中   | 牟善磊   |
| 12   | SASRec     | sequential    | 未认领   |          |
| 13   | FPMC       | sequential    | 未认领   |          |

### 2.代码编写

1) 在 model/模型类别/ 目录下新建py模型文件，若不存在相对应的模型类别目录，需要新建目录。
2) 模型需要继承 AbstractRecommender 这个类，这个类要求实现 forward(), calculate_loss(), predict()三个方法。
forward() 为模型前向传播逻辑，calculate_loss() 为训练时调用的方法，输入为PyTorch常见训练类型数据，输出为模型Loss。
predict() 为评测时用到的方法，输出要为score或者其他任务适配的用来评测的内容。
3) 编写模型文件前，可以查看一下是否有与待实现模型输入输出以及结构相似的模型，如果有可以参考进行编写，事半功倍。
4) 在写模型结构时，包括各种layers和loss，如果为常见的layers和loss可以查看model/layers.py和model/loss.py看有没有已经实现的相应模块，
如果有可以直接调用。如果没有且该模块非常常用，可以添加到layers.py或loss.py文件中。
5) 模型相关的超参数，需要写入到配置文件中，配置文件目录：properties/model/
6) 代码规范请参考 Python PEP8编码规范

### 3.测试上线

1) 首先需要保证模型能够顺利运行，可选取ml-100k这个数据集进行测试。简单测试方式:调整好数据集配置文件，模型配置文件，overall配置文件，
运行run_test.py，或直接运行`python run_test.py --model='model_name' --dataset=ml-100k`,检查是否报错。
2) 保证模型顺利运行后，需要逐字检查模型文件，看是否有逻辑错误或其他错误。
3) 在上线测试数据集上，进行训练评测。一种方式（推荐）：利用RecBox自动调参工具，按照模型类别找到相应的数据集及评测方式，设置好相应的配置文件，
保证要测试的设置在run_test.py中能够无误跑通，随后在hyper.test 中按照要求设置要调整的超参范围，调整好后执行`run_hyper.py --max_evals 'eval_nums'`，
`max_evals`控制搜索次数，返回得到最优的超参和测试集结果，将结果填入对应模型的配置文件中以及下表。另一种方式：自行调参。
4) 检查3）得到的结果是否异常（是否与其他模型相差过大以及其他判断方式），正常无误模型可以顺利上线，结果异常需进一步检查代码，若还未发现问题，请及时与同学和老师进行沟通。

## **测试使用的数据集及评价指标**

### **general类 模型**

使用数据集： ml-1m、yelp、Amazon-game  （所有数据集均过滤评分小于3的数据项）

评测方式：8：1：1 ,全排序

评价指标 ：Recall@20、NGCG@20、MRR

|             | ml-1m     | ml-1m   | ml-1m  |
| ----------- | --------- | ------- | :----- |
| Method      | Recall@10 | NDCG@10 | MRR@10 |
| **Pop**     | 0.0706    | 0.1008  | 0.2045 |
| **ItemKNN** |           |         |        |
| **BPRMF**   | 0.1667    | 0.2043  | 0.3620 |
| **NCF**     | 0.1591    | 0.1936  | 0.3459 |
| **NCF**-pre | 0.1545    | 0.1944  | 0.3479 |
| **NGCF**    |           |         |        |

BPRMF：{learning_rate=0.0001,embedding_size=512}

NeuMF：{dropout=0.1,learning_rate=0.0001,mf_embedding_size=512,mlp_embedding_size=128,mlp_hidden_size:[128,64,32]}

|             | yelp      | yelp    | yelp   |
| ----------- | --------- | ------- | :----- |
| Method      | Recall@10 | NDCG@10 | MRR@10 |
| **Pop**     | 0.0066    | 0.0069  | 0.0095 |
| **ItemKNN** |           |         |        |
| **BPRMF**   | 0.0984    | 0.0962  | 0.1653 |
| **NCF**     |           |         |        |
| **NGCF**    |           |         |        |

BPRMF：{learning_rate=0.0005,embedding_size=512}

|             | Amazon-game | Amazon-game | Amazon-game |
| ----------- | ----------- | ----------- | :---------- |
| Method      | Recall@10   | NDCG@10     | MRR@10      |
| **Pop**     |             |             |             |
| **ItemKNN** |             |             |             |
| **BPRMF**   |             |             |             |
| **NCF**     |             |             |             |
| **NGCF**    |             |             |             |

### **context-aware类模型**

使用数据集：ml-1m（评分小于3的数据项为正样本，大于等于3的数据项目为负样本）、

评测方式：8：1：1

评价指标：AUC、Logloss

|               | ml-1m | ml-1m   |
| ------------- | ----- | ------- |
| Method        | AUC   | Logloss |
| **FM**        |       |         |
| **DeepFM**    |       |         |
| **NFM**       |       |         |
| **AFM**       |       |         |
| **Wide&Deep** |       |         |