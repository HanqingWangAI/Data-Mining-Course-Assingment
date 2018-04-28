
# 作业三：分类与聚类


## 汇报人
- 汪汗青 2120171064

## 环境依赖
- python3.5
- numpy
- sklearn
- matplotlib

## 目录
1. [数据说明](#数据说明)
2. [数据预处理](#数据预处理)
3. [分类模型挖掘](#分类模型挖掘)
    - [Gaussian Naive Bayes](#Gaussian Naive Bayes)
    - [SVM](#SVM)
    - [分类结果分析](#分类结果分析)
4. [聚类模型挖掘](#聚类模型挖掘)
    - [KMeans](#KMeans)
    - [GMM](#GMM)
    - [聚类结果分析](#聚类结果分析)




## 数据说明

本实践使用的数据集为Titanic: Machine Learning from Disaster数据集，该数据集记录了Titanic号海难的部分船员信息。该数据集分为两个部分，`training set`和`testing set`。其中`test.csv`包含418个数据条目，11个属性。`training.csv`,包含891个数据条目，12个属性，比`testing`数据集多一个`Survival`属性，表示该船员是否幸存。数据的各项属性描述如下:

<table>
<tbody>
<tr><th><b>Variable</b></th><th><b>Definition</b></th><th><b>Key</b></th></tr>
<tr>
<td>survival</td>
<td>Survival</td>
<td>0 = No, 1 = Yes</td>
</tr>
<tr>
<td>pclass</td>
<td>Ticket class</td>
<td>1 = 1st, 2 = 2nd, 3 = 3rd</td>
</tr>
<tr>
<td>sex</td>
<td>Sex</td>
<td></td>
</tr>
<tr>
<td>Age</td>
<td>Age in years</td>
<td></td>
</tr>
<tr>
<td>sibsp</td>
<td># of siblings / spouses aboard the Titanic</td>
<td></td>
</tr>
<tr>
<td>parch</td>
<td># of parents / children aboard the Titanic</td>
<td></td>
</tr>
<tr>
<td>ticket</td>
<td>Ticket number</td>
<td></td>
</tr>
<tr>
<td>fare</td>
<td>Passenger fare</td>
<td></td>
</tr>
<tr>
<td>cabin</td>
<td>Cabin number</td>
<td></td>
</tr>
<tr>
<td>embarked</td>
<td>Port of Embarkation</td>
<td>C = Cherbourg, Q = Queenstown, S = Southampton</td>
</tr>
</tbody>
</table>


## 数据预处理

为了方便对数据进行分类和聚类的挖掘，我选取了以下的一些属性。


1. pclass
2. sex
3. Age
4. bibsp
5. parch
6. fare
7. embarked

对于数据中部分的缺失项，我首先对它们进行了分析，发现缺失项主要出现在Age和embarked两个属性上。对于第一个Age属性的缺失，为了防止填充过程对总的数据分布的影响，我采用填0的方式。对于第二个属性，由于其是标称属性，我对其增加了一个标签值NA来填补缺失项。

## 分类模型挖掘
在分类模型挖掘这一部分，我采用了两种不同的有监督的分类算法进行分类。一种是基于高斯模型的朴素贝叶斯分类器，第二种是高斯核的支持向量机分类。我们的分类任务是将筛选出的7个属性作为特征进行输入，对其是否幸存进行预测。
### Gaussian Naive Bayes

### SVM

### 分类结果分析

## 聚类模型挖掘

### KMeans

### GMM



### 聚类结果分析