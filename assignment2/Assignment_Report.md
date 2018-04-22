# 作业二：关联规则挖掘
## 汇报人
- 汪汗青 2120171064

## 环境依赖
- python3.5
- numpy
- pickle
- [apyori](https://github.com/ymoch/apyori)

## 数据说明

本实践使用的数据集为San Francisco Building Permits数据集，共计43项属性，198900个数据条目。

## 数据预处理

为了方便进行关联规则挖掘，必须对数据进行预处理。这里选择了7个具有较多功能描述信息的标称属性进行关联规则挖掘：

1. Permit Type(Definition)
2. Street Number Suffix
3. Current Status
4. Existing Use
5. Proposed Use
6. Existing Construction Type(Definition)
7. Proposed Construction Type(Definition)

由于数据中存在NA(缺失)项，为了不至于混淆，将缺失项用该类的NA值进行填充。数据预处理详见 `data_preprocess.py`。

## 处理频繁项集

在这里我们使用Apriori算法来计算频繁项集，算法实现使用了apyori包中的apriori模块，支持度阈值设置为0.1。计算出的频繁项集根据项集的元素个数分别按支持度降序排序，如下表所示。
