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

<br>

---------

**<center>1-Itemset</center>**

| Itemset | Support |
| :-----: | :-----: |
| {"Street Number Suffix NA"} | 0.988854 |
| {"otc alterations permit"} | 0.899161 |
| {"wood frame (5)"} | 0.582149 |
| {"complete"} | 0.488067 |
| {"issued"} | 0.420103 |
| {"1 family dwelling"} | 0.239984 |
| {"apartments"} | 0.218124 |
| {"Existing Construction Type Description NA"} | 0.218028 |
| {"Proposed Construction Type Description NA"} | 0.217002 |
| {"Proposed Use NA"} | 0.213367 |
| {"Existing Use NA"} | 0.206706 |
| {"constr type 1"} | 0.146294 |
| {"office"} | 0.126576 |
| {"2 family dwelling"} | 0.115213 |
<br>

---------

**<center>2-Itemset</center>**

| Itemset | Support |
| :-----: | :-----: |
| {"otc alterations permit", "Street Number Suffix NA"} | 0.889799 |
| {"Street Number Suffix NA", "wood frame (5)"} | 0.573889 |
| {"otc alterations permit", "wood frame (5)"} | 0.514472 |
| {"Street Number Suffix NA", "complete"} | 0.482984 |
| {"otc alterations permit", "complete"} | 0.454658 |
| {"Street Number Suffix NA", "issued"} | 0.415438 |
| {"otc alterations permit", "issued"} | 0.387635 |
| {"wood frame (5)", "complete"} | 0.349893 |
| {"Street Number Suffix NA", "1 family dwelling"} | 0.238506 |
| {"1 family dwelling", "wood frame (5)"} | 0.238003 |
| {"otc alterations permit", "1 family dwelling"} | 0.216022 |
| {"Street Number Suffix NA", "Existing Construction Type Description NA"} | 0.215027 |
| {"Street Number Suffix NA", "Proposed Construction Type Description NA"} | 0.214308 |
| {"Street Number Suffix NA", "apartments"} | 0.213920 |
| {"Proposed Use NA", "Proposed Construction Type Description NA"} | 0.212121 |
| {"Street Number Suffix NA", "Proposed Use NA"} | 0.210904 |
| {"Existing Use NA", "Existing Construction Type Description NA"} | 0.206057 |
| {"otc alterations permit", "Existing Construction Type Description NA"} | 0.204499 |
| {"Street Number Suffix NA", "Existing Use NA"} | 0.204021 |
| {"Existing Construction Type Description NA", "Proposed Construction Type Description NA"} | 0.199210 |
| {"issued", "Proposed Construction Type Description NA"} | 0.196555 |
| {"Existing Construction Type Description NA", "issued"} | 0.195580 |
| {"otc alterations permit", "Proposed Construction Type Description NA"} | 0.195268 |
| {"Existing Use NA", "Proposed Use NA"} | 0.195092 |
| {"otc alterations permit", "Existing Use NA"} | 0.195042 |
| {"Proposed Use NA", "issued"} | 0.194981 |
| {"Existing Use NA", "Proposed Construction Type Description NA"} | 0.194795 |
| {"Proposed Use NA", "Existing Construction Type Description NA"} | 0.194770 |
| {"otc alterations permit", "Proposed Use NA"} | 0.192553 |
| {"Existing Use NA", "issued"} | 0.190889 |
| {"otc alterations permit", "apartments"} | 0.189371 |
| {"apartments", "wood frame (5)"} | 0.180125 |
| {"wood frame (5)", "issued"} | 0.166716 |
| {"1 family dwelling", "complete"} | 0.146827 |
| {"Street Number Suffix NA", "constr type 1"} | 0.146103 |
| {"otc alterations permit", "constr type 1"} | 0.129788 |
| {"apartments", "complete"} | 0.127873 |
| {"Street Number Suffix NA", "office"} | 0.126435 |
| {"otc alterations permit", "office"} | 0.116380 |
| {"2 family dwelling", "wood frame (5)"} | 0.114092 |
| {"2 family dwelling", "Street Number Suffix NA"} | 0.112569 |
| {"otc alterations permit", "2 family dwelling"} | 0.100864 |
<br>

---------

**<center>3-Itemset</center>**

| Itemset | Support |
| :-----: | :-----: |
| {"otc alterations permit", "Street Number Suffix NA", "wood frame (5)"} | 0.507810 |
| {"otc alterations permit", "Street Number Suffix NA", "complete"} | 0.450063 |
| {"otc alterations permit", "Street Number Suffix NA", "issued"} | 0.383553 |
| {"Street Number Suffix NA", "wood frame (5)", "complete"} | 0.345071 |
| {"otc alterations permit", "wood frame (5)", "complete"} | 0.329124 |
| {"Street Number Suffix NA", "1 family dwelling", "wood frame (5)"} | 0.236535 |
| {"otc alterations permit", "Street Number Suffix NA", "1 family dwelling"} | 0.214946 |
| {"otc alterations permit", "1 family dwelling", "wood frame (5)"} | 0.214217 |
| {"Street Number Suffix NA", "Proposed Use NA", "Proposed Construction Type Description NA"} | 0.209672 |
| {"Street Number Suffix NA", "Existing Use NA", "Existing Construction Type Description NA"} | 0.203383 |
| {"otc alterations permit", "Street Number Suffix NA", "Existing Construction Type Description NA"} | 0.201894 |
| {"Street Number Suffix NA", "Existing Construction Type Description NA", "Proposed Construction Type Description NA"} | 0.196625 |
| {"otc alterations permit", "Existing Construction Type Description NA", "Proposed Construction Type Description NA"} | 0.194841 |
| {"Existing Use NA", "Existing Construction Type Description NA", "Proposed Construction Type Description NA"} | 0.194790 |
| {"Proposed Use NA", "Existing Construction Type Description NA", "Proposed Construction Type Description NA"} | 0.194740 |
| {"issued", "Proposed Use NA", "Proposed Construction Type Description NA"} | 0.194665 |
| {"Proposed Use NA", "Existing Use NA", "Existing Construction Type Description NA"} | 0.194594 |
| {"Existing Use NA", "Proposed Use NA", "Proposed Construction Type Description NA"} | 0.194574 |
| {"otc alterations permit", "Existing Use NA", "Existing Construction Type Description NA"} | 0.194469 |
| {"Street Number Suffix NA", "issued", "Proposed Construction Type Description NA"} | 0.194127 |
| {"Street Number Suffix NA", "Existing Construction Type Description NA", "issued"} | 0.193111 |
| {"otc alterations permit", "Street Number Suffix NA", "Proposed Construction Type Description NA"} | 0.192759 |
| {"Street Number Suffix NA", "Existing Use NA", "Proposed Use NA"} | 0.192749 |
| {"otc alterations permit", "Street Number Suffix NA", "Existing Use NA"} | 0.192694 |
| {"Street Number Suffix NA", "Proposed Use NA", "issued"} | 0.192649 |
| {"Street Number Suffix NA", "Existing Use NA", "Proposed Construction Type Description NA"} | 0.192442 |
| {"Proposed Use NA", "Street Number Suffix NA", "Existing Construction Type Description NA"} | 0.192432 |
| {"otc alterations permit", "Existing Use NA", "Proposed Use NA"} | 0.191804 |
| {"otc alterations permit", "Existing Use NA", "Proposed Construction Type Description NA"} | 0.191472 |
| {"otc alterations permit", "Proposed Use NA", "Proposed Construction Type Description NA"} | 0.191392 |
| {"Proposed Use NA", "otc alterations permit", "Existing Construction Type Description NA"} | 0.191357 |
| {"Existing Use NA", "Existing Construction Type Description NA", "issued"} | 0.190778 |
| {"otc alterations permit", "Existing Construction Type Description NA", "issued"} | 0.190441 |
| {"otc alterations permit", "Street Number Suffix NA", "Proposed Use NA"} | 0.190220 |
| {"Street Number Suffix NA", "Existing Use NA", "issued"} | 0.188541 |
| {"issued", "Existing Construction Type Description NA", "Proposed Construction Type Description NA"} | 0.188516 |
| {"Existing Use NA", "Proposed Use NA", "issued"} | 0.186832 |
| {"Proposed Use NA", "Existing Construction Type Description NA", "issued"} | 0.186827 |
| {"otc alterations permit", "issued", "Proposed Construction Type Description NA"} | 0.186827 |
| {"Existing Use NA", "issued", "Proposed Construction Type Description NA"} | 0.186801 |
| {"otc alterations permit", "Existing Use NA", "issued"} | 0.186570 |
| {"otc alterations permit", "Street Number Suffix NA", "apartments"} | 0.186088 |
| {"otc alterations permit", "Proposed Use NA", "issued"} | 0.185615 |
| {"Street Number Suffix NA", "apartments", "wood frame (5)"} | 0.175997 |
| {"Street Number Suffix NA", "wood frame (5)", "issued"} | 0.164489 |
| {"otc alterations permit", "apartments", "wood frame (5)"} | 0.156414 |
| {"otc alterations permit", "wood frame (5)", "issued"} | 0.146158 |
| {"Street Number Suffix NA", "1 family dwelling", "complete"} | 0.146058 |
| {"1 family dwelling", "wood frame (5)", "complete"} | 0.145630 |
| {"otc alterations permit", "1 family dwelling", "complete"} | 0.139929 |
| {"otc alterations permit", "Street Number Suffix NA", "constr type 1"} | 0.129673 |
| {"Street Number Suffix NA", "apartments", "complete"} | 0.125505 |
| {"otc alterations permit", "apartments", "complete"} | 0.119783 |
| {"otc alterations permit", "Street Number Suffix NA", "office"} | 0.116279 |
| {"2 family dwelling", "Street Number Suffix NA", "wood frame (5)"} | 0.111468 |
| {"apartments", "wood frame (5)", "complete"} | 0.104841 |
<br>

---------

**<center>4-Itemset</center>**

| Itemset | Support |
| :-----: | :-----: |
| {"otc alterations permit", "Street Number Suffix NA", "wood frame (5)", "complete"} | 0.324734 |
| {"otc alterations permit", "Street Number Suffix NA", "1 family dwelling", "wood frame (5)"} | 0.213141 |
| {"Proposed Use NA", "Existing Use NA", "Existing Construction Type Description NA", "Proposed Construction Type Description NA"} | 0.194569 |
| {"Street Number Suffix NA", "Existing Use NA", "Existing Construction Type Description NA", "Proposed Construction Type Description NA"} | 0.192437 |
| {"Proposed Use NA", "Street Number Suffix NA", "Existing Construction Type Description NA", "Proposed Construction Type Description NA"} | 0.192402 |
| {"otc alterations permit", "Street Number Suffix NA", "Existing Construction Type Description NA", "Proposed Construction Type Description NA"} | 0.192332 |
| {"Street Number Suffix NA", "issued", "Proposed Use NA", "Proposed Construction Type Description NA"} | 0.192332 |
| {"Proposed Use NA", "Street Number Suffix NA", "Existing Use NA", "Existing Construction Type Description NA"} | 0.192256 |
| {"Street Number Suffix NA", "Existing Use NA", "Proposed Use NA", "Proposed Construction Type Description NA"} | 0.192236 |
| {"otc alterations permit", "Street Number Suffix NA", "Existing Use NA", "Existing Construction Type Description NA"} | 0.192126 |
| {"otc alterations permit", "Existing Use NA", "Existing Construction Type Description NA", "Proposed Construction Type Description NA"} | 0.191472 |
| {"Proposed Use NA", "otc alterations permit", "Existing Use NA", "Existing Construction Type Description NA"} | 0.191351 |
| {"Proposed Use NA", "otc alterations permit", "Existing Construction Type Description NA", "Proposed Construction Type Description NA"} | 0.191346 |
| {"otc alterations permit", "Existing Use NA", "Proposed Use NA", "Proposed Construction Type Description NA"} | 0.191341 |
| {"otc alterations permit", "Street Number Suffix NA", "Existing Use NA", "Proposed Use NA"} | 0.189481 |
| {"otc alterations permit", "Street Number Suffix NA", "Existing Use NA", "Proposed Construction Type Description NA"} | 0.189144 |
| {"otc alterations permit", "Street Number Suffix NA", "Proposed Use NA", "Proposed Construction Type Description NA"} | 0.189074 |
| {"Proposed Use NA", "otc alterations permit", "Street Number Suffix NA", "Existing Construction Type Description NA"} | 0.189039 |
| {"Street Number Suffix NA", "Existing Use NA", "Existing Construction Type Description NA", "issued"} | 0.188430 |
| {"otc alterations permit", "Street Number Suffix NA", "Existing Construction Type Description NA", "issued"} | 0.188058 |
| {"Proposed Use NA", "issued", "Existing Construction Type Description NA", "Proposed Construction Type Description NA"} | 0.186817 |
| {"issued", "Existing Use NA", "Existing Construction Type Description NA", "Proposed Construction Type Description NA"} | 0.186801 |
| {"Proposed Use NA", "Existing Use NA", "Existing Construction Type Description NA", "issued"} | 0.186741 |
| {"issued", "Existing Use NA", "Proposed Use NA", "Proposed Construction Type Description NA"} | 0.186736 |
| {"otc alterations permit", "issued", "Existing Construction Type Description NA", "Proposed Construction Type Description NA"} | 0.186651 |
| {"otc alterations permit", "Existing Use NA", "Existing Construction Type Description NA", "issued"} | 0.186460 |
| {"Street Number Suffix NA", "issued", "Existing Construction Type Description NA", "Proposed Construction Type Description NA"} | 0.186128 |
| {"otc alterations permit", "Existing Use NA", "Proposed Use NA", "issued"} | 0.185394 |
| {"otc alterations permit", "Existing Use NA", "issued", "Proposed Construction Type Description NA"} | 0.185348 |
| {"otc alterations permit", "issued", "Proposed Use NA", "Proposed Construction Type Description NA"} | 0.185308 |
| {"Proposed Use NA", "otc alterations permit", "Existing Construction Type Description NA", "issued"} | 0.185303 |
| {"Street Number Suffix NA", "Existing Use NA", "Proposed Use NA", "issued"} | 0.184539 |
| {"Proposed Use NA", "Street Number Suffix NA", "Existing Construction Type Description NA", "issued"} | 0.184534 |
| {"Street Number Suffix NA", "Existing Use NA", "issued", "Proposed Construction Type Description NA"} | 0.184504 |
| {"otc alterations permit", "Street Number Suffix NA", "issued", "Proposed Construction Type Description NA"} | 0.184479 |
| {"otc alterations permit", "Street Number Suffix NA", "Existing Use NA", "issued"} | 0.184273 |
| {"otc alterations permit", "Street Number Suffix NA", "Proposed Use NA", "issued"} | 0.183332 |
| {"otc alterations permit", "Street Number Suffix NA", "apartments", "wood frame (5)"} | 0.153172 |
| {"Street Number Suffix NA", "1 family dwelling", "wood frame (5)", "complete"} | 0.144866 |
| {"otc alterations permit", "Street Number Suffix NA", "wood frame (5)", "issued"} | 0.144474 |
| {"otc alterations permit", "Street Number Suffix NA", "1 family dwelling", "complete"} | 0.139240 |
| {"otc alterations permit", "1 family dwelling", "wood frame (5)", "complete"} | 0.138757 |
| {"otc alterations permit", "Street Number Suffix NA", "apartments", "complete"} | 0.117662 |
| {"Street Number Suffix NA", "apartments", "complete", "wood frame (5)"} | 0.102498 |
<br>

---------

**<center>5-Itemset</center>**

| Itemset | Support |
| :-----: | :-----: |
| {"Proposed Construction Type Description NA", "Existing Use NA", "Proposed Use NA", "Existing Construction Type Description NA", "Street Number Suffix NA"} | 0.192231 |
| {"Proposed Construction Type Description NA", "Existing Use NA", "Proposed Use NA", "Existing Construction Type Description NA", "otc alterations permit"} | 0.191341 |
| {"Proposed Construction Type Description NA", "Existing Use NA", "Existing Construction Type Description NA", "otc alterations permit", "Street Number Suffix NA"} | 0.189144 |
| {"Existing Use NA", "Proposed Use NA", "Existing Construction Type Description NA", "otc alterations permit", "Street Number Suffix NA"} | 0.189034 |
| {"Proposed Construction Type Description NA", "Proposed Use NA", "Existing Construction Type Description NA", "otc alterations permit", "Street Number Suffix NA"} | 0.189029 |
| {"Proposed Construction Type Description NA", "Existing Use NA", "Proposed Use NA", "otc alterations permit", "Street Number Suffix NA"} | 0.189024 |
| {"Proposed Construction Type Description NA", "Existing Use NA", "Proposed Use NA", "Existing Construction Type Description NA", "issued"} | 0.186736 |
| {"Proposed Construction Type Description NA", "Existing Use NA", "Existing Construction Type Description NA", "otc alterations permit", "issued"} | 0.185348 |
| {"Existing Use NA", "Proposed Use NA", "Existing Construction Type Description NA", "otc alterations permit", "issued"} | 0.185303 |
| {"Proposed Construction Type Description NA", "Proposed Use NA", "Existing Construction Type Description NA", "otc alterations permit", "issued"} | 0.185303 |
| {"Proposed Construction Type Description NA", "Existing Use NA", "Proposed Use NA", "otc alterations permit", "issued"} | 0.185303 |
| {"Proposed Construction Type Description NA", "Proposed Use NA", "Existing Construction Type Description NA", "Street Number Suffix NA", "issued"} | 0.184524 |
| {"Proposed Construction Type Description NA", "Existing Use NA", "Existing Construction Type Description NA", "Street Number Suffix NA", "issued"} | 0.184504 |
| {"Existing Use NA", "Proposed Use NA", "Existing Construction Type Description NA", "Street Number Suffix NA", "issued"} | 0.184449 |
| {"Proposed Construction Type Description NA", "Existing Use NA", "Proposed Use NA", "Street Number Suffix NA", "issued"} | 0.184444 |
| {"Proposed Construction Type Description NA", "Existing Construction Type Description NA", "otc alterations permit", "Street Number Suffix NA", "issued"} | 0.184303 |
| {"Existing Use NA", "Existing Construction Type Description NA", "otc alterations permit", "Street Number Suffix NA", "issued"} | 0.184162 |
| {"Existing Use NA", "Proposed Use NA", "otc alterations permit", "Street Number Suffix NA", "issued"} | 0.183111 |
| {"Proposed Construction Type Description NA", "Existing Use NA", "otc alterations permit", "Street Number Suffix NA", "issued"} | 0.183061 |
| {"Proposed Construction Type Description NA", "Proposed Use NA", "otc alterations permit", "Street Number Suffix NA", "issued"} | 0.183026 |
| {"Proposed Use NA", "Existing Construction Type Description NA", "otc alterations permit", "Street Number Suffix NA", "issued"} | 0.183021 |
| {"1 family dwelling", "wood frame (5)", "otc alterations permit", "Street Number Suffix NA", "complete"} | 0.138069 |
<br>

---------

**<center>6-Itemset</center>**

| Itemset | Support |
| :-----: | :-----: |
| {"Proposed Construction Type Description NA", "Existing Use NA", "Proposed Use NA", "Existing Construction Type Description NA", "otc alterations permit", "Street Number Suffix NA"} | 0.189024 |
| {"Proposed Construction Type Description NA", "Existing Use NA", "Proposed Use NA", "Existing Construction Type Description NA", "otc alterations permit", "issued"} | 0.185303 |
| {"Proposed Construction Type Description NA", "Existing Use NA", "Proposed Use NA", "Existing Construction Type Description NA", "Street Number Suffix NA", "issued"} | 0.184444 |
| {"Proposed Construction Type Description NA", "Existing Use NA", "Existing Construction Type Description NA", "otc alterations permit", "Street Number Suffix NA", "issued"} | 0.183061 |
| {"Existing Use NA", "Proposed Use NA", "Existing Construction Type Description NA", "otc alterations permit", "Street Number Suffix NA", "issued"} | 0.183021 |
| {"Proposed Construction Type Description NA", "Proposed Use NA", "Existing Construction Type Description NA", "otc alterations permit", "Street Number Suffix NA", "issued"} | 0.183021 |
| {"Proposed Construction Type Description NA", "Existing Use NA", "Proposed Use NA", "otc alterations permit", "Street Number Suffix NA", "issued"} | 0.183021 |
<br>

---------

**<center>7-Itemset</center>**

| Itemset | Support |
| :-----: | :-----: |
| {"Proposed Construction Type Description NA", "Existing Use NA", "Proposed Use NA", "Existing Construction Type Description NA", "otc alterations permit", "Street Number Suffix NA", "issued"} | 0.183021 |


## 