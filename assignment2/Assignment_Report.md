
# 作业二：关联规则挖掘

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>


## 汇报人
- 汪汗青 2120171064

## 环境依赖
- python3.5
- numpy
- pickle
- [apyori](https://github.com/ymoch/apyori)

## 目录
1. [数据说明](#数据说明)
2. [数据预处理](#数据预处理)
3. [处理频繁项集](#处理频繁项集)
4. [关联规则的导出与评估](#关联规则的导出与评估)
5. [对关联规则的结果分析](#对关联规则的结果分析)
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

由于数据中存在NA(缺失)项,为了防止NA之间的混淆，这里将缺失项用该属性的NA值进行填充，之后再进行剔除。数据预处理详见 `data_preprocess.py`。

## 处理频繁项集

在这里我们使用Apriori算法来计算频繁项集，算法实现使用了apyori包中的apriori模块，支持度阈值设置为0.01。计算出的频繁项集根据项集的元素个数分别按支持度降序排序，如下表所示。

<br>

---------

**<center>1-Itemset</center>**

| Itemset | Support |
| :-----: | :-----: |
| {"(Permit Type Definition) otc alterations permit"} | 0.899161 |
| {"(Proposed Construction Type Description) wood frame (5)"} | 0.575070 |
| {"(Existing Construction Type Description) wood frame (5)"} | 0.569881 |
| {"(Current Status) complete"} | 0.488067 |
| {"(Current Status) issued"} | 0.420103 |
| {"(Existing Use) 1 family dwelling"} | 0.235122 |
| {"(Proposed Use) 1 family dwelling"} | 0.233010 |
| {"(Existing Construction Type Description) NA"} | 0.218028 |
| {"(Proposed Construction Type Description) NA"} | 0.217002 |
| {"(Proposed Use) apartments"} | 0.216349 |
| {"(Proposed Use) NA"} | 0.213367 |
| {"(Existing Use) NA"} | 0.206706 |
| {"(Existing Use) apartments"} | 0.205117 |
| {"(Existing Construction Type Description) constr type 1"} | 0.141136 |
| {"(Proposed Construction Type Description) constr type 1"} | 0.139974 |
| {"(Existing Use) office"} | 0.123760 |
| {"(Proposed Use) office"} | 0.120472 |
| {"(Proposed Use) 2 family dwelling"} | 0.110914 |
| {"(Existing Use) 2 family dwelling"} | 0.105515 |
| {"(Permit Type Definition) additions alterations or repairs"} | 0.073720 |
| {"(Current Status) filed"} | 0.060548 |
| {"(Existing Construction Type Description) constr type 3"} | 0.048582 |
| {"(Proposed Construction Type Description) constr type 3"} | 0.047059 |
| {"(Existing Use) retail sales"} | 0.034741 |
| {"(Street Name) Market"} | 0.027365 |
| {"(Proposed Use) retail sales"} | 0.025535 |
| {"(Proposed Use) food/beverage hndlng"} | 0.025405 |
| {"(Existing Use) food/beverage hndlng"} | 0.024565 |
| {"(Street Name) California"} | 0.023062 |
| {"(Street Name) Mission"} | 0.021161 |
| {"(Existing Construction Type Description) constr type 2"} | 0.020452 |
| {"(Proposed Construction Type Description) constr type 2"} | 0.018994 |
| {"(Permit Type Definition) sign - erect"} | 0.014540 |
| {"(Street Name) Montgomery"} | 0.012081 |
<br>

---------

**<center>2-Itemset</center>**

| Itemset | Support |
| :-----: | :-----: |
| {"(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)"} | 0.562803 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) wood frame (5)"} | 0.513959 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) wood frame (5)"} | 0.508288 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit"} | 0.454658 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued"} | 0.387635 |
| {"(Current Status) complete", "(Proposed Construction Type Description) wood frame (5)"} | 0.347575 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)"} | 0.344604 |
| {"(Existing Construction Type Description) wood frame (5)", "(Existing Use) 1 family dwelling"} | 0.233131 |
| {"(Existing Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)"} | 0.232322 |
| {"(Proposed Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)"} | 0.230929 |
| {"(Proposed Use) 1 family dwelling", "(Existing Use) 1 family dwelling"} | 0.228149 |
| {"(Proposed Use) 1 family dwelling", "(Existing Construction Type Description) wood frame (5)"} | 0.227269 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Use) 1 family dwelling"} | 0.214071 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Use) 1 family dwelling"} | 0.213674 |
| {"(Proposed Construction Type Description) NA", "(Proposed Use) NA"} | 0.212121 |
| {"(Existing Construction Type Description) NA", "(Existing Use) NA"} | 0.206057 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) NA"} | 0.204499 |
| {"(Proposed Use) apartments", "(Existing Use) apartments"} | 0.203342 |
| {"(Existing Construction Type Description) NA", "(Proposed Construction Type Description) NA"} | 0.199210 |
| {"(Current Status) issued", "(Proposed Construction Type Description) NA"} | 0.196555 |
| {"(Current Status) issued", "(Existing Construction Type Description) NA"} | 0.195580 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) NA"} | 0.195268 |
| {"(Existing Use) NA", "(Proposed Use) NA"} | 0.195092 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Use) NA"} | 0.195042 |
| {"(Current Status) issued", "(Proposed Use) NA"} | 0.194981 |
| {"(Existing Use) NA", "(Proposed Construction Type Description) NA"} | 0.194795 |
| {"(Existing Construction Type Description) NA", "(Proposed Use) NA"} | 0.194770 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Use) NA"} | 0.192553 |
| {"(Current Status) issued", "(Existing Use) NA"} | 0.190889 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Use) apartments"} | 0.188833 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Use) apartments"} | 0.183046 |
| {"(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments"} | 0.178898 |
| {"(Existing Construction Type Description) wood frame (5)", "(Proposed Use) apartments"} | 0.175580 |
| {"(Existing Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | 0.171698 |
| {"(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | 0.171352 |
| {"(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)"} | 0.163700 |
| {"(Current Status) issued", "(Existing Construction Type Description) wood frame (5)"} | 0.162121 |
| {"(Current Status) complete", "(Existing Use) 1 family dwelling"} | 0.145208 |
| {"(Current Status) complete", "(Proposed Use) 1 family dwelling"} | 0.144610 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1"} | 0.134816 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1"} | 0.129662 |
| {"(Current Status) complete", "(Proposed Use) apartments"} | 0.127169 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1"} | 0.127028 |
| {"(Current Status) complete", "(Existing Use) apartments"} | 0.122523 |
| {"(Proposed Use) office", "(Existing Use) office"} | 0.117657 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Use) office"} | 0.114419 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Use) office"} | 0.113825 |
| {"(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) 2 family dwelling"} | 0.109778 |
| {"(Existing Construction Type Description) wood frame (5)", "(Proposed Use) 2 family dwelling"} | 0.106053 |
| {"(Existing Use) 2 family dwelling", "(Existing Construction Type Description) wood frame (5)"} | 0.104570 |
| {"(Existing Use) 2 family dwelling", "(Proposed Construction Type Description) wood frame (5)"} | 0.104494 |
| {"(Existing Use) 2 family dwelling", "(Proposed Use) 2 family dwelling"} | 0.101216 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Use) 2 family dwelling"} | 0.098858 |
| {"(Existing Use) 2 family dwelling", "(Permit Type Definition) otc alterations permit"} | 0.096490 |
| {"(Existing Construction Type Description) constr type 1", "(Existing Use) office"} | 0.092544 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Use) office"} | 0.091327 |
| {"(Proposed Construction Type Description) constr type 1", "(Proposed Use) office"} | 0.091297 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1"} | 0.091005 |
| {"(Current Status) complete", "(Proposed Construction Type Description) constr type 1"} | 0.090794 |
| {"(Proposed Use) office", "(Existing Construction Type Description) constr type 1"} | 0.090532 |
| {"(Current Status) complete", "(Existing Use) office"} | 0.081015 |
| {"(Current Status) complete", "(Proposed Use) office"} | 0.079617 |
| {"(Current Status) complete", "(Proposed Use) 2 family dwelling"} | 0.069633 |
| {"(Proposed Use) 1 family dwelling", "(Current Status) issued"} | 0.068235 |
| {"(Current Status) issued", "(Existing Use) 1 family dwelling"} | 0.068104 |
| {"(Current Status) complete", "(Existing Use) 2 family dwelling"} | 0.067511 |
| {"(Current Status) issued", "(Proposed Use) apartments"} | 0.061111 |
| {"(Current Status) issued", "(Existing Use) apartments"} | 0.057858 |
| {"(Proposed Construction Type Description) wood frame (5)", "(Permit Type Definition) additions alterations or repairs"} | 0.056380 |
| {"(Existing Construction Type Description) wood frame (5)", "(Permit Type Definition) additions alterations or repairs"} | 0.055188 |
| {"(Existing Construction Type Description) constr type 3", "(Proposed Construction Type Description) constr type 3"} | 0.045676 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 3"} | 0.041528 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 3"} | 0.040880 |
| {"(Current Status) issued", "(Existing Construction Type Description) constr type 1"} | 0.040648 |
| {"(Current Status) filed", "(Proposed Construction Type Description) wood frame (5)"} | 0.039673 |
| {"(Proposed Construction Type Description) constr type 1", "(Current Status) issued"} | 0.039613 |
| {"(Current Status) filed", "(Existing Construction Type Description) wood frame (5)"} | 0.039241 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) filed"} | 0.036883 |
| {"(Current Status) issued", "(Existing Use) office"} | 0.034444 |
| {"(Proposed Use) office", "(Current Status) issued"} | 0.033187 |
| {"(Current Status) issued", "(Proposed Use) 2 family dwelling"} | 0.030191 |
| {"(Existing Use) 2 family dwelling", "(Current Status) issued"} | 0.028275 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 3"} | 0.028200 |
| {"(Current Status) complete", "(Proposed Construction Type Description) constr type 3"} | 0.028044 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Use) retail sales"} | 0.025555 |
| {"(Current Status) complete", "(Permit Type Definition) additions alterations or repairs"} | 0.025178 |
| {"(Proposed Use) apartments", "(Permit Type Definition) additions alterations or repairs"} | 0.025063 |
| {"(Proposed Use) retail sales", "(Existing Use) retail sales"} | 0.024128 |
| {"(Permit Type Definition) otc alterations permit", "(Street Name) Market"} | 0.023806 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Use) retail sales"} | 0.022810 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Use) food/beverage hndlng"} | 0.022122 |
| {"(Proposed Construction Type Description) constr type 1", "(Proposed Use) apartments"} | 0.021825 |
| {"(Permit Type Definition) otc alterations permit", "(Street Name) California"} | 0.021815 |
| {"(Current Status) issued", "(Permit Type Definition) additions alterations or repairs"} | 0.021066 |
| {"(Existing Use) apartments", "(Permit Type Definition) additions alterations or repairs"} | 0.021005 |
| {"(Existing Use) food/beverage hndlng", "(Proposed Use) food/beverage hndlng"} | 0.020774 |
| {"(Existing Use) 1 family dwelling", "(Permit Type Definition) additions alterations or repairs"} | 0.020362 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Use) food/beverage hndlng"} | 0.019532 |
| {"(Current Status) filed", "(Proposed Use) apartments"} | 0.019497 |
| {"(Existing Construction Type Description) constr type 1", "(Existing Use) apartments"} | 0.018914 |
| {"(Current Status) complete", "(Existing Use) retail sales"} | 0.018567 |
| {"(Proposed Construction Type Description) constr type 2", "(Existing Construction Type Description) constr type 2"} | 0.018461 |
| {"(Permit Type Definition) otc alterations permit", "(Street Name) Mission"} | 0.018446 |
| {"(Existing Construction Type Description) constr type 1", "(Proposed Use) apartments"} | 0.018366 |
| {"(Current Status) filed", "(Permit Type Definition) additions alterations or repairs"} | 0.018316 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Use) apartments"} | 0.018265 |
| {"(Existing Construction Type Description) constr type 1", "(Street Name) Market"} | 0.017692 |
| {"(Proposed Use) 1 family dwelling", "(Permit Type Definition) additions alterations or repairs"} | 0.017461 |
| {"(Proposed Construction Type Description) constr type 1", "(Street Name) Market"} | 0.017215 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 2"} | 0.017008 |
| {"(Current Status) filed", "(Existing Use) apartments"} | 0.016958 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 2"} | 0.016787 |
| {"(Current Status) complete", "(Street Name) Market"} | 0.015616 |
| {"(Current Status) issued", "(Existing Construction Type Description) constr type 3"} | 0.015048 |
| {"(Existing Use) office", "(Street Name) Market"} | 0.014615 |
| {"(Current Status) complete", "(Proposed Use) food/beverage hndlng"} | 0.014575 |
| {"(Permit Type Definition) sign - erect", "(Proposed Construction Type Description) NA"} | 0.014540 |
| {"(Permit Type Definition) sign - erect", "(Proposed Use) NA"} | 0.014540 |
| {"(Current Status) complete", "(Proposed Use) retail sales"} | 0.014238 |
| {"(Current Status) issued", "(Proposed Construction Type Description) constr type 3"} | 0.014223 |
| {"(Existing Construction Type Description) wood frame (5)", "(Existing Use) retail sales"} | 0.014158 |
| {"(Proposed Use) office", "(Street Name) Market"} | 0.014072 |
| {"(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) food/beverage hndlng"} | 0.013670 |
| {"(Existing Construction Type Description) constr type 3", "(Existing Use) office"} | 0.013605 |
| {"(Existing Construction Type Description) constr type 1", "(Street Name) California"} | 0.013565 |
| {"(Existing Construction Type Description) wood frame (5)", "(Proposed Use) food/beverage hndlng"} | 0.013560 |
| {"(Current Status) complete", "(Existing Use) food/beverage hndlng"} | 0.013404 |
| {"(Proposed Construction Type Description) constr type 1", "(Street Name) California"} | 0.013343 |
| {"(Current Status) complete", "(Street Name) California"} | 0.013172 |
| {"(Existing Construction Type Description) wood frame (5)", "(Existing Use) food/beverage hndlng"} | 0.013167 |
| {"(Existing Use) office", "(Proposed Construction Type Description) constr type 3"} | 0.013112 |
| {"(Proposed Use) office", "(Proposed Construction Type Description) constr type 3"} | 0.013017 |
| {"(Proposed Use) office", "(Existing Construction Type Description) constr type 3"} | 0.012966 |
| {"(Street Name) California", "(Existing Use) office"} | 0.012745 |
| {"(Proposed Construction Type Description) wood frame (5)", "(Existing Use) retail sales"} | 0.012629 |
| {"(Proposed Use) office", "(Street Name) California"} | 0.012604 |
| {"(Current Status) filed", "(Existing Use) 1 family dwelling"} | 0.012267 |
| {"(Existing Construction Type Description) NA", "(Proposed Construction Type Description) wood frame (5)"} | 0.012127 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 2"} | 0.011855 |
| {"(Proposed Construction Type Description) wood frame (5)", "(Existing Use) food/beverage hndlng"} | 0.011830 |
| {"(Permit Type Definition) otc alterations permit", "(Street Name) Montgomery"} | 0.011473 |
| {"(Current Status) complete", "(Proposed Construction Type Description) constr type 2"} | 0.011322 |
| {"(Current Status) complete", "(Existing Construction Type Description) NA"} | 0.011212 |
| {"(Current Status) issued", "(Existing Use) retail sales"} | 0.011076 |
| {"(Proposed Use) apartments", "(Proposed Construction Type Description) constr type 3"} | 0.011051 |
| {"(Proposed Use) 1 family dwelling", "(Current Status) filed"} | 0.011016 |
| {"(Proposed Use) 2 family dwelling", "(Permit Type Definition) additions alterations or repairs"} | 0.010819 |
| {"(Existing Construction Type Description) wood frame (5)", "(Existing Use) office"} | 0.010372 |
| {"(Existing Construction Type Description) constr type 1", "(Existing Use) retail sales"} | 0.010206 |
| {"(Existing Construction Type Description) constr type 3", "(Proposed Use) apartments"} | 0.010146 |
| {"(Proposed Use) retail sales", "(Proposed Construction Type Description) wood frame (5)"} | 0.010141 |
| {"(Existing Construction Type Description) wood frame (5)", "(Proposed Use) retail sales"} | 0.010111 |
| {"(Existing Construction Type Description) constr type 3", "(Existing Use) apartments"} | 0.010085 |
| {"(Current Status) complete", "(Street Name) Mission"} | 0.010075 |
| {"(Proposed Construction Type Description) constr type 3", "(Existing Use) apartments"} | 0.010025 |
<br>

---------

**<center>3-Itemset</center>**

| Itemset | Support |
| :-----: | :-----: |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)"} | 0.507775 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)"} | 0.342286 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) wood frame (5)"} | 0.328862 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Permit Type Definition) otc alterations permit"} | 0.325785 |
| {"(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) 1 family dwelling"} | 0.232181 |
| {"(Proposed Use) 1 family dwelling", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)"} | 0.227053 |
| {"(Proposed Use) 1 family dwelling", "(Existing Construction Type Description) wood frame (5)", "(Existing Use) 1 family dwelling"} | 0.226288 |
| {"(Proposed Use) 1 family dwelling", "(Existing Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)"} | 0.226198 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) wood frame (5)", "(Existing Use) 1 family dwelling"} | 0.212191 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)"} | 0.212095 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)"} | 0.211759 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Use) 1 family dwelling", "(Proposed Use) 1 family dwelling"} | 0.211723 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) wood frame (5)", "(Proposed Use) 1 family dwelling"} | 0.210401 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) NA", "(Proposed Construction Type Description) NA"} | 0.194841 |
| {"(Existing Construction Type Description) NA", "(Existing Use) NA", "(Proposed Construction Type Description) NA"} | 0.194790 |
| {"(Existing Construction Type Description) NA", "(Proposed Construction Type Description) NA", "(Proposed Use) NA"} | 0.194740 |
| {"(Current Status) issued", "(Proposed Construction Type Description) NA", "(Proposed Use) NA"} | 0.194665 |
| {"(Existing Construction Type Description) NA", "(Existing Use) NA", "(Proposed Use) NA"} | 0.194594 |
| {"(Existing Use) NA", "(Proposed Construction Type Description) NA", "(Proposed Use) NA"} | 0.194574 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) NA", "(Existing Use) NA"} | 0.194469 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Use) NA", "(Proposed Use) NA"} | 0.191804 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Use) NA", "(Proposed Construction Type Description) NA"} | 0.191472 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) NA", "(Proposed Use) NA"} | 0.191392 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) NA", "(Proposed Use) NA"} | 0.191357 |
| {"(Current Status) issued", "(Existing Construction Type Description) NA", "(Existing Use) NA"} | 0.190778 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Construction Type Description) NA"} | 0.190441 |
| {"(Current Status) issued", "(Existing Construction Type Description) NA", "(Proposed Construction Type Description) NA"} | 0.188516 |
| {"(Current Status) issued", "(Existing Use) NA", "(Proposed Use) NA"} | 0.186832 |
| {"(Current Status) issued", "(Existing Construction Type Description) NA", "(Proposed Use) NA"} | 0.186827 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Construction Type Description) NA"} | 0.186827 |
| {"(Current Status) issued", "(Existing Use) NA", "(Proposed Construction Type Description) NA"} | 0.186801 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Use) NA"} | 0.186570 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Use) NA"} | 0.185615 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Use) apartments", "(Existing Use) apartments"} | 0.182508 |
| {"(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments"} | 0.175283 |
| {"(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | 0.171181 |
| {"(Proposed Use) apartments", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | 0.170778 |
| {"(Existing Construction Type Description) wood frame (5)", "(Proposed Use) apartments", "(Existing Use) apartments"} | 0.170768 |
| {"(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)", "(Existing Construction Type Description) wood frame (5)"} | 0.159104 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments"} | 0.155786 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) wood frame (5)", "(Proposed Use) apartments"} | 0.154077 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | 0.152347 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | 0.152307 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)"} | 0.145942 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Existing Use) 1 family dwelling"} | 0.143941 |
| {"(Current Status) complete", "(Existing Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)"} | 0.143720 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Construction Type Description) wood frame (5)"} | 0.143680 |
| {"(Current Status) complete", "(Proposed Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)"} | 0.143373 |
| {"(Current Status) complete", "(Existing Use) 1 family dwelling", "(Proposed Use) 1 family dwelling"} | 0.142991 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Use) 1 family dwelling"} | 0.142186 |
| {"(Current Status) complete", "(Existing Use) 1 family dwelling", "(Permit Type Definition) otc alterations permit"} | 0.138933 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Use) 1 family dwelling"} | 0.138793 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Proposed Construction Type Description) constr type 1"} | 0.126902 |
| {"(Current Status) complete", "(Proposed Use) apartments", "(Existing Use) apartments"} | 0.121819 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Use) apartments"} | 0.119466 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Existing Use) apartments"} | 0.116078 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Use) office", "(Existing Use) office"} | 0.111865 |
| {"(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) 2 family dwelling"} | 0.106013 |
| {"(Existing Use) 2 family dwelling", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)"} | 0.104399 |
| {"(Current Status) complete", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments"} | 0.104338 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Use) apartments"} | 0.102599 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | 0.101412 |
| {"(Current Status) complete", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | 0.101362 |
| {"(Existing Use) 2 family dwelling", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) 2 family dwelling"} | 0.100361 |
| {"(Existing Use) 2 family dwelling", "(Existing Construction Type Description) wood frame (5)", "(Proposed Use) 2 family dwelling"} | 0.100296 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) 2 family dwelling"} | 0.097828 |
| {"(Existing Use) 2 family dwelling", "(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) wood frame (5)"} | 0.095656 |
| {"(Existing Use) 2 family dwelling", "(Existing Construction Type Description) wood frame (5)", "(Permit Type Definition) otc alterations permit"} | 0.095590 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) wood frame (5)", "(Proposed Use) 2 family dwelling"} | 0.095585 |
| {"(Existing Use) 2 family dwelling", "(Permit Type Definition) otc alterations permit", "(Proposed Use) 2 family dwelling"} | 0.094484 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1", "(Existing Use) office"} | 0.091272 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1", "(Proposed Use) office"} | 0.090482 |
| {"(Proposed Use) office", "(Existing Construction Type Description) constr type 1", "(Existing Use) office"} | 0.090261 |
| {"(Proposed Construction Type Description) constr type 1", "(Proposed Use) office", "(Existing Use) office"} | 0.090236 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Proposed Construction Type Description) constr type 1"} | 0.088798 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Existing Use) office"} | 0.087883 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Existing Use) office"} | 0.087873 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Proposed Use) office"} | 0.087757 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Proposed Use) office"} | 0.087234 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1"} | 0.086440 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit"} | 0.085148 |
| {"(Current Status) complete", "(Proposed Use) office", "(Existing Use) office"} | 0.078235 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Existing Use) office"} | 0.076903 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Use) office"} | 0.076450 |
| {"(Current Status) complete", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) 2 family dwelling"} | 0.069004 |
| {"(Proposed Use) 1 family dwelling", "(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)"} | 0.067571 |
| {"(Current Status) issued", "(Existing Use) 1 family dwelling", "(Existing Construction Type Description) wood frame (5)"} | 0.067491 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Use) 2 family dwelling"} | 0.067350 |
| {"(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) 1 family dwelling"} | 0.067224 |
| {"(Current Status) complete", "(Existing Use) 2 family dwelling", "(Proposed Construction Type Description) wood frame (5)"} | 0.066993 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Existing Use) 2 family dwelling"} | 0.066983 |
| {"(Proposed Use) 1 family dwelling", "(Current Status) issued", "(Existing Use) 1 family dwelling"} | 0.066133 |
| {"(Current Status) complete", "(Existing Use) 2 family dwelling", "(Proposed Use) 2 family dwelling"} | 0.066053 |
| {"(Proposed Use) 1 family dwelling", "(Current Status) issued", "(Existing Construction Type Description) wood frame (5)"} | 0.065827 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Use) 2 family dwelling"} | 0.065585 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Existing Use) 2 family dwelling"} | 0.064675 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Existing Use) office"} | 0.062559 |
| {"(Current Status) complete", "(Proposed Construction Type Description) constr type 1", "(Existing Use) office"} | 0.062202 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Use) 1 family dwelling"} | 0.062111 |
| {"(Current Status) complete", "(Proposed Construction Type Description) constr type 1", "(Proposed Use) office"} | 0.062106 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Use) 1 family dwelling"} | 0.062076 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Proposed Use) office"} | 0.061785 |
| {"(Current Status) issued", "(Proposed Use) apartments", "(Existing Use) apartments"} | 0.057074 |
| {"(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Permit Type Definition) additions alterations or repairs"} | 0.055022 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Use) apartments"} | 0.052579 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Use) apartments"} | 0.050628 |
| {"(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments"} | 0.049980 |
| {"(Current Status) issued", "(Proposed Use) apartments", "(Existing Construction Type Description) wood frame (5)"} | 0.049175 |
| {"(Current Status) issued", "(Existing Use) apartments", "(Existing Construction Type Description) wood frame (5)"} | 0.048059 |
| {"(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | 0.047843 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 3", "(Proposed Construction Type Description) constr type 3"} | 0.040779 |
| {"(Current Status) filed", "(Proposed Construction Type Description) wood frame (5)", "(Existing Construction Type Description) wood frame (5)"} | 0.038034 |
| {"(Proposed Construction Type Description) constr type 1", "(Current Status) issued", "(Existing Construction Type Description) constr type 1"} | 0.037561 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Construction Type Description) constr type 1"} | 0.036536 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Construction Type Description) constr type 1"} | 0.035400 |
| {"(Proposed Use) office", "(Current Status) issued", "(Existing Use) office"} | 0.032287 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Use) office"} | 0.031599 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Use) office"} | 0.031523 |
| {"(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) 2 family dwelling"} | 0.029784 |
| {"(Current Status) issued", "(Proposed Use) 2 family dwelling", "(Existing Construction Type Description) wood frame (5)"} | 0.028245 |
| {"(Existing Use) 2 family dwelling", "(Current Status) issued", "(Existing Construction Type Description) wood frame (5)"} | 0.027928 |
| {"(Existing Use) 2 family dwelling", "(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)"} | 0.027908 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 3", "(Proposed Construction Type Description) constr type 3"} | 0.027310 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Use) 2 family dwelling"} | 0.027044 |
| {"(Existing Use) 2 family dwelling", "(Current Status) issued", "(Proposed Use) 2 family dwelling"} | 0.026973 |
| {"(Existing Use) 2 family dwelling", "(Current Status) issued", "(Permit Type Definition) otc alterations permit"} | 0.025812 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 3"} | 0.025525 |
| {"(Existing Use) office", "(Current Status) issued", "(Existing Construction Type Description) constr type 1"} | 0.025369 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 3"} | 0.025022 |
| {"(Proposed Construction Type Description) constr type 1", "(Current Status) issued", "(Proposed Use) office"} | 0.024796 |
| {"(Proposed Construction Type Description) constr type 1", "(Current Status) issued", "(Existing Use) office"} | 0.024706 |
| {"(Proposed Use) office", "(Current Status) issued", "(Existing Construction Type Description) constr type 1"} | 0.024394 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) filed", "(Proposed Construction Type Description) wood frame (5)"} | 0.022921 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) filed", "(Existing Construction Type Description) wood frame (5)"} | 0.022780 |
| {"(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments", "(Permit Type Definition) additions alterations or repairs"} | 0.021905 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Use) retail sales", "(Existing Use) retail sales"} | 0.021729 |
| {"(Existing Construction Type Description) wood frame (5)", "(Proposed Use) apartments", "(Permit Type Definition) additions alterations or repairs"} | 0.021503 |
| {"(Proposed Use) apartments", "(Existing Use) apartments", "(Permit Type Definition) additions alterations or repairs"} | 0.020834 |
| {"(Existing Construction Type Description) wood frame (5)", "(Permit Type Definition) additions alterations or repairs", "(Existing Use) 1 family dwelling"} | 0.020251 |
| {"(Existing Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)", "(Permit Type Definition) additions alterations or repairs"} | 0.020226 |
| {"(Existing Construction Type Description) wood frame (5)", "(Existing Use) apartments", "(Permit Type Definition) additions alterations or repairs"} | 0.019040 |
| {"(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments", "(Permit Type Definition) additions alterations or repairs"} | 0.019004 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Use) food/beverage hndlng", "(Proposed Use) food/beverage hndlng"} | 0.018964 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Proposed Use) apartments"} | 0.018678 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1", "(Proposed Use) apartments"} | 0.018296 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1", "(Existing Use) apartments"} | 0.018225 |
| {"(Existing Construction Type Description) constr type 1", "(Proposed Use) apartments", "(Existing Use) apartments"} | 0.018160 |
| {"(Proposed Construction Type Description) constr type 1", "(Proposed Use) apartments", "(Existing Use) apartments"} | 0.018150 |
| {"(Proposed Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)", "(Permit Type Definition) additions alterations or repairs"} | 0.017335 |
| {"(Current Status) filed", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments"} | 0.017230 |
| {"(Current Status) complete", "(Proposed Construction Type Description) wood frame (5)", "(Permit Type Definition) additions alterations or repairs"} | 0.017179 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Proposed Use) apartments"} | 0.017039 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Existing Use) apartments"} | 0.016993 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Existing Use) apartments"} | 0.016963 |
| {"(Proposed Use) 1 family dwelling", "(Existing Construction Type Description) wood frame (5)", "(Permit Type Definition) additions alterations or repairs"} | 0.016868 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1", "(Street Name) Market"} | 0.016853 |
| {"(Current Status) filed", "(Proposed Use) apartments", "(Existing Use) apartments"} | 0.016817 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Permit Type Definition) additions alterations or repairs"} | 0.016782 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 2", "(Existing Construction Type Description) constr type 2"} | 0.016747 |
| {"(Current Status) filed", "(Proposed Use) apartments", "(Existing Construction Type Description) wood frame (5)"} | 0.016727 |
| {"(Proposed Use) 1 family dwelling", "(Existing Use) 1 family dwelling", "(Permit Type Definition) additions alterations or repairs"} | 0.016425 |
| {"(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)", "(Permit Type Definition) additions alterations or repairs"} | 0.016395 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Street Name) Market"} | 0.015917 |
| {"(Current Status) issued", "(Permit Type Definition) additions alterations or repairs", "(Existing Construction Type Description) wood frame (5)"} | 0.015711 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Street Name) Market"} | 0.015691 |
| {"(Current Status) filed", "(Existing Use) apartments", "(Existing Construction Type Description) wood frame (5)"} | 0.015485 |
| {"(Current Status) filed", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | 0.015440 |
| {"(Current Status) filed", "(Permit Type Definition) additions alterations or repairs", "(Existing Construction Type Description) wood frame (5)"} | 0.015329 |
| {"(Current Status) filed", "(Proposed Construction Type Description) wood frame (5)", "(Permit Type Definition) additions alterations or repairs"} | 0.015324 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Existing Use) retail sales"} | 0.014867 |
| {"(Permit Type Definition) sign - erect", "(Proposed Construction Type Description) NA", "(Proposed Use) NA"} | 0.014540 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Street Name) Market"} | 0.014148 |
| {"(Proposed Use) office", "(Existing Use) office", "(Street Name) Market"} | 0.014012 |
| {"(Current Status) issued", "(Existing Construction Type Description) constr type 3", "(Proposed Construction Type Description) constr type 3"} | 0.013771 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Use) office", "(Street Name) Market"} | 0.013595 |
| {"(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) food/beverage hndlng"} | 0.013544 |
| {"(Current Status) complete", "(Proposed Use) retail sales", "(Existing Use) retail sales"} | 0.013479 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Use) office", "(Street Name) Market"} | 0.013358 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1", "(Street Name) California"} | 0.013338 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Use) food/beverage hndlng"} | 0.013243 |
| {"(Current Status) complete", "(Proposed Construction Type Description) constr type 1", "(Proposed Use) apartments"} | 0.013203 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Street Name) California"} | 0.013067 |
| {"(Existing Construction Type Description) constr type 3", "(Existing Use) office", "(Proposed Construction Type Description) constr type 3"} | 0.013067 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Street Name) California"} | 0.013062 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Use) retail sales"} | 0.013042 |
| {"(Proposed Use) office", "(Existing Construction Type Description) constr type 3", "(Proposed Construction Type Description) constr type 3"} | 0.012921 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Construction Type Description) constr type 3"} | 0.012846 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Street Name) California"} | 0.012735 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Construction Type Description) constr type 3"} | 0.012720 |
| {"(Existing Construction Type Description) constr type 1", "(Existing Use) office", "(Street Name) Market"} | 0.012690 |
| {"(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) retail sales"} | 0.012619 |
| {"(Proposed Use) office", "(Street Name) California", "(Existing Use) office"} | 0.012589 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Use) office", "(Street Name) Market"} | 0.012574 |
| {"(Permit Type Definition) otc alterations permit", "(Street Name) California", "(Existing Use) office"} | 0.012418 |
| {"(Permit Type Definition) otc alterations permit", "(Street Name) California", "(Proposed Use) office"} | 0.012373 |
| {"(Proposed Construction Type Description) constr type 1", "(Proposed Use) office", "(Street Name) Market"} | 0.012343 |
| {"(Proposed Use) office", "(Existing Construction Type Description) constr type 1", "(Street Name) Market"} | 0.012333 |
| {"(Proposed Use) office", "(Existing Construction Type Description) constr type 3", "(Existing Use) office"} | 0.012298 |
| {"(Proposed Use) office", "(Existing Use) office", "(Proposed Construction Type Description) constr type 3"} | 0.012288 |
| {"(Current Status) filed", "(Existing Use) 1 family dwelling", "(Existing Construction Type Description) wood frame (5)"} | 0.012202 |
| {"(Current Status) complete", "(Existing Use) food/beverage hndlng", "(Proposed Use) food/beverage hndlng"} | 0.012056 |
| {"(Existing Construction Type Description) constr type 1", "(Street Name) California", "(Existing Use) office"} | 0.012031 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Existing Use) apartments"} | 0.012026 |
| {"(Proposed Construction Type Description) constr type 1", "(Street Name) California", "(Existing Use) office"} | 0.011961 |
| {"(Current Status) filed", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) 1 family dwelling"} | 0.011946 |
| {"(Proposed Use) office", "(Existing Construction Type Description) constr type 1", "(Street Name) California"} | 0.011931 |
| {"(Proposed Construction Type Description) constr type 1", "(Street Name) California", "(Proposed Use) office"} | 0.011926 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Proposed Use) apartments"} | 0.011860 |
| {"(Current Status) complete", "(Proposed Construction Type Description) constr type 1", "(Existing Use) apartments"} | 0.011820 |
| {"(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) food/beverage hndlng"} | 0.011820 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) filed", "(Proposed Use) apartments"} | 0.011795 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 3", "(Existing Use) office"} | 0.011704 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Existing Use) food/beverage hndlng"} | 0.011694 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Use) office", "(Proposed Construction Type Description) constr type 3"} | 0.011684 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) food/beverage hndlng"} | 0.011624 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) wood frame (5)", "(Proposed Use) food/beverage hndlng"} | 0.011584 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) filed", "(Existing Use) apartments"} | 0.011559 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 3", "(Proposed Use) office"} | 0.011543 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Use) office", "(Proposed Construction Type Description) constr type 3"} | 0.011543 |
| {"(Existing Construction Type Description) wood frame (5)", "(Existing Use) food/beverage hndlng", "(Proposed Use) food/beverage hndlng"} | 0.011473 |
| {"(Proposed Construction Type Description) wood frame (5)", "(Existing Use) food/beverage hndlng", "(Proposed Use) food/beverage hndlng"} | 0.011468 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Street Name) Market"} | 0.011443 |
| {"(Current Status) complete", "(Proposed Construction Type Description) constr type 1", "(Street Name) Market"} | 0.011337 |
| {"(Current Status) complete", "(Proposed Construction Type Description) constr type 2", "(Existing Construction Type Description) constr type 2"} | 0.011121 |
| {"(Proposed Use) 1 family dwelling", "(Current Status) filed", "(Proposed Construction Type Description) wood frame (5)"} | 0.010895 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) retail sales"} | 0.010774 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) wood frame (5)", "(Existing Use) retail sales"} | 0.010769 |
| {"(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) 2 family dwelling", "(Permit Type Definition) additions alterations or repairs"} | 0.010754 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) wood frame (5)", "(Existing Use) food/beverage hndlng"} | 0.010704 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) food/beverage hndlng"} | 0.010699 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 2"} | 0.010498 |
| {"(Existing Construction Type Description) wood frame (5)", "(Proposed Use) 2 family dwelling", "(Permit Type Definition) additions alterations or repairs"} | 0.010468 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 2"} | 0.010372 |
| {"(Proposed Use) 1 family dwelling", "(Current Status) filed", "(Existing Construction Type Description) wood frame (5)"} | 0.010342 |
| {"(Proposed Use) 1 family dwelling", "(Current Status) filed", "(Existing Use) 1 family dwelling"} | 0.010211 |
| {"(Existing Construction Type Description) wood frame (5)", "(Proposed Use) retail sales", "(Proposed Construction Type Description) wood frame (5)"} | 0.010111 |
| {"(Existing Construction Type Description) constr type 3", "(Proposed Use) apartments", "(Proposed Construction Type Description) constr type 3"} | 0.010101 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Use) apartments", "(Proposed Construction Type Description) constr type 3"} | 0.010050 |
| {"(Proposed Construction Type Description) constr type 3", "(Proposed Use) apartments", "(Existing Use) apartments"} | 0.010015 |
| {"(Existing Construction Type Description) constr type 3", "(Proposed Use) apartments", "(Existing Use) apartments"} | 0.010010 |
<br>

---------

**<center>4-Itemset</center>**

| Itemset | Support |
| :-----: | :-----: |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Permit Type Definition) otc alterations permit"} | 0.325524 |
| {"(Proposed Use) 1 family dwelling", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) 1 family dwelling"} | 0.226072 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) 1 family dwelling"} | 0.211965 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) wood frame (5)", "(Proposed Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)"} | 0.210205 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) wood frame (5)", "(Proposed Use) 1 family dwelling", "(Existing Use) 1 family dwelling"} | 0.209934 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Use) 1 family dwelling", "(Proposed Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)"} | 0.209863 |
| {"(Existing Construction Type Description) NA", "(Existing Use) NA", "(Proposed Construction Type Description) NA", "(Proposed Use) NA"} | 0.194569 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) NA", "(Existing Use) NA", "(Proposed Construction Type Description) NA"} | 0.191472 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) NA", "(Existing Use) NA", "(Proposed Use) NA"} | 0.191351 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) NA", "(Proposed Construction Type Description) NA", "(Proposed Use) NA"} | 0.191346 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Use) NA", "(Proposed Construction Type Description) NA", "(Proposed Use) NA"} | 0.191341 |
| {"(Current Status) issued", "(Existing Construction Type Description) NA", "(Proposed Construction Type Description) NA", "(Proposed Use) NA"} | 0.186817 |
| {"(Current Status) issued", "(Proposed Construction Type Description) NA", "(Existing Construction Type Description) NA", "(Existing Use) NA"} | 0.186801 |
| {"(Current Status) issued", "(Existing Construction Type Description) NA", "(Existing Use) NA", "(Proposed Use) NA"} | 0.186741 |
| {"(Current Status) issued", "(Existing Use) NA", "(Proposed Construction Type Description) NA", "(Proposed Use) NA"} | 0.186736 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Construction Type Description) NA", "(Proposed Construction Type Description) NA"} | 0.186651 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Construction Type Description) NA", "(Existing Use) NA"} | 0.186460 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Use) NA", "(Proposed Use) NA"} | 0.185394 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Use) NA", "(Proposed Construction Type Description) NA"} | 0.185348 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Construction Type Description) NA", "(Proposed Use) NA"} | 0.185308 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Construction Type Description) NA", "(Proposed Use) NA"} | 0.185303 |
| {"(Proposed Use) apartments", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | 0.170607 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments"} | 0.153891 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | 0.152181 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Use) apartments", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | 0.151910 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) wood frame (5)", "(Proposed Use) apartments", "(Existing Use) apartments"} | 0.151864 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) 1 family dwelling"} | 0.143619 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)", "(Existing Construction Type Description) wood frame (5)"} | 0.143463 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)"} | 0.142086 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Use) 1 family dwelling", "(Existing Use) 1 family dwelling"} | 0.141784 |
| {"(Current Status) complete", "(Existing Use) 1 family dwelling", "(Proposed Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)"} | 0.141784 |
| {"(Current Status) complete", "(Existing Use) 1 family dwelling", "(Proposed Use) 1 family dwelling", "(Permit Type Definition) otc alterations permit"} | 0.137797 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Permit Type Definition) otc alterations permit", "(Existing Use) 1 family dwelling"} | 0.137687 |
| {"(Current Status) complete", "(Existing Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)", "(Permit Type Definition) otc alterations permit"} | 0.137677 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)"} | 0.137571 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Use) 1 family dwelling", "(Permit Type Definition) otc alterations permit"} | 0.136872 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Use) apartments", "(Existing Use) apartments"} | 0.115761 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments"} | 0.102493 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | 0.101236 |
| {"(Current Status) complete", "(Proposed Use) apartments", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | 0.101085 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Use) apartments", "(Existing Use) apartments"} | 0.101015 |
| {"(Existing Use) 2 family dwelling", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) 2 family dwelling"} | 0.100266 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments"} | 0.098049 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Use) apartments", "(Permit Type Definition) otc alterations permit"} | 0.096867 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | 0.096023 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Existing Use) apartments", "(Permit Type Definition) otc alterations permit"} | 0.095957 |
| {"(Existing Use) 2 family dwelling", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Permit Type Definition) otc alterations permit"} | 0.095560 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) 2 family dwelling"} | 0.095555 |
| {"(Existing Use) 2 family dwelling", "(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) 2 family dwelling"} | 0.093665 |
| {"(Existing Use) 2 family dwelling", "(Existing Construction Type Description) wood frame (5)", "(Proposed Use) 2 family dwelling", "(Permit Type Definition) otc alterations permit"} | 0.093594 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1", "(Proposed Use) office", "(Existing Use) office"} | 0.090211 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Proposed Construction Type Description) constr type 1", "(Existing Use) office"} | 0.087833 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Proposed Use) office", "(Proposed Construction Type Description) constr type 1"} | 0.087189 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Proposed Use) office", "(Existing Use) office"} | 0.087028 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Proposed Use) office", "(Existing Use) office"} | 0.087008 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Proposed Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit"} | 0.085062 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Use) office", "(Existing Use) office"} | 0.075339 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) 2 family dwelling"} | 0.067330 |
| {"(Current Status) issued", "(Existing Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)", "(Existing Construction Type Description) wood frame (5)"} | 0.067194 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Existing Use) 2 family dwelling", "(Proposed Construction Type Description) wood frame (5)"} | 0.066933 |
| {"(Proposed Use) 1 family dwelling", "(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)", "(Existing Construction Type Description) wood frame (5)"} | 0.065746 |
| {"(Current Status) complete", "(Existing Use) 2 family dwelling", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) 2 family dwelling"} | 0.065580 |
| {"(Proposed Use) 1 family dwelling", "(Current Status) issued", "(Existing Use) 1 family dwelling", "(Existing Construction Type Description) wood frame (5)"} | 0.065570 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Existing Use) 2 family dwelling", "(Proposed Use) 2 family dwelling"} | 0.065535 |
| {"(Proposed Use) 1 family dwelling", "(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) 1 family dwelling"} | 0.065510 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) 2 family dwelling"} | 0.064982 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Existing Use) 2 family dwelling", "(Proposed Construction Type Description) wood frame (5)"} | 0.064203 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Existing Use) 2 family dwelling", "(Permit Type Definition) otc alterations permit"} | 0.064158 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Use) 2 family dwelling", "(Permit Type Definition) otc alterations permit"} | 0.064072 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Existing Use) 2 family dwelling", "(Proposed Use) 2 family dwelling"} | 0.063620 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Proposed Construction Type Description) constr type 1", "(Existing Use) office"} | 0.062182 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Proposed Use) office", "(Proposed Construction Type Description) constr type 1"} | 0.061754 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Proposed Use) office", "(Existing Use) office"} | 0.061609 |
| {"(Current Status) complete", "(Proposed Construction Type Description) constr type 1", "(Proposed Use) office", "(Existing Use) office"} | 0.061593 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Use) 1 family dwelling", "(Existing Construction Type Description) wood frame (5)"} | 0.061518 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) 1 family dwelling"} | 0.061443 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)"} | 0.061443 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Use) 1 family dwelling", "(Existing Use) 1 family dwelling"} | 0.061257 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Use) 1 family dwelling", "(Existing Construction Type Description) wood frame (5)"} | 0.060849 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Existing Use) office", "(Permit Type Definition) otc alterations permit"} | 0.060422 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Existing Use) office"} | 0.060407 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Use) office", "(Proposed Construction Type Description) constr type 1"} | 0.060291 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Proposed Use) office", "(Permit Type Definition) otc alterations permit"} | 0.060055 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Use) apartments", "(Existing Use) apartments"} | 0.050437 |
| {"(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments", "(Existing Construction Type Description) wood frame (5)"} | 0.049039 |
| {"(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments", "(Existing Construction Type Description) wood frame (5)"} | 0.047803 |
| {"(Current Status) issued", "(Proposed Use) apartments", "(Existing Use) apartments", "(Existing Construction Type Description) wood frame (5)"} | 0.047682 |
| {"(Proposed Use) apartments", "(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | 0.047652 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments"} | 0.042996 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Use) apartments", "(Existing Construction Type Description) wood frame (5)"} | 0.042579 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Use) apartments", "(Existing Construction Type Description) wood frame (5)"} | 0.041890 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | 0.041870 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1"} | 0.035364 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Use) office", "(Existing Use) office"} | 0.030875 |
| {"(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) 2 family dwelling", "(Existing Construction Type Description) wood frame (5)"} | 0.028230 |
| {"(Existing Use) 2 family dwelling", "(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)", "(Existing Construction Type Description) wood frame (5)"} | 0.027878 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) 2 family dwelling"} | 0.026667 |
| {"(Existing Use) 2 family dwelling", "(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) 2 family dwelling"} | 0.026646 |
| {"(Existing Use) 2 family dwelling", "(Current Status) issued", "(Proposed Use) 2 family dwelling", "(Existing Construction Type Description) wood frame (5)"} | 0.026626 |
| {"(Existing Use) 2 family dwelling", "(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)", "(Permit Type Definition) otc alterations permit"} | 0.025495 |
| {"(Existing Use) 2 family dwelling", "(Current Status) issued", "(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) wood frame (5)"} | 0.025475 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Use) 2 family dwelling", "(Existing Construction Type Description) wood frame (5)"} | 0.025460 |
| {"(Existing Use) 2 family dwelling", "(Current Status) issued", "(Proposed Use) 2 family dwelling", "(Permit Type Definition) otc alterations permit"} | 0.025043 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 3", "(Proposed Construction Type Description) constr type 3"} | 0.024967 |
| {"(Existing Use) office", "(Current Status) issued", "(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1"} | 0.024671 |
| {"(Proposed Construction Type Description) constr type 1", "(Current Status) issued", "(Proposed Use) office", "(Existing Construction Type Description) constr type 1"} | 0.024379 |
| {"(Existing Use) office", "(Current Status) issued", "(Proposed Use) office", "(Existing Construction Type Description) constr type 1"} | 0.024349 |
| {"(Proposed Construction Type Description) constr type 1", "(Current Status) issued", "(Proposed Use) office", "(Existing Use) office"} | 0.024344 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Use) office", "(Proposed Construction Type Description) constr type 1"} | 0.023811 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Construction Type Description) constr type 1", "(Existing Use) office"} | 0.023766 |
| {"(Existing Use) office", "(Current Status) issued", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit"} | 0.023761 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Use) office", "(Existing Construction Type Description) constr type 1"} | 0.023529 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) filed", "(Proposed Construction Type Description) wood frame (5)", "(Existing Construction Type Description) wood frame (5)"} | 0.022765 |
| {"(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments", "(Permit Type Definition) additions alterations or repairs"} | 0.021393 |
| {"(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Permit Type Definition) additions alterations or repairs", "(Existing Use) 1 family dwelling"} | 0.020216 |
| {"(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments", "(Permit Type Definition) additions alterations or repairs"} | 0.018999 |
| {"(Existing Construction Type Description) wood frame (5)", "(Proposed Use) apartments", "(Existing Use) apartments", "(Permit Type Definition) additions alterations or repairs"} | 0.018904 |
| {"(Proposed Use) apartments", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments", "(Permit Type Definition) additions alterations or repairs"} | 0.018869 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1", "(Proposed Use) apartments", "(Existing Use) apartments"} | 0.018115 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Proposed Construction Type Description) constr type 1", "(Proposed Use) apartments"} | 0.016978 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Proposed Construction Type Description) constr type 1", "(Existing Use) apartments"} | 0.016953 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Proposed Use) apartments", "(Existing Use) apartments"} | 0.016913 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Proposed Use) apartments", "(Existing Use) apartments"} | 0.016883 |
| {"(Proposed Use) 1 family dwelling", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Permit Type Definition) additions alterations or repairs"} | 0.016848 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Permit Type Definition) additions alterations or repairs"} | 0.016757 |
| {"(Current Status) filed", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments", "(Existing Construction Type Description) wood frame (5)"} | 0.016687 |
| {"(Proposed Use) 1 family dwelling", "(Existing Construction Type Description) wood frame (5)", "(Permit Type Definition) additions alterations or repairs", "(Existing Use) 1 family dwelling"} | 0.016355 |
| {"(Proposed Use) 1 family dwelling", "(Existing Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)", "(Permit Type Definition) additions alterations or repairs"} | 0.016335 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Proposed Construction Type Description) constr type 1", "(Street Name) Market"} | 0.015681 |
| {"(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)", "(Permit Type Definition) additions alterations or repairs", "(Existing Construction Type Description) wood frame (5)"} | 0.015641 |
| {"(Current Status) filed", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments", "(Existing Construction Type Description) wood frame (5)"} | 0.015440 |
| {"(Current Status) filed", "(Proposed Use) apartments", "(Existing Use) apartments", "(Existing Construction Type Description) wood frame (5)"} | 0.015395 |
| {"(Proposed Use) apartments", "(Current Status) filed", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | 0.015374 |
| {"(Current Status) filed", "(Proposed Construction Type Description) wood frame (5)", "(Permit Type Definition) additions alterations or repairs", "(Existing Construction Type Description) wood frame (5)"} | 0.015269 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Use) office", "(Existing Use) office", "(Street Name) Market"} | 0.013313 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Street Name) California", "(Proposed Construction Type Description) constr type 1"} | 0.013062 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Construction Type Description) constr type 3", "(Proposed Construction Type Description) constr type 3"} | 0.012675 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1", "(Existing Use) office", "(Street Name) Market"} | 0.012559 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Use) retail sales", "(Existing Use) retail sales"} | 0.012403 |
| {"(Permit Type Definition) otc alterations permit", "(Street Name) California", "(Proposed Use) office", "(Existing Use) office"} | 0.012358 |
| {"(Proposed Use) office", "(Existing Construction Type Description) constr type 1", "(Existing Use) office", "(Street Name) Market"} | 0.012333 |
| {"(Proposed Construction Type Description) constr type 1", "(Proposed Use) office", "(Existing Use) office", "(Street Name) Market"} | 0.012333 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1", "(Proposed Use) office", "(Street Name) Market"} | 0.012328 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Proposed Use) apartments"} | 0.012272 |
| {"(Proposed Use) office", "(Existing Construction Type Description) constr type 3", "(Existing Use) office", "(Proposed Construction Type Description) constr type 3"} | 0.012257 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1", "(Street Name) California", "(Existing Use) office"} | 0.011961 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Existing Use) office", "(Street Name) Market"} | 0.011946 |
| {"(Current Status) filed", "(Existing Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)", "(Existing Construction Type Description) wood frame (5)"} | 0.011941 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Existing Use) office", "(Street Name) Market"} | 0.011936 |
| {"(Proposed Use) office", "(Existing Construction Type Description) constr type 1", "(Street Name) California", "(Existing Use) office"} | 0.011926 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1", "(Street Name) California", "(Proposed Use) office"} | 0.011926 |
| {"(Proposed Construction Type Description) constr type 1", "(Street Name) California", "(Proposed Use) office", "(Existing Use) office"} | 0.011921 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Proposed Construction Type Description) constr type 1", "(Proposed Use) apartments"} | 0.011815 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Proposed Construction Type Description) constr type 1", "(Existing Use) apartments"} | 0.011810 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Street Name) California", "(Existing Use) office"} | 0.011790 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Street Name) California", "(Existing Use) office"} | 0.011785 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Proposed Use) office", "(Street Name) Market"} | 0.011765 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Street Name) California", "(Proposed Use) office"} | 0.011760 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Proposed Use) office", "(Street Name) Market"} | 0.011755 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Street Name) California", "(Proposed Use) office"} | 0.011755 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Proposed Use) apartments", "(Existing Use) apartments"} | 0.011740 |
| {"(Current Status) complete", "(Proposed Construction Type Description) constr type 1", "(Proposed Use) apartments", "(Existing Use) apartments"} | 0.011729 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 3", "(Existing Use) office", "(Proposed Construction Type Description) constr type 3"} | 0.011659 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) food/beverage hndlng"} | 0.011569 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) filed", "(Proposed Use) apartments", "(Existing Use) apartments"} | 0.011533 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 3", "(Proposed Use) office", "(Proposed Construction Type Description) constr type 3"} | 0.011498 |
| {"(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) food/beverage hndlng", "(Proposed Use) food/beverage hndlng"} | 0.011458 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Proposed Use) apartments", "(Permit Type Definition) otc alterations permit"} | 0.011403 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Existing Use) apartments", "(Permit Type Definition) otc alterations permit"} | 0.011367 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Existing Use) apartments"} | 0.011352 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Existing Use) food/beverage hndlng", "(Proposed Use) food/beverage hndlng"} | 0.011307 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Proposed Construction Type Description) constr type 1", "(Street Name) Market"} | 0.011151 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 3", "(Proposed Use) office", "(Existing Use) office"} | 0.011091 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Use) office", "(Existing Use) office", "(Proposed Construction Type Description) constr type 3"} | 0.011071 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) retail sales"} | 0.010764 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) food/beverage hndlng"} | 0.010689 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Street Name) Market"} | 0.010603 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) filed", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments"} | 0.010593 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) filed", "(Proposed Use) apartments", "(Existing Construction Type Description) wood frame (5)"} | 0.010533 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Street Name) Market", "(Permit Type Definition) otc alterations permit"} | 0.010462 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) filed", "(Existing Use) apartments", "(Existing Construction Type Description) wood frame (5)"} | 0.010457 |
| {"(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) 2 family dwelling", "(Permit Type Definition) additions alterations or repairs"} | 0.010457 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) filed", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | 0.010452 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) wood frame (5)", "(Existing Use) food/beverage hndlng", "(Proposed Use) food/beverage hndlng"} | 0.010437 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) food/beverage hndlng", "(Proposed Use) food/beverage hndlng"} | 0.010432 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 2", "(Existing Construction Type Description) constr type 2"} | 0.010347 |
| {"(Proposed Use) 1 family dwelling", "(Current Status) filed", "(Proposed Construction Type Description) wood frame (5)", "(Existing Construction Type Description) wood frame (5)"} | 0.010317 |
| {"(Proposed Use) 1 family dwelling", "(Current Status) filed", "(Existing Use) 1 family dwelling", "(Existing Construction Type Description) wood frame (5)"} | 0.010161 |
| {"(Proposed Use) 1 family dwelling", "(Current Status) filed", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) 1 family dwelling"} | 0.010141 |
<br>

---------

**<center>5-Itemset</center>**

| Itemset | Support |
| :-----: | :-----: |
| {"(Proposed Use) 1 family dwelling", "(Permit Type Definition) otc alterations permit", "(Existing Use) 1 family dwelling", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)"} | 0.209738 |
| {"(Proposed Construction Type Description) NA", "(Proposed Use) NA", "(Permit Type Definition) otc alterations permit", "(Existing Use) NA", "(Existing Construction Type Description) NA"} | 0.191341 |
| {"(Proposed Construction Type Description) NA", "(Proposed Use) NA", "(Current Status) issued", "(Existing Use) NA", "(Existing Construction Type Description) NA"} | 0.186736 |
| {"(Proposed Construction Type Description) NA", "(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Use) NA", "(Existing Construction Type Description) NA"} | 0.185348 |
| {"(Proposed Use) NA", "(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Use) NA", "(Existing Construction Type Description) NA"} | 0.185303 |
| {"(Proposed Construction Type Description) NA", "(Proposed Use) NA", "(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Construction Type Description) NA"} | 0.185303 |
| {"(Proposed Construction Type Description) NA", "(Proposed Use) NA", "(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Use) NA"} | 0.185303 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Use) apartments", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments"} | 0.151744 |
| {"(Proposed Use) 1 family dwelling", "(Existing Use) 1 family dwelling", "(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)"} | 0.141684 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Use) 1 family dwelling", "(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)"} | 0.137576 |
| {"(Proposed Use) 1 family dwelling", "(Permit Type Definition) otc alterations permit", "(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)"} | 0.136772 |
| {"(Proposed Use) 1 family dwelling", "(Permit Type Definition) otc alterations permit", "(Existing Use) 1 family dwelling", "(Current Status) complete", "(Existing Construction Type Description) wood frame (5)"} | 0.136601 |
| {"(Proposed Use) 1 family dwelling", "(Permit Type Definition) otc alterations permit", "(Current Status) complete", "(Existing Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)"} | 0.136601 |
| {"(Existing Use) apartments", "(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments"} | 0.100960 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments"} | 0.096767 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | 0.095897 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) complete", "(Existing Use) apartments", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments"} | 0.095791 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Use) apartments", "(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Use) apartments"} | 0.095721 |
| {"(Existing Use) 2 family dwelling", "(Permit Type Definition) otc alterations permit", "(Proposed Use) 2 family dwelling", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)"} | 0.093569 |
| {"(Proposed Construction Type Description) constr type 1", "(Proposed Use) office", "(Existing Use) office", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit"} | 0.086983 |
| {"(Existing Use) 2 family dwelling", "(Proposed Use) 2 family dwelling", "(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)"} | 0.065520 |
| {"(Proposed Use) 1 family dwelling", "(Current Status) issued", "(Existing Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)", "(Existing Construction Type Description) wood frame (5)"} | 0.065490 |
| {"(Existing Use) 2 family dwelling", "(Permit Type Definition) otc alterations permit", "(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)"} | 0.064142 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Use) 2 family dwelling", "(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)"} | 0.064057 |
| {"(Existing Use) 2 family dwelling", "(Permit Type Definition) otc alterations permit", "(Proposed Use) 2 family dwelling", "(Current Status) complete", "(Proposed Construction Type Description) wood frame (5)"} | 0.063157 |
| {"(Existing Use) 2 family dwelling", "(Permit Type Definition) otc alterations permit", "(Proposed Use) 2 family dwelling", "(Current Status) complete", "(Existing Construction Type Description) wood frame (5)"} | 0.063112 |
| {"(Proposed Construction Type Description) constr type 1", "(Proposed Use) office", "(Existing Use) office", "(Existing Construction Type Description) constr type 1", "(Current Status) complete"} | 0.061578 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)", "(Existing Construction Type Description) wood frame (5)"} | 0.061423 |
| {"(Proposed Use) 1 family dwelling", "(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)"} | 0.060774 |
| {"(Proposed Use) 1 family dwelling", "(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Use) 1 family dwelling", "(Existing Construction Type Description) wood frame (5)"} | 0.060699 |
| {"(Proposed Use) 1 family dwelling", "(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)"} | 0.060643 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Use) office", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit", "(Current Status) complete"} | 0.060387 |
| {"(Proposed Use) office", "(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit", "(Current Status) complete"} | 0.060025 |
| {"(Proposed Use) office", "(Existing Use) office", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit", "(Current Status) complete"} | 0.059904 |
| {"(Proposed Construction Type Description) constr type 1", "(Proposed Use) office", "(Existing Use) office", "(Permit Type Definition) otc alterations permit", "(Current Status) complete"} | 0.059889 |
| {"(Current Status) issued", "(Existing Use) apartments", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments", "(Existing Construction Type Description) wood frame (5)"} | 0.047612 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments"} | 0.042504 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | 0.041835 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Use) apartments", "(Proposed Use) apartments", "(Existing Construction Type Description) wood frame (5)"} | 0.041714 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Use) apartments", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments"} | 0.041694 |
| {"(Existing Use) 2 family dwelling", "(Current Status) issued", "(Proposed Use) 2 family dwelling", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)"} | 0.026616 |
| {"(Existing Use) 2 family dwelling", "(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)"} | 0.025465 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Use) 2 family dwelling", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)"} | 0.025445 |
| {"(Existing Use) 2 family dwelling", "(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Use) 2 family dwelling", "(Proposed Construction Type Description) wood frame (5)"} | 0.024726 |
| {"(Existing Use) 2 family dwelling", "(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Use) 2 family dwelling", "(Existing Construction Type Description) wood frame (5)"} | 0.024706 |
| {"(Proposed Construction Type Description) constr type 1", "(Proposed Use) office", "(Existing Use) office", "(Existing Construction Type Description) constr type 1", "(Current Status) issued"} | 0.024334 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Use) office", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit", "(Current Status) issued"} | 0.023745 |
| {"(Proposed Construction Type Description) constr type 1", "(Proposed Use) office", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit", "(Current Status) issued"} | 0.023514 |
| {"(Proposed Use) office", "(Existing Use) office", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit", "(Current Status) issued"} | 0.023494 |
| {"(Proposed Construction Type Description) constr type 1", "(Proposed Use) office", "(Existing Use) office", "(Permit Type Definition) otc alterations permit", "(Current Status) issued"} | 0.023489 |
| {"(Permit Type Definition) additions alterations or repairs", "(Existing Use) apartments", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments"} | 0.018864 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit", "(Existing Use) apartments", "(Proposed Use) apartments"} | 0.016873 |
| {"(Proposed Use) 1 family dwelling", "(Permit Type Definition) additions alterations or repairs", "(Existing Use) 1 family dwelling", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)"} | 0.016335 |
| {"(Existing Use) apartments", "(Current Status) filed", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments", "(Existing Construction Type Description) wood frame (5)"} | 0.015374 |
| {"(Proposed Construction Type Description) constr type 1", "(Proposed Use) office", "(Existing Use) office", "(Street Name) Market", "(Existing Construction Type Description) constr type 1"} | 0.012328 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Use) office", "(Street Name) Market", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit"} | 0.011931 |
| {"(Proposed Construction Type Description) constr type 1", "(Proposed Use) office", "(Existing Use) office", "(Existing Construction Type Description) constr type 1", "(Street Name) California"} | 0.011921 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Use) office", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit", "(Street Name) California"} | 0.011785 |
| {"(Proposed Use) office", "(Existing Use) office", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit", "(Street Name) California"} | 0.011755 |
| {"(Proposed Use) office", "(Existing Use) office", "(Street Name) Market", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit"} | 0.011755 |
| {"(Proposed Use) office", "(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit", "(Street Name) California"} | 0.011755 |
| {"(Proposed Construction Type Description) constr type 1", "(Proposed Use) office", "(Existing Use) office", "(Street Name) Market", "(Permit Type Definition) otc alterations permit"} | 0.011755 |
| {"(Proposed Use) office", "(Proposed Construction Type Description) constr type 1", "(Street Name) Market", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit"} | 0.011750 |
| {"(Proposed Construction Type Description) constr type 1", "(Proposed Use) office", "(Existing Use) office", "(Permit Type Definition) otc alterations permit", "(Street Name) California"} | 0.011750 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1", "(Current Status) complete", "(Existing Use) apartments", "(Proposed Use) apartments"} | 0.011719 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit", "(Current Status) complete", "(Proposed Use) apartments"} | 0.011362 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit", "(Current Status) complete", "(Existing Use) apartments"} | 0.011347 |
| {"(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit", "(Current Status) complete", "(Existing Use) apartments", "(Proposed Use) apartments"} | 0.011302 |
| {"(Proposed Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit", "(Current Status) complete", "(Existing Use) apartments", "(Proposed Use) apartments"} | 0.011287 |
| {"(Proposed Use) office", "(Existing Use) office", "(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 3", "(Proposed Construction Type Description) constr type 3"} | 0.011051 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) filed", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments", "(Existing Construction Type Description) wood frame (5)"} | 0.010528 |
| {"(Proposed Construction Type Description) constr type 1", "(Street Name) Market", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit", "(Current Status) complete"} | 0.010452 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) filed", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments", "(Existing Construction Type Description) wood frame (5)"} | 0.010452 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Use) apartments", "(Current Status) filed", "(Proposed Use) apartments", "(Existing Construction Type Description) wood frame (5)"} | 0.010432 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Use) apartments", "(Current Status) filed", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments"} | 0.010427 |
| {"(Proposed Use) food/beverage hndlng", "(Permit Type Definition) otc alterations permit", "(Existing Use) food/beverage hndlng", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)"} | 0.010422 |
| {"(Proposed Use) 1 family dwelling", "(Existing Use) 1 family dwelling", "(Current Status) filed", "(Proposed Construction Type Description) wood frame (5)", "(Existing Construction Type Description) wood frame (5)"} | 0.010136 |
<br>

---------

**<center>6-Itemset</center>**

| Itemset | Support |
| :-----: | :-----: |
| {"(Proposed Construction Type Description) NA", "(Proposed Use) NA", "(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Use) NA", "(Existing Construction Type Description) NA"} | 0.185303 |
| {"(Proposed Use) 1 family dwelling", "(Permit Type Definition) otc alterations permit", "(Existing Use) 1 family dwelling", "(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)"} | 0.136500 |
| {"(Proposed Use) apartments", "(Permit Type Definition) otc alterations permit", "(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | 0.095666 |
| {"(Existing Use) 2 family dwelling", "(Permit Type Definition) otc alterations permit", "(Proposed Use) 2 family dwelling", "(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)"} | 0.063097 |
| {"(Proposed Use) 1 family dwelling", "(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)", "(Existing Construction Type Description) wood frame (5)"} | 0.060623 |
| {"(Proposed Construction Type Description) constr type 1", "(Proposed Use) office", "(Existing Use) office", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit", "(Current Status) complete"} | 0.059874 |
| {"(Proposed Use) apartments", "(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | 0.041659 |
| {"(Existing Use) 2 family dwelling", "(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Use) 2 family dwelling", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)"} | 0.024696 |
| {"(Proposed Construction Type Description) constr type 1", "(Proposed Use) office", "(Existing Use) office", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit", "(Current Status) issued"} | 0.023479 |
| {"(Proposed Construction Type Description) constr type 1", "(Proposed Use) office", "(Existing Use) office", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit", "(Street Name) California"} | 0.011750 |
| {"(Proposed Construction Type Description) constr type 1", "(Proposed Use) office", "(Existing Use) office", "(Street Name) Market", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit"} | 0.011750 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1", "(Proposed Use) apartments", "(Permit Type Definition) otc alterations permit", "(Current Status) complete", "(Existing Use) apartments"} | 0.011282 |
| {"(Proposed Use) apartments", "(Permit Type Definition) otc alterations permit", "(Current Status) filed", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments", "(Existing Construction Type Description) wood frame (5)"} | 0.010427 |
<br>

---------


## 关联规则的导出与评估

一般我们使用三个指标来度量一个关联规则，这三个指标分别是：支持度、置信度和提升度。

- **Support（支持度）**：表示同时包含 A 和 B 的事务占所有事务的比例。如果用 P(A) 表示包含 A 的事务的比例，那么 $$Support=P(A\&B)$$
- **Confidence（置信度）**：表示包含 A 的事务中同时包含 B 的事务的比例，即同时包含 A 和 B 的事务占包含 A 的事务的比例。公式表达：$$Confidence = \frac{P(A \& B)}{P(A)}$$
- **Lift（提升度）**：表示“包含 A 的事务中同时包含 B 的事务的比例”与“包含 B 的事务的比例”的比值。公式表达：$$Lift = \frac{P(A \& B)}{P(A)P(B)}$$。提升度反映了关联规则中的 A 与 B 的相关性，提升度 > 1 且越高表明正相关性越高，提升度 < 1 且越低表明负相关性越高，提升度 = 1 表明没有相关性。

为了挖掘出可信的强关联信息，在这里将置信度阈值设置为0.6, 将支持度阈值设置为3。为了方便导出，我们做出了一些限制，设置后继项只包含一个事件。因为对缺失值我们采用不同的NA进行填充，为了避免出现先导项和后继项均为NA的关联规则出现，我们将这种情况的关联规则筛去。经过筛选，我们得到了一些关联规则，按照前导项集的大小组织，如下表所示。

---------

**<center>1-Itemset Relation Rules</center>**

| LHS | RHS | Support | Confidence | Lift |
| :-----: | :-----: | :-----: | :-----: | :-----: |
| {"(Existing Use) 1 family dwelling"} | {"(Proposed Use) 1 family dwelling"}  | 0.228149 | 0.970342 | 4.164371 |
| {"(Proposed Use) 1 family dwelling"} | {"(Existing Use) 1 family dwelling"}  | 0.228149 | 0.979135 | 4.164371 |
| {"(Existing Use) apartments"} | {"(Proposed Use) apartments"}  | 0.203342 | 0.991348 | 4.582172 |
| {"(Proposed Use) apartments"} | {"(Existing Use) apartments"}  | 0.203342 | 0.939882 | 4.582172 |
| {"(Existing Construction Type Description) constr type 1"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.134816 | 0.955222 | 6.824276 |
| {"(Proposed Construction Type Description) constr type 1"} | {"(Existing Construction Type Description) constr type 1"}  | 0.134816 | 0.963148 | 6.824276 |
| {"(Existing Use) office"} | {"(Proposed Use) office"}  | 0.117657 | 0.950682 | 7.891315 |
| {"(Proposed Use) office"} | {"(Existing Use) office"}  | 0.117657 | 0.976630 | 7.891315 |
| {"(Existing Use) 2 family dwelling"} | {"(Proposed Use) 2 family dwelling"}  | 0.101216 | 0.959260 | 8.648650 |
| {"(Proposed Use) 2 family dwelling"} | {"(Existing Use) 2 family dwelling"}  | 0.101216 | 0.912561 | 8.648650 |
| {"(Existing Construction Type Description) constr type 1"} | {"(Existing Use) office"}  | 0.092544 | 0.655707 | 5.298210 |
| {"(Existing Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.092544 | 0.747766 | 5.298210 |
| {"(Existing Use) office"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.091327 | 0.737935 | 5.271935 |
| {"(Proposed Construction Type Description) constr type 1"} | {"(Existing Use) office"}  | 0.091327 | 0.652455 | 5.271935 |
| {"(Proposed Construction Type Description) constr type 1"} | {"(Proposed Use) office"}  | 0.091297 | 0.652240 | 5.414034 |
| {"(Proposed Use) office"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.091297 | 0.757825 | 5.414034 |
| {"(Existing Construction Type Description) constr type 1"} | {"(Proposed Use) office"}  | 0.090532 | 0.641458 | 5.324538 |
| {"(Proposed Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.090532 | 0.751482 | 5.324538 |
| {"(Existing Construction Type Description) constr type 3"} | {"(Proposed Construction Type Description) constr type 3"}  | 0.045676 | 0.940184 | 19.979015 |
| {"(Proposed Construction Type Description) constr type 3"} | {"(Existing Construction Type Description) constr type 3"}  | 0.045676 | 0.970620 | 19.979015 |
| {"(Existing Use) retail sales"} | {"(Proposed Use) retail sales"}  | 0.024128 | 0.694501 | 27.197655 |
| {"(Proposed Use) retail sales"} | {"(Existing Use) retail sales"}  | 0.024128 | 0.944871 | 27.197655 |
| {"(Existing Use) food/beverage hndlng"} | {"(Proposed Use) food/beverage hndlng"}  | 0.020774 | 0.845682 | 33.288522 |
| {"(Proposed Use) food/beverage hndlng"} | {"(Existing Use) food/beverage hndlng"}  | 0.020774 | 0.817732 | 33.288522 |
| {"(Existing Construction Type Description) constr type 2"} | {"(Proposed Construction Type Description) constr type 2"}  | 0.018461 | 0.902655 | 47.522222 |
| {"(Proposed Construction Type Description) constr type 2"} | {"(Existing Construction Type Description) constr type 2"}  | 0.018461 | 0.971943 | 47.522222 |
| {"(Street Name) Market"} | {"(Existing Construction Type Description) constr type 1"}  | 0.017692 | 0.646518 | 4.580834 |
| {"(Street Name) Market"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.017215 | 0.629065 | 4.494150 |
<br>

---------

**<center>2-Itemset Relation Rules</center>**

| LHS | RHS | Support | Confidence | Lift |
| :-----: | :-----: | :-----: | :-----: | :-----: |
| {"(Existing Construction Type Description) wood frame (5)", "(Existing Use) 1 family dwelling"} | {"(Proposed Use) 1 family dwelling"}  | 0.226288 | 0.970649 | 4.165690 |
| {"(Proposed Use) 1 family dwelling", "(Existing Construction Type Description) wood frame (5)"} | {"(Existing Use) 1 family dwelling"}  | 0.226288 | 0.995686 | 4.234764 |
| {"(Existing Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)"} | {"(Proposed Use) 1 family dwelling"}  | 0.226198 | 0.973641 | 4.178533 |
| {"(Proposed Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)"} | {"(Existing Use) 1 family dwelling"}  | 0.226198 | 0.979513 | 4.165979 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Use) 1 family dwelling"} | {"(Proposed Use) 1 family dwelling"}  | 0.211723 | 0.989032 | 4.244584 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Use) 1 family dwelling"} | {"(Existing Use) 1 family dwelling"}  | 0.211723 | 0.990871 | 4.214283 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Use) apartments"} | {"(Proposed Use) apartments"}  | 0.182508 | 0.997061 | 4.608581 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Use) apartments"} | {"(Existing Use) apartments"}  | 0.182508 | 0.966506 | 4.711972 |
| {"(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | {"(Proposed Use) apartments"}  | 0.170778 | 0.996655 | 4.606704 |
| {"(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments"} | {"(Existing Use) apartments"}  | 0.170778 | 0.954613 | 4.653991 |
| {"(Existing Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | {"(Proposed Use) apartments"}  | 0.170768 | 0.994583 | 4.597126 |
| {"(Existing Construction Type Description) wood frame (5)", "(Proposed Use) apartments"} | {"(Existing Use) apartments"}  | 0.170768 | 0.972597 | 4.741666 |
| {"(Current Status) complete", "(Existing Use) 1 family dwelling"} | {"(Proposed Use) 1 family dwelling"}  | 0.142991 | 0.984731 | 4.226125 |
| {"(Current Status) complete", "(Proposed Use) 1 family dwelling"} | {"(Existing Use) 1 family dwelling"}  | 0.142991 | 0.988805 | 4.205498 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.126902 | 0.999011 | 7.137107 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1"} | {"(Existing Construction Type Description) constr type 1"}  | 0.126902 | 0.978713 | 6.934559 |
| {"(Current Status) complete", "(Existing Use) apartments"} | {"(Proposed Use) apartments"}  | 0.121819 | 0.994255 | 4.595612 |
| {"(Current Status) complete", "(Proposed Use) apartments"} | {"(Existing Use) apartments"}  | 0.121819 | 0.957935 | 4.670184 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Use) office"} | {"(Proposed Use) office"}  | 0.111865 | 0.977678 | 8.115398 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Use) office"} | {"(Existing Use) office"}  | 0.111865 | 0.982774 | 7.940961 |
| {"(Existing Use) 2 family dwelling", "(Proposed Construction Type Description) wood frame (5)"} | {"(Proposed Use) 2 family dwelling"}  | 0.100361 | 0.960450 | 8.659378 |
| {"(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) 2 family dwelling"} | {"(Existing Use) 2 family dwelling"}  | 0.100361 | 0.914220 | 8.664379 |
| {"(Existing Use) 2 family dwelling", "(Existing Construction Type Description) wood frame (5)"} | {"(Proposed Use) 2 family dwelling"}  | 0.100296 | 0.959133 | 8.647498 |
| {"(Existing Construction Type Description) wood frame (5)", "(Proposed Use) 2 family dwelling"} | {"(Existing Use) 2 family dwelling"}  | 0.100296 | 0.945719 | 8.962905 |
| {"(Existing Use) 2 family dwelling", "(Permit Type Definition) otc alterations permit"} | {"(Proposed Use) 2 family dwelling"}  | 0.094484 | 0.979210 | 8.828515 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Use) 2 family dwelling"} | {"(Existing Use) 2 family dwelling"}  | 0.094484 | 0.955754 | 9.058013 |
| {"(Existing Construction Type Description) constr type 1", "(Existing Use) office"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.091272 | 0.986255 | 7.045981 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1"} | {"(Existing Use) office"}  | 0.091272 | 0.677009 | 5.470336 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.091272 | 0.999394 | 7.081097 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1"} | {"(Proposed Use) office"}  | 0.090482 | 0.671154 | 5.571039 |
| {"(Proposed Use) office", "(Existing Construction Type Description) constr type 1"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.090482 | 0.999445 | 7.140208 |
| {"(Proposed Construction Type Description) constr type 1", "(Proposed Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.090482 | 0.991079 | 7.022177 |
| {"(Existing Construction Type Description) constr type 1", "(Existing Use) office"} | {"(Proposed Use) office"}  | 0.090261 | 0.975335 | 8.095952 |
| {"(Proposed Use) office", "(Existing Construction Type Description) constr type 1"} | {"(Existing Use) office"}  | 0.090261 | 0.997001 | 8.055920 |
| {"(Proposed Use) office", "(Existing Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.090261 | 0.767157 | 5.435602 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Use) office"} | {"(Proposed Use) office"}  | 0.090236 | 0.988054 | 8.201524 |
| {"(Proposed Use) office", "(Existing Use) office"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.090236 | 0.766943 | 5.479176 |
| {"(Proposed Construction Type Description) constr type 1", "(Proposed Use) office"} | {"(Existing Use) office"}  | 0.090236 | 0.988380 | 7.986263 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.088798 | 0.975747 | 6.970910 |
| {"(Current Status) complete", "(Proposed Construction Type Description) constr type 1"} | {"(Existing Construction Type Description) constr type 1"}  | 0.088798 | 0.978017 | 6.929626 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1"} | {"(Existing Use) office"}  | 0.087883 | 0.691839 | 5.590162 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.087883 | 0.768082 | 5.442155 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Use) office"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.087873 | 0.767994 | 5.486682 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1"} | {"(Existing Use) office"}  | 0.087873 | 0.677705 | 5.475955 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1"} | {"(Proposed Use) office"}  | 0.087757 | 0.676813 | 5.618009 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Use) office"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.087757 | 0.770981 | 5.508021 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1"} | {"(Proposed Use) office"}  | 0.087234 | 0.686733 | 5.700355 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.087234 | 0.766387 | 5.430148 |
| {"(Current Status) complete", "(Existing Use) office"} | {"(Proposed Use) office"}  | 0.078235 | 0.965682 | 8.015822 |
| {"(Current Status) complete", "(Proposed Use) office"} | {"(Existing Use) office"}  | 0.078235 | 0.982635 | 7.939835 |
| {"(Current Status) issued", "(Existing Use) 1 family dwelling"} | {"(Proposed Use) 1 family dwelling"}  | 0.066133 | 0.971062 | 4.167460 |
| {"(Proposed Use) 1 family dwelling", "(Current Status) issued"} | {"(Existing Use) 1 family dwelling"}  | 0.066133 | 0.969201 | 4.122121 |
| {"(Current Status) complete", "(Existing Use) 2 family dwelling"} | {"(Proposed Use) 2 family dwelling"}  | 0.066053 | 0.978403 | 8.821241 |
| {"(Current Status) complete", "(Proposed Use) 2 family dwelling"} | {"(Existing Use) 2 family dwelling"}  | 0.066053 | 0.948592 | 8.990132 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1"} | {"(Existing Use) office"}  | 0.062559 | 0.687421 | 5.554462 |
| {"(Current Status) complete", "(Existing Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.062559 | 0.772186 | 5.471235 |
| {"(Current Status) complete", "(Existing Use) office"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.062202 | 0.767780 | 5.485152 |
| {"(Current Status) complete", "(Proposed Construction Type Description) constr type 1"} | {"(Existing Use) office"}  | 0.062202 | 0.685088 | 5.535613 |
| {"(Current Status) complete", "(Proposed Construction Type Description) constr type 1"} | {"(Proposed Use) office"}  | 0.062106 | 0.684036 | 5.677964 |
| {"(Current Status) complete", "(Proposed Use) office"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.062106 | 0.780058 | 5.572872 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1"} | {"(Proposed Use) office"}  | 0.061785 | 0.678913 | 5.635441 |
| {"(Current Status) complete", "(Proposed Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.061785 | 0.776017 | 5.498379 |
| {"(Current Status) issued", "(Existing Use) apartments"} | {"(Proposed Use) apartments"}  | 0.057074 | 0.986444 | 4.559508 |
| {"(Current Status) issued", "(Proposed Use) apartments"} | {"(Existing Use) apartments"}  | 0.057074 | 0.933937 | 4.553187 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 3"} | {"(Proposed Construction Type Description) constr type 3"}  | 0.040779 | 0.997540 | 21.197837 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 3"} | {"(Existing Construction Type Description) constr type 3"}  | 0.040779 | 0.981961 | 20.212468 |
| {"(Current Status) issued", "(Existing Construction Type Description) constr type 1"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.037561 | 0.924057 | 6.601625 |
| {"(Proposed Construction Type Description) constr type 1", "(Current Status) issued"} | {"(Existing Construction Type Description) constr type 1"}  | 0.037561 | 0.948217 | 6.718483 |
| {"(Current Status) issued", "(Existing Use) office"} | {"(Proposed Use) office"}  | 0.032287 | 0.937381 | 7.780907 |
| {"(Proposed Use) office", "(Current Status) issued"} | {"(Existing Use) office"}  | 0.032287 | 0.972883 | 7.861041 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 3"} | {"(Proposed Construction Type Description) constr type 3"}  | 0.027310 | 0.968444 | 20.579529 |
| {"(Current Status) complete", "(Proposed Construction Type Description) constr type 3"} | {"(Existing Construction Type Description) constr type 3"}  | 0.027310 | 0.973826 | 20.045008 |
| {"(Existing Use) 2 family dwelling", "(Current Status) issued"} | {"(Proposed Use) 2 family dwelling"}  | 0.026973 | 0.953947 | 8.600747 |
| {"(Current Status) issued", "(Proposed Use) 2 family dwelling"} | {"(Existing Use) 2 family dwelling"}  | 0.026973 | 0.893422 | 8.467268 |
| {"(Current Status) issued", "(Existing Construction Type Description) constr type 1"} | {"(Existing Use) office"}  | 0.025369 | 0.624119 | 5.042974 |
| {"(Current Status) issued", "(Existing Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.025369 | 0.736535 | 5.218635 |
| {"(Proposed Construction Type Description) constr type 1", "(Current Status) issued"} | {"(Proposed Use) office"}  | 0.024796 | 0.625968 | 5.195961 |
| {"(Proposed Use) office", "(Current Status) issued"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.024796 | 0.747160 | 5.337839 |
| {"(Current Status) issued", "(Existing Use) office"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.024706 | 0.717268 | 5.124286 |
| {"(Proposed Construction Type Description) constr type 1", "(Current Status) issued"} | {"(Existing Use) office"}  | 0.024706 | 0.623683 | 5.039455 |
| {"(Current Status) issued", "(Existing Construction Type Description) constr type 1"} | {"(Proposed Use) office"}  | 0.024394 | 0.600124 | 4.981437 |
| {"(Proposed Use) office", "(Current Status) issued"} | {"(Existing Construction Type Description) constr type 1"}  | 0.024394 | 0.735040 | 5.208044 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Use) retail sales"} | {"(Proposed Use) retail sales"}  | 0.021729 | 0.850285 | 33.298403 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Use) retail sales"} | {"(Existing Use) retail sales"}  | 0.021729 | 0.952612 | 27.420471 |
| {"(Existing Use) apartments", "(Permit Type Definition) additions alterations or repairs"} | {"(Proposed Use) apartments"}  | 0.020834 | 0.991862 | 4.584550 |
| {"(Proposed Use) apartments", "(Permit Type Definition) additions alterations or repairs"} | {"(Existing Use) apartments"}  | 0.020834 | 0.831294 | 4.052777 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Use) food/beverage hndlng"} | {"(Proposed Use) food/beverage hndlng"}  | 0.018964 | 0.970914 | 38.218033 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Use) food/beverage hndlng"} | {"(Existing Use) food/beverage hndlng"}  | 0.018964 | 0.857273 | 34.898159 |
| {"(Existing Construction Type Description) constr type 1", "(Proposed Use) apartments"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.018296 | 0.996168 | 7.116796 |
| {"(Proposed Construction Type Description) constr type 1", "(Proposed Use) apartments"} | {"(Existing Construction Type Description) constr type 1"}  | 0.018296 | 0.838286 | 5.939582 |
| {"(Existing Construction Type Description) constr type 1", "(Existing Use) apartments"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.018225 | 0.963583 | 6.884008 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Use) apartments"} | {"(Existing Construction Type Description) constr type 1"}  | 0.018225 | 0.997798 | 7.069785 |
| {"(Existing Construction Type Description) constr type 1", "(Existing Use) apartments"} | {"(Proposed Use) apartments"}  | 0.018160 | 0.960128 | 4.437868 |
| {"(Existing Construction Type Description) constr type 1", "(Proposed Use) apartments"} | {"(Existing Use) apartments"}  | 0.018160 | 0.988776 | 4.820545 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Use) apartments"} | {"(Proposed Use) apartments"}  | 0.018150 | 0.993669 | 4.592903 |
| {"(Proposed Construction Type Description) constr type 1", "(Proposed Use) apartments"} | {"(Existing Use) apartments"}  | 0.018150 | 0.831606 | 4.054297 |
| {"(Existing Construction Type Description) constr type 1", "(Street Name) Market"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.016853 | 0.952543 | 6.805137 |
| {"(Proposed Construction Type Description) constr type 1", "(Street Name) Market"} | {"(Existing Construction Type Description) constr type 1"}  | 0.016853 | 0.978972 | 6.936396 |
| {"(Current Status) filed", "(Existing Use) apartments"} | {"(Proposed Use) apartments"}  | 0.016817 | 0.991699 | 4.583795 |
| {"(Current Status) filed", "(Proposed Use) apartments"} | {"(Existing Use) apartments"}  | 0.016817 | 0.862558 | 4.205198 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 2"} | {"(Proposed Construction Type Description) constr type 2"}  | 0.016747 | 0.997604 | 52.521029 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 2"} | {"(Existing Construction Type Description) constr type 2"}  | 0.016747 | 0.984629 | 48.142502 |
| {"(Existing Use) 1 family dwelling", "(Permit Type Definition) additions alterations or repairs"} | {"(Proposed Use) 1 family dwelling"}  | 0.016425 | 0.806667 | 3.461934 |
| {"(Proposed Use) 1 family dwelling", "(Permit Type Definition) additions alterations or repairs"} | {"(Existing Use) 1 family dwelling"}  | 0.016425 | 0.940685 | 4.000839 |
| {"(Permit Type Definition) otc alterations permit", "(Street Name) Market"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.015917 | 0.668638 | 4.776866 |
| {"(Permit Type Definition) otc alterations permit", "(Street Name) Market"} | {"(Existing Construction Type Description) constr type 1"}  | 0.015691 | 0.659134 | 4.670221 |
| {"(Existing Use) office", "(Street Name) Market"} | {"(Proposed Use) office"}  | 0.014012 | 0.958720 | 7.958035 |
| {"(Proposed Use) office", "(Street Name) Market"} | {"(Existing Use) office"}  | 0.014012 | 0.995713 | 8.045510 |
| {"(Current Status) issued", "(Existing Construction Type Description) constr type 3"} | {"(Proposed Construction Type Description) constr type 3"}  | 0.013771 | 0.915135 | 19.446723 |
| {"(Current Status) issued", "(Proposed Construction Type Description) constr type 3"} | {"(Existing Construction Type Description) constr type 3"}  | 0.013771 | 0.968187 | 19.928934 |
| {"(Current Status) complete", "(Existing Use) retail sales"} | {"(Proposed Use) retail sales"}  | 0.013479 | 0.725968 | 28.429961 |
| {"(Current Status) complete", "(Proposed Use) retail sales"} | {"(Existing Use) retail sales"}  | 0.013479 | 0.946681 | 27.249748 |
| {"(Existing Construction Type Description) constr type 1", "(Street Name) California"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.013338 | 0.983321 | 7.025018 |
| {"(Proposed Construction Type Description) constr type 1", "(Street Name) California"} | {"(Existing Construction Type Description) constr type 1"}  | 0.013338 | 0.999623 | 7.082718 |
| {"(Existing Construction Type Description) constr type 3", "(Existing Use) office"} | {"(Proposed Construction Type Description) constr type 3"}  | 0.013067 | 0.960458 | 20.409840 |
| {"(Existing Use) office", "(Proposed Construction Type Description) constr type 3"} | {"(Existing Construction Type Description) constr type 3"}  | 0.013067 | 0.996549 | 20.512740 |
| {"(Proposed Use) office", "(Existing Construction Type Description) constr type 3"} | {"(Proposed Construction Type Description) constr type 3"}  | 0.012921 | 0.996510 | 21.175950 |
| {"(Proposed Use) office", "(Proposed Construction Type Description) constr type 3"} | {"(Existing Construction Type Description) constr type 3"}  | 0.012921 | 0.992661 | 20.432714 |
| {"(Existing Construction Type Description) constr type 1", "(Street Name) Market"} | {"(Existing Use) office"}  | 0.012690 | 0.717249 | 5.795482 |
| {"(Existing Use) office", "(Street Name) Market"} | {"(Existing Construction Type Description) constr type 1"}  | 0.012690 | 0.868249 | 6.151881 |
| {"(Street Name) California", "(Existing Use) office"} | {"(Proposed Use) office"}  | 0.012589 | 0.987771 | 8.199177 |
| {"(Proposed Use) office", "(Street Name) California"} | {"(Existing Use) office"}  | 0.012589 | 0.998803 | 8.070482 |
| {"(Existing Use) office", "(Street Name) Market"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.012574 | 0.860337 | 6.146400 |
| {"(Proposed Construction Type Description) constr type 1", "(Street Name) Market"} | {"(Existing Use) office"}  | 0.012574 | 0.730432 | 5.902003 |
| {"(Proposed Construction Type Description) constr type 1", "(Street Name) Market"} | {"(Proposed Use) office"}  | 0.012343 | 0.716998 | 5.951571 |
| {"(Proposed Use) office", "(Street Name) Market"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.012343 | 0.877099 | 6.266149 |
| {"(Existing Construction Type Description) constr type 1", "(Street Name) Market"} | {"(Proposed Use) office"}  | 0.012333 | 0.697073 | 5.786183 |
| {"(Proposed Use) office", "(Street Name) Market"} | {"(Existing Construction Type Description) constr type 1"}  | 0.012333 | 0.876384 | 6.209523 |
| {"(Existing Construction Type Description) constr type 3", "(Existing Use) office"} | {"(Proposed Use) office"}  | 0.012298 | 0.903917 | 7.503132 |
| {"(Proposed Use) office", "(Existing Construction Type Description) constr type 3"} | {"(Existing Use) office"}  | 0.012298 | 0.948430 | 7.663455 |
| {"(Existing Use) office", "(Proposed Construction Type Description) constr type 3"} | {"(Proposed Use) office"}  | 0.012288 | 0.937117 | 7.778709 |
| {"(Proposed Use) office", "(Proposed Construction Type Description) constr type 3"} | {"(Existing Use) office"}  | 0.012288 | 0.943994 | 7.627613 |
| {"(Current Status) complete", "(Existing Use) food/beverage hndlng"} | {"(Proposed Use) food/beverage hndlng"}  | 0.012056 | 0.899475 | 35.405987 |
| {"(Current Status) complete", "(Proposed Use) food/beverage hndlng"} | {"(Existing Use) food/beverage hndlng"}  | 0.012056 | 0.827182 | 33.673206 |
| {"(Existing Construction Type Description) constr type 1", "(Street Name) California"} | {"(Existing Use) office"}  | 0.012031 | 0.886953 | 7.166717 |
| {"(Street Name) California", "(Existing Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.012031 | 0.943984 | 6.688494 |
| {"(Street Name) California", "(Existing Use) office"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.011961 | 0.938462 | 6.704534 |
| {"(Proposed Construction Type Description) constr type 1", "(Street Name) California"} | {"(Existing Use) office"}  | 0.011961 | 0.896383 | 7.242909 |
| {"(Existing Construction Type Description) constr type 1", "(Street Name) California"} | {"(Proposed Use) office"}  | 0.011931 | 0.879540 | 7.300787 |
| {"(Proposed Use) office", "(Street Name) California"} | {"(Existing Construction Type Description) constr type 1"}  | 0.011931 | 0.946550 | 6.706671 |
| {"(Proposed Construction Type Description) constr type 1", "(Street Name) California"} | {"(Proposed Use) office"}  | 0.011926 | 0.893745 | 7.418698 |
| {"(Proposed Use) office", "(Street Name) California"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.011926 | 0.946151 | 6.759468 |
| {"(Existing Construction Type Description) wood frame (5)", "(Existing Use) food/beverage hndlng"} | {"(Proposed Use) food/beverage hndlng"}  | 0.011473 | 0.871325 | 34.297922 |
| {"(Existing Construction Type Description) wood frame (5)", "(Proposed Use) food/beverage hndlng"} | {"(Existing Use) food/beverage hndlng"}  | 0.011473 | 0.846125 | 34.444366 |
| {"(Proposed Construction Type Description) wood frame (5)", "(Existing Use) food/beverage hndlng"} | {"(Proposed Use) food/beverage hndlng"}  | 0.011468 | 0.969401 | 38.158476 |
| {"(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) food/beverage hndlng"} | {"(Existing Use) food/beverage hndlng"}  | 0.011468 | 0.838911 | 34.150698 |
| {"(Current Status) complete", "(Street Name) Market"} | {"(Existing Construction Type Description) constr type 1"}  | 0.011443 | 0.732775 | 5.191997 |
| {"(Current Status) complete", "(Street Name) Market"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.011337 | 0.726014 | 5.186773 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 2"} | {"(Proposed Construction Type Description) constr type 2"}  | 0.011121 | 0.938083 | 49.387420 |
| {"(Current Status) complete", "(Proposed Construction Type Description) constr type 2"} | {"(Existing Construction Type Description) constr type 2"}  | 0.011121 | 0.982238 | 48.025596 |
| {"(Current Status) filed", "(Existing Use) 1 family dwelling"} | {"(Proposed Use) 1 family dwelling"}  | 0.010211 | 0.832377 | 3.572274 |
| {"(Proposed Use) 1 family dwelling", "(Current Status) filed"} | {"(Existing Use) 1 family dwelling"}  | 0.010211 | 0.926974 | 3.942523 |
| {"(Existing Construction Type Description) constr type 3", "(Proposed Use) apartments"} | {"(Proposed Construction Type Description) constr type 3"}  | 0.010101 | 0.995540 | 21.155334 |
| {"(Proposed Use) apartments", "(Proposed Construction Type Description) constr type 3"} | {"(Existing Construction Type Description) constr type 3"}  | 0.010101 | 0.914013 | 18.813831 |
| {"(Proposed Construction Type Description) constr type 3", "(Existing Use) apartments"} | {"(Proposed Use) apartments"}  | 0.010015 | 0.998997 | 4.617529 |
| {"(Proposed Use) apartments", "(Proposed Construction Type Description) constr type 3"} | {"(Existing Use) apartments"}  | 0.010015 | 0.906278 | 4.418346 |
| {"(Existing Construction Type Description) constr type 3", "(Existing Use) apartments"} | {"(Proposed Use) apartments"}  | 0.010010 | 0.992522 | 4.587602 |
| {"(Existing Construction Type Description) constr type 3", "(Proposed Use) apartments"} | {"(Existing Use) apartments"}  | 0.010010 | 0.986620 | 4.810034 |
<br>

---------

**<center>3-Itemset Relation Rules</center>**

| LHS | RHS | Support | Confidence | Lift |
| :-----: | :-----: | :-----: | :-----: | :-----: |
| {"(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) 1 family dwelling"} | {"(Proposed Use) 1 family dwelling"}  | 0.226072 | 0.973690 | 4.178743 |
| {"(Proposed Use) 1 family dwelling", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)"} | {"(Existing Use) 1 family dwelling"}  | 0.226072 | 0.995682 | 4.234747 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) wood frame (5)", "(Existing Use) 1 family dwelling"} | {"(Proposed Use) 1 family dwelling"}  | 0.209934 | 0.989361 | 4.245997 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) wood frame (5)", "(Proposed Use) 1 family dwelling"} | {"(Existing Use) 1 family dwelling"}  | 0.209934 | 0.997778 | 4.243660 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)"} | {"(Proposed Use) 1 family dwelling"}  | 0.209863 | 0.989475 | 4.246485 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)"} | {"(Existing Use) 1 family dwelling"}  | 0.209863 | 0.991049 | 4.215042 |
| {"(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | {"(Proposed Use) apartments"}  | 0.170607 | 0.996652 | 4.606689 |
| {"(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments"} | {"(Existing Use) apartments"}  | 0.170607 | 0.973325 | 4.745215 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | {"(Proposed Use) apartments"}  | 0.151910 | 0.997129 | 4.608894 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments"} | {"(Existing Use) apartments"}  | 0.151910 | 0.975118 | 4.753956 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | {"(Proposed Use) apartments"}  | 0.151864 | 0.997095 | 4.608738 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) wood frame (5)", "(Proposed Use) apartments"} | {"(Existing Use) apartments"}  | 0.151864 | 0.985642 | 4.805267 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Existing Use) 1 family dwelling"} | {"(Proposed Use) 1 family dwelling"}  | 0.141784 | 0.985016 | 4.227347 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Use) 1 family dwelling"} | {"(Existing Use) 1 family dwelling"}  | 0.141784 | 0.997171 | 4.241080 |
| {"(Current Status) complete", "(Existing Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)"} | {"(Proposed Use) 1 family dwelling"}  | 0.141784 | 0.986532 | 4.233854 |
| {"(Current Status) complete", "(Proposed Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)"} | {"(Existing Use) 1 family dwelling"}  | 0.141784 | 0.988919 | 4.205982 |
| {"(Current Status) complete", "(Existing Use) 1 family dwelling", "(Permit Type Definition) otc alterations permit"} | {"(Proposed Use) 1 family dwelling"}  | 0.137797 | 0.991822 | 4.256556 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Use) 1 family dwelling"} | {"(Existing Use) 1 family dwelling"}  | 0.137797 | 0.992828 | 4.222606 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Existing Use) apartments"} | {"(Proposed Use) apartments"}  | 0.115761 | 0.997271 | 4.609552 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Use) apartments"} | {"(Existing Use) apartments"}  | 0.115761 | 0.968984 | 4.724053 |
| {"(Current Status) complete", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | {"(Proposed Use) apartments"}  | 0.101085 | 0.997272 | 4.609555 |
| {"(Current Status) complete", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments"} | {"(Existing Use) apartments"}  | 0.101085 | 0.968824 | 4.723271 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | {"(Proposed Use) apartments"}  | 0.101015 | 0.996083 | 4.604062 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Use) apartments"} | {"(Existing Use) apartments"}  | 0.101015 | 0.984564 | 4.800010 |
| {"(Existing Use) 2 family dwelling", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)"} | {"(Proposed Use) 2 family dwelling"}  | 0.100266 | 0.960414 | 8.659052 |
| {"(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) 2 family dwelling"} | {"(Existing Use) 2 family dwelling"}  | 0.100266 | 0.945793 | 8.963609 |
| {"(Existing Use) 2 family dwelling", "(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) wood frame (5)"} | {"(Proposed Use) 2 family dwelling"}  | 0.093665 | 0.979186 | 8.828301 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) 2 family dwelling"} | {"(Existing Use) 2 family dwelling"}  | 0.093665 | 0.957447 | 9.074052 |
| {"(Existing Use) 2 family dwelling", "(Existing Construction Type Description) wood frame (5)", "(Permit Type Definition) otc alterations permit"} | {"(Proposed Use) 2 family dwelling"}  | 0.093594 | 0.979120 | 8.827699 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) wood frame (5)", "(Proposed Use) 2 family dwelling"} | {"(Existing Use) 2 family dwelling"}  | 0.093594 | 0.979171 | 9.279940 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1", "(Existing Use) office"} | {"(Proposed Use) office"}  | 0.090211 | 0.988377 | 8.204207 |
| {"(Proposed Use) office", "(Existing Construction Type Description) constr type 1", "(Existing Use) office"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.090211 | 0.999443 | 7.140196 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1", "(Proposed Use) office"} | {"(Existing Use) office"}  | 0.090211 | 0.996999 | 8.055907 |
| {"(Proposed Construction Type Description) constr type 1", "(Proposed Use) office", "(Existing Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.090211 | 0.999721 | 7.083414 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Existing Use) office"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.087833 | 0.999428 | 7.140089 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Proposed Construction Type Description) constr type 1"} | {"(Existing Use) office"}  | 0.087833 | 0.692128 | 5.592498 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Existing Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.087833 | 0.999542 | 7.082144 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Proposed Construction Type Description) constr type 1"} | {"(Proposed Use) office"}  | 0.087189 | 0.687057 | 5.703041 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Proposed Use) office"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.087189 | 0.999481 | 7.140470 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Proposed Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.087189 | 0.993526 | 7.039518 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Existing Use) office"} | {"(Proposed Use) office"}  | 0.087028 | 0.990275 | 8.219957 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Proposed Use) office"} | {"(Existing Use) office"}  | 0.087028 | 0.997637 | 8.061058 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Use) office", "(Existing Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.087028 | 0.777978 | 5.512272 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Existing Use) office"} | {"(Proposed Use) office"}  | 0.087008 | 0.990159 | 8.218998 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Use) office", "(Existing Use) office"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.087008 | 0.777798 | 5.556724 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Proposed Use) office"} | {"(Existing Use) office"}  | 0.087008 | 0.991464 | 8.011177 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.085062 | 0.998996 | 7.137005 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1"} | {"(Existing Construction Type Description) constr type 1"}  | 0.085062 | 0.984063 | 6.972470 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Existing Use) office"} | {"(Proposed Use) office"}  | 0.075339 | 0.979668 | 8.131914 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Use) office"} | {"(Existing Use) office"}  | 0.075339 | 0.985466 | 7.962716 |
| {"(Current Status) complete", "(Existing Use) 2 family dwelling", "(Proposed Construction Type Description) wood frame (5)"} | {"(Proposed Use) 2 family dwelling"}  | 0.065580 | 0.978912 | 8.825826 |
| {"(Current Status) complete", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) 2 family dwelling"} | {"(Existing Use) 2 family dwelling"}  | 0.065580 | 0.950383 | 9.007101 |
| {"(Current Status) issued", "(Existing Use) 1 family dwelling", "(Existing Construction Type Description) wood frame (5)"} | {"(Proposed Use) 1 family dwelling"}  | 0.065570 | 0.971544 | 4.169529 |
| {"(Proposed Use) 1 family dwelling", "(Current Status) issued", "(Existing Construction Type Description) wood frame (5)"} | {"(Existing Use) 1 family dwelling"}  | 0.065570 | 0.996105 | 4.236544 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Existing Use) 2 family dwelling"} | {"(Proposed Use) 2 family dwelling"}  | 0.065535 | 0.978383 | 8.821060 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Use) 2 family dwelling"} | {"(Existing Use) 2 family dwelling"}  | 0.065535 | 0.973052 | 9.221944 |
| {"(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) 1 family dwelling"} | {"(Proposed Use) 1 family dwelling"}  | 0.065510 | 0.974497 | 4.182204 |
| {"(Proposed Use) 1 family dwelling", "(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)"} | {"(Existing Use) 1 family dwelling"}  | 0.065510 | 0.969494 | 4.123366 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Existing Use) 2 family dwelling"} | {"(Proposed Use) 2 family dwelling"}  | 0.063620 | 0.983675 | 8.868774 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Use) 2 family dwelling"} | {"(Existing Use) 2 family dwelling"}  | 0.063620 | 0.970027 | 9.193277 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Existing Use) office"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.062182 | 0.993973 | 7.101114 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Proposed Construction Type Description) constr type 1"} | {"(Existing Use) office"}  | 0.062182 | 0.700260 | 5.658210 |
| {"(Current Status) complete", "(Proposed Construction Type Description) constr type 1", "(Existing Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.062182 | 0.999677 | 7.083097 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Proposed Construction Type Description) constr type 1"} | {"(Proposed Use) office"}  | 0.061754 | 0.695448 | 5.772693 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Proposed Use) office"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.061754 | 0.999512 | 7.140688 |
| {"(Current Status) complete", "(Proposed Construction Type Description) constr type 1", "(Proposed Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.061754 | 0.994333 | 7.045237 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Existing Use) office"} | {"(Proposed Use) office"}  | 0.061609 | 0.984811 | 8.174603 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Proposed Use) office"} | {"(Existing Use) office"}  | 0.061609 | 0.997152 | 8.057138 |
| {"(Current Status) complete", "(Proposed Use) office", "(Existing Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.061609 | 0.787482 | 5.579612 |
| {"(Current Status) complete", "(Proposed Construction Type Description) constr type 1", "(Existing Use) office"} | {"(Proposed Use) office"}  | 0.061593 | 0.990220 | 8.219502 |
| {"(Current Status) complete", "(Proposed Use) office", "(Existing Use) office"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.061593 | 0.787289 | 5.624529 |
| {"(Current Status) complete", "(Proposed Construction Type Description) constr type 1", "(Proposed Use) office"} | {"(Existing Use) office"}  | 0.061593 | 0.991743 | 8.013432 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Use) 1 family dwelling"} | {"(Proposed Use) 1 family dwelling"}  | 0.061257 | 0.986239 | 4.232598 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Use) 1 family dwelling"} | {"(Existing Use) 1 family dwelling"}  | 0.061257 | 0.986798 | 4.196963 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit"} | {"(Existing Use) office"}  | 0.060422 | 0.709613 | 5.733778 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Existing Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.060422 | 0.785696 | 5.566958 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Existing Use) office"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.060407 | 0.785499 | 5.611746 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1"} | {"(Existing Use) office"}  | 0.060407 | 0.698831 | 5.646659 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1"} | {"(Proposed Use) office"}  | 0.060291 | 0.697493 | 5.789671 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Use) office"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.060291 | 0.788636 | 5.634155 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit"} | {"(Proposed Use) office"}  | 0.060055 | 0.705302 | 5.854492 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.060055 | 0.785545 | 5.565892 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Use) apartments"} | {"(Proposed Use) apartments"}  | 0.050437 | 0.996226 | 4.604723 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Use) apartments"} | {"(Existing Use) apartments"}  | 0.050437 | 0.959266 | 4.676673 |
| {"(Current Status) issued", "(Existing Use) apartments", "(Existing Construction Type Description) wood frame (5)"} | {"(Proposed Use) apartments"}  | 0.047682 | 0.992154 | 4.585899 |
| {"(Current Status) issued", "(Proposed Use) apartments", "(Existing Construction Type Description) wood frame (5)"} | {"(Existing Use) apartments"}  | 0.047682 | 0.969635 | 4.727226 |
| {"(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | {"(Proposed Use) apartments"}  | 0.047652 | 0.996007 | 4.603707 |
| {"(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments"} | {"(Existing Use) apartments"}  | 0.047652 | 0.953425 | 4.648199 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Construction Type Description) constr type 1"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.035364 | 0.999006 | 7.137073 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Construction Type Description) constr type 1"} | {"(Existing Construction Type Description) constr type 1"}  | 0.035364 | 0.967937 | 6.858211 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Use) office"} | {"(Proposed Use) office"}  | 0.030875 | 0.977088 | 8.110502 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Use) office"} | {"(Existing Use) office"}  | 0.030875 | 0.979426 | 7.913909 |
| {"(Existing Use) 2 family dwelling", "(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)"} | {"(Proposed Use) 2 family dwelling"}  | 0.026646 | 0.954783 | 8.608281 |
| {"(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) 2 family dwelling"} | {"(Existing Use) 2 family dwelling"}  | 0.026646 | 0.894666 | 8.479054 |
| {"(Existing Use) 2 family dwelling", "(Current Status) issued", "(Existing Construction Type Description) wood frame (5)"} | {"(Proposed Use) 2 family dwelling"}  | 0.026626 | 0.953375 | 8.595590 |
| {"(Current Status) issued", "(Proposed Use) 2 family dwelling", "(Existing Construction Type Description) wood frame (5)"} | {"(Existing Use) 2 family dwelling"}  | 0.026626 | 0.942684 | 8.934142 |
| {"(Existing Use) 2 family dwelling", "(Current Status) issued", "(Permit Type Definition) otc alterations permit"} | {"(Proposed Use) 2 family dwelling"}  | 0.025043 | 0.970199 | 8.747268 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Use) 2 family dwelling"} | {"(Existing Use) 2 family dwelling"}  | 0.025043 | 0.926009 | 8.776101 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 3"} | {"(Proposed Construction Type Description) constr type 3"}  | 0.024967 | 0.997790 | 21.203141 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 3"} | {"(Existing Construction Type Description) constr type 3"}  | 0.024967 | 0.978137 | 20.133744 |
| {"(Existing Use) office", "(Current Status) issued", "(Existing Construction Type Description) constr type 1"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.024671 | 0.972453 | 6.947378 |
| {"(Proposed Construction Type Description) constr type 1", "(Current Status) issued", "(Existing Construction Type Description) constr type 1"} | {"(Existing Use) office"}  | 0.024671 | 0.656806 | 5.307094 |
| {"(Proposed Construction Type Description) constr type 1", "(Current Status) issued", "(Existing Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.024671 | 0.998575 | 7.075294 |
| {"(Proposed Construction Type Description) constr type 1", "(Current Status) issued", "(Existing Construction Type Description) constr type 1"} | {"(Proposed Use) office"}  | 0.024379 | 0.649043 | 5.387501 |
| {"(Proposed Use) office", "(Current Status) issued", "(Existing Construction Type Description) constr type 1"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.024379 | 0.999382 | 7.139759 |
| {"(Proposed Construction Type Description) constr type 1", "(Current Status) issued", "(Proposed Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.024379 | 0.983171 | 6.966148 |
| {"(Existing Use) office", "(Current Status) issued", "(Existing Construction Type Description) constr type 1"} | {"(Proposed Use) office"}  | 0.024349 | 0.959770 | 7.966749 |
| {"(Proposed Use) office", "(Current Status) issued", "(Existing Construction Type Description) constr type 1"} | {"(Existing Use) office"}  | 0.024349 | 0.998145 | 8.065163 |
| {"(Proposed Use) office", "(Current Status) issued", "(Existing Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.024349 | 0.754126 | 5.343278 |
| {"(Proposed Construction Type Description) constr type 1", "(Current Status) issued", "(Existing Use) office"} | {"(Proposed Use) office"}  | 0.024344 | 0.985348 | 8.179063 |
| {"(Proposed Use) office", "(Current Status) issued", "(Existing Use) office"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.024344 | 0.753971 | 5.386499 |
| {"(Proposed Construction Type Description) constr type 1", "(Current Status) issued", "(Proposed Use) office"} | {"(Existing Use) office"}  | 0.024344 | 0.981752 | 7.932703 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Construction Type Description) constr type 1"} | {"(Proposed Use) office"}  | 0.023811 | 0.651713 | 5.409666 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Use) office"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.023811 | 0.755343 | 5.396303 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Use) office"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.023766 | 0.752108 | 5.373193 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Construction Type Description) constr type 1"} | {"(Existing Use) office"}  | 0.023766 | 0.650475 | 5.255934 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Construction Type Description) constr type 1"} | {"(Existing Use) office"}  | 0.023761 | 0.671211 | 5.423490 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.023761 | 0.751949 | 5.327851 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Construction Type Description) constr type 1"} | {"(Proposed Use) office"}  | 0.023529 | 0.664678 | 5.517285 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.023529 | 0.746411 | 5.288615 |
| {"(Existing Construction Type Description) wood frame (5)", "(Existing Use) apartments", "(Permit Type Definition) additions alterations or repairs"} | {"(Proposed Use) apartments"}  | 0.018904 | 0.992870 | 4.589210 |
| {"(Existing Construction Type Description) wood frame (5)", "(Proposed Use) apartments", "(Permit Type Definition) additions alterations or repairs"} | {"(Existing Use) apartments"}  | 0.018904 | 0.879121 | 4.285946 |
| {"(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments", "(Permit Type Definition) additions alterations or repairs"} | {"(Proposed Use) apartments"}  | 0.018869 | 0.992857 | 4.589149 |
| {"(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments", "(Permit Type Definition) additions alterations or repairs"} | {"(Existing Use) apartments"}  | 0.018869 | 0.861373 | 4.199418 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1", "(Existing Use) apartments"} | {"(Proposed Use) apartments"}  | 0.018115 | 0.993931 | 4.594113 |
| {"(Existing Construction Type Description) constr type 1", "(Proposed Use) apartments", "(Existing Use) apartments"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.018115 | 0.997508 | 7.126375 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1", "(Proposed Use) apartments"} | {"(Existing Use) apartments"}  | 0.018115 | 0.990107 | 4.827033 |
| {"(Proposed Construction Type Description) constr type 1", "(Proposed Use) apartments", "(Existing Use) apartments"} | {"(Existing Construction Type Description) constr type 1"}  | 0.018115 | 0.998061 | 7.071649 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Proposed Use) apartments"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.016978 | 0.996459 | 7.118879 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Proposed Use) apartments"} | {"(Existing Construction Type Description) constr type 1"}  | 0.016978 | 0.909017 | 6.440741 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Existing Use) apartments"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.016953 | 0.997633 | 7.127267 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Existing Use) apartments"} | {"(Existing Construction Type Description) constr type 1"}  | 0.016953 | 0.999407 | 7.081188 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Existing Use) apartments"} | {"(Proposed Use) apartments"}  | 0.016913 | 0.995266 | 4.600285 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Proposed Use) apartments"} | {"(Existing Use) apartments"}  | 0.016913 | 0.992623 | 4.839300 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Existing Use) apartments"} | {"(Proposed Use) apartments"}  | 0.016883 | 0.995258 | 4.600246 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Proposed Use) apartments"} | {"(Existing Use) apartments"}  | 0.016883 | 0.903903 | 4.406766 |
| {"(Existing Construction Type Description) wood frame (5)", "(Permit Type Definition) additions alterations or repairs", "(Existing Use) 1 family dwelling"} | {"(Proposed Use) 1 family dwelling"}  | 0.016355 | 0.807597 | 3.465926 |
| {"(Proposed Use) 1 family dwelling", "(Existing Construction Type Description) wood frame (5)", "(Permit Type Definition) additions alterations or repairs"} | {"(Existing Use) 1 family dwelling"}  | 0.016355 | 0.969598 | 4.123807 |
| {"(Existing Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)", "(Permit Type Definition) additions alterations or repairs"} | {"(Proposed Use) 1 family dwelling"}  | 0.016335 | 0.807606 | 3.465967 |
| {"(Proposed Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)", "(Permit Type Definition) additions alterations or repairs"} | {"(Existing Use) 1 family dwelling"}  | 0.016335 | 0.942285 | 4.007645 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Street Name) Market"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.015681 | 0.999359 | 7.139598 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Street Name) Market"} | {"(Existing Construction Type Description) constr type 1"}  | 0.015681 | 0.985155 | 6.980203 |
| {"(Current Status) filed", "(Existing Use) apartments", "(Existing Construction Type Description) wood frame (5)"} | {"(Proposed Use) apartments"}  | 0.015395 | 0.994156 | 4.595152 |
| {"(Current Status) filed", "(Proposed Use) apartments", "(Existing Construction Type Description) wood frame (5)"} | {"(Existing Use) apartments"}  | 0.015395 | 0.920349 | 4.486942 |
| {"(Current Status) filed", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | {"(Proposed Use) apartments"}  | 0.015374 | 0.995767 | 4.602599 |
| {"(Current Status) filed", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments"} | {"(Existing Use) apartments"}  | 0.015374 | 0.892326 | 4.350323 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Use) office", "(Street Name) Market"} | {"(Proposed Use) office"}  | 0.013313 | 0.979290 | 8.128777 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Use) office", "(Street Name) Market"} | {"(Existing Use) office"}  | 0.013313 | 0.996613 | 8.052781 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Street Name) California"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.013062 | 0.999615 | 7.141427 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Street Name) California"} | {"(Existing Construction Type Description) constr type 1"}  | 0.013062 | 1.000000 | 7.085388 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Construction Type Description) constr type 3"} | {"(Proposed Construction Type Description) constr type 3"}  | 0.012675 | 0.996443 | 21.174514 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Construction Type Description) constr type 3"} | {"(Existing Construction Type Description) constr type 3"}  | 0.012675 | 0.986693 | 20.309860 |
| {"(Existing Construction Type Description) constr type 1", "(Existing Use) office", "(Street Name) Market"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.012559 | 0.989699 | 7.070583 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1", "(Street Name) Market"} | {"(Existing Use) office"}  | 0.012559 | 0.745227 | 6.021545 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Use) office", "(Street Name) Market"} | {"(Existing Construction Type Description) constr type 1"}  | 0.012559 | 0.998800 | 7.076889 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Existing Use) retail sales"} | {"(Proposed Use) retail sales"}  | 0.012403 | 0.834292 | 32.672065 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Use) retail sales"} | {"(Existing Use) retail sales"}  | 0.012403 | 0.951041 | 27.375250 |
| {"(Permit Type Definition) otc alterations permit", "(Street Name) California", "(Existing Use) office"} | {"(Proposed Use) office"}  | 0.012358 | 0.995142 | 8.260357 |
| {"(Permit Type Definition) otc alterations permit", "(Street Name) California", "(Proposed Use) office"} | {"(Existing Use) office"}  | 0.012358 | 0.998781 | 8.070301 |
| {"(Existing Construction Type Description) constr type 1", "(Existing Use) office", "(Street Name) Market"} | {"(Proposed Use) office"}  | 0.012333 | 0.971870 | 8.067187 |
| {"(Proposed Use) office", "(Existing Construction Type Description) constr type 1", "(Street Name) Market"} | {"(Existing Use) office"}  | 0.012333 | 1.000000 | 8.080151 |
| {"(Proposed Use) office", "(Existing Use) office", "(Street Name) Market"} | {"(Existing Construction Type Description) constr type 1"}  | 0.012333 | 0.880158 | 6.236260 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Use) office", "(Street Name) Market"} | {"(Proposed Use) office"}  | 0.012333 | 0.980808 | 8.141375 |
| {"(Proposed Use) office", "(Existing Use) office", "(Street Name) Market"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.012333 | 0.880158 | 6.288003 |
| {"(Proposed Construction Type Description) constr type 1", "(Proposed Use) office", "(Street Name) Market"} | {"(Existing Use) office"}  | 0.012333 | 0.999185 | 8.073569 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1", "(Street Name) Market"} | {"(Proposed Use) office"}  | 0.012328 | 0.731504 | 6.071980 |
| {"(Proposed Use) office", "(Existing Construction Type Description) constr type 1", "(Street Name) Market"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.012328 | 0.999592 | 7.141263 |
| {"(Proposed Construction Type Description) constr type 1", "(Proposed Use) office", "(Street Name) Market"} | {"(Existing Construction Type Description) constr type 1"}  | 0.012328 | 0.998778 | 7.076729 |
| {"(Existing Construction Type Description) constr type 3", "(Existing Use) office", "(Proposed Construction Type Description) constr type 3"} | {"(Proposed Use) office"}  | 0.012257 | 0.938053 | 7.786483 |
| {"(Proposed Use) office", "(Existing Construction Type Description) constr type 3", "(Existing Use) office"} | {"(Proposed Construction Type Description) constr type 3"}  | 0.012257 | 0.996729 | 21.180605 |
| {"(Proposed Use) office", "(Existing Construction Type Description) constr type 3", "(Proposed Construction Type Description) constr type 3"} | {"(Existing Use) office"}  | 0.012257 | 0.948638 | 7.665139 |
| {"(Proposed Use) office", "(Existing Use) office", "(Proposed Construction Type Description) constr type 3"} | {"(Existing Construction Type Description) constr type 3"}  | 0.012257 | 0.997545 | 20.533240 |
| {"(Existing Construction Type Description) constr type 1", "(Street Name) California", "(Existing Use) office"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.011961 | 0.994150 | 7.102380 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1", "(Street Name) California"} | {"(Existing Use) office"}  | 0.011961 | 0.896721 | 7.245639 |
| {"(Proposed Construction Type Description) constr type 1", "(Street Name) California", "(Existing Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.011961 | 1.000000 | 7.085388 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Use) office", "(Street Name) Market"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.011946 | 0.878698 | 6.277575 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Street Name) Market"} | {"(Existing Use) office"}  | 0.011946 | 0.750474 | 6.063942 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Street Name) Market"} | {"(Existing Use) office"}  | 0.011936 | 0.760654 | 6.146196 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Use) office", "(Street Name) Market"} | {"(Existing Construction Type Description) constr type 1"}  | 0.011936 | 0.877959 | 6.220677 |
| {"(Existing Construction Type Description) constr type 1", "(Street Name) California", "(Existing Use) office"} | {"(Proposed Use) office"}  | 0.011926 | 0.991224 | 8.227841 |
| {"(Proposed Use) office", "(Existing Construction Type Description) constr type 1", "(Street Name) California"} | {"(Existing Use) office"}  | 0.011926 | 0.999579 | 8.076746 |
| {"(Proposed Use) office", "(Street Name) California", "(Existing Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.011926 | 0.947284 | 6.711877 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1", "(Street Name) California"} | {"(Proposed Use) office"}  | 0.011926 | 0.894082 | 7.421494 |
| {"(Proposed Use) office", "(Existing Construction Type Description) constr type 1", "(Street Name) California"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.011926 | 0.999579 | 7.141165 |
| {"(Proposed Construction Type Description) constr type 1", "(Street Name) California", "(Proposed Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.011926 | 1.000000 | 7.085388 |
| {"(Proposed Construction Type Description) constr type 1", "(Street Name) California", "(Existing Use) office"} | {"(Proposed Use) office"}  | 0.011921 | 0.996637 | 8.272771 |
| {"(Proposed Use) office", "(Street Name) California", "(Existing Use) office"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.011921 | 0.946885 | 6.764713 |
| {"(Proposed Construction Type Description) constr type 1", "(Street Name) California", "(Proposed Use) office"} | {"(Existing Use) office"}  | 0.011921 | 0.999578 | 8.076745 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Proposed Use) apartments"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.011815 | 0.996185 | 7.116920 |
| {"(Current Status) complete", "(Proposed Construction Type Description) constr type 1", "(Proposed Use) apartments"} | {"(Existing Construction Type Description) constr type 1"}  | 0.011815 | 0.894897 | 6.340693 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Existing Use) apartments"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.011810 | 0.982023 | 7.015748 |
| {"(Current Status) complete", "(Proposed Construction Type Description) constr type 1", "(Existing Use) apartments"} | {"(Existing Construction Type Description) constr type 1"}  | 0.011810 | 0.999149 | 7.079360 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Street Name) California"} | {"(Existing Use) office"}  | 0.011790 | 0.902270 | 7.290479 |
| {"(Permit Type Definition) otc alterations permit", "(Street Name) California", "(Existing Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.011790 | 0.949393 | 6.726815 |
| {"(Permit Type Definition) otc alterations permit", "(Street Name) California", "(Existing Use) office"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.011785 | 0.948988 | 6.779736 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Street Name) California"} | {"(Existing Use) office"}  | 0.011785 | 0.902232 | 7.290175 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Street Name) Market"} | {"(Proposed Use) office"}  | 0.011765 | 0.739103 | 6.135060 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Use) office", "(Street Name) Market"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.011765 | 0.880693 | 6.291822 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Street Name) California"} | {"(Proposed Use) office"}  | 0.011760 | 0.899962 | 7.470297 |
| {"(Permit Type Definition) otc alterations permit", "(Street Name) California", "(Proposed Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.011760 | 0.950427 | 6.734141 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Street Name) Market"} | {"(Proposed Use) office"}  | 0.011755 | 0.749119 | 6.218199 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Use) office", "(Street Name) Market"} | {"(Existing Construction Type Description) constr type 1"}  | 0.011755 | 0.879940 | 6.234714 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Street Name) California"} | {"(Proposed Use) office"}  | 0.011755 | 0.899923 | 7.469977 |
| {"(Permit Type Definition) otc alterations permit", "(Street Name) California", "(Proposed Use) office"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.011755 | 0.950020 | 6.787112 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Existing Use) apartments"} | {"(Proposed Use) apartments"}  | 0.011740 | 0.976171 | 4.512021 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Proposed Use) apartments"} | {"(Existing Use) apartments"}  | 0.011740 | 0.989826 | 4.825664 |
| {"(Current Status) complete", "(Proposed Construction Type Description) constr type 1", "(Existing Use) apartments"} | {"(Proposed Use) apartments"}  | 0.011729 | 0.992344 | 4.586776 |
| {"(Current Status) complete", "(Proposed Construction Type Description) constr type 1", "(Proposed Use) apartments"} | {"(Existing Use) apartments"}  | 0.011729 | 0.888423 | 4.331298 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 3", "(Existing Use) office"} | {"(Proposed Construction Type Description) constr type 3"}  | 0.011659 | 0.996134 | 21.167954 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Use) office", "(Proposed Construction Type Description) constr type 3"} | {"(Existing Construction Type Description) constr type 3"}  | 0.011659 | 0.997849 | 20.539488 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) filed", "(Existing Use) apartments"} | {"(Proposed Use) apartments"}  | 0.011533 | 0.997825 | 4.612112 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) filed", "(Proposed Use) apartments"} | {"(Existing Use) apartments"}  | 0.011533 | 0.977835 | 4.767201 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 3", "(Proposed Use) office"} | {"(Proposed Construction Type Description) constr type 3"}  | 0.011498 | 0.996080 | 21.166809 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Use) office", "(Proposed Construction Type Description) constr type 3"} | {"(Existing Construction Type Description) constr type 3"}  | 0.011498 | 0.996080 | 20.503088 |
| {"(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) food/beverage hndlng"} | {"(Proposed Use) food/beverage hndlng"}  | 0.011458 | 0.969375 | 38.157452 |
| {"(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) food/beverage hndlng"} | {"(Existing Use) food/beverage hndlng"}  | 0.011458 | 0.845954 | 34.437391 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Existing Use) food/beverage hndlng"} | {"(Proposed Use) food/beverage hndlng"}  | 0.011307 | 0.966896 | 38.059880 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Use) food/beverage hndlng"} | {"(Existing Use) food/beverage hndlng"}  | 0.011307 | 0.853834 | 34.758193 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Street Name) Market"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.011151 | 0.974517 | 6.962119 |
| {"(Current Status) complete", "(Proposed Construction Type Description) constr type 1", "(Street Name) Market"} | {"(Existing Construction Type Description) constr type 1"}  | 0.011151 | 0.983592 | 6.969131 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 3", "(Existing Use) office"} | {"(Proposed Use) office"}  | 0.011091 | 0.947595 | 7.865683 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 3", "(Proposed Use) office"} | {"(Existing Use) office"}  | 0.011091 | 0.960801 | 7.763420 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Use) office", "(Proposed Construction Type Description) constr type 3"} | {"(Proposed Use) office"}  | 0.011071 | 0.947504 | 7.864934 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Use) office", "(Proposed Construction Type Description) constr type 3"} | {"(Existing Use) office"}  | 0.011071 | 0.959059 | 7.749344 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Street Name) Market"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.010603 | 0.749467 | 5.354324 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Street Name) Market"} | {"(Existing Construction Type Description) constr type 1"}  | 0.010462 | 0.739517 | 5.239762 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) wood frame (5)", "(Existing Use) food/beverage hndlng"} | {"(Proposed Use) food/beverage hndlng"}  | 0.010437 | 0.975106 | 38.383039 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) wood frame (5)", "(Proposed Use) food/beverage hndlng"} | {"(Existing Use) food/beverage hndlng"}  | 0.010437 | 0.901042 | 36.679920 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) food/beverage hndlng"} | {"(Proposed Use) food/beverage hndlng"}  | 0.010432 | 0.975094 | 38.382578 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) food/beverage hndlng"} | {"(Existing Use) food/beverage hndlng"}  | 0.010432 | 0.897491 | 36.535392 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 2"} | {"(Proposed Construction Type Description) constr type 2"}  | 0.010347 | 0.997576 | 52.519569 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 2"} | {"(Existing Construction Type Description) constr type 2"}  | 0.010347 | 0.985632 | 48.191550 |
| {"(Current Status) filed", "(Existing Use) 1 family dwelling", "(Existing Construction Type Description) wood frame (5)"} | {"(Proposed Use) 1 family dwelling"}  | 0.010161 | 0.832715 | 3.573726 |
| {"(Proposed Use) 1 family dwelling", "(Current Status) filed", "(Existing Construction Type Description) wood frame (5)"} | {"(Existing Use) 1 family dwelling"}  | 0.010161 | 0.982499 | 4.178677 |
| {"(Current Status) filed", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) 1 family dwelling"} | {"(Proposed Use) 1 family dwelling"}  | 0.010141 | 0.848906 | 3.643210 |
| {"(Proposed Use) 1 family dwelling", "(Current Status) filed", "(Proposed Construction Type Description) wood frame (5)"} | {"(Existing Use) 1 family dwelling"}  | 0.010141 | 0.930780 | 3.958710 |
<br>

---------

**<center>4-Itemset Relation Rules</center>**

| LHS | RHS | Support | Confidence | Lift |
| :-----: | :-----: | :-----: | :-----: | :-----: |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) 1 family dwelling"} | {"(Proposed Use) 1 family dwelling"}  | 0.209738 | 0.989492 | 4.246559 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) wood frame (5)", "(Proposed Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)"} | {"(Existing Use) 1 family dwelling"}  | 0.209738 | 0.997776 | 4.243651 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | {"(Proposed Use) apartments"}  | 0.151744 | 0.997126 | 4.608880 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments"} | {"(Existing Use) apartments"}  | 0.151744 | 0.986050 | 4.807253 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) 1 family dwelling"} | {"(Proposed Use) 1 family dwelling"}  | 0.141684 | 0.986522 | 4.233813 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)"} | {"(Existing Use) 1 family dwelling"}  | 0.141684 | 0.997169 | 4.241072 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Permit Type Definition) otc alterations permit", "(Existing Use) 1 family dwelling"} | {"(Proposed Use) 1 family dwelling"}  | 0.136601 | 0.992113 | 4.257805 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Use) 1 family dwelling", "(Permit Type Definition) otc alterations permit"} | {"(Existing Use) 1 family dwelling"}  | 0.136601 | 0.998016 | 4.244675 |
| {"(Current Status) complete", "(Existing Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)", "(Permit Type Definition) otc alterations permit"} | {"(Proposed Use) 1 family dwelling"}  | 0.136601 | 0.992185 | 4.258116 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)"} | {"(Existing Use) 1 family dwelling"}  | 0.136601 | 0.992947 | 4.223113 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | {"(Proposed Use) apartments"}  | 0.100960 | 0.997269 | 4.609540 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments"} | {"(Existing Use) apartments"}  | 0.100960 | 0.985039 | 4.802323 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | {"(Proposed Use) apartments"}  | 0.095791 | 0.997591 | 4.611032 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments"} | {"(Existing Use) apartments"}  | 0.095791 | 0.976977 | 4.763019 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Existing Use) apartments", "(Permit Type Definition) otc alterations permit"} | {"(Proposed Use) apartments"}  | 0.095721 | 0.997537 | 4.610783 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Use) apartments", "(Permit Type Definition) otc alterations permit"} | {"(Existing Use) apartments"}  | 0.095721 | 0.988166 | 4.817571 |
| {"(Existing Use) 2 family dwelling", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Permit Type Definition) otc alterations permit"} | {"(Proposed Use) 2 family dwelling"}  | 0.093569 | 0.979166 | 8.828113 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) 2 family dwelling"} | {"(Existing Use) 2 family dwelling"}  | 0.093569 | 0.979217 | 9.280376 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Proposed Construction Type Description) constr type 1", "(Existing Use) office"} | {"(Proposed Use) office"}  | 0.086983 | 0.990326 | 8.220386 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Proposed Use) office", "(Existing Use) office"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.086983 | 0.999480 | 7.140461 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Proposed Use) office", "(Proposed Construction Type Description) constr type 1"} | {"(Existing Use) office"}  | 0.086983 | 0.997636 | 8.061048 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Proposed Use) office", "(Existing Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.086983 | 0.999711 | 7.083340 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Existing Use) 2 family dwelling", "(Proposed Construction Type Description) wood frame (5)"} | {"(Proposed Use) 2 family dwelling"}  | 0.065520 | 0.978893 | 8.825654 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) 2 family dwelling"} | {"(Existing Use) 2 family dwelling"}  | 0.065520 | 0.973118 | 9.222576 |
| {"(Current Status) issued", "(Existing Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)", "(Existing Construction Type Description) wood frame (5)"} | {"(Proposed Use) 1 family dwelling"}  | 0.065490 | 0.974635 | 4.182797 |
| {"(Proposed Use) 1 family dwelling", "(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)", "(Existing Construction Type Description) wood frame (5)"} | {"(Existing Use) 1 family dwelling"}  | 0.065490 | 0.996100 | 4.236524 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Existing Use) 2 family dwelling", "(Proposed Construction Type Description) wood frame (5)"} | {"(Proposed Use) 2 family dwelling"}  | 0.063157 | 0.983712 | 8.869102 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) 2 family dwelling"} | {"(Existing Use) 2 family dwelling"}  | 0.063157 | 0.971915 | 9.211171 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Existing Use) 2 family dwelling", "(Permit Type Definition) otc alterations permit"} | {"(Proposed Use) 2 family dwelling"}  | 0.063112 | 0.983700 | 8.868999 |
| {"(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Use) 2 family dwelling", "(Permit Type Definition) otc alterations permit"} | {"(Existing Use) 2 family dwelling"}  | 0.063112 | 0.985013 | 9.335302 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Proposed Construction Type Description) constr type 1", "(Existing Use) office"} | {"(Proposed Use) office"}  | 0.061578 | 0.990298 | 8.220147 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Proposed Use) office", "(Existing Use) office"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.061578 | 0.999510 | 7.140678 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Proposed Use) office", "(Proposed Construction Type Description) constr type 1"} | {"(Existing Use) office"}  | 0.061578 | 0.997151 | 8.057127 |
| {"(Current Status) complete", "(Proposed Construction Type Description) constr type 1", "(Proposed Use) office", "(Existing Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.061578 | 0.999755 | 7.083653 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Use) 1 family dwelling", "(Existing Construction Type Description) wood frame (5)"} | {"(Proposed Use) 1 family dwelling"}  | 0.060699 | 0.986679 | 4.234483 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Use) 1 family dwelling", "(Existing Construction Type Description) wood frame (5)"} | {"(Existing Use) 1 family dwelling"}  | 0.060699 | 0.997521 | 4.242569 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) 1 family dwelling"} | {"(Proposed Use) 1 family dwelling"}  | 0.060643 | 0.986990 | 4.235818 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)"} | {"(Existing Use) 1 family dwelling"}  | 0.060643 | 0.986990 | 4.197777 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Existing Use) office", "(Permit Type Definition) otc alterations permit"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.060387 | 0.999418 | 7.140015 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Proposed Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit"} | {"(Existing Use) office"}  | 0.060387 | 0.709912 | 5.736196 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Existing Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.060387 | 0.999667 | 7.083029 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Proposed Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit"} | {"(Proposed Use) office"}  | 0.060025 | 0.705656 | 5.857431 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Proposed Use) office", "(Permit Type Definition) otc alterations permit"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.060025 | 0.999498 | 7.140587 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Use) office", "(Proposed Construction Type Description) constr type 1"} | {"(Existing Construction Type Description) constr type 1"}  | 0.060025 | 0.995580 | 7.054073 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Existing Use) office", "(Permit Type Definition) otc alterations permit"} | {"(Proposed Use) office"}  | 0.059904 | 0.991430 | 8.229544 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Proposed Use) office", "(Permit Type Definition) otc alterations permit"} | {"(Existing Use) office"}  | 0.059904 | 0.997488 | 8.059858 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Use) office", "(Existing Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.059904 | 0.795128 | 5.633793 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Existing Use) office"} | {"(Proposed Use) office"}  | 0.059889 | 0.991427 | 8.229526 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Use) office", "(Existing Use) office"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.059889 | 0.794928 | 5.679107 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Use) office", "(Proposed Construction Type Description) constr type 1"} | {"(Existing Use) office"}  | 0.059889 | 0.993329 | 8.026248 |
| {"(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments", "(Existing Construction Type Description) wood frame (5)"} | {"(Proposed Use) apartments"}  | 0.047612 | 0.996003 | 4.603692 |
| {"(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments", "(Existing Construction Type Description) wood frame (5)"} | {"(Existing Use) apartments"}  | 0.047612 | 0.970884 | 4.733314 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Use) apartments", "(Existing Construction Type Description) wood frame (5)"} | {"(Proposed Use) apartments"}  | 0.041714 | 0.995799 | 4.602749 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Use) apartments", "(Existing Construction Type Description) wood frame (5)"} | {"(Existing Use) apartments"}  | 0.041714 | 0.979691 | 4.776250 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | {"(Proposed Use) apartments"}  | 0.041694 | 0.995797 | 4.602739 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments"} | {"(Existing Use) apartments"}  | 0.041694 | 0.969715 | 4.727615 |
| {"(Existing Use) 2 family dwelling", "(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)", "(Existing Construction Type Description) wood frame (5)"} | {"(Proposed Use) 2 family dwelling"}  | 0.026616 | 0.954734 | 8.607839 |
| {"(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) 2 family dwelling", "(Existing Construction Type Description) wood frame (5)"} | {"(Existing Use) 2 family dwelling"}  | 0.026616 | 0.942832 | 8.935540 |
| {"(Existing Use) 2 family dwelling", "(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)", "(Permit Type Definition) otc alterations permit"} | {"(Proposed Use) 2 family dwelling"}  | 0.024726 | 0.969828 | 8.743930 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) 2 family dwelling"} | {"(Existing Use) 2 family dwelling"}  | 0.024726 | 0.927225 | 8.787627 |
| {"(Existing Use) 2 family dwelling", "(Current Status) issued", "(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) wood frame (5)"} | {"(Proposed Use) 2 family dwelling"}  | 0.024706 | 0.969805 | 8.743716 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Use) 2 family dwelling", "(Existing Construction Type Description) wood frame (5)"} | {"(Existing Use) 2 family dwelling"}  | 0.024706 | 0.970379 | 9.196616 |
| {"(Existing Use) office", "(Current Status) issued", "(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1"} | {"(Proposed Use) office"}  | 0.024334 | 0.986346 | 8.187347 |
| {"(Existing Use) office", "(Current Status) issued", "(Proposed Use) office", "(Existing Construction Type Description) constr type 1"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.024334 | 0.999381 | 7.139750 |
| {"(Proposed Construction Type Description) constr type 1", "(Current Status) issued", "(Proposed Use) office", "(Existing Construction Type Description) constr type 1"} | {"(Existing Use) office"}  | 0.024334 | 0.998144 | 8.065154 |
| {"(Proposed Construction Type Description) constr type 1", "(Current Status) issued", "(Proposed Use) office", "(Existing Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.024334 | 0.999587 | 7.082461 |
| {"(Existing Use) office", "(Current Status) issued", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.023745 | 0.999365 | 7.139641 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1"} | {"(Existing Use) office"}  | 0.023745 | 0.671453 | 5.425441 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Construction Type Description) constr type 1", "(Existing Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.023745 | 0.999154 | 7.079392 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1"} | {"(Proposed Use) office"}  | 0.023514 | 0.664913 | 5.519235 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Use) office", "(Existing Construction Type Description) constr type 1"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.023514 | 0.999359 | 7.139596 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Use) office", "(Proposed Construction Type Description) constr type 1"} | {"(Existing Construction Type Description) constr type 1"}  | 0.023514 | 0.987542 | 6.997119 |
| {"(Existing Use) office", "(Current Status) issued", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit"} | {"(Proposed Use) office"}  | 0.023494 | 0.988785 | 8.207596 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Use) office", "(Existing Construction Type Description) constr type 1"} | {"(Existing Use) office"}  | 0.023494 | 0.998504 | 8.068065 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Use) office", "(Existing Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.023494 | 0.760951 | 5.391633 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Construction Type Description) constr type 1", "(Existing Use) office"} | {"(Proposed Use) office"}  | 0.023489 | 0.988365 | 8.204104 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Use) office", "(Existing Use) office"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.023489 | 0.760788 | 5.435204 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Use) office", "(Proposed Construction Type Description) constr type 1"} | {"(Existing Use) office"}  | 0.023489 | 0.986486 | 7.970960 |
| {"(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments", "(Permit Type Definition) additions alterations or repairs"} | {"(Proposed Use) apartments"}  | 0.018864 | 0.992855 | 4.589141 |
| {"(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments", "(Permit Type Definition) additions alterations or repairs"} | {"(Existing Use) apartments"}  | 0.018864 | 0.881786 | 4.298940 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Proposed Construction Type Description) constr type 1", "(Existing Use) apartments"} | {"(Proposed Use) apartments"}  | 0.016873 | 0.995255 | 4.600233 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Proposed Use) apartments", "(Existing Use) apartments"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.016873 | 0.997622 | 7.127186 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Proposed Construction Type Description) constr type 1", "(Proposed Use) apartments"} | {"(Existing Use) apartments"}  | 0.016873 | 0.993781 | 4.844946 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Proposed Use) apartments", "(Existing Use) apartments"} | {"(Existing Construction Type Description) constr type 1"}  | 0.016873 | 0.999404 | 7.081168 |
| {"(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Permit Type Definition) additions alterations or repairs", "(Existing Use) 1 family dwelling"} | {"(Proposed Use) 1 family dwelling"}  | 0.016335 | 0.808008 | 3.467691 |
| {"(Proposed Use) 1 family dwelling", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Permit Type Definition) additions alterations or repairs"} | {"(Existing Use) 1 family dwelling"}  | 0.016335 | 0.969561 | 4.123652 |
| {"(Current Status) filed", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments", "(Existing Construction Type Description) wood frame (5)"} | {"(Proposed Use) apartments"}  | 0.015374 | 0.995767 | 4.602599 |
| {"(Current Status) filed", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments", "(Existing Construction Type Description) wood frame (5)"} | {"(Existing Use) apartments"}  | 0.015374 | 0.921362 | 4.491882 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1", "(Existing Use) office", "(Street Name) Market"} | {"(Proposed Use) office"}  | 0.012328 | 0.981585 | 8.147830 |
| {"(Proposed Use) office", "(Existing Construction Type Description) constr type 1", "(Existing Use) office", "(Street Name) Market"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.012328 | 0.999592 | 7.141263 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1", "(Proposed Use) office", "(Street Name) Market"} | {"(Existing Use) office"}  | 0.012328 | 1.000000 | 8.080151 |
| {"(Proposed Construction Type Description) constr type 1", "(Proposed Use) office", "(Existing Use) office", "(Street Name) Market"} | {"(Existing Construction Type Description) constr type 1"}  | 0.012328 | 0.999592 | 7.082499 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Existing Use) office", "(Street Name) Market"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.011931 | 0.999579 | 7.141167 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Proposed Construction Type Description) constr type 1", "(Street Name) Market"} | {"(Existing Use) office"}  | 0.011931 | 0.760821 | 6.147547 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Existing Use) office", "(Street Name) Market"} | {"(Existing Construction Type Description) constr type 1"}  | 0.011931 | 0.998737 | 7.076441 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1", "(Street Name) California", "(Existing Use) office"} | {"(Proposed Use) office"}  | 0.011921 | 0.996637 | 8.272771 |
| {"(Proposed Use) office", "(Existing Construction Type Description) constr type 1", "(Street Name) California", "(Existing Use) office"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.011921 | 0.999578 | 7.141164 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1", "(Street Name) California", "(Proposed Use) office"} | {"(Existing Use) office"}  | 0.011921 | 0.999578 | 8.076745 |
| {"(Proposed Construction Type Description) constr type 1", "(Street Name) California", "(Proposed Use) office", "(Existing Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.011921 | 1.000000 | 7.085388 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Street Name) California", "(Existing Use) office"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.011785 | 0.999574 | 7.141129 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Street Name) California", "(Proposed Construction Type Description) constr type 1"} | {"(Existing Use) office"}  | 0.011785 | 0.902232 | 7.290175 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Street Name) California", "(Existing Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.011785 | 1.000000 | 7.085388 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Street Name) California", "(Existing Use) office"} | {"(Proposed Use) office"}  | 0.011755 | 0.997015 | 8.275906 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Street Name) California", "(Proposed Use) office"} | {"(Existing Use) office"}  | 0.011755 | 0.999572 | 8.076697 |
| {"(Permit Type Definition) otc alterations permit", "(Street Name) California", "(Proposed Use) office", "(Existing Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.011755 | 0.951180 | 6.739478 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Existing Use) office", "(Street Name) Market"} | {"(Proposed Use) office"}  | 0.011755 | 0.984836 | 8.174811 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Proposed Use) office", "(Street Name) Market"} | {"(Existing Use) office"}  | 0.011755 | 1.000000 | 8.080151 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Use) office", "(Existing Use) office", "(Street Name) Market"} | {"(Existing Construction Type Description) constr type 1"}  | 0.011755 | 0.882931 | 6.255905 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Street Name) California", "(Proposed Construction Type Description) constr type 1"} | {"(Proposed Use) office"}  | 0.011755 | 0.899923 | 7.469977 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Street Name) California", "(Proposed Use) office"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.011755 | 0.999572 | 7.141121 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Street Name) California", "(Proposed Use) office"} | {"(Existing Construction Type Description) constr type 1"}  | 0.011755 | 1.000000 | 7.085388 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Existing Use) office", "(Street Name) Market"} | {"(Proposed Use) office"}  | 0.011755 | 0.984007 | 8.167929 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Use) office", "(Existing Use) office", "(Street Name) Market"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.011755 | 0.882931 | 6.307811 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Proposed Use) office", "(Street Name) Market"} | {"(Existing Use) office"}  | 0.011755 | 0.999145 | 8.073245 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Proposed Construction Type Description) constr type 1", "(Street Name) Market"} | {"(Proposed Use) office"}  | 0.011750 | 0.749279 | 6.219525 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 1", "(Proposed Use) office", "(Street Name) Market"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.011750 | 0.999572 | 7.141120 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Proposed Use) office", "(Street Name) Market"} | {"(Existing Construction Type Description) constr type 1"}  | 0.011750 | 0.998718 | 7.076304 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Street Name) California", "(Existing Use) office"} | {"(Proposed Use) office"}  | 0.011750 | 0.997014 | 8.275896 |
| {"(Permit Type Definition) otc alterations permit", "(Street Name) California", "(Proposed Use) office", "(Existing Use) office"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.011750 | 0.950773 | 6.792489 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Street Name) California", "(Proposed Use) office"} | {"(Existing Use) office"}  | 0.011750 | 0.999572 | 8.076695 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Proposed Construction Type Description) constr type 1", "(Existing Use) apartments"} | {"(Proposed Use) apartments"}  | 0.011719 | 0.992337 | 4.586746 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Proposed Use) apartments", "(Existing Use) apartments"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.011719 | 0.998287 | 7.131937 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Proposed Construction Type Description) constr type 1", "(Proposed Use) apartments"} | {"(Existing Use) apartments"}  | 0.011719 | 0.991915 | 4.835846 |
| {"(Current Status) complete", "(Proposed Construction Type Description) constr type 1", "(Proposed Use) apartments", "(Existing Use) apartments"} | {"(Existing Construction Type Description) constr type 1"}  | 0.011719 | 0.999143 | 7.079314 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Proposed Use) apartments", "(Permit Type Definition) otc alterations permit"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.011362 | 0.996473 | 7.118976 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Proposed Use) apartments"} | {"(Existing Construction Type Description) constr type 1"}  | 0.011362 | 0.925850 | 6.560007 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Existing Use) apartments", "(Permit Type Definition) otc alterations permit"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.011347 | 0.998231 | 7.131537 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Existing Use) apartments"} | {"(Existing Construction Type Description) constr type 1"}  | 0.011347 | 0.999557 | 7.082250 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Existing Use) apartments", "(Permit Type Definition) otc alterations permit"} | {"(Proposed Use) apartments"}  | 0.011302 | 0.994250 | 4.595589 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Proposed Use) apartments", "(Permit Type Definition) otc alterations permit"} | {"(Existing Use) apartments"}  | 0.011302 | 0.991182 | 4.832272 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Existing Use) apartments"} | {"(Proposed Use) apartments"}  | 0.011287 | 0.994243 | 4.595554 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Proposed Use) apartments"} | {"(Existing Use) apartments"}  | 0.011287 | 0.919705 | 4.483804 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 3", "(Existing Use) office", "(Proposed Construction Type Description) constr type 3"} | {"(Proposed Use) office"}  | 0.011051 | 0.947822 | 7.867574 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 3", "(Proposed Use) office", "(Existing Use) office"} | {"(Proposed Construction Type Description) constr type 3"}  | 0.011051 | 0.996374 | 21.173044 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) constr type 3", "(Proposed Use) office", "(Proposed Construction Type Description) constr type 3"} | {"(Existing Use) office"}  | 0.011051 | 0.961084 | 7.765707 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Use) office", "(Existing Use) office", "(Proposed Construction Type Description) constr type 3"} | {"(Existing Construction Type Description) constr type 3"}  | 0.011051 | 0.998183 | 20.546382 |
| {"(Current Status) complete", "(Existing Construction Type Description) constr type 1", "(Street Name) Market", "(Permit Type Definition) otc alterations permit"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.010452 | 0.999039 | 7.137310 |
| {"(Current Status) complete", "(Permit Type Definition) otc alterations permit", "(Proposed Construction Type Description) constr type 1", "(Street Name) Market"} | {"(Existing Construction Type Description) constr type 1"}  | 0.010452 | 0.985775 | 6.984600 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) filed", "(Existing Use) apartments", "(Existing Construction Type Description) wood frame (5)"} | {"(Proposed Use) apartments"}  | 0.010432 | 0.997596 | 4.611054 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) filed", "(Proposed Use) apartments", "(Existing Construction Type Description) wood frame (5)"} | {"(Existing Use) apartments"}  | 0.010432 | 0.990453 | 4.828722 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) filed", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | {"(Proposed Use) apartments"}  | 0.010427 | 0.997595 | 4.611049 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) filed", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments"} | {"(Existing Use) apartments"}  | 0.010427 | 0.984338 | 4.798907 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) food/beverage hndlng"} | {"(Proposed Use) food/beverage hndlng"}  | 0.010422 | 0.975071 | 38.381656 |
| {"(Permit Type Definition) otc alterations permit", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) food/beverage hndlng"} | {"(Existing Use) food/beverage hndlng"}  | 0.010422 | 0.900913 | 36.674668 |
| {"(Current Status) filed", "(Existing Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)", "(Existing Construction Type Description) wood frame (5)"} | {"(Proposed Use) 1 family dwelling"}  | 0.010136 | 0.848842 | 3.642937 |
| {"(Proposed Use) 1 family dwelling", "(Current Status) filed", "(Proposed Construction Type Description) wood frame (5)", "(Existing Construction Type Description) wood frame (5)"} | {"(Existing Use) 1 family dwelling"}  | 0.010136 | 0.982456 | 4.178495 |
<br>

---------

**<center>5-Itemset Relation Rules</center>**

| LHS | RHS | Support | Confidence | Lift |
| :-----: | :-----: | :-----: | :-----: | :-----: |
| {"(Permit Type Definition) otc alterations permit", "(Existing Use) 1 family dwelling", "(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)"} | {"(Proposed Use) 1 family dwelling"}  | 0.136500 | 0.992180 | 4.258091 |
| {"(Proposed Use) 1 family dwelling", "(Permit Type Definition) otc alterations permit", "(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)"} | {"(Existing Use) 1 family dwelling"}  | 0.136500 | 0.998015 | 4.244669 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | {"(Proposed Use) apartments"}  | 0.095666 | 0.997588 | 4.611018 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments"} | {"(Existing Use) apartments"}  | 0.095666 | 0.988622 | 4.819791 |
| {"(Existing Use) 2 family dwelling", "(Permit Type Definition) otc alterations permit", "(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)"} | {"(Proposed Use) 2 family dwelling"}  | 0.063097 | 0.983697 | 8.868964 |
| {"(Permit Type Definition) otc alterations permit", "(Proposed Use) 2 family dwelling", "(Current Status) complete", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)"} | {"(Existing Use) 2 family dwelling"}  | 0.063097 | 0.985009 | 9.335269 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Use) 1 family dwelling", "(Proposed Construction Type Description) wood frame (5)", "(Existing Construction Type Description) wood frame (5)"} | {"(Proposed Use) 1 family dwelling"}  | 0.060623 | 0.986985 | 4.235800 |
| {"(Proposed Use) 1 family dwelling", "(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)"} | {"(Existing Use) 1 family dwelling"}  | 0.060623 | 0.997518 | 4.242556 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Use) office", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit", "(Current Status) complete"} | {"(Proposed Use) office"}  | 0.059874 | 0.991508 | 8.230193 |
| {"(Proposed Use) office", "(Existing Use) office", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit", "(Current Status) complete"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.059874 | 0.999496 | 7.140578 |
| {"(Proposed Use) office", "(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit", "(Current Status) complete"} | {"(Existing Use) office"}  | 0.059874 | 0.997487 | 8.059848 |
| {"(Proposed Construction Type Description) constr type 1", "(Proposed Use) office", "(Existing Use) office", "(Permit Type Definition) otc alterations permit", "(Current Status) complete"} | {"(Existing Construction Type Description) constr type 1"}  | 0.059874 | 0.999748 | 7.083603 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments"} | {"(Proposed Use) apartments"}  | 0.041659 | 0.995794 | 4.602723 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments"} | {"(Existing Use) apartments"}  | 0.041659 | 0.980128 | 4.778381 |
| {"(Existing Use) 2 family dwelling", "(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)"} | {"(Proposed Use) 2 family dwelling"}  | 0.024696 | 0.969793 | 8.743608 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) issued", "(Proposed Use) 2 family dwelling", "(Existing Construction Type Description) wood frame (5)", "(Proposed Construction Type Description) wood frame (5)"} | {"(Existing Use) 2 family dwelling"}  | 0.024696 | 0.970559 | 9.198322 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Use) office", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit", "(Current Status) issued"} | {"(Proposed Use) office"}  | 0.023479 | 0.988778 | 8.207537 |
| {"(Proposed Use) office", "(Existing Use) office", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit", "(Current Status) issued"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.023479 | 0.999358 | 7.139589 |
| {"(Proposed Construction Type Description) constr type 1", "(Proposed Use) office", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit", "(Current Status) issued"} | {"(Existing Use) office"}  | 0.023479 | 0.998503 | 8.068058 |
| {"(Proposed Construction Type Description) constr type 1", "(Proposed Use) office", "(Existing Use) office", "(Permit Type Definition) otc alterations permit", "(Current Status) issued"} | {"(Existing Construction Type Description) constr type 1"}  | 0.023479 | 0.999572 | 7.082354 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Use) office", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit", "(Street Name) California"} | {"(Proposed Use) office"}  | 0.011750 | 0.997014 | 8.275896 |
| {"(Proposed Use) office", "(Existing Use) office", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit", "(Street Name) California"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.011750 | 0.999572 | 7.141120 |
| {"(Proposed Use) office", "(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit", "(Street Name) California"} | {"(Existing Use) office"}  | 0.011750 | 0.999572 | 8.076695 |
| {"(Proposed Construction Type Description) constr type 1", "(Proposed Use) office", "(Existing Use) office", "(Permit Type Definition) otc alterations permit", "(Street Name) California"} | {"(Existing Construction Type Description) constr type 1"}  | 0.011750 | 1.000000 | 7.085388 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Use) office", "(Street Name) Market", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit"} | {"(Proposed Use) office"}  | 0.011750 | 0.984829 | 8.174757 |
| {"(Proposed Use) office", "(Existing Use) office", "(Street Name) Market", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.011750 | 0.999572 | 7.141120 |
| {"(Proposed Use) office", "(Proposed Construction Type Description) constr type 1", "(Street Name) Market", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit"} | {"(Existing Use) office"}  | 0.011750 | 1.000000 | 8.080151 |
| {"(Proposed Construction Type Description) constr type 1", "(Proposed Use) office", "(Existing Use) office", "(Street Name) Market", "(Permit Type Definition) otc alterations permit"} | {"(Existing Construction Type Description) constr type 1"}  | 0.011750 | 0.999572 | 7.082357 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit", "(Current Status) complete", "(Existing Use) apartments"} | {"(Proposed Use) apartments"}  | 0.011282 | 0.994240 | 4.595542 |
| {"(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit", "(Current Status) complete", "(Existing Use) apartments", "(Proposed Use) apartments"} | {"(Proposed Construction Type Description) constr type 1"}  | 0.011282 | 0.998221 | 7.131464 |
| {"(Proposed Construction Type Description) constr type 1", "(Existing Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit", "(Current Status) complete", "(Proposed Use) apartments"} | {"(Existing Use) apartments"}  | 0.011282 | 0.992920 | 4.840748 |
| {"(Proposed Construction Type Description) constr type 1", "(Permit Type Definition) otc alterations permit", "(Current Status) complete", "(Existing Use) apartments", "(Proposed Use) apartments"} | {"(Existing Construction Type Description) constr type 1"}  | 0.011282 | 0.999555 | 7.082232 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) filed", "(Proposed Construction Type Description) wood frame (5)", "(Existing Use) apartments", "(Existing Construction Type Description) wood frame (5)"} | {"(Proposed Use) apartments"}  | 0.010427 | 0.997595 | 4.611049 |
| {"(Permit Type Definition) otc alterations permit", "(Current Status) filed", "(Proposed Construction Type Description) wood frame (5)", "(Proposed Use) apartments", "(Existing Construction Type Description) wood frame (5)"} | {"(Existing Use) apartments"}  | 0.010427 | 0.990449 | 4.828699 |
<br>

---------

## 对关联规则的结果分析

通过对导出的关联规则的分析，我们发现了一些规则，以下是一些例举：
1. 建筑的当前用途和新提出的用途是基本保持一致的，例如原来用作单个单个家庭的居所的建筑，新提出的用途申请基本也是和单个家庭居所、原来用作公寓提出的新用途申请也基本用作公寓。
2. 建筑的新建设类型与当前基本保持一致。
3. Market街的建筑主要建设类型为type1.
4. 当前已完成的建设类型为type1的建筑大都用作office。
5. ...