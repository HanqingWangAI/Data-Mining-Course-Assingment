import numpy as np
import pickle
from apyori import apriori


def test():
    transactions = [
        ['beer','nuts','yes'],
        ['beer','cheese'],
        ['beer','nuts','no'],
        ['beer','nuts','no'],
        ['beer','nuts','no'],
        ['beer','nuts','no'],
        ['beer','nuts','no'],
        ['beer','nuts','no'],
        ['beer','nuts','no'],
        ['beer','nuts','no'],
        ['beer','nuts','no'],
        ['beer','nuts','no']
    ]

    # relationRecord = {items=frozenset(), support=, ordered_statistics=[OrderedStatistic(items_base=frozenset(), items_add=frozenset(), confidence, lift)]}

    results = list(apriori(transactions))
    for event in results:
        # print(event)
        fs = event.items
        for i in fs:
            print(i)
        print('================================')

def check_set(set_):
    cnt = 0
    for item in set_:
        if item[-2:] != 'NA':
            cnt += 1
    return cnt

def main():
    with open('Data/preprocessed.pkl','rb') as fp:
        transactions = pickle.load(fp,encoding='iso-8859-1')
    results = list(apriori(transactions,min_support=0.01,max_length=1000))
    # print(len(results))
    
    items_by_k = []
    relations_by_k = []

    for i in range(7):
        items_by_k.append([])
        relations_by_k.append([])

    for event in results:
        items = event.items
        n = len(items)
        # print(n)
        support = event.support
        ordered_statistics = event.ordered_statistics
        items_by_k[n-1].append((support, items))
        relations_by_k[n-1].append((support, ordered_statistics))

    with open('frequent_items.md','w') as fp:
        for _, items in enumerate(items_by_k):
            items = sorted(items, reverse=True)
            fp.write('<br>\n\n---------\n\n**<center>%d-Itemset</center>**\n\n'%(_+1))
            fp.write('| Itemset | Support |\n')
            fp.write('| :-----: | :-----: |\n')
            for item in items:
                set_ = item[1]
                support_ = item[0]
                fp.write('| {')
                for i, ele in enumerate(set_):
                    if i != 0:
                        fp.write(', ')
                    fp.write('"%s"'%ele)
                
                fp.write('} |')
                fp.write(' %f |\n'%support_)


    with open('relation_rules.md','w') as fp:
        for _, relations in enumerate(relations_by_k):
            relations = sorted(relations, reverse=True)
            fp.write('<br>\n\n---------\n\n**<center>%d-Itemset Relation Rules</center>**\n\n'%(_))
            fp.write('| LHS | RHS | Support | Confidence | Lift |\n')
            fp.write('| :-----: | :-----: | :-----: | :-----: | :-----: |\n')
            for relation in relations:
                set_list = relation[1]
                support = relation[0]
                for set_ in set_list:
                    lhs = set_.items_base
                    rhs = set_.items_add
                    
                    confidence = set_.confidence
                    lift = set_.lift

                    if check_set(lhs) == 0 or check_set(rhs) == 0:
                        continue
                    if confidence < 0.6 or lift < 3:
                        continue
                    fp.write('| {')
                    for i, ele in enumerate(lhs):
                        if i != 0:
                            fp.write(', ')
                        fp.write('"%s"'%ele)
                    fp.write('} | {')
                    for i, ele in enumerate(rhs):
                        if i != 0:
                            fp.write(', ')
                        fp.write('"%s"'%ele)
                    fp.write('} ')
                    fp.write(' | %f | %f | %f |\n'%(support,confidence,lift))



if __name__ == '__main__':
    main()