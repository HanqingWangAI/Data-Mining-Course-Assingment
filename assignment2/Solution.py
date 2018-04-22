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

def main():
    with open('Data/preprocessed.pkl','rb') as fp:
        transactions = pickle.load(fp,encoding='iso-8859-1')
    results = list(apriori(transactions,max_length=1000))
    # print(len(results))
    with open('Results.md','w') as fp:
        items_by_k = []
        for i in range(7):
            items_by_k.append([])

        for event in results:
            items = event.items
            n = len(items)
            # print(n)
            support = event.support
            items_by_k[n-1].append((support, items))
        
        for _, items in enumerate(items_by_k):
            items = sorted(items, reverse=True)
            fp.write('<br>\n\n---------\n\n**<center>%d-Itemset</center>**\n\n'%(_+1))
            fp.write('| Itemset | Support |\n')
            fp.write('| :-----: | :-----: |\n')
            for item in items:
                set_ = item[1]
                support_ = item[0]
                fp.write('| {')
                for i,ele in enumerate(set_):
                    if i != 0:
                        fp.write(', ')
                    fp.write('"%s"'%ele)
                
                fp.write('} |')

                fp.write(' %f |\n'%support_)
            


if __name__ == '__main__':
    main()