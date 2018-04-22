import numpy as numpy
import csv
import pickle

database = 'Data/Building_Permits.csv'

NA = ['NA','None','','NONE','none','Na']

def preprocess():
    event = []
    
    with open('Data/item','r') as fp:
        attr = [int(i)-1 for i in fp.read().split(' ')]

    with open(database,encoding='utf-8') as fp:
        reader = csv.reader(fp)
        for _,row in enumerate(reader):
            item = []
            if _ == 0:
                name = row
            for i in attr:
                at = row[i]
                if at in NA:
                    at = '%s NA'%name[i]
                item.append(at)

            event.append(item)
            print(item)

    with open('Data/preprocessed.pkl','wb') as fp:
        pickle.dump(event,fp)

if __name__ == '__main__':
    preprocess()