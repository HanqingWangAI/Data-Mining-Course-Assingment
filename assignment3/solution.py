import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
import csv
import matplotlib.pyplot as plt


train_data = []
test_data = []
train_label = []


database_train = 'Data/train.csv'
database_test = 'Data/test.csv'
attr = [2,4,5,6,7,9,11]
bin_num = [3,2,20,4,4,20,3]
attr_name = []


NA = ['NA','None','','NONE','none','Na']

sex = {'male':0,'female':1}
embarked = {'C':0,'Q':1,'S':2,'':3}

def data_preprocess():
    global train_data, test_data, train_label, attr_name
    with open(database_train,encoding='utf-8') as fp:
        reader = csv.reader(fp)
        events = []
        for i, row in enumerate(reader):
            event = []
            for j in attr:
                if i != 0:
                    if j == 4:
                        row[j] = sex[row[j]]
                    elif j == 11:
                        row[j] = embarked[row[j]]
                    else:
                        if row[j] in NA:
                            row[j] = -1
                        else:
                            row[j] = float(row[j])
                # print(row[j])
                event.append(row[j])
                

            if i == 0:
                attr_name = event
            else:
                train_label.append(int(row[1]))
                events.append(event)
        
        train_data = np.array(events)

    with open(database_test,encoding='utf-8') as fp:
        reader = csv.reader(fp)
        events = []
        for i, row in enumerate(reader):
            event = []
            if i != 0:
                for j in attr:
                    j = j-1
                   
                    if j == 3:
                        row[j] = sex[row[j]]
                    elif j == 10:
                        row[j] = embarked[row[j]]
                    else:
                        if row[j] in NA:
                            row[j] = 0
                        else:
                            row[j] = float(row[j])

                    event.append(row[j])
                
                events.append(event)
        
        test_data = np.array(events)
    
def classification_svm():
    clf = svm.SVC(C=0.8)
    clf.fit(train_data, train_label)

    return clf.predict(test_data)

def classification_gnb():
    gnb = GaussianNB()
    test_label = gnb.fit(train_data, train_label).predict(test_data)

    return test_label

def cluster_kmeans():
    cluster = KMeans(n_clusters=2)
    cluster.fit(test_data)

    return cluster.labels_

def cluster_gmm():
    cluster = GMM(n_components=2)
    cluster.fit(test_data)

    return cluster.predict(test_data)

def plot_classificaton_result(test_label,classifier=''): 
    pred = []
    labelled = []
    pred.append([])
    pred.append([])
    labelled.append([])
    labelled.append([])


    for i,l in enumerate(test_label):
        pred[l].append(test_data[i])
    
    for i,l in enumerate(train_label):
        labelled[l].append(train_data[i])

    for i in range(2):
        pred[i] = np.array(pred[i]).transpose()
    
    for i in range(2):
        labelled[i] = np.array(labelled[i]).transpose()

    for i in range(7):
        
        fig,axes = plt.subplots(ncols=2)
        fig.set_size_inches(8,5)
        ax0,ax1 = axes.flatten()
        ax0.hist([pred[0][i],pred[1][i]],bins=bin_num[i],label=['Death','Survival'])
        ax0.set_title('Predicted Survival over %s'%attr_name[i])
        
        ax1.hist([labelled[0][i],labelled[1][i]],bins=bin_num[i],label=['Death','Survival'])
        ax1.set_title('Labelled Survival over %s'%attr_name[i])
        
        plt.legend()


        # plt.show()
        plt.savefig('Figures/%s_%s_hist.png'%(classifier,attr_name[i]))
        plt.close()

def plot_pca_result(low_d_data,label,model='',legends=['First Class', 'Second Class']):
    from matplotlib.ticker import NullFormatter
    tmp = np.abs(low_d_data).flatten()
    sorted(tmp)
    xymax = tmp[int(len(tmp)*0.99)-1]+2
    print(xymax)

    datas = []
    datas.append([])
    datas.append([])

    for i,l in enumerate(label):
        datas[l].append(low_d_data[i])

    for i in range(2):
        datas[i] = np.array(datas[i])

    X = [datas[0][:,0], datas[1][:,0]]
    Y = [datas[0][:,1], datas[1][:,1]]

    nullfmt = NullFormatter() 

    # definitions for the axes
    left, width = 0.1, 0.60
    bottom, height = 0.1, 0.60
    bottom_h = left_h = left + width + 0.04

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    plt.figure(1, figsize=(8, 8))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    axScatter.scatter(X[0], Y[0],label='Death')
    axScatter.scatter(X[1], Y[1],label='Survival')
    # now determine nice limits by hand:
    binwidth = 5
    
    lim = (xymax/binwidth + 1) * binwidth

    axScatter.set_xlim((-lim, lim))
    axScatter.set_ylim((-lim, lim))

    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(X, bins=bins,label=legends)
    axHisty.hist(Y, bins=bins, orientation='horizontal')

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())
    axHistx.legend()
    axHistx.set_title('First principal component')
    axHisty.set_title('Second principal component')
    axScatter.set_title('Distribution of %s after PCA'%model)

    # plt.show()
    plt.savefig('Figures/%s_scatter.png'%model)
    plt.close()



def main():
    data_preprocess()
    

    label_gmm = cluster_gmm()
    label_kmeans = cluster_kmeans()
    
    label_gnb = classification_gnb()
    label_svm = classification_svm()
    # plot_classificaton_result(label_gnb,'gnb')
    # plot_classificaton_result(label_svm,'svm')

    # plot_classificaton_result(label_gmm,'gmm')
    plot_classificaton_result(label_kmeans,'kmeans')

    pca = PCA(2)
    low_d_test = pca.fit_transform(test_data)
    
    # plot_pca_result(low_d_test,label_gmm,'GMM')
    plot_pca_result(low_d_test,label_kmeans,'Kmeans')
    # plot_pca_result(low_d_test,label_svm,'SVM',legends=['Death','Survival'])
    # plot_pca_result(low_d_test,label_gnb,'GNB',legends=['Death','Survival'])

    

if __name__ == '__main__':
    main()