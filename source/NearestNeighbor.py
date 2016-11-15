__author__ = 'deepak'

import numpy as np
import csv

class NearestNeighbor:
    def __init__(self):
        pass

    def train(self,X,y):
        self.Xtr = X
        self.Ytr = y

    def predict(self,X):
        num_test = X.shape
        Ypred = np.zeros(40);

        for i in range(40):
            distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]),axis=1))
            min_index = np.argmin(distances)
            Ypred[i] = self.Ytr[min_index]


        return  Ypred

    def loadData(self,location):

        data = []
        labels = []
        csv_data = csv.reader(open(location,"r"),delimiter=",")
        index = 0
        for row in csv_data:
            index=index+1
            if index==1:
                continue
            labels.append(int(row[0]))
            row = row[1:]
            data.append(np.array(np.int64(row)))

        return (np.mat(data),labels)

if __name__ == "__main__":
    nn = NearestNeighbor()
    Xtr,Ytr = nn.loadData("/home/deepak/Documents/kaggle/digit recognizer/train.csv")
    Xte,Yte = nn.loadData("/home/deepak/Documents/kaggle/digit recognizer/test1.csv")
    Xtr_rows = Xtr.reshape(Xtr.shape[0],28*28)
    Xte_rows = Xte.reshape(Xte.shape[0],28*28)
    nn.train(Xtr_rows,Ytr)
    Yte_predict = nn.predict(Xte_rows)
    for i in range(40):
        print Yte[i],"=",Yte_predict[i]
    print "accuracy: %f" % (np.mean(Yte_predict==Yte))




