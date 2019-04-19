from __future__ import division
import numpy as np
import pandas as pd 
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix

def get_data_features(file_name) :
    matrix = np.array([])
    path = os.getcwd() + "\\" + file_name
    data_frame = pd.read_csv(path, skiprows=[0,1,3],usecols=['DT','GR','LLD','NPHI'])
    data_frame = data_frame.astype(float)
    matrix = data_frame.values
    return matrix
def get_data_labels(file_name) :
    matrix = np.array([])
    path = os.getcwd() + "\\" + file_name
    data_frame = pd.read_csv(path, skiprows=[0,1,3],usecols=['FACIES'])
    data_frame = data_frame.astype(float)
    matrix = data_frame.values
    return matrix

def smooth_data(data) :
    for row in data :
        row[0] = row[0] // 2.5 * 2.5
        row[1] = row[1] // 2.5 * 2.5
        row[2] = row[2] // 0.5 * 0.5
    return data

if __name__ == "__main__" :
    data1 = get_data_features("\RUBY-1X_RUBY-1X.csv")
    data2 = get_data_features("\RUBY-2X_RUBY-2X.csv")
    data3 = get_data_features("\RUBY-3X_RUBY-3X.csv")
    data4 = get_data_features("\RUBY-4X_RUBY-4X.csv")
    data = np.vstack([data2,data3,data4,data1])
    data = smooth_data(data)

    labels_1 = get_data_labels("\RUBY-1X_RUBY-1X.csv")
    labels_2 = get_data_labels("\RUBY-2X_RUBY-2X.csv")
    labels_3 = get_data_labels("\RUBY-3X_RUBY-3X.csv")
    labels_4 = get_data_labels("\RUBY-4X_RUBY-4X.csv")
    labels = np.vstack([labels_2,labels_3,labels_4,labels_1])
    labels = labels.ravel()  #convert 2D array to 1D

    X_train,X_test,y_train,y_test = train_test_split(data,labels,shuffle=False,test_size=len(data1)/len(data))
    gnb = GaussianNB()
    gnb.fit(X_train,y_train)
    y_pred = gnb.predict(X_test)

    print X_train
    print y_train
    #print gnb.predict([[83.8900,52.4816,2.2845,0.2072]])
    print "accuracy :" ,metrics.accuracy_score(y_test,y_pred)
    #print "recall : " ,metrics.recall_score(y_test,y_pred,average='macro')
    #print "precision : ", metrics.precision_score(y_test,y_pred,average='macro')
    #print "f1 :" ,metrics.f1_score(y_test,y_pred,average='macro')
    print confusion_matrix(y_test,y_pred)
    print len(data)

    