from __future__ import division
import math
import numpy as np
import pandas as pd 
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
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

def state_transition_probability(label,state1,state2) :
    count = 0
    for i in range (len(label)-1) :
        if label[i] == state1 :
            if label[i+1] == state2 :
                count = count + 1
    return count/(len(label))

def state_transition_matrix(label) :
    matrix = []
    for i in range(5) :
        matrix.append([state_transition_probability(label,i+1,j+1) for j in range(5)])
    return matrix

"""
def add_hmm(log_prob, matrix):
    max_previous = 0
    for i in range(2,len(log_prob)) :
        max_previous = max(log_prob[i-1])
        if max_previous == log_prob[i-1][0] :
            for j in range(5) :
                if matrix[0][j] != 0 :
                    log_prob[i][j] = log_prob[i][j] + math.log(matrix[0][j])
        elif max_previous == log_prob[i-1][1] :
            for j in range(5) :
                if matrix[1][j] != 0 :
                    log_prob[i][j] = log_prob[i][j] + math.log(matrix[1][j])
        elif max_previous == log_prob[i-1][2] :
            for j in range(5) :
                if matrix[2][j] != 0 :
                    log_prob[i][j] = log_prob[i][j] + math.log(matrix[2][j])
        elif max_previous == log_prob[i-1][3] :
            for j in range(5) :
                if matrix[3][j] != 0 :
                    log_prob[i][j] = log_prob[i][j] + math.log(matrix[3][j])
        elif max_previous == log_prob[i-1][4] :
            for j in range(5) :
                if matrix[4][j] != 0 :
                    log_prob[i][j] = log_prob[i][j] + math.log(matrix[4][j])
    return log_prob
"""
def fit_again(log_prob,real_state) :
    count = 0
    for i in range (1,len(log_prob)) :
        m = max(log_prob[i])
        if log_prob[i][0] == m :
            if real_state[i-1] == 1 :
                count = count + 1
        if log_prob[i][1] == m :
            if real_state[i-1] == 2 :
                count = count + 1
        if log_prob[i][2] == m :
            if real_state[i-1] == 3 :
                count = count + 1  
        if log_prob[i][3] == m :
            if real_state[i-1] == 4 :
                count = count + 1      
    return count/(len(log_prob)-1)

def log_prob(X_test,clf) :
    log_prob = np.array([0,0,0,0,0])
    for x in X_test :
        m = (clf.predict_log_proba([x]))
        log_prob = np.vstack([log_prob,m])
    return log_prob


if __name__ == "__main__" :
    data1 = get_data_features("\RUBY-1X_RUBY-1X.csv")
    data2 = get_data_features("\RUBY-2X_RUBY-2X.csv")
    data3 = get_data_features("\RUBY-3X_RUBY-3X.csv")
    data4 = get_data_features("\RUBY-4X_RUBY-4X.csv")
    data = np.vstack([data2,data3,data4,data1])

    labels_1 = get_data_labels("\RUBY-1X_RUBY-1X.csv")
    labels_2 = get_data_labels("\RUBY-2X_RUBY-2X.csv")
    labels_3 = get_data_labels("\RUBY-3X_RUBY-3X.csv")
    labels_4 = get_data_labels("\RUBY-4X_RUBY-4X.csv")
    labels = np.vstack([labels_2,labels_3,labels_4,labels_1])
    labels = labels.ravel()  #convert 2D array to 1D

    X_train,X_test,y_train,y_test = train_test_split(data,labels,shuffle=False,test_size=len(data1)/len(data))
    clf = MLPClassifier(activation="tanh")
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print "accuracy :" ,metrics.accuracy_score(y_test,y_pred)
    #print "recall : " ,metrics.recall_score(y_test,y_pred,average='macro')
    #print "precision : ", metrics.precision_score(y_test,y_pred,average='macro')
    #print "f1 :" ,metrics.f1_score(y_test,y_pred,average='macro')
    print confusion_matrix(y_test,y_pred)
    prob = np.array([0,0,0,0,0])
    for x in X_test :
        m = (clf.predict_proba([x]))
        prob = np.vstack([prob,m])
    print prob
    #print log_prob(X_test,clf)
    print state_transition_matrix(y_train)
    #print state_transition_matrix(y_train)
    #print log_prob[3][3]
    