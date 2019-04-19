from __future__ import division
import numpy as np
import pandas as pd 
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
import math

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

def count_state(label,state) :
    count = 0 
    for i in range (len(label)) :
        if label[i] == state :
            count = count+1
    return count
def state_transition_probability(label,state1,state2) :
    count = 0
    for i in range (len(label)-1) :
        if label[i] == state1 :
            if label[i+1] == state2 :
                count = count + 1
    return count/count_state(label,state1)

def state_transition_matrix(label) :
    matrix = []
    for i in range(5) :
        matrix.append([state_transition_probability(label,i+1,j+1) for j in range(5)])
    return matrix

"""
def p_DT_and_s(data,state,value) :
    count = 0
    for row in data :
        if row[1] == state :
            if row[0] >= value // 2.5 * 2.5 :
                if row[0] < value // 2.5 * 2.5 + 2.5 :
                    count = count + 1
    return count/len(data)

def p_LLD_and_s(data,state,value) :
    count = 0
    for row in data :
        if row[1] == state :
            if row[3] >= value // 1 * 1 :
                if row[3] < value // 1 * 1 + 1 :
                    count = count + 1
    return count/len(data)

def p_NPHI_and_s(data,state,value) :
    count = 0
    for row in data :
        if row[1] == state :
            if row[4] >= value // 0.1 * 0.1 :
                if row[4] < value // 0.1 * 0.1 + 0.1 :
                    count = count + 1
    return count/len(data)

def p_GR_and_s(data,state,value) :
    count = 0
    for row in data :
        if row[1] == state :
            if row[2] >= value // 2.5 * 2.5 :
                if row[2] < value // 2.5 * 2.5 + 2.5 :
                    count = count + 1
    return count/len(data)

def p_o_and_s(data,state,value) :
    probability = p_DT_and_s(data,state,value[0]) * \
                    p_GR_and_s(data,state,value[2]) * \
                    p_LLD_and_s(data,state,value[3]) * \
                    p_NPHI_and_s(data,state,value[4])
    return probability

def p_DT(data,value) :
    count = 0
    for row in data :
        if row[0] >= value// 2.5 * 2.5 :
            if row[0] < value// 2.5 * 2.5 + 2.5 :
                count = count + 1 
    return count / len(data)

def p_GR(data,value) :
    count = 0
    for row in data :
        if row[2] >= value// 2.5 * 2.5 :
            if row[2] < value// 2.5 * 2.5 + 2.5 :
                count = count + 1 
    return count / len(data)

def p_LLD(data,value) :
    count = 0
    for row in data :
        if row[3] >= value// 1 * 1 :
            if row[3] < value// 1 * 1 + 1 :
                count = count + 1 
    return count / len(data)

def p_NPHI(data,value) :
    count = 0
    for row in data :
        if row[4] >= value// 0.1 * 0.1 :
            if row[4] < value// 0.1 * 0.1 + 0.1 :
                count = count + 1 
    return count / len(data)            
    
def p_ob(data,value) :
    probability = p_DT(data,value[0]) * p_GR(data,value[2]) * \
                  p_LLD(data,value[3]) + p_NPHI(data,value[4])
    return probability

def likelihood(data,state,value) :
    return p_o_and_s(data,state,value) / p_ob(data,value)
"""
def add_hmm(prob, matrix):
    max_previous = 0
    for i in range(2,len(prob)) :
        max_previous = max(prob[i-1])
        if max_previous == prob[i-1][0] :
            for j in range(5) :
                    prob[i][j] = (prob[i][j] + (matrix[0][j]) ) /2
        elif max_previous == prob[i-1][1] :
            for j in range(5) :
                    prob[i][j] = (prob[i][j] + (matrix[1][j]) ) / 2
        elif max_previous == prob[i-1][2] :
            for j in range(5) :
                    prob[i][j] = ( prob[i][j] + (matrix[2][j]) ) / 2
        elif max_previous == prob[i-1][3] :
            for j in range(5) :
                    prob[i][j] = ( prob[i][j] + (matrix[3][j]) ) /2
        elif max_previous == prob[i-1][4] :
            for j in range(5) :
                    prob[i][j] = ( prob[i][j] + (matrix[4][j]) )/ 2
    return prob

def fit_again(prob,real_state) :
    count = 0
    for i in range (1,len(prob)) :
        m = max(prob[i])
        if prob[i][0] == m :
            if real_state[i-1] == 1 :
                count = count + 1
        if prob[i][1] == m :
            if real_state[i-1] == 2 :
                count = count + 1
        if prob[i][2] == m :
            if real_state[i-1] == 3 :
                count = count + 1  
        if prob[i][3] == m :
            if real_state[i-1] == 4 :
                count = count + 1      
    return count/(len(prob)-1)

def add_prob(prob_hmm,prob_nn) :
    prob = prob_nn
    for i in range(len(prob_hmm)) :
        for j in range(5) :
            prob[i][j] = (prob[i][j] + prob_hmm[i][j]) / 2
    return prob

def multiply_prob(prob_hmm,prob_nn) :
    prob = prob_nn
    for i in range(len(prob_hmm)) :
        for j in range(5) :
            prob[i][j] = prob[i][j] * prob_hmm[i][j]
    return prob

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
    gnb = GaussianNB()
    gnb.fit(X_train,y_train)
    y_pred = gnb.predict(X_test)

    #print X_train
    #print y_train
    #print "accuracy :" ,metrics.accuracy_score(y_test,y_pred)
    #print confusion_matrix(y_test,y_pred)
    
    matrix = np.array([0,0,0,0,0])
    matrix = np.vstack(state_transition_matrix(y_train))
    print matrix

    prob_nb = np.array([0,0,0,0,0])
    for x in X_test :
        m = (gnb.predict_proba([x]))
        prob_nb = np.vstack([prob_nb,m])
    print prob_nb[1:]
    prob_hmm = add_hmm(prob_nb,matrix)
    print prob_hmm[1:]
    print "accuracy hmm :" ,fit_again(prob_hmm,y_test)


    clf = MLPClassifier()
    clf.fit(X_train,y_train)
    prob_nn = np.array([0,0,0,0,0])
    for x in X_test :
        m = (clf.predict_proba([x]))
        prob_nn = np.vstack([prob_nn,m])
    print "probnn" ,prob_nn[1:]
    print fit_again(prob_nn,y_test)

    prob_avg = add_prob(prob_hmm,prob_nn)
    prob_mul = multiply_prob(prob_hmm,prob_nn)
    print prob_avg[1:]
    print fit_again(prob_avg,y_test) 
    
    #print 3.46548405e-03 +1.30491672e-01 +2.63319493e-01 +3.90686280e-01 +1.96406122e-04
    