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


if __name__ == "__main__":
    pass
