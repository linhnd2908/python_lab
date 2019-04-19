from __future__ import division
import numpy as np
import pandas as pd 
import os
from sklearn.model_selection import train_test_split
import xgboost as xgb
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

if __name__ == "__main__" :
    data1 = get_data_features("\RUBY-1X_RUBY-1X.csv")
    data2 = get_data_features("\RUBY-2X_RUBY-2X.csv")
    data3 = get_data_features("\RUBY-3X_RUBY-3X.csv")
    data4 = get_data_features("\RUBY-4X_RUBY-4X.csv")
    data = np.vstack([data1,data3,data4,data2])

    labels_1 = get_data_labels("\RUBY-1X_RUBY-1X.csv")
    labels_2 = get_data_labels("\RUBY-2X_RUBY-2X.csv")
    labels_3 = get_data_labels("\RUBY-3X_RUBY-3X.csv")
    labels_4 = get_data_labels("\RUBY-4X_RUBY-4X.csv")
    labels = np.vstack([labels_1,labels_3,labels_4,labels_2])
    labels = labels.ravel()  #convert 2D array to 1D

    X_train,X_test,y_train,y_test = train_test_split(data,labels,shuffle=False,test_size=len(data2)/len(data))
    xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
    xg_reg.fit(X_train,y_train)
    y_pred = xg_reg.predict(X_test)
    print "accuracy :" ,metrics.accuracy_score(y_test,y_pred)