# -*- coding: utf-8 -*-
"""
Created on Fri Jan 01 14:05:51 2016
Benchmarking of the results on online News Popularity as reported in :- 

K. Fernandes, P. Vinagre and P. Cortez. A Proactive Intelligent Decision
Support System for Predicting the Popularity of Online News. Proceedings
of the 17th EPIA 2015 - Portuguese Conference on Artificial Intelligence,
September,Coimbra, Portugal. 

@author: Rupak Chakraborty
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler 
from sklearn import cross_validation
from sklearn import metrics
import pandas as pd
import numpy as np 

filename = "Online News Popularity/OnlineNewsPopularity.csv"
dataFrame = pd.read_csv(filename)
class_label_map = {True:1,False:0}
dataFrame.columns = map(lambda x: x.strip(),dataFrame.columns)
del dataFrame["url"]
dataFrame = dataFrame.iloc[np.random.permutation(len(dataFrame))]
dataFrame["shares"] = map(lambda x: class_label_map[x],map(lambda x: x >= 1400,dataFrame["shares"]))
class_labels = dataFrame["shares"].values
del dataFrame["shares"]

standard_scaler = StandardScaler()
min_max_scaler = MinMaxScaler()
standard_data = standard_scaler.fit_transform(dataFrame.values)
min_max_data = min_max_scaler.fit_transform(dataFrame.values)
unprocessed_data = dataFrame.values 

train_standard,test_standard,train_label,test_label = cross_validation.train_test_split(standard_data,class_labels,test_size=0.3)
train_min_max,test_min_max,train_label,test_label = cross_validation.train_test_split(min_max_data,class_labels,test_size=0.3)
train_raw,test_raw,train_label,test_label = cross_validation.train_test_split(unprocessed_data,class_labels,test_size=0.3)

processed_train_data = [train_raw,train_standard,train_min_max]
processed_test_data = [test_raw,test_standard,test_min_max]
processed_data_names = ["Raw Data","Standardization","Min-Max Scaling"]

rf = RandomForestClassifier(n_estimators=71)
ada = AdaBoostClassifier(n_estimators=101)
gnb = GaussianNB()
knn = KNeighborsClassifier(n_neighbors=7)

classifier_list = [rf,ada,gnb,knn]
classifier_name_list = ["Random Forest","AdaBoost","Gaussian NB","KNN(K=7)"]

for train_data,test_data,preprocess_name in zip(processed_train_data,processed_test_data,processed_data_names): 
    
    print "-----------------------------------------------------\n"
    print "For Data Preprocessing by : ",preprocess_name
    print "-----------------------------------------------------\n" 
    
    for classifier,classifier_name in zip(classifier_list,classifier_name_list):
        
        classifier.fit(train_data,train_label)
        predicted_label = classifier.predict(test_data)
        print "-----------------------------------------------------\n"
        print "Accuracy for ",classifier_name," : ",metrics.accuracy_score(test_label,predicted_label)
        print "-----------------------------------------------------\n"
        
    
 


