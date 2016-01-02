# -*- coding: utf-8 -*-
"""
Created on Sat Jan 02 19:29:01 2016
Benchmarking of Forest Type Prediction Dataset on UCI Repository
@author: Rupak Chakraborty
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis 
from sklearn.svm import SVC
from sklearn import metrics 

test_file = "Forest Type Mapping/testing.csv"
train_file = "Forest Type Mapping/training.csv"
test_frame = pd.read_csv(test_file)
train_frame = pd.read_csv(train_file)
forest_map = {"s":0,"h":1,"d":2,"o":3}

test_frame["class"] = test_frame["class"].str.strip()
train_frame["class"] = train_frame["class"].str.strip()

test_frame["class"] = map(lambda x:forest_map[x],test_frame["class"])
train_frame["class"] = map(lambda x:forest_map[x],train_frame["class"])

train_labels = np.array(train_frame["class"].values)
test_labels = np.array(test_frame["class"].values)

del train_frame["class"]
del test_frame["class"]

train_data = np.array(train_frame.values)
test_data = np.array(test_frame.values) 

rf = RandomForestClassifier(n_estimators=101)
ada = AdaBoostClassifier(n_estimators=101)
bagging = BaggingClassifier(n_estimators=101)
grad_boost = GradientBoostingClassifier(n_estimators=101)
gnb = GaussianNB()
knn = KNeighborsClassifier(n_neighbors=7)
qda = QuadraticDiscriminantAnalysis()
svm = SVC()

classifiers = [rf,ada,bagging,grad_boost,gnb,knn,qda,svm]
classifier_names = ["Random Forests","AdaBoost","Bagging","Gradient Boosting"\
,"Gaussian NB","KNN(k=7)","Quadratic Discriminant Analysis","SVM (rbf)"]

for classifier,classifier_name in zip(classifiers,classifier_names): 
    
    classifier.fit(train_data,train_labels)
    predicted_labels = classifier.predict(test_data) 
    
    print "------------------------------------------------\n"
    print "Accuracy for ",classifier_name," : ",metrics.accuracy_score(test_labels,predicted_labels)
    print "Confusion Matrix for ",classifier_name," :\n ",metrics.confusion_matrix(test_labels,predicted_labels)
    print "Classification Report for ",classifier_name, " :\n ",metrics.classification_report(test_labels,predicted_labels)
    print "--------------------------------------------------\n"