# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 23:26:01 2016
Benchmark for Activity Recognition using Smartphone accelerometer,gyrometer data
The dataset has been taken from the following paper:- 

Jorge L. ReyesOrtiz(1,2), Davide Anguita(1), Luca Oneto(1) and Xavier Parra(2)
1 Smartlab,DIBRIS UniversitÃ degli Studi di Genova, Genoa (16145), Italy CETpD Universitat
PolitÃ¨cnica de Catalunya. Vilanova i la GeltrÃº (08800), Spainhar '@' smartlab.wswww.smartlab.ws

@author: Rupak Chakraborty
"""
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
import time

train_file_name = "Smart Phone Activity Recognition/HAPT Data Set/Train/X_train.txt"
train_label_file = "Smart Phone Activity Recognition/HAPT Data Set/Train/y_train.txt"
test_file_name = "Smart Phone Activity Recognition/HAPT Data Set/Test/X_test.txt"
test_label_file = "Smart Phone Activity Recognition/HAPT Data Set/Test/y_test.txt"

train_data = pd.read_csv(train_file_name,sep=" ",header=None)
train_labels = pd.read_csv(train_label_file,sep=" ",header=None)
test_data = pd.read_csv(test_file_name,sep=" ",header=None)
test_labels = pd.read_csv(test_label_file,sep=" ",header=None)

rf = RandomForestClassifier(n_estimators=101)
bagging = BaggingClassifier(n_estimators=101)
extra = ExtraTreesClassifier(n_estimators=101)
grad = GradientBoostingClassifier(n_estimators=101)
ada = AdaBoostClassifier(n_estimators=101)
gnb = GaussianNB()

classifiers = [rf,extra,grad,ada,gnb,bagging]
classifier_names = ["Random Forest","Extra Trees","Grad Boost","AdaBoost","Gaussian NB","Bagging"]

for classifier,classifier_name in zip(classifiers,classifier_names):
    
    classifier.fit(train_data.values,train_labels.values)
    predicted_labels = classifier.predict(test_data.values)
    print "-----------------------------\n"
    print "Accuracy for ",classifier_name, " : ",metrics.accuracy_score(test_labels.values,predicted_labels)
    print "-----------------------------\n"