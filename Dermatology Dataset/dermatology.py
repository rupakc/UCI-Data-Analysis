# -*- coding: utf-8 -*-
"""
Created on Sun Jan 03 17:08:14 2016
TODO - Find important features and plot the decision boundaries
@author: Rupak Chakraborty
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from sklearn import metrics

filename = "Dermatology Dataset/dermatology.data"
dataFrame = pd.read_csv(filename,header=None)
dataFrame = dataFrame.iloc[np.random.permutation(len(dataFrame))]
class_labels = dataFrame[(len(dataFrame.columns)-1)]
del dataFrame[(len(dataFrame.columns)-1)]

train_data,test_data,train_labels,test_labels = cross_validation.train_test_split(dataFrame,class_labels,test_size=0.2)

rf = RandomForestClassifier(n_estimators = 101)
ada = AdaBoostClassifier(n_estimators = 101)
grad_boost = GradientBoostingClassifier(n_estimators=101)
bagging = BaggingClassifier(n_estimators=101)
svm = SVC()
knn = KNeighborsClassifier(n_neighbors=7)

classifiers = [rf,ada,grad_boost,bagging,svm,knn]
classifier_names = ["Random Forest","AdaBoost","Gradient Boost","Bagging","SVC(rbf)","KNN (k=7)"]

for classifier,classifier_name in zip(classifiers,classifier_names):
    
    classifier.fit(train_data,train_labels)
    predicted_labels = classifier.predict(test_data)
    
    print ("--------------------------------------\n")
    print "Accuracy for Classifier ",classifier_name," : ",metrics.accuracy_score(test_labels,predicted_labels)
    print "Confusion Matrix for ",classifier_name, " :\n ",metrics.confusion_matrix(test_labels,predicted_labels)
    print "Classification Report for ",classifier_name, " :\n ",metrics.classification_report(test_labels,predicted_labels)
    print "----------------------------------------\n"
