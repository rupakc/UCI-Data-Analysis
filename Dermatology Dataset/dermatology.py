# -*- coding: utf-8 -*-
"""
Created on Sun Jan 03 17:08:14 2016
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
import matplotlib.pyplot as plt

filename = "Dermatology Dataset/dermatology.data"
step_size = 0.02 

dataFrame = pd.read_csv(filename,header=None)
dataFrame = dataFrame.iloc[np.random.permutation(len(dataFrame))]
class_labels = dataFrame[(len(dataFrame.columns)-1)]
del dataFrame[(len(dataFrame.columns)-1)]

train_data,test_data,train_labels,test_labels = cross_validation.train_test_split(dataFrame.values,class_labels.values,test_size=0.2)

rf = RandomForestClassifier(n_estimators = 101)
ada = AdaBoostClassifier(n_estimators = 101)
grad_boost = GradientBoostingClassifier(n_estimators=101)
bagging = BaggingClassifier(n_estimators=101)
svm = SVC()
knn = KNeighborsClassifier(n_neighbors=7)

classifiers = [rf,ada,grad_boost,bagging,svm,knn]
classifier_names = ["Random Forest","AdaBoost","Gradient Boost","Bagging","SVC(rbf)","KNN (k=7)"]
feature_weights = []

for classifier,classifier_name in zip(classifiers,classifier_names):
    
    classifier.fit(train_data,train_labels)
    predicted_labels = classifier.predict(test_data)
    if classifier_name == "Random Forest":
        feature_weights = classifier.feature_importances_
    print ("--------------------------------------\n")
    print "Accuracy for Classifier ",classifier_name," : ",metrics.accuracy_score(test_labels,predicted_labels)
    print "Confusion Matrix for ",classifier_name, " :\n ",metrics.confusion_matrix(test_labels,predicted_labels)
    print "Classification Report for ",classifier_name, " :\n ",metrics.classification_report(test_labels,predicted_labels)
    print "----------------------------------------\n"

print "Weights of the selected features : \n", feature_weights 

#Plotting the decision boundaries of each classifier 

dataFrame = dataFrame[[19,21]]
train_data,test_data,train_labels,test_labels = cross_validation.train_test_split(dataFrame.values,class_labels.values,test_size=0.2) 

x_min,x_max = train_data[:,0].min()-1,train_data[:,0].max()+1
y_min,y_max = train_data[:,1].min()-1,train_data[:,1].max()+1
xx,yy = np.meshgrid(np.arange(x_min,x_max,step_size),np.arange(y_min,y_max,step_size))

for index,classifier in enumerate(classifiers):
    
    classifier.fit(train_data,train_labels)
    Z = classifier.predict(zip(xx.ravel(),yy.ravel()))
    Z = Z.reshape(xx.shape)
    
    plt.subplot(2,3,index+1)
    plt.subplots_adjust(wspace=0.5,hspace=0.5)
    plt.contourf(xx,yy,Z,cmap=plt.cm.CMRmap,alpha=0.6)
    plt.scatter(train_data[:,0],train_data[:,1],c=train_labels,cmap=plt.cm.gist_rainbow,alpha=0.7)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.xlim(xx.min(),xx.max())
    plt.ylim(yy.min(),yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(classifier_names[index])
    
plt.show()

