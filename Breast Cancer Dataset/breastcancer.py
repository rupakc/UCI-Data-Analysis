# -*- coding: utf-8 -*-
"""
Created on Sat Jan 02 13:54:19 2016
Analysis of the wisconsin breast cancer dataset
@author: Rupak Chakraborty
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier 
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import Perceptron 
from sklearn import metrics
from sklearn import cross_validation 

filename = "Winsconsin Breast Cancer/breast-cancer-wisconsin.data"
cancer_data = pd.read_csv(filename)
column_names = ["Sample Code Number","Clump Thickness","Uniformity of Cell Size"\
,"Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size"\
,"Bare Nuclei","Bland Chromatin","Normal Nuclei","Mitoses","Class Label"]
cancer_data.columns = column_names
del cancer_data["Sample Code Number"]
cancer_data = cancer_data.iloc[np.random.permutation(len(cancer_data))]
class_labels = cancer_data["Class Label"]
del cancer_data["Class Label"] 

train_data,test_data,train_labels,test_labels = cross_validation.train_test_split(cancer_data,class_labels,test_size=0.3)

#Initializing classifiers 

rf = RandomForestClassifier(n_estimators=101)
ada = AdaBoostClassifier(n_estimators=101)
bagging = BaggingClassifier(n_estimators=101)
grad_boost = GradientBoostingClassifier(n_estimators=101)
mnb = MultinomialNB()
gnb = GaussianNB()
bnb = BernoulliNB()
brm = BernoulliRBM()
percept = Perceptron()
svm = SVC()
knn = KNeighborsClassifier(n_neighbors=5)
radnn = RadiusNeighborsClassifier(radius=10.3)

classifiers = [rf,ada,bagging,grad_boost,mnb,gnb,bnb,percept,svm,knn,radnn]
classifier_names = ["Random Forests","Adaboost","Bagging","Gradient Boost","Multinomial NB"\
,"Gaussian NB","Bernoulli NB","Perceptron","SVM (RBF)","KNN (K=5)","RadiusNN(r=10.3)"]

for classifier,classifier_name in zip(classifiers,classifier_names):
    
    classifier.fit(train_data.values,train_labels.values)
    predicted_labels = classifier.predict(test_data.values) 
    
    print "-------------------------------------\n"
    print "Accuracy for ",classifier_name," : ",metrics.accuracy_score(test_labels.values,predicted_labels)
    print "Confusion Matrix for ",classifier_name," : \n",metrics.confusion_matrix(test_labels.values,predicted_labels)
    print "Classification Report for ",classifier_name," : \n",metrics.classification_report(test_labels.values,predicted_labels)
    print cross_validation.cross_val_score(classifier,cancer_data.values,class_labels.values,cv=5)
    print "-------------------------------------\n"

