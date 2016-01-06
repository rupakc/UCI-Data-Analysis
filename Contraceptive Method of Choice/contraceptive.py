# -*- coding: utf-8 -*-
"""
Created on Wed Jan 06 07:24:53 2016
Predicting the use of contraceptive from survey data the details are as follows:-
This dataset is a subset of the 1987 National Indonesia Contraceptive
Prevalence Survey. The samples are married women who were either not 
pregnant or do not know if they were at the time of interview. The 
problem is to predict the current contraceptive method choice 
(no use, long-term methods, or short-term methods) of a woman based 
on her demographic and socio-economic characteristics.

This is the baseline providing just higher than 50% accuracy on the data

@author: Rupak Chakraborty
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn import cross_validation

filename = "Contraceptive Method of Choice/cmc.data" # Replace this with your own filename 

contraFrame = pd.read_csv(filename,header=None)
column_names = ["Wife's age","Wife's education","Husband's education","Number of children born" \
,"Wife's religion","Wife's now working","Husband's occupation","Standard of living" \
,"Media exposure","Contraceptive Method Used"]

contraFrame.columns = column_names
contraFrame = contraFrame.iloc[np.random.permutation(len(contraFrame))]
target_label = contraFrame["Contraceptive Method Used"]
del contraFrame["Contraceptive Method Used"]

train_data,test_data,train_labels,test_labels = cross_validation.train_test_split(contraFrame.values,target_label.values,test_size=0.1)

rf = RandomForestClassifier(n_estimators=101)
bagging = BaggingClassifier(n_estimators=101)
ada = AdaBoostClassifier(n_estimators=101)
grad_boost = GradientBoostingClassifier(n_estimators=101)
extra_trees = ExtraTreesClassifier(n_estimators=101)
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()
percept = Perceptron()
knn = KNeighborsClassifier(n_neighbors=5)
lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()

classifier_names = ["Random Forest","Bagging","Adaboost","Gradient Boost","Extra Trees","Gaussian NB","Multinomial NB","Bernoulli NB","Perceptron","KNN (k=5)","Linear Discriminant Analysis","Quadratic Discriminant Analysis"]

classifiers = [rf,bagging,ada,grad_boost,extra_trees,gnb,mnb,bnb,percept,knn,lda,qda] 

for classifier,classifier_name in zip(classifiers,classifier_names):
    
    classifier.fit(train_data,train_labels)
    predicted_labels = classifier.predict(test_data)
    print "-----------------------------\n"
    print "Accuracy for ",classifier_name," : ",metrics.accuracy_score(test_labels,predicted_labels)
    print "Confusion Matrix for ",classifier_name," : \n",metrics.confusion_matrix(test_labels,predicted_labels)
    print "Cross Validation Score for ",classifier_name," : \n",cross_validation.cross_val_score(classifier,contraFrame.values,target_label.values,cv=5)
    print "----------------------------\n"
