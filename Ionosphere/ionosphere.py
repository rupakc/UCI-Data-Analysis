# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 23:22:40 2017
Ensemble methods to try out on the ionosphere data from UCI archives
@author: Rupak Chakraborty
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

rf = RandomForestClassifier(n_estimators=101)
ada = AdaBoostClassifier()
grad = GradientBoostingClassifier()
lr = LogisticRegression()
gnb = GaussianNB()
knn = KNeighborsClassifier(n_neighbors=5)
svm = SVC()

classifiers = [rf,ada,grad,lr,gnb,knn,svm]
classifier_names = ['Random Forests','Adaboost','Gradient Boost','Logistic Regression', \
'Gaussian NB','KNN','SVM (Rbf)']

ion_frame = pd.read_csv('ionosphere.data',header=None)
class_labels = ion_frame[len(ion_frame.columns)-1].values
del ion_frame[len(ion_frame.columns)-1]

X_train,X_test,y_train,y_test = train_test_split(ion_frame.values,class_labels,test_size=0.5,random_state=42)

for classifier,classifier_name in zip(classifiers,classifier_names):
    classifier.fit(X_train,y_train)
    predicted_values = classifier.predict(X_test)
    print 'For ',classifier_name,' accuracy is - ',metrics.accuracy_score(predicted_values,y_test)