# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 22:17:40 2017
A baseline solution for the Hill Valley data set from UCI archive
@author: Rupak Chakraborty
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pandas as pd
from sklearn import metrics

rf = RandomForestClassifier(n_estimators=101)
ada = AdaBoostClassifier()
grad = GradientBoostingClassifier()
lr = LogisticRegression()
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()
knn = KNeighborsClassifier(n_neighbors=5)
svm = SVC()

classifiers = [rf,ada,grad,lr,gnb,mnb,bnb,knn,svm]
classifier_names = ['Random Forests','Adaboost','Gradient Boost','Logistic Regression', \
'Gaussian NB','Multinomial NB','Bernoulli NB','KNN','SVM (Rbf)']

hill_test_frame = pd.read_csv('Hill_Valley_without_noise_Training.data')
hill_train_frame = pd.read_csv('Hill_Valley_without_noise_Testing.data')

y_train = hill_train_frame['class'].values
y_test = hill_test_frame['class'].values

del hill_train_frame['class']
del hill_test_frame['class']

X_train = hill_train_frame.values
X_test = hill_test_frame.values

for classifier,classifier_name in zip(classifiers,classifier_names):
    classifier.fit(X_train,y_train)
    predicted_values = classifier.predict(X_test)
    print 'For ',classifier_name,' accuracy is - ',metrics.accuracy_score(predicted_values,y_test)
