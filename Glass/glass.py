# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 21:34:38 2017
A sample approach to the UCI glass dataset
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
from sklearn.model_selection import train_test_split

column_names = ['Id','RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Type']
glass_frame = pd.read_csv('glass.data',header=None)
glass_frame.columns = column_names
class_labels = glass_frame['Type'].values
del glass_frame['Id']
del glass_frame['Type']
X_train,X_test,y_train,y_test = train_test_split(glass_frame.values,class_labels,test_size=0.2,random_state=42)

rf = RandomForestClassifier(n_estimators=51)
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

for classifier,classifier_name in zip(classifiers,classifier_names):
    classifier.fit(X_train,y_train)
    predicted_values = classifier.predict(X_test)
    print 'For ',classifier_name,' accuracy is - ',metrics.accuracy_score(predicted_values,y_test)
