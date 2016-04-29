# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 22:14:02 2016
Classification of leaf data set using sklearn toolkit
@author: Rupak Chakraborty
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from sklearn import metrics 

leaf_data_frame = pd.read_csv('leaf.csv')
column_names = ['Class','Specimen Num','Eccentricity','Aspect Ratio','Elongation','Solidity','Stochastic Convexity',
'Isometric Factor','Maximal Indentation Depth','Lobedness','Average Intensity','Average Contrast','Smoothness',
'Third Moment','Uniformity','Entropy']
leaf_data_frame.columns = column_names
Y = leaf_data_frame['Class'].values
del leaf_data_frame['Class']
X = leaf_data_frame.values

train_data,test_data,train_labels,test_labels = cross_validation.train_test_split(X,Y,test_size=0.2)

rf = RandomForestClassifier(n_estimators=101)
grad = GradientBoostingClassifier(n_estimators=151)
ada = AdaBoostClassifier(n_estimators=101)
bagging = BaggingClassifier(n_estimators=191)
lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()
bnb = BernoulliNB()
gnb = GaussianNB()
mnb = MultinomialNB()
knn = KNeighborsClassifier(n_neighbors=7)

classifiers = [rf,grad,ada,bagging,lda,qda,bnb,gnb,mnb,knn]
classifier_names = ['Random Forests','Gradient Boosting','AdaBoost','Bagging','LDA','QDA','BNB','GNB','MNB','KNN']

for classifier,classifier_name in zip(classifiers,classifier_names):
    classifier.fit(train_data,train_labels)
    predicted_labels = classifier.predict(test_data)
    print '----------- For Classifier ',classifier_name,' ------------\n',metrics.accuracy_score(test_labels,predicted_labels)
    print '----------------------------------------------------\n'
