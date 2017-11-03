# -*- coding: utf-8 -*-
"""
Created on Sat Jan 02 13:54:19 2016
Analysis of the wisconsin breast cancer dataset
@author: Rupak Chakraborty
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

filename = "Winsconsin Breast Cancer/breast-cancer-wisconsin.data"
cancer_data = pd.read_csv(filename)
step_size = 0.02
column_names = ["Sample Code Number","Clump Thickness","Uniformity of Cell Size"
,"Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size"
,"Bare Nuclei","Bland Chromatin","Normal Nuclei","Mitoses","Class Label"]
cancer_data.columns = column_names
del cancer_data["Sample Code Number"]
cancer_data = cancer_data.iloc[np.random.permutation(len(cancer_data))]
class_labels = cancer_data["Class Label"]
del cancer_data["Class Label"] 

train_data,test_data,train_labels,test_labels = train_test_split(cancer_data,class_labels,test_size=0.3,random_state=42)

#Initializing classifiers 

for trees in range(10,50,5):
    rf = RandomForestClassifier(n_estimators=trees,oob_score=True,random_state=42,verbose=1)
    rf.fit(train_data,train_labels)
    predicted_values = rf.predict(test_data)
    print "For N Estimators = ",trees
    print metrics.classification_report(test_labels,predicted_values)
    print metrics.accuracy_score(test_labels,predicted_values)
    print "===============================================\n"


# mnb = MultinomialNB()
# gnb = GaussianNB()
# bnb = BernoulliNB()
# svm = SVC(random_state=42)
#
#
# classifiers = [rf,mnb,gnb,bnb,svm]
# classifier_names = ["Random Forests","Multinomial NB" ,"Gaussian NB","Bernoulli NB","SVM (RBF)"]
#
# for classifier,classifier_name in zip(classifiers,classifier_names):
#
#     classifier.fit(train_data.values,train_labels.values)
#     predicted_labels = classifier.predict(test_data.values)
#
#     print "-------------------------------------\n"
#     print "Accuracy for ",classifier_name," : ",metrics.accuracy_score(test_labels.values,predicted_labels)
#     print "Confusion Matrix for ",classifier_name," : \n",metrics.confusion_matrix(test_labels.values,predicted_labels)
#     print "Classification Report for ",classifier_name," : \n",metrics.classification_report(test_labels.values,predicted_labels)
#     print "Cross Validation Score for ",classifier_name,":\n",cross_val_score(classifier,cancer_data.values,class_labels.values,cv=5)
#     if classifier_name == 'Random Forests':
#         feature_importances = rf.feature_importances_
#         for index,value in enumerate(feature_importances):
#             print column_names[index], " => ", value
#     print "-------------------------------------\n"
#
#
dataFrame = cancer_data[['Clump Thickness','Uniformity of Cell Size']]
train_data, test_data, train_labels, test_labels = train_test_split(dataFrame.values, class_labels.values,
                                                                    test_size=0.2, random_state=42)

x_min, x_max = train_data[:, 0].min() - 1, train_data[:, 0].max() + 1
y_min, y_max = train_data[:, 1].min() - 1, train_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))


for index, classifier in enumerate([1,2,3,4,5,6,7,8]):
    rf.fit(train_data, train_labels)
    Z = rf.predict(zip(xx.ravel(), yy.ravel()))
    Z = Z.reshape(xx.shape)
    plt.subplot(4, 4, index + 1)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.contourf(xx, yy, Z, cmap=plt.cm.CMRmap, alpha=0.6)
    plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels, cmap=plt.cm.gist_rainbow, alpha=0.3)
    plt.xlabel("Clump Thickness")
    plt.ylabel("Uniformity of Cell Size")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(str(index))

plt.show()
