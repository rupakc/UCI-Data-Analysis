# -*- coding: utf-8 -*-
"""
Created on Wed Jan 06 21:52:50 2016
Fun with the badges dataset from UCI Archives
@author: Rupak Chakraborty
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier 
 
from sklearn import metrics
from sklearn import cross_validation
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class_label_map = {"+":1,"-":0}
filename = "Badges/badges.data"
f = open(filename,'r')
data_content = f.read()
data_list = data_content.split('\n')
sign_list = []
name_list = []

def convertListToString(name_list):
    s = ""
    for name in name_list:
        s = s + name.lower().strip()
        s = s.replace(".","")
        s = s.replace("-","")
        s = s.replace("'","")
    return s
    
for data in data_list:
    temp_list = data.split(' ')
    sign_list.append(temp_list[0].strip())
    name_list.append(convertListToString(temp_list[1:]))

class_labels = map(lambda x:class_label_map[x],sign_list)
feature_set = np.zeros((len(class_labels),26)) 

c = 0
for name in name_list:
    for character in name:
        feature_set[c][ord(character)-ord('a')] = feature_set[c][ord(character)-ord('a')] + 1.0
    c = c + 1

train_data,test_data,train_labels,test_labels = cross_validation.train_test_split(feature_set,class_labels,test_size=0.2)

rf = RandomForestClassifier(n_estimators=101)
ada = AdaBoostClassifier(n_estimators=101)
grad_boost = GradientBoostingClassifier(n_estimators=101)
bagging = BaggingClassifier(n_estimators=101)
svm = SVC(kernel='rbf')
knn = KNeighborsClassifier(n_neighbors=5)

classifiers = [rf,ada,grad_boost,bagging,svm,knn]
classifier_names = ["Random Forest","AdaBoost","Gradient Boost","Bagging","SVM","KNN"]

for classifier,classifier_name in zip(classifiers,classifier_names):
    classifier.fit(train_data,train_labels)
    predicted_labels = classifier.predict(test_data) 
    
    print "--------------------------------\n"
    print "Accuracy for ",classifier_name," : ",metrics.accuracy_score(test_labels,predicted_labels)
    print "Confusion Matrix for ",classifier_name, ":\n",metrics.confusion_matrix(test_labels,predicted_labels)
    print "Classification Report for ",classifier_name, ":\n", metrics.classification_report(test_labels,predicted_labels)
    print "--------------------------------\n"

def plotTSNEDecisionBoundaries(): 
    
    tsne = TSNE()
    tsne_data = tsne.fit_transform(feature_set)
    x_min,x_max = tsne_data[:,0].min()-1, tsne_data[:,0].max() + 1
    y_min,y_max = tsne_data[:,1].min()-1, tsne_data[:,1].max() + 1
    step_size = 0.2
    
    xx,yy = np.meshgrid(np.arange(x_min,x_max,step_size),np.arange(y_min,y_max,step_size))
    
    for index,classifier in enumerate(classifiers):
        
        plt.subplot(2,3,index+1)
        plt.subplots_adjust(wspace=0.5,hspace=0.5)
        classifier.fit(tsne_data,class_labels)
        
        Z = classifier.predict(zip(xx.ravel(),yy.ravel()))
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx,yy,Z,cmap=plt.cm.Paired,alpha=0.7)
        plt.scatter(tsne_data[:,0],tsne_data[:,1],c=class_labels,cmap=plt.cm.rainbow,alpha=0.6)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.xlim(x_min,x_max)
        plt.ylim(y_min,y_max)
        plt.xticks(())
        plt.yticks(())
        plt.title(classifier_names[index])
        
    plt.show()
        
        
