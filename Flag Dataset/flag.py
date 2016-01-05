# -*- coding: utf-8 -*-
"""
Created on Tue Jan 05 17:11:19 2016
Using the flag of a country to predict its religion 
Mostly for fun there are certain visualizations involved here
@author: Rupak Chakraborty
"""

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import cross_validation
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 

filename = "Flag/flag.data" 
flagFrame = pd.read_csv(filename,header=None) 
color_map = {"orange":0,"black":1,"white":2,"blue":3,"green":4,"brown":5,"red":6,"gold":7} 
column_names = ["name","landmass","zone","area","population","language","religion","bars","stripes","colours","red","green","blue","gold","white","black","orange","mainhue","circles","crosses","saltires","quarters","sunstars","crescent","triangle","icon","animate","text","topleft","botright"]

# Adding the column names and shuffling the dataset 

flagFrame.columns = column_names
flagFrame = flagFrame.iloc[np.random.permutation(len(flagFrame))]
nominal_columns = ["mainhue","topleft","botright"]

country_names = flagFrame["name"]
del flagFrame["name"]
religion_labels = flagFrame["religion"]
del flagFrame["religion"]

# Converting nominal values into numeric ones 

for column in nominal_columns:
    flagFrame[column] = map(lambda x:color_map[x.strip()],flagFrame[column])

# Reducing the dimensionality of the data

pca = PCA(n_components=25)
kpca = KernelPCA(n_components=25,kernel='rbf')
trunc_svd = TruncatedSVD(n_components=25)
tsne = TSNE() 

pca_data = pca.fit_transform(flagFrame.values)
kpca_data = kpca.fit_transform(flagFrame.values)
svd_data = trunc_svd.fit_transform(flagFrame.values)
raw_data = flagFrame.values 
tsne_data = tsne.fit_transform(flagFrame.values)

data_list = [pca_data,kpca_data,svd_data,raw_data]
data_list_names = ["PCA","KPCA (Kernel=rbf)","SVD","Raw Data"] 

rf = RandomForestClassifier(n_estimators=101)
ada = AdaBoostClassifier(n_estimators=101)
grad_boost = GradientBoostingClassifier(n_estimators=101)
bagging = BaggingClassifier(n_estimators=101)
gnb = GaussianNB()
bnb = BernoulliNB()
percept = Perceptron()
logit = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=5)

classifiers = [rf,ada,grad_boost,bagging,gnb,mnb,bnb,percept,logit,knn]

classifier_names = ["Random Forest","AdaBoost","Gradient Boost","Bagging","Gaussian NB"\
,"Bernoulli NB","Perceptron","Logistic Regression","KNN (k=5)"]

# Iterating over classifiers to analyze the performance metrics 

for data,data_reduced in zip(data_list,data_list_names):
    
    print "For Data Dimensionality Reduction Using ",data_reduced
    
    train_data,test_data,train_labels,test_labels = cross_validation.train_test_split(data,religion_labels.values,test_size=0.3)
    
    for classifier,classifier_name in zip(classifiers,classifier_names):
        
        try:
            classifier.fit(train_data,train_labels)
            predicted_labels = classifier.predict(test_data)
        except:
            pass 
        
        print "--------------------------------\n"
        print "Accuracy for ",classifier_name," : ",metrics.accuracy_score(test_labels,predicted_labels)
        print "Confusion Matrix for ",classifier_name," :\n ",metrics.confusion_matrix(test_labels,predicted_labels)
        print "Classification Report for ",classifier_name, ":\n",metrics.classification_report(test_labels,predicted_labels)
        print "--------------------------------\n"
        print "Cross Validation Score for ",classifier_name," : \n",cross_validation.cross_val_score(estimator=classifier,X=flagFrame.values,y=religion_labels.values,cv=5)
        print "--------------------------------\n"

#Using t-SNE for visualization of the dataset and decision boundaries

def t_SNEVisualization():
        
    x_min,x_max = tsne_data[:,0].min()-1,tsne_data[:,0].max()+1
    y_min,y_max = tsne_data[:,1].min()-1,tsne_data[:,1].max()+1
    step_size = 0.02
    
    xx,yy = np.meshgrid(np.arange(x_min,x_max,step_size),np.arange(y_min,y_max,step_size))
    
    for index,classifier in enumerate(classifiers):
            
            try:
                classifier.fit(tsne_data,train_labels)
                predicted_labels = classifier.predict(zip(xx.ravel(),yy.ravel()))
                predicted_labels = predicted_labels.reshape(xx.shape)
            except:
                pass 
            
            plt.subplot(3,3,index+1)
            plt.subplots_adjust(wspace=0.5,hspace=0.5)
            plt.contourf(xx,yy,predicted_labels,cmap=plt.cm.Paired,alpha=0.6)
            plt.scatter(tsne_data[:,0],tsne_data[:,1],c=religion_labels.values,cmap=plt.cm.rainbow)
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.xticks(())
            plt.yticks(())
            plt.xlim(xx.min(),xx.max())
            plt.ylim(yy.min(),yy.max())
            plt.title(classifier_names[index])
    
    plt.show()        
