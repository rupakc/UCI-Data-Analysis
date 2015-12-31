# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 19:50:51 2015
Experiments on the famous wine data set :- 
Feature Preprocessing
Feature Selection
Plotting of Classifier Accuracies
Cross Validation
The Features are as follows :-
    1) Alcohol
 	2) Malic acid
 	3) Ash
	4) Alcalinity of ash  
 	5) Magnesium
	6) Total phenols
 	7) Flavanoids
 	8) Nonflavanoid phenols
 	9) Proanthocyanins
	10)Color intensity
 	11)Hue
 	12)OD280/OD315 of diluted wines
 	13)Proline 
@author: Rupak Chakraborty
"""

import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler 
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn import cross_validation
import matplotlib.pyplot as plt 

filepath = "Wine/"
filename = "wine.data"
fullpath = filepath+filename
dataFrame = pd.read_csv(fullpath,header=None)
dataFrame = dataFrame.iloc[np.random.permutation(len(dataFrame))]
TRAIN_SIZE = 128
class_labels = list(dataFrame[0]) 

normalizer = Normalizer()
standardscale = StandardScaler()
minmaxscaler = MinMaxScaler() 

del dataFrame[0] 

column_names = ["alcohol","malic acid","ash","alcalinity of ash","magnesium","total phenols",\
"falavanoids","nonflavanoid phenols","proanthocyanins","color intensity","hue","od280/od315","proline"]
dataFrame.columns = column_names

print "------------ Description of DataFrame values ---------------"
print dataFrame.describe()
print "------------ Basic Information About the Data Frame ------------"
print dataFrame.info()

# Calculating the correlations between the variables

pearson_corr = dataFrame.corr(method = 'pearson')
spearman_corr = dataFrame.corr(method ='spearman')
kendall_corr = dataFrame.corr(method ='kendall')
correlation_list = [pearson_corr,spearman_corr,kendall_corr]
correlation_method_names = ["pearson","spearman","kendall"]

for correlation,method_name,col_name in zip(correlation_list,correlation_method_names,column_names):
    
    print "-------- For Correlation Using ",method_name," ---------" 
    
    for compare_column in column_names:
        if (compare_column != col_name):
            print correlation.loc[(correlation[col_name] > 0) & (correlation[compare_column] > 0)]

# Initializing the classifiers 

lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()
svm = SVC()
perceptron = Perceptron()
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()
knn = KNeighborsClassifier()
rf = RandomForestClassifier(n_estimators=51)
ada = AdaBoostClassifier()

classifiers = [lda,qda,svm,perceptron,gnb,mnb,bnb,knn,rf,ada]
classifier_names = ["LDA","QDA","SVM (RBF)","Perceptron","Gaussian NB","Multinomial NB",\
"Bernoulli NB","KNN (K=5)","Random Forests","Ada Boost"]

index = np.arange(len(classifier_names))  

#Extracting the data values in a numpy array and Preprocessing it

data = dataFrame.values
data_normalized = normalizer.fit_transform(data) 
data_standard = standardscale.fit_transform(data)
data_minmax = minmaxscaler.fit_transform(data) 
preprocess_names = ["Unscaled","Normalized","Standardized","MinMax"] 
preprocessors = [data,data_normalized,data_standard,data_minmax] 
train_labels = class_labels[:128]
test_labels = class_labels[128:]
performance_all_preprocess = list([]) 
count = 0

#Defines the Recursive Feature Selector for best feature selection 

def recursiveFeatureSelector(classifier_model,train_data,train_labels,test_data,number_of_features):
    
    rfe = RFE(classifier_model,number_of_features)
    transformed_train_data = rfe.fit_transform(train_data,train_labels)
    transformed_test_data = rfe.transform(test_data)
    
    return transformed_train_data,transformed_test_data 
    
#Defines the recursive feature selector for choosing the best feature using Cross Validation   
    
def recursiveFeatureSelectorCV(classifier_model,train_data,train_labels,test_data,number_of_features):
    
    rfe = RFECV(classifier_model,number_of_features)
    transformed_train_data = rfe.fit_transform(train_data,train_labels)
    transformed_test_data = rfe.transform(test_data)
    
    return transformed_train_data,transformed_test_data

#Iterating over all feature preprocessors and classifiers in turn 

for data_process,preprocess_type in zip(preprocessors,preprocess_names):
    
    print "\n-------------------------------------------------"
    print "    For Preprocess Data Type : ", preprocess_type
    print "\n-------------------------------------------------"
    train_data = data_process[:128]
    test_data = data_process[128:]
    
    performance = list([])
    
    print "\n ----------- Accuracy for Different Classifiers ------------" 
    
    for classifier,classifier_name in zip(classifiers,classifier_names):
        
        try:
            train_data,test_data = recursiveFeatureSelector(classifier,train_data,train_labels,test_data,5)
            classifier.fit(train_data,train_labels)
            predicted_labels = classifier.predict(test_data)
            performance.append(metrics.accuracy_score(test_labels,predicted_labels))
            
            print "\n -----------------------------------------------------------------"
            print "Accuracy for ",classifier_name," : ",metrics.accuracy_score(test_labels,predicted_labels)
            print "Confusion Matrix for ",classifier_name, " :\n ",metrics.confusion_matrix(test_labels,predicted_labels)
            
            #Cross Validation Scores for each classifier
            
            scores = cross_validation.cross_val_score(classifier,train_data,train_labels,cv=5,scoring='f1_weighted')
            print "Score Array for ",classifier_name, " : \n",scores
            print "Mean Score for ", classifier_name, " : ", scores.mean() 
            print "Stanard Deviation of Scores for ", classifier_name, " : ", scores.std()
            print "-------------------------------------------------------------------\n"
            
        except:
            pass
        
    performance_all_preprocess.append(performance)
    
def plotBarChartAccuracyAll():
    
    try:
        width = 0.5
        fig, ax = plt.subplots(2,2)
        plt.subplot(221)
        plt.bar(index,performance_all_preprocess[0],width,alpha=0.5)
        plt.xticks(index,classifier_names)
        plt.xlabel("Classifiers")
        plt.ylabel("Accuracy in Percentage")
        plt.tight_layout()  
        
        plt.subplot(222)
        plt.bar(index,performance_all_preprocess[1],width,alpha=0.45)
        plt.xticks(index,classifier_names)
        plt.xlabel("Classifiers")
        plt.ylabel("Accuracy in Percentage")
        plt.tight_layout()  
        
        plt.subplot(223)
        plt.bar(index,performance_all_preprocess[2],width,alpha=0.75)
        plt.xticks(index,classifier_names)
        plt.xlabel("Classifiers")
        plt.ylabel("Accuracy in Percentage")
        plt.tight_layout()  
        
        plt.subplot(224)
        plt.bar(index,performance_all_preprocess[3],width,alpha=0.65)
        plt.xticks(index,classifier_names)
        plt.xlabel("Classifiers")
        plt.ylabel("Accuracy in Percentage")
        plt.tight_layout()  
    except:
        pass
    
    