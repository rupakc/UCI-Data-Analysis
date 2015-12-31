# -*- coding: utf-8 -*-
"""
Experiments on the car dataset available on the UCI repository
Created on Thu Dec 31 00:08:29 2015
@author: Rupak Chakraborty
"""

import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn import metrics 
from sklearn import cross_validation
from sklearn.feature_selection import SelectFromModel

start = time.time()
filename = "car.data"
filepath = "Car Evaluation/"
full_path = filepath + filename
class_label_map = {"unacc":0,"acc":1,"good":2,"vgood":3}
buying_label_map = {"vhigh":3,"high":2,"med":1,"low":0}
maintain_label_map = buying_label_map
doors_label_map = {'2':2,'3':3,'4':4,"5more":5}
person_label_map = {'2':2,'4':4,"more":5}
lug_boot_map = {"small":0,"med":1,"big":2}
safety_label_map = {"low":0,"med":1,"high":2}
column_names = ["buying","maintain","doors","persons","lug_boot","safety"]
column_maps = [buying_label_map,maintain_label_map,doors_label_map,\
person_label_map,lug_boot_map,safety_label_map]

dataFrame = pd.read_csv(full_path,header=None)
dataFrame = dataFrame.iloc[np.random.permutation(len(dataFrame))]
labels = dataFrame[len(dataFrame.columns)-1]
del dataFrame[len(dataFrame.columns)-1]
dataFrame.columns = column_names
labels = map(lambda x:class_label_map[x],labels) 

for column_map,col_name in zip(column_maps,column_names):
    dataFrame[col_name] = map(lambda x:column_map[x],dataFrame[col_name])

#Initializing the classifiers (All are tree based classifiers)

dt = DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators=51)
extree = ExtraTreeClassifier()
classifier_list = [dt,rf,extree]
classifier_name_list = ["Decision Tree","Random Forests","Extra Trees"] 

data = dataFrame.values 

# Initializing Cross Validation Models 

kf = cross_validation.KFold(len(labels),n_folds=5)
stratifiedkf = cross_validation.StratifiedKFold(labels,n_folds=4)
labeledkf = cross_validation.LabelKFold(labels,n_folds=4)
leavePout = cross_validation.LeavePOut(len(labels),p=100)
cross_validation_model_list = [kf,stratifiedkf,labeledkf,leavePout]
cross_validation_model_names = ["K-Fold","Stratified K-fold","Labeled K-Fold","Leave P Out"]

# Cross Validating each given classifier 

for classifier,classifier_name in zip(classifier_list,classifier_name_list):
    scores = cross_validation.cross_val_score(classifier,data,labels,cv=10)
    print "-------- For Classifier : ",classifier_name," ---------------"
    print "Score Array : ",scores
    print "Mean Score : ",scores.mean()
    print "Standard Deviation : ",scores.std()
    print "------------------------------------------------------"
    
label_array = np.array(labels) 

def crossValidationTestAll():
    
    for cross_valid_model,validation_name in zip(cross_validation_model_list,cross_validation_model_names):
        
        print "---------- For Cross Validation Method : ",validation_name ," -------------"
        
        for train,test in cross_valid_model:
           
            train_data = data[train]
            train_labels = label_array[train]
            test_data = data[test]
            test_labels = label_array[test]
            
            for classifier,classifier_name in zip(classifier_list,classifier_name_list): 
                
                classifier.fit(train_data,train_labels)
                predicted_labels = classifier.predict(test_data)
                print "-------------------------------------------------\n"
                print "Accuracy for Classifier ",classifier_name," : ",metrics.accuracy_score(test_labels,predicted_labels)
                print "Classification Report for Classifier ",classifier_name, " :\n ",metrics.classification_report(test_labels,predicted_labels)
                print "-------------------------------------------------\n" 
            
X_train,X_test,y_train,y_test = cross_validation.train_test_split(data,label_array,test_size=0.3) 

#Selecting features using the selectFrom Model 

for classifier,classifier_name in zip(classifier_list,classifier_name_list): 
    
    classifier.fit(X_train,y_train)
    select = SelectFromModel(classifier,prefit=True)
    X_train_new = select.transform(X_train)
    X_test_new = select.transform(X_test)
    classifier.fit(X_train_new,y_train)
    predicted_labels = classifier.predict(X_test_new) 
    
    print "-----------------------------------------------------\n"
    print "Accuracy for ",classifier_name, " : ", metrics.accuracy_score(y_test,predicted_labels)
    print "Classification Report for ",classifier_name, " :\n", metrics.classification_report(y_test,predicted_labels)
    print "-----------------------------------------------------\n"

end = time.time()
print "Total Time Taken For the code : ",end-start