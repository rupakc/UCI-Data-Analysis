# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 18:16:59 2016
Baseline Algorithm for predicting student grades given a collection of attributes
@author: Rupak Chakraborty
"""
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import Hinge
from sklearn.linear_model import Huber

filename = "Student/student-mat.csv"
data = pd.read_csv(filename,sep=";")

binary_features = ["schoolsup","famsup","paid","activities","nursery","higher","internet","romantic"]
binary_map = {"yes":1,"no":0}

other_nominal_features = ["school","sex","address","famsize","Pstatus","Mjob","Fjob","reason","guardian"]
school_map = {"GP":0,"MS":1}
sex_map = {"M":0,"F":1}
address_map = {"U":0,"R":1}
famsize_map = {"LE3":0,"GT3":1}
Pstatus = {"T":1,"A":0}
M_job_map = {"teacher":0,"health":1,"services":2,"at_home":3,"other":4}
F_job_map = {"teacher":0,"health":1,"services":2,"at_home":3,"other":4}
reason_map = {"home":0,"reputation":1,"course":3,"other":4}
guardian_map = {"father":0,"mother":1,"other":2}

other_nominal_map_list = [school_map,sex_map,address_map,famsize_map,Pstatus,M_job_map,F_job_map,reason_map,guardian_map]

for binary_feature in binary_features:
    data[binary_feature] = map(lambda x:binary_map[x],data[binary_feature])

for index,feature in enumerate(other_nominal_features):
    feature_map = other_nominal_map_list[index]
    data[feature] = map(lambda x:feature_map[x],data[feature]) 
    
data["G1"] = map(lambda x: float(x),data["G1"])
data["G2"] = map(lambda x: float(x),data["G2"])
predictor = data["G3"]
del data["G3"]

knn = KNeighborsRegressor(n_neighbors=5)
rf = RandomForestRegressor(n_estimators=101)
ada = AdaBoostRegressor(n_estimators=101)
grad = GradientBoostingRegressor(n_estimators=101)
bagging = BaggingRegressor(n_estimators=101)
bayes = BayesianRidge()

regressors = [knn,rf,ada,grad,bagging,bayes]
regressor_names = ["KNN","Random Forests","AdaBoost","Gradient Boost","Bagging","Bayes"] 

X_train,X_test,y_train,y_test = cross_validation.train_test_split(data.values,predictor.values,test_size=0.2)
feature_importances = []

for regressor,name in zip(regressors,regressor_names):
    
    regressor.fit(X_train,y_train)
    predicted_values = regressor.predict(X_test)
    if name == "Random Forests":
        feature_importances = regressor.feature_importances_
    
    print "---------------------------\n"
    print "Absolute Mean Error for ", name , " : ", metrics.mean_absolute_error(y_test,predicted_values)
    print "Median Error for ", name ," : ", metrics.median_absolute_error(y_test,predicted_values)
    print "Mean Squared Error for ",name," : ",metrics.mean_squared_error(y_test,predicted_values)
    print "R2 Score for ",name," : ",metrics.r2_score(y_test,predicted_values)
    print "---------------------------\n"

print "\n------------- Feature Importances------------\n"
for name,importance in zip(data.columns,feature_importances):
    print name," : ",importance

