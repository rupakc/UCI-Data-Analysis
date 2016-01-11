# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 18:13:04 2016
Benchmarking on the boston housing dataset
@author: Rupak Chakraborty
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ElasticNet
from sklearn import metrics
from sklearn import cross_validation 
import random

datafile = "Boston Housing/housing.data"
dataFrame = pd.read_csv(datafile,header=None,sep='\t')
dataArray = np.zeros((len(dataFrame),13))
dataOutput = np.zeros((len(dataFrame),1)) 
 
column_names = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B" \
,"LSTAT","MEDV"]

for i in range(len(dataFrame)):  
    
    feature_string = dataFrame.iloc[i][0]
    feature_list = feature_string.split() 
    
    for j in range(len(feature_list)): 
        
        if j != len(feature_list)-1:
            dataArray[i][j] = float(feature_list[j])
        else:
            dataOutput[i] = float(feature_list[j])

random.shuffle(dataArray)

ada = AdaBoostRegressor()
rf = RandomForestRegressor()
bagging = BaggingRegressor()
grad = GradientBoostingRegressor()
svr = SVR()
bayes_ridge = BayesianRidge()
elastic_net = ElasticNet()

regressors = [ada,rf,bagging,grad,svr,bayes_ridge,elastic_net]
regressor_names = ["AdaBoost","Random Forest","Bagging","Gradient Boost","SVR","Bayesian Ridge","Elastic Net"]

train_data,test_data,train_values,test_values = cross_validation.train_test_split(dataArray,dataOutput,test_size=0.3)

for regressor,regressor_name in zip(regressors,regressor_names):
    
    regressor.fit(train_data,train_values)
    predicted_values = list(regressor.predict(test_data))
    
    print "-----------------------------------\n"
    print "For Regressor : ",regressor_name
    print "Mean Absolute Error : ",metrics.mean_absolute_error(list(test_values),predicted_values)
    print "Median Absolute Error : ",metrics.median_absolute_error(list(test_values),predicted_values)
    print "Mean Squared Error : ",metrics.mean_squared_error(list(test_values),predicted_values)
    print "R2 Score : ",metrics.r2_score(list(test_values),predicted_values)
    print "---------------------------------\n"
