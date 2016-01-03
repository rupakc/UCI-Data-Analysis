# -*- coding: utf-8 -*-
"""
Created on Sun Jan 03 22:13:19 2016
TODO - Reduce Dimensionality of data and preprocess data to decrease error
And test on the individual files provided
@author: Rupak Chakraborty
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn import metrics
from sklearn import cross_validation
from sklearn import preprocessing

filename = "Blog Feedback/BlogFeedback/blogData_train.csv"
data = pd.read_csv(filename,header=None)
data = data.iloc[np.random.permutation(len(data))]
output_variables = data[len(data.columns)-1]
del data[len(data.columns)-1]

train_data,test_data,train_output,test_output = cross_validation.train_test_split(data.values,output_variables.values,test_size=0.3)

rf = RandomForestRegressor(n_estimators=101)
ada = AdaBoostRegressor(n_estimators=101)
bagging = BaggingRegressor(n_estimators=101)
gradBoost = GradientBoostingRegressor(n_estimators=101)
bayes = BayesianRidge()

regressors = [rf,ada,bagging,gradBoost,bayes]
regressor_names = ["Random Forests","Adaboost","Bagging","Gradient Boosting","Bayesian Ridge"]

for regressor,regressor_name in zip(regressors,regressor_names):
    
    regressor.fit(train_data,train_output)
    predicted_values = regressor.predict(test_data)
    
    print "--------------------------------\n"
    print "Mean Absolute Error for ",regressor_name," : ",metrics.mean_absolute_error(test_output,predicted_values)
    print "Median Absolute Error for ",regressor_name, " : ",metrics.median_absolute_error(test_output,predicted_values)
    print "Mean Squared Error for ",regressor_name, " : ",metrics.mean_squared_error(test_output,predicted_values)
    print "R2 score for ",regressor_name, " : ",metrics.r2_score(test_output,predicted_values)
    print "--------------------------------\n"