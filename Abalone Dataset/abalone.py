# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 23:19:53 2015
Benchmarking of the abalone age prediction using several regressors
@author: Rupak Chakraborty
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LassoLars
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn import cross_validation

filename = "Abalone/abalone.data"
abalone_data = pd.read_csv(filename,header=None)
sex_map = {"M":0,"F":1,"I":2}
column_names = ["Sex","Length","Diameter","Height","Whole Weight","Shucked Weight"\
,"Viscera Weight","Shell Weight","Ring"]
abalone_data.columns = column_names
output_variables = abalone_data["Ring"]
del abalone_data["Ring"]
column_names.remove("Ring")

abalone_data["Sex"] = map(lambda x:sex_map[x],abalone_data["Sex"])

# If there are any missing values filling them by the mean value for that sex

for column in column_names:
    abalone_data[column].fillna(abalone_data.groupby("Sex")[column].transform("mean"),inplace=True) 
    
# Randomly shuffling the data to reduce variance
    
abalone_data = abalone_data.iloc[np.random.permutation(len(abalone_data))]
train_data,test_data,train_output,test_output = cross_validation.train_test_split(abalone_data,output_variables,test_size=0.3)

# Initializing the regressors

rf_reg = RandomForestRegressor(n_estimators=71)
extree_reg = ExtraTreesRegressor(n_estimators=51)
gradboost_reg = GradientBoostingRegressor()
bag_reg = BaggingRegressor(n_estimators=71)
adaboost_reg = AdaBoostRegressor(n_estimators=100)
bayes_ridge = BayesianRidge()
lasso = LassoLars()
sgd = SGDRegressor()
linear = LinearRegression()

classifiers = [rf_reg,extree_reg,gradboost_reg,bag_reg,adaboost_reg,bayes_ridge,lasso,sgd,linear]
classifier_names = ["Random Forest","Extra Trees","Gradent Boosting","Bagging",\
"AdaBoost","Bayesian Ridge","Lasso LARS","Stochastic Gradient Descent","Linear Regression"]

# Iterating over regressors to calculate the performance metrics 

for classifier,classifier_name in zip(classifiers,classifier_names): 
    
    classifier.fit(train_data.values,train_output.values)
    predicted_values = classifier.predict(test_data.values)
    
    print ("----------------------------------------------\n")
    print "Mean Absoulte Error of ",classifier_name, " : ", metrics.mean_absolute_error(test_output.values,predicted_values)
    print "Median Absolute Error of ",classifier_name, " : ",metrics.median_absolute_error(test_output.values,predicted_values)
    print "Mean Squared Error of ",classifier_name, " : ",metrics.mean_squared_error(test_output.values,predicted_values)
    print "R2 Score of :",classifier_name, " : ",metrics.r2_score(test_output.values,predicted_values)
    print "Explained Variance of ",classifier_name, " : ",metrics.explained_variance_score(test_output.values,predicted_values)
    print ("----------------------------------------------\n")
