# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 23:22:13 2016

@author: Rupak Chakraborty
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler 
from sklearn import metrics
from sklearn import cross_validation 

rscaler = RobustScaler()
air_frame = pd.read_csv('airfoil_self_noise.dat',sep='\t')
column_names = ['Frequency','Attack Angle','Chord Length','Free Velocity','Suction Side','Scaled Sound']
air_frame.column = column_names
scaled_data = rscaler.fit_transform(air_frame.values)
X = scaled_data[:,:5]
Y = scaled_data[:,5]

train_data,test_data,train_regressor,test_regressor = cross_validation.train_test_split(X,Y,test_size=0.3)
rf = RandomForestRegressor()
grad = GradientBoostingRegressor()
bag = BaggingRegressor()
ada = AdaBoostRegressor()
bayes = BayesianRidge()
svr = SVR()
lin_reg = LinearRegression()

regressors_names = ['Random Forests','Gradient Boost','Bagging','Ada Boost','Bayesian Ridge','SVR','Linear Reg']
regressors = [rf,grad,bag,ada,bayes,svr,lin_reg]

for regressor,name in zip(regressors,regressors_names):
    regressor.fit(train_data,train_regressor)
    predicted_values = regressor.predict(test_data)
    print '-------- For Regressor ',name,' ------------\n'
    print 'Median Absolute Error : ',metrics.median_absolute_error(predicted_values,test_regressor)
    print 'Mean Squared Error : ',metrics.mean_squared_error(predicted_values,test_regressor)
    print 'R2 Score : ',metrics.r2_score(predicted_values,test_regressor)
    print '-------------------------------------------\n'
