# -*- coding: utf-8 -*-
"""
Created on Mon Jan 04 18:12:32 2016
TODO-Increase the accuracy its abysmal at the moment
@author: Rupak Chakraborty
"""

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

filename_train = "Adult/adult.data"
filename_test = "Adult/adult.test"

trainDataFrame = pd.read_csv(filename_train,header=None)
testDataFrame = pd.read_csv(filename_test,header=None) 

column_names = ["age","workclass","fnlwgt","education","education-num","marital-status"\
,"occupation","relationship","race","sex","capital-gain","captial-loss","hours-per-week"\
,"native-country","salary"]

trainDataFrame.columns = column_names
testDataFrame.columns = column_names

workclass_map = {"Private":0, "Self-emp-not-inc":1,"Self-emp-inc":2,"Federal-gov":3\
,"Local-gov":4,"State-gov":5,"Without-pay":6,"Never-worked":7,"0":0}

education_map = {"Bachelors":0,"Some-college":1,"11th":2,"HS-grad":3,"Prof-school":4\
,"Assoc-acdm":5,"Assoc-voc":6,"9th":7,"7th-8th":8,"12th":9,"Masters":10,"1st-4th":11\
,"10th":12,"Doctorate":13,"5th-6th":14,"Preschool":15,"0":0}

marital_status_map = {"Married-civ-spouse":0,"Divorced":1,"Never-married":2,"Separated":3\
,"Widowed":4,"Married-spouse-absent":5,"Married-AF-spouse":6,"0":0}

occupation_map = {"Tech-support":0,"Craft-repair":1,"Other-service":2,"Sales":3 \
,"Exec-managerial":4,"Prof-specialty":5,"Handlers-cleaners":6,"Machine-op-inspct":7 \
,"Adm-clerical":8,"Farming-fishing":9,"Transport-moving":10,"Priv-house-serv":11 \
,"Protective-serv":12,"Armed-Forces":13,"0":0}

relationship_map = {"Wife":0,"Own-child":1,"Husband":2,"Not-in-family":3,"Other-relative":4 \
,"Unmarried":5,"0":0}

race_map = {"Asian-Pac-Islander":0,"Amer-Indian-Eskimo":1,"Other":2,"Black":3,"0":0,"White":4} 

sex_map = {"Female":0,"Male":1,"0":0} 

native_country_map = {"United-States":0,"Cambodia":1,"England":2,"Puerto-Rico":3,"Canada":4 \
,"Germany":5,"Outlying-US(Guam-USVI-etc)":6,"India":7,"Japan":8,"Greece":9,"South":10 \
,"China":11,"Cuba":12,"Iran":13,"Honduras":14,"Philippines":15,"Italy":16,"Poland":17,"Jamaica":18 \
,"Vietnam":19,"Mexico":20,"Portugal":21,"Ireland":22,"France":23,"Dominican-Republic":24,"Laos":25 \
,"Ecuador":26,"Taiwan":27,"Haiti":28,"Columbia":29,"Hungary":30,"Guatemala":31,"Nicaragua":32 \
,"Scotland":33,"Thailand":34,"Yugoslavia":35,"El-Salvador":36,"Trinadad&Tobago":37,"Peru":38 \
,"Hong":39,"Holand-Netherlands":40,"0":0}

class_label_map = {True:1,False:0}

label_list = ["workclass","education","marital-status","occupation","relationship","race" \
,"sex","native-country"]

map_list = [workclass_map,education_map,marital_status_map,occupation_map,relationship_map,race_map\
,sex_map,native_country_map]

def convertLabels(dataFrame,label_list,label_maps):
    for label,map_type in zip(label_list,label_maps):
        dataFrame[label] = map(lambda x:map_type[x.strip()],dataFrame[label])
    return dataFrame
        
trainDataFrame = convertLabels(trainDataFrame,label_list,map_list)
trainDataFrame['salary'] = map(lambda x : class_label_map[x], map(lambda x: x.strip() == "<=50K",trainDataFrame['salary']))

testDataFrame = convertLabels(testDataFrame,label_list,map_list)
testDataFrame['salary'] = map(lambda x : class_label_map[x], map(lambda x: x.strip() == "<=50K",testDataFrame['salary']))

trainLabels = trainDataFrame['salary']
testLabels = testDataFrame['salary']

del trainDataFrame['salary']
del testDataFrame['salary']

rf = RandomForestClassifier(n_estimators=51)
ada = AdaBoostClassifier(n_estimators=51)
bagging = BaggingClassifier(n_estimators=101)
grad = GradientBoostingClassifier(n_estimators=101)

classifiers = [rf,ada,bagging,grad]

for classifier in classifiers:
    print "Fitting Classifier "
    classifier.fit(trainDataFrame.values,trainLabels.values)
    print "Starting Prediction:"
    predicted_labels = classifier.predict(testDataFrame.values)
    print metrics.accuracy_score(testLabels.values,predicted_labels)

