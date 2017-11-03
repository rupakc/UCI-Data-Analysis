# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 18:56:35 2015

Spam Detection on Email Dataset Provided in:-
[1] Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. 
Contributions to the study of SMS Spam Filtering: New Collection and Results.
Proceedings of the 2011 ACM Symposium on Document Engineering (ACM DOCENG'11)
Mountain View, CA, USA, 2011 

The following features have been used:- 
1. Bag of words
2. Tf-Idf 

Classifier Details are the following:-

1. SVM (Linear and RBF kernel)
2. Multinomial Naive Bayes
3. Bernoulli Naive Bayes
4. Perceptron
5. Logistic Regression

@author: Rupak Chakraborty
"""

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import string
import numpy as np
import random

stop_words = set(stopwords.words('english'))
label_map = {"ham":1,"spam":0}
stemmer = PorterStemmer()
TRAIN_LIMIT = 4500
FILENAME = "SMSSpamCollection"

def is_ascii(s):
    return all(ord(c) < 128 for c in s)    

def remove_punctuations(tokenized_summary): 
    
    reduced_list = []
    punctuations = list(string.punctuation)
    punctuations.append('\n')
    punctuations.append('\r')
    punctuations.append('\r\n')
    
    for word in tokenized_summary:
        if word not in punctuations:
            reduced_list.append(word) 
            
    return reduced_list
    
def preProcessingPipeline(sentence): 
    
    sentence = sentence.lower()
    words = []
    try:
        words = word_tokenize(sentence)
        words = remove_punctuations(words)
        words = [w for w in words if w not in stop_words] 
    except:
        pass
    
    try:
        words = filter(is_ascii,words)
        words = map(lambda word:stemmer.stem_word(word),words)
    except:
        pass
    
    sanitized_string = ""
    
    for word in words:
        if len(word.strip()) >= 3:
            sanitized_string = sanitized_string + word.strip() + " "
    
    return sanitized_string.strip()

def splitTrainTestSet(filename):  
    
    f = open(filename,"r")
    data = f.read()
    data = data.split("\n")
    random.shuffle(data)
    train_sample = data[0:TRAIN_LIMIT]
    test_sample = data[TRAIN_LIMIT:]
    
    return train_sample,test_sample 
    
#Spliting the train and test module
    
train_sample,test_sample = splitTrainTestSet(FILENAME)    
train_labels = np.zeros(len(train_sample))
test_labels = np.zeros(len(test_sample))
train_list = list([])
test_list = list([])

c = 0

#Preprocessing Training Data

for sample in train_sample:
    label_text = sample.split("\t")
    train_labels[c] = label_map[label_text[0]]
    text = label_text[1]
    text = preProcessingPipeline(text)
    train_list.append(text)
    c = c + 1 
    
c = 0 

# Preprocessing Test Data 

for sample in test_sample:
    label_text = sample.split("\t")
    test_labels[c] = label_map[label_text[0]]
    text = label_text[1]
    text = preProcessingPipeline(text)
    test_list.append(text)
    c = c + 1

#Extracting Feature Vectors from training data
    
count_vec = CountVectorizer()
tf_idf = TfidfVectorizer()

count_features_train = count_vec.fit_transform(train_list)
tfidf_features_train = tf_idf.fit_transform(train_list)

#Extracting features from test data

count_features_test = count_vec.transform(test_list)
tfidf_features_test = tf_idf.transform(test_list)

# Initializing Classifiers 

svm = SVC()
mnb = MultinomialNB()
gnb = GaussianNB()
bnb = BernoulliNB()
logit = LogisticRegression()
percept = Perceptron()
sgd = SGDClassifier()

feature_list_train = [count_features_train,tfidf_features_train]
feature_list_test = [count_features_test,tfidf_features_test]

for features_train,features_test in zip(feature_list_train,feature_list_test):
    
    #Training Classifiers
    
    svm.fit(features_train,train_labels)
    mnb.fit(features_train,train_labels)
    bnb.fit(features_train,train_labels)
    logit.fit(features_train,train_labels)
    percept.fit(features_train,train_labels)
    sgd.fit(features_train,train_labels) 
    
    #Predicting output on test data
    
    svm_predict = svm.predict(features_test)
    mnb_predict = mnb.predict(features_test)
    bnb_predict = bnb.predict(features_test)
    logit_predict = logit.predict(features_test)
    percept_predict = percept.predict(features_test)
    sgd_predict = sgd.predict(features_test)
    
    classifier_names = ["SVM","Multinomial NB","Bernoulli NB","Logistic Regression","Perceptron","SGD"]
    classifiers_predictions = [svm_predict,mnb_predict,bnb_predict,logit_predict,percept_predict,sgd_predict]
    
    for predict,name in zip(classifiers_predictions,classifier_names):
        
        #Performance Metrics of Classifiers 
    
        print "------ Classification Report of ",name," ---------"
        print metrics.classification_report(test_labels,predict)
        print "-------- Confusion Matrix of ",name," --------"
        print metrics.confusion_matrix(test_labels,predict)
        print "-------- Accuracy Score of the " ,name, " ---------"
        print metrics.accuracy_score(test_labels,predict)
