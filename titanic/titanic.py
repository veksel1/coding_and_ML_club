# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 22:27:59 2018

@author: Ilja
"""
from __future__ import division
#import math
#import string
import pandas as pd
#import numpy as np
#import sklearn as sk
from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#cleaning and formatting data
def formatData(data):
    data['class_1'] = data['Pclass'].apply(lambda pclass: 1 if pclass == 1 else 0)
    data['class_2'] = data['Pclass'].apply(lambda pclass: 1 if pclass == 2 else 0)
    data['class_3'] = data['Pclass'].apply(lambda pclass: 1 if pclass == 3 else 0)
    
    data['female'] = data['Sex'].apply(lambda gender: 1 if gender == 'female' else 0)
    #data['cabin_indicated'] = data['Cabin'].apply(lambda cabin: 0 if cabin.isnull().any() else 0)
    
    #C = Cherbourg, Q = Queenstown, S = Southampton
    data['cherbourg'] = data['Embarked'].apply(lambda city: 1 if city == 'C' else 0)
    data['queenstown'] = data['Embarked'].apply(lambda city: 1 if city == 'Q' else 0)
    data['southampton'] = data['Embarked'].apply(lambda city: 1 if city == 'S' else 0)
      
     # !!!!! address missing age data
    data['people_amount_per_ticket'] = data['Ticket'].apply(lambda ticketName: len(data[data['Ticket']==ticketName]))
    data['price_per_person'] = data['Fare']/data['people_amount_per_ticket']
    
    data['child_travelled_with_nanny'] = data['Parch'].apply(lambda parentsNum: 1 if parentsNum == 0 else 0)
    data['child_travelled_with_nanny'][data['Age']>=14] = 0
    #replace missing data with means - !!!!! can improve here!!!!!!!!!!!!
    for feature in feature_labels_basic:
        data[feature].fillna(data[feature].mean(), inplace=True)
    
    #remove missing data
#    data = data[data.isNull=False]
    
    return data

def addNonLinearity(data):
    data['age^2'] = data['Age']**2
    data['age^3'] = data['Age']**3
    data['age^4'] = data['Age']**4
    
    data['female_class_1'] = data['female']*data['class_1']
    data['female_class_2'] = data['female']*data['class_2']
    data['female_class_3'] = data['female']*data['class_3']

    data['SibSp^2'] = data['SibSp']**2    
    data['SibSp^3'] = data['SibSp']**3    
    data['SibSp^4'] = data['SibSp']**4    
    
    data['parch^2'] = data['Parch']**2    
    data['parch^3'] = data['Parch']**3    
    data['parch^4'] = data['Parch']**4    
    
    data['price_per_person^2'] = data['price_per_person']**2   
    data['price_per_person^3'] = data['price_per_person']**3      
    data['price_per_person^4'] = data['price_per_person']**4      
    
    return data


#showing regression results
def printRegResults(log_reg, data_set, predictions, score, target='Survived'):
    survived = data_set[target]
    #print('regression coefficients: ', str(log_reg.coef_))
    #print('intercept: ', log_reg.intercept_)
    
    #print('my predictions: \n', predictions) 
    #print('actual sentiments: \n', survived) 
#    print('-------------------------------------------------------------------------------------')
#    print('\n actual sentiments: mean=', np.mean(survived), ', median=', np.median(survived), ', min=', np.min(survived), ', max=',  np.max(survived))
#    print('\n predictions: mean=', np.mean(predictions), ', median=', np.median(predictions), ', min=', np.min(predictions), ', max=',np.max(predictions))
    
    observation_amount = len(survived)
#    print('total number of obsevations=', observation_amount)
    print('\nscore / mean accuracy (as given by package) = ',int(score*100))
#    print('accuracy (calculated manually) =', len(survived[survived==predictions]) / observation_amount)
#    print('majority class accuracy=', len(survived[survived==0]) / observation_amount)
#    print('predicting that all females survived, but males died =', len(data_set[data_set['female']==0]) / observation_amount)
    print('-------------------------------------------------------------------------------------')
    print()
    
    
#defining features and the target
feature_labels = ['Age',                    
            'class_1',              
            'class_2',                 
            'class_3',            
            'female',         
            'cherbourg',                       
            'queenstown',                 
            'southampton',                    
            'SibSp',
            'price_per_person',
            'people_amount_per_ticket',
            'Parch',
            'Fare',
            'child_travelled_with_nanny',
           ]

feature_labels_basic = feature_labels

#additional features
feature_labels_non_linear = [
        'age^2',
        'age^3',
#        'age^4',
        'female_class_1',
        'female_class_2',
        'female_class_3',
        'SibSp^2',
        'SibSp^3', 
#        'SibSp^4', 
        'parch^2', 
        'parch^3',
#        'parch^4', 
        'price_per_person^2', 
        'price_per_person^3',
#        'price_per_person^4',
        ]

#adding non linearity to features (uncomment to remove)
feature_labels = feature_labels + feature_labels_non_linear

target = 'Survived'

# dataset import in Pandas
given_train_data = pd.read_csv('C:\\Users\\ilja.surikovs\\Documents\\GitHub\\coding_club\\titanic\\train.csv') #, dtype = {'name':str,'review':str,'rating':int})

#formatting data    
clean_data = formatData(given_train_data)
#adding non-linearity to data
clean_data = addNonLinearity(clean_data)

#to check the sparse matrix size
#clean_data.shape

#train - test split
train_set, valid_set = train_test_split(given_train_data, test_size=0.2)

train_set = formatData(train_set)
train_set = addNonLinearity(train_set)

valid_set = formatData(valid_set)
valid_set = addNonLinearity(valid_set)

print()
print('train-set size', len(train_set))
print('valid-set size', len(valid_set))
print('full-set size', len(clean_data))
print()

# defining Y and X for train and valid sets
survived_train = train_set[target]
features_train = train_set[feature_labels]

survived_valid = valid_set[target]
features_valid = valid_set[feature_labels]

##--------------------------------------------------------------------
##learning - running regressions
#log_reg = linear_model.LogisticRegression()
#log_reg.fit(features_train, survived_train)
#score_train = log_reg.score(features_train, survived_train)
#predictions_train = log_reg.predict(features_train)
##prob_predictions = log_reg.predict_proba(features)
#    
##printing results for train_set
#print('regression - train-set results:')
#printRegResults(log_reg, train_set, predictions_train, score_train)
#
##--------------------------------------------------------------------
##forecasting with validation data

#predictions_valid = log_reg.predict(features_valid)
#score_valid = log_reg.score(features_valid, survived_valid)
#
##printing results for valid_set
#print('regression - validation-set results:')
#printRegResults(log_reg, valid_set, predictions_valid, score_valid)

##----------------------------------------------------------------------------------
# try decision trees
for minImpur in range(0, 1, 1):
    print('minImpur =', minImpur)
    decTree= DecisionTreeClassifier(min_samples_split=18, min_samples_leaf=9)
    
    decTree.fit(features_train, survived_train) 
    score_train = decTree.score(features_train, survived_train)
    predictions_train = decTree.predict(features_train)
#    print('decTree - train-set results:')
#    printRegResults(decTree, train_set, predictions_train, score_train)
    
    #valid-set
    predictions_valid = decTree.predict(features_valid)
    score_valid = decTree.score(features_valid, survived_valid)
    print('decTree - validation-set results:')
    printRegResults(decTree, valid_set, predictions_valid, score_valid)

#---------------------------------------------------------------------------------
#training Decision Tree on full data set and predicting on training set
survived_full_train = clean_data[target]
features_full_train = clean_data[feature_labels]

decTree2 = DecisionTreeClassifier(min_samples_split=18, min_samples_leaf=9)
decTree2.fit(features_full_train, survived_full_train)
predictions_full_train = decTree2.predict(features_full_train)
score_full_train = decTree2.score(features_full_train, survived_full_train)
print('SVM - from full initain "train" dataset results:')
printRegResults(decTree2, clean_data, predictions_full_train, score_full_train)

# predicting test data with SVM
test_data = pd.read_csv('C:\\Users\\ilja.surikovs\\Documents\\GitHub\\coding_club\\titanic\\test.csv') 
test_data = formatData(test_data)
test_data = addNonLinearity(test_data)

features_test = test_data[feature_labels]
predictions_test = decTree2.predict(features_test)

test_data['Survived'] = predictions_test

answers = test_data[['PassengerId','Survived']]
answers.to_csv('C:\\Users\\ilja.surikovs\\Documents\\GitHub\\coding_club\\titanic\\answers.csv')

##----------------------------------------------------------------------------------

## try support vector machines
## train-set
##penalty = 0.1 * i
##print('penalty =',penalty)
#svm1 = SVC() # kernel = ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
#
#svm1.fit(features_train, survived_train) 
#score_svm_train = svm1.score(features_train, survived_train)
#predictions_train = svm1.predict(features_train)
#print('SVM - train-set results:')
#printRegResults(svm1, train_set, predictions_train, score_svm_train)
#
##valid-set
#predictions_valid = svm1.predict(features_valid)
#score_svm_valid = svm1.score(features_valid, survived_valid)
#print('SVM - validation-set results:')
#printRegResults(svm1, valid_set, predictions_valid, score_svm_valid)

#---------------------------------------------------------------------------------
##training SVM on full data set and predicting on training set
#survived_full_train = clean_data[target]
#features_full_train = clean_data[feature_labels]
#
#svm2 = SVC()
#svm2.fit(features_full_train, survived_full_train)
#predictions_full_train = svm2.predict(features_full_train)
#score_svm_full_train = svm2.score(features_full_train, survived_full_train)
#print('SVM - from full initain "train" dataset results:')
#printRegResults(svm2, clean_data, predictions_full_train, score_svm_full_train)
#
## predicting test data with SVM
#test_data = pd.read_csv('C:\\Users\\ilja.surikovs\\Documents\\GitHub\\coding_club\\titanic\\test.csv') 
#test_data = formatData(test_data)
#test_data = addNonLinearity(test_data)
#
#features_test = test_data[feature_labels]
#predictions_test = svm2.predict(features_test)
#
#test_data['Survived'] = predictions_test
#
#answers = test_data[['PassengerId','Survived']]
#answers.to_csv('C:\\Users\\ilja.surikovs\\Documents\\GitHub\\coding_club\\titanic\\answers.csv')
