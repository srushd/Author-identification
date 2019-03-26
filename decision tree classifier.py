#!/usr/bin/python
from sklearn.model_selection import train_test_split
""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("C:/Users/Srushti/Desktop/New folder (3)/ML udacity/ud120-projects-master/tools")
#from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###


#########################################################
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
clf = DecisionTreeClassifier(min_samples_split=40)
clf.fit(features_train,labels_train)
pred=clf.predict(features_test)



accuracy = accuracy_score(pred, labels_test)

print ('\naccuracy = {0}'.format(accuracy))
    
