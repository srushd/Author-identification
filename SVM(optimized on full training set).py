output----->
no. of Chris training emails: 7936
no. of Sara training emails: 7884

training time: 138.434 s
predicting time: 13.375 s

accuracy = 0.9908987485779295




Input--->>

#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("C:/Users/Srushti/Desktop/New folder (3)/ML udacity/ud120-projects-master/tools")



### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
from sklearn.svm import SVC

clf = SVC(C=10000,kernel='rbf',gamma='auto')

t0 = time()

clf.fit(features_train, labels_train)
print ("\ntraining time:", round(time()-t0,3), "s")

# predict
t0 = time()
pred = clf.predict(features_test)
print ("predicting time:", round(time()-t0, 3), "s")

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred, labels_test)

print ('\naccuracy = {0}'.format(accuracy))
print(sum(pred))
