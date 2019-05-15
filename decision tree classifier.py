from time import time
features_train, features_test, labels_train, labels_test = preprocess()
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
clf= DecisionTreeClassifier(min_samples_split=40)
clf.fit(features_train,labels_train)
t0=time()
pred=clf.predict(features_test)

print('predicting time: ', round(time()-t0,3),'s')



accuracy = accuracy_score(pred, labels_test)

print ('\naccuracy = {0}'.format(accuracy))
len(features_train[0])    
