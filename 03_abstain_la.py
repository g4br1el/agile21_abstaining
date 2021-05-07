from sklearn.datasets import load_files
import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix


from abstaining import aCL, NI
    

def grid_search(P,y, verbose=False):
    ''' A pretty simple grid search with visualization of the 2D space'''
    img = np.zeros([25,25]);
    r = np.linspace(0,1,25);
    best_indices=None
    best_NI = 0
    
    for i1,t1 in enumerate(r):
        for i2,t2 in enumerate(r):
            c = aCL(P,np.array([t1,t2]))
            ni= NI(y,c,2)
            this_NI =  ni.NI()
            img[i1,i2]  = this_NI
            if this_NI > best_NI:
                best_NI = this_NI
                best_T = np.array([t1,t2])
            if verbose:
                print("%f %f --- %f" % (t1,t2,ni.NI()))
    print( "Optimization Result (Grid Search):%f %f --- %f" %(best_T[0],best_T[1], best_NI) )
    return best_NI, best_T, img

def optimize_kernel(x,args):
    ''' A kernel to be minimized, args are P and y and verbose  '''
    c=aCL(args[0], np.array(x))
    if (args[2]):
        print("params",x);
    ni = NI(args[1],c,2) # information with respect to target.
    return 1-ni.NI(); # minimizing this maximizes the function




print("Load Training")
data_train = load_files("./la/left")
print("Load Testing")
data_test = load_files("./la/right")

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
X_train = vectorizer.fit_transform(data_train.data)
y_train = data_train.target

X_test = vectorizer.transform(data_test.data)
y_test = data_test.target


for clf, name in ( 
             #RidgeClassifier(tol=1e-2, solver="lsqr") # no proba
#             Perceptron(n_iter=50),#no proba
            (MultinomialNB(alpha=.001),"Multinomial Naive Bayes"),
            (MultinomialNB(alpha=.01),"Multinomial Naive Bayes"),
            (MultinomialNB(alpha=.1),"Multinomial Naive Bayes"),
            (BernoulliNB(alpha=.001), "Bernoulli Bayes"),
            (BernoulliNB(alpha=.01), "Bernoulli Bayes"),
    #-        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
    #-        (Perceptron(n_iter=50), "Perceptron"),
    #-        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
#            (KNeighborsClassifier(n_neighbors=10), "kNN"),
#            (RandomForestClassifier(n_estimators=100), "Random forest"),
     #-       (ExtraTreesClassifier(n_estimators=100), "ExtraTree"),
            (SGDClassifier(alpha=.001, max_iter=500,loss="modified_huber",penalty="l2"), "SGD-l2"),
            (SGDClassifier(alpha=.001, max_iter=500,loss="modified_huber",penalty="l1"), "SGD-l1"),
            ):
    print(name)
    print("="*60)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_train)
    print("Training error (BIAS)")
    print(metrics.classification_report(y_train, pred))


    pred = clf.predict(X_test)
    print("Validation")
    print(metrics.classification_report(y_test, pred))


    P = clf.predict_proba(X_test)
    direc = np.random.rand(10,2)
    res = minimize(optimize_kernel, [0.1,0.13],[P,y_test,False], method='Powell', tol=1e-4, options={'disp':False, 'direc':direc})
    pred = aCL(P,res.x)


    print("Abstained Validation")
    print(pred.shape)
    print(metrics.classification_report(y_test, pred))

    print("abstained in %d of %d cases" % (np.sum(pred==2), len(y_test)))
    print("A = %f"% (float(np.sum(pred==2)) /  len(y_test)))
    print(metrics.confusion_matrix(y_test, pred))
