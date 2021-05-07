from sklearn.datasets import load_files
import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import bz2;
from tqdm import tqdm;
from math import exp;

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier, LogisticRegression
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

from sklearn.calibration import CalibratedClassifierCV

class Options:
    resample = True
    weighting = 'score'
    score="f1"

opts = Options()

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
 

def vote(X_train, y_train, X_test, y_test):
    for clf, name in (
            (MultinomialNB(alpha=.001),"Multinomial Naive Bayes"),
            (MultinomialNB(alpha=.01),"Multinomial Naive Bayes"),
            (MultinomialNB(alpha=.1),"Multinomial Naive Bayes"),
            (BernoulliNB(alpha=.001), "Bernoulli Bayes"),
            (BernoulliNB(alpha=.01), "Bernoulli Bayes"),
            (BernoulliNB(alpha=.1), "Bernoulli Bayes"),
    #-        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
    #-        (Perceptron(n_iter=50), "Perceptron"),
    #-        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
#            (KNeighborsClassifier(n_neighbors=10), "kNN"),
#            (RandomForestClassifier(n_estimators=100), "Random forest"),
     #-       (ExtraTreesClassifier(n_estimators=100), "ExtraTree"),
            (SGDClassifier(alpha=.001, max_iter=500,loss="modified_huber",penalty="l2"), "SGD-l2"),
            (SGDClassifier(alpha=.001, max_iter=500,loss="modified_huber",penalty="l1"), "SGD-l1"),
            (LogisticRegression(penalty="l2",
                                    dual=False,
                                    tol=0.0001,
                                    C=1.0,
                                    fit_intercept=True,
                                    intercept_scaling=1,
                                    class_weight=None,
                                    random_state=None,
                                    solver="liblinear",
                                    max_iter=100,
                                    multi_class="ovr",
                                    verbose=0,
                                    warm_start=False,
                                    n_jobs=1), "MaxEnt"),
#            (SGDClassifier(alpha=.001, n_iter=500,loss="log",penalty="elasticnet"), "SGD-elastic"),
#            (CalibratedClassifierCV(SGDClassifier(alpha=.001, n_iter=500,penalty="elasticnet")), "SGD-elastic"),
#            (CalibratedClassifierCV(LinearSVC(penalty="l2", dual=False,tol=1e-3)),"L-SVC-l2"),  # turns decision_function to predict_proba

            ):
        print(clf)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_train)
        print("Training error (BIAS)")
        print(metrics.classification_report(y_train, pred))
    
        pred = clf.predict(X_test)
        print("Validation")
        print(pred.shape)
        print(y_test.shape)
        print(metrics.classification_report(y_test, pred))
    
        P = clf.predict_proba(X_test)
    
        direc = np.random.rand(10,2)
        res = minimize(optimize_kernel, [0.01,0.01],[P,y_test,False], method='Powell', tol=1e-4, options={'disp':False, 'direc':direc})
    
        pred = aCL(P,res.x)
    
        print("Abstained Validation")
        print(metrics.classification_report(y_test, pred))
    
        print("abstained in %d of %d cases (%f)" % (np.sum(pred==2), len(y_test),np.sum(pred==2)/ len(y_test) ))
        print(metrics.confusion_matrix(y_test, pred))

        if opts.score=="precision":
            ps = metrics.precision_score(y_test, pred, average=None)
        elif opts.score=="f1":
            ps = metrics.f1_score(y_test, pred, average=None)
        elif opts.score=='f1squared':
            ps =  metrics.f1_score(y_test, pred, average=None)
            ps = [ x*x for x in ps]
        elif opts.score=='f1exp':
            ps =  metrics.f1_score(y_test, pred, average=None)
            ps = [ exp(x) for x in ps]
        else:
            raise "unknown score "+opts.score
        yield ps, pred




        


print("Load...")
with bz2.BZ2File("la-large-full/single-file.txt.bz2") as f:
    lines = f.readlines()
print("Found %d records" % len(lines))
print("Transform to NPY")
lines = [x.decode() for x in tqdm(lines)]
ds = [[l.split(" ")[0], l.split(" ")[1]," ".join(l.split(" ")[2:])] for l in tqdm(lines)]
ds = np.array(ds)
print(ds.shape)


print("Transform to sklearn sets")

class TextDataset:
    target=None
    data=None
    target_names=None

data_train = TextDataset();
data_train.target = (ds[ds[:,0] == 'left',1]=='residential')*1.0
data_train.data = ds[ds[:,0] == 'left',2]

data_train.target_names = ["commercial", "residential"]

data_test = TextDataset();
data_test.target=(ds[ds[:,0] == 'right',1]=='residential')*1.0
data_test.data = ds[ds[:,0] == 'right',2]
data_test.target_names=["commercial", "residential"]

#possibly resample here:
_, counts = np.unique(data_train.target, return_counts=True)
print(counts)
N = np.min(counts)
_, counts = np.unique(data_test.target, return_counts=True)
print(counts)
N = min(N, np.min(counts))
print("Sampling to %d" % (N))
np.random.seed(42);
if opts.resample:
    print("resampling")
    # selector for N
    select = np.hstack([
        np.random.choice(np.argwhere(data_train.target==0).squeeze(),N),
        np.random.choice(np.argwhere(data_train.target==1).squeeze(),N)
        ])
    data_train.target = data_train.target[select]
    data_train.data = data_train.data[select]

    select = np.hstack([
        np.random.choice(np.argwhere(data_test.target==0).squeeze(),N),
        np.random.choice(np.argwhere(data_test.target==1).squeeze(),N)
        ])
    data_test.target = data_test.target[select]
    data_test.data = data_test.data[select]
    print("finished resampling")






print("Data Setup complete")
print("Vectorize")
vectorizer = TfidfVectorizer(sublinear_tf=True, min_df = 0.001, max_df=0.2,
                                 stop_words='english')
X_train = vectorizer.fit_transform(data_train.data)
y_train = data_train.target

X_test = vectorizer.transform(data_test.data)
y_test = data_test.target


print(X_train.shape)
print(X_test.shape)

#
votes = [x for x in vote(X_train,y_train, X_test,y_test)]
v = [x[1] for x in votes]
w_0 = [x[0][0] for x in votes] # focus weight on residential layer
w_1 = [x[0][1] for x in votes] # focus weight on residential layer

w_0 = w_0 / np.sum(w_0)
w_1 = w_1 / np.sum(w_1)

if opts.weighting is None:
    votes_for_0 = np.average((np.array(v) == 0),axis=0)
    votes_for_1 = np.average((np.array(v) == 1),axis=0)
elif opts.weighting=='score':
    print("Using score" + opts.score)
    votes_for_0 = np.average((np.array(v) == 0),axis=0, weights = w_0)
    votes_for_1 = np.average((np.array(v) == 1),axis=0, weights= w_1)
    
votes_for_any = (votes_for_0 + votes_for_1 / 2)

P = np.transpose(np.vstack([
    votes_for_0 / (votes_for_0 + votes_for_1),
    votes_for_1 / (votes_for_0 + votes_for_1)]
    ))


pred = (votes_for_1 > votes_for_0)*1
#pred[votes_for_any <0.5] = 2

print("Directly Voted Abstained Validation")
print(metrics.classification_report(y_test, pred))
    
print("abstained in %d of %d cases" % (np.sum(pred==2), len(y_test)))
print("Abstaining Rate %f" % (float(np.sum(pred==2)) / len(y_test)))
print(metrics.confusion_matrix(y_test, pred))
#
#
#
#
#### now abstain from the ensemble
direc = np.random.rand(10,2)
res = minimize(optimize_kernel, [0.01,0.01],[P,y_test,False], method='Powell', tol=1e-4, options={'disp':False, 'direc':direc})
pred = aCL(P,res.x)

print("Abstained Ensemble of Abstaining Classifiers")
print(metrics.classification_report(y_test, pred))
    
print("abstained in %d of %d cases" % (np.sum(pred==2), len(y_test)))
print("Abstaining Rate %f" % (float(np.sum(pred==2)) / len(y_test)))
print(metrics.confusion_matrix(y_test, pred))

## Finally, resolve these to actual classifications of tweets that can be rendered in the end.
#
#len(data_test.filenames[pred==1])
#
#
#
#
## question: can we abstain such that we maximize not the entropy, but the performance on the interesting class?
#
#
##def resolve_vovtes(votes,weights=None):
##    result=dict()
# #   v = [x[1] for x in votes]
# #   for i,item in enumerate(np.transpose(v)):
# #       unique, counts = np.unique(item, return_counts=True)
# #       result[i] = dict(zip(unique,counts))
# #   return result
#
#
#
##res = resolve_votes(v)
#
#
#
#
#
