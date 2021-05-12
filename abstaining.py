"""
Big Geospatial Data Management 
(c) 2021 M.Werner <martinw.werner@tum.de> 
This is the core functionality shared across the various experiments and data generators
"""
import numpy as np
import json
from sklearn.metrics import confusion_matrix
import re
"""
Section 1: Data Generation for the Synthesis of Example Datasets
"""


def infinite_generator(lst):
    """
    infinite_generator (lst): turns a list into an endless iterator starting over from the beginning
    """
    i = 0
    while True:
        result = lst[i % len(lst)]
        i += 1
        yield result
    
def tweet_gen(document, length=12):
    """
    tweet_gen (doc, length): takes a list, turns it into an infinite generator and yields documents with length words.
    """ 
    words = document.split(" ")
    iterator = infinite_generator(words)
    while True:
        yield " ".join([next(iterator) for _ in range(length)])
    
def get_datasets(length = 12):
    """
    This function just provides iterators for our three datasets, each sampling k words.
    """
    # dr faust
    lines = " ".join([re.sub('[\.\r\n]',' ',x) for x in open("input/drfaust.txt","r")])
    document = re.sub(' +'," ",lines)
    drfaust = tweet_gen(document, length)
    # goehtes faust
    lines = " ".join([re.sub('[\.\r\n]',' ',x) for x in open("input/faust.txt","r")])
    document = re.sub(' +'," ",lines)
    faust = tweet_gen(document,length)
    # reviews (first 100k)
    lines = [json.loads(x)["reviewText"] for x in open("input/Books_5_100k.json","r")]
    lines = " ".join([re.sub('[\.\r\n]',' ',x) for x in lines])
    document = re.sub(' +'," ",lines)
    reviews = tweet_gen(document,length)
    return drfaust,faust,reviews

        

#"""
#Section 2: The abstaining methodology
#"""


def aCL(phi, T=None):
    ''' Abstaining Classification using rejection threshold vector T. '''
    if T is None:
        q = np.argmax(phi,axis=1)
        return(q)

    assert(phi.shape[1] == len(T))
    tv = np.max(phi / (1-T), axis=1)
    abstained_decisions = len(T) * (tv < 1) # set all of those abstaining to two
    cv = np.argmax(phi / (1-T), axis=1) * (tv >= 1) # non-abstained arg-max decisions
    decision = cv + abstained_decisions     #combination
    return (decision) ## just a hack


class NI:
    """
    Our Normalized Information Class
    """
    def __init__(self,t,y, m):
      np.seterr(all='ignore')
      self.cm = confusion_matrix(t,y)[0:m,:];
      if (self.cm.shape[1] != m+1):
        self.cm = np.c_[self.cm, np.zeros(m)]
      self.m = m
      self.n = np.sum(self.cm).astype(np.float32)           # total number
      self.C = np.sum(self.cm,axis=1).astype(np.float32)    # number in ith class
      self.D = np.sum(self.cm,axis=0).astype(np.float32)    # number in ith class

    def pj(self,i,j):
        ''' joint distribution'''
        val = self.cm[i][j] / self.n
#        print("p_j (%d,%d) = %f"% (i,j,val))
        return(val)
    def pt (self,i):
        '''marginal distribution related to target t'''
        val = self.C[i]/self.n
#        print("p_t (%d) = %f"% (i,val))
        return(val)
    
    def py (self,j):
        ''' marginal distribution related to target t'''
        val = self.D[j]/self.n
#        print("p_y (%d) = %f"% (j,val))
        return(val)
    def I(self):
        ''' The mutual information'''
        I=0
        for i in range(self.m):
            for j in range(self.m+1):
                term = self.pj(i,j) * np.log2(self.pj(i,j) / (self.pt(i) * self.py(j)))
                I += term
        if np.isnan(I):
            I=0
        return(I)
    def H(self):
        ''' The entropy of the one marginal distribution'''
        H=0
        for i in range(self.m):
            H -= self.pt(i) * np.log2(self.pt(i))
        return (H)
    def NI(self):
        '''Normalized Empirical Mutual Information with Abstaining Correction'''
        return (self.I() / self.H())

