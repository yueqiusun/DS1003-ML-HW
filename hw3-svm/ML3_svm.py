import os
import pickle
import random
import pandas as pd
import logging
import numpy as np
import sys
import matplotlib.pyplot as plt
import time
from sklearn.cross_validation import train_test_split
from random import shuffle
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.optimize import minimize
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, make_scorer
from collections import Counter
import time

#matrix inverse
from numpy.linalg import inv
# calculate norm linalg.norm
from numpy import linalg
import copy

'''
Note:  This code is just a hint for people who are not familiar with text processing in python. There is no obligation to use this code, though you may if you like. 
'''


def folder_list(path,label):
    '''
    PARAMETER PATH IS THE PATH OF YOUR LOCAL FOLDER
    '''
    filelist = os.listdir(path)
    review = []
    for infile in filelist:
        file = os.path.join(path,infile)
        r = read_data(file)
        r = list(r)
        r.append(label)
        review.append(r)
    return review

def read_data(file):
    '''
    Read each file into a list of strings. 
    Example:
    ["it's", 'a', 'curious', 'thing', "i've", 'found', 'that', 'when', 'willis', 'is', 'not', 'called', 'on', 
    ...'to', 'carry', 'the', 'whole', 'movie', "he's", 'much', 'better', 'and', 'so', 'is', 'the', 'movie']
    '''
    f = open(file)
    lines = f.read().split(' ')
    symbols = '${}()[].,:;+-*/&|<>=~" '
    table = str.maketrans("", "", symbols)
    words = map(lambda Element: Element.translate(table).strip(), lines)
    words = filter(None, words)
    return words
	
###############################################
######## YOUR CODE STARTS FROM HERE. ##########
###############################################

def shuffle_data():
    '''
    pos_path is where you save positive review data.
    neg_path is where you save negative review data.
    '''
    pos_path = "/Users/hp/GitHub/ML_HW/hw3-svm/data/neg"
    neg_path = "/Users/hp/GitHub/ML_HW/hw3-svm/data/pos"
	
    pos_review = folder_list(pos_path,1)
    neg_review = folder_list(neg_path,-1)
	
    review = pos_review + neg_review
    random.shuffle(review)
    return review 	
'''
Now you have read all the files into list 'review' and it has been shuffled.
Save your shuffled result by pickle.
*Pickle is a useful module to serialize a python object structure. 
*Check it out. https://wiki.python.org/moin/UsingPickle
'''
def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items())

def increment(d1, scale, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.

    NOTE: This function does not return anything, but rather
    increments d1 in place. We do this because it is much faster to
    change elements of d1 in place than to build a new dictionary and
    return it.
    """
    for f, v in d2.items():
        d1[f] = d1.get(f, 0) + v * scale
        
def bow_rep(word_list):
    return Counter(word_list)        
        
        
def loss_func(X, y, w, lambda_reg):
    num_instances = len(X)
    loss = 0
    for i in range(num_instances):
        loss += max(0, 1 - y[i] * dotProduct(X[i], w))
    loss /= num_instances
    loss += lambda_reg * dotProduct(w,w) / 2
    return loss

def percent_loss_func(X, y, w, lambda_reg):
    num_instances = len(X)
    loss = 0
    for i in range(num_instances):
        pred = dotProduct(X[i], w)
        if(pred * y[i] < 0):
            loss += 1
    loss /= num_instances
    return loss



def Pegasos_64(X, y, X_t, y_t, lambda_reg = 1, num_iter = 10, stop_diff = 10**-6):
    start = time.time()
    w={}
    t = 0
    num_instances = len(X)
    loss_hist = np.zeros(num_iter + 1) #initialize loss_hist
    loss_hist[0] = loss_func(X_t, y_t, w, lambda_reg)
    index = np.arange(num_instances)
    for i in range(num_iter):
        #np.random.shuffle(index)
        for j in index:
            t += 1
            eta = 1/(t*lambda_reg)
            if y[j]*dotProduct(X[j], w) < 1:
                increment(w, -eta*lambda_reg , w)
                increment(w, eta*y[j], X[j])
            else:
                increment(w, -eta*lambda_reg , w)
        loss_hist[i+1] = loss_func(X_t, y_t, w, lambda_reg)
        if abs(loss_hist[i+1]-loss_hist[i]) < stop_diff:
            print ('when lambda = {0}, Pegasos converge in {1} iteration\n'.format(lambda_reg, i))
            break
        if(i == num_iter - 1):
            print ('don\'t converge')
    end = time.time()
    print ('the time to run {0} iteration is {1}'.format(num_iter, end - start))
    return w, loss_hist[i+1]

def Pegasos_65(X, y, X_t, y_t, lambda_reg = 1, start_t = 2, num_iter = 100, stop_diff = 10**-8, loss_function = percent_loss_func):
    start = time.time()
    w={}
    w_test = {}
    t = start_t
    num_instances = len(X)
    loss_hist = np.zeros(num_iter + 1) #initialize loss_hist
    loss_hist[0] = loss_func(X_t, y_t, w, lambda_reg)
    index = np.arange(num_instances)
    s = 1
    w_small = {}
    for i in range(num_iter):
        start = time.time()
        #np.random.shuffle(index)
        
        for j in index:
            t += 1
            eta = 1/(t*lambda_reg)
            s *= (1 - eta * lambda_reg)
            if y[j]*dotProduct(X[j], w) < 1/s:
                increment(w, 1/s*eta*y[j], X[j])
        w_small = {}
        increment(w_small, s, w)
        loss_hist[i+1] = loss_function(X_t, y_t, w_small, lambda_reg)
        if abs(loss_hist[i+1]-loss_hist[i]) < stop_diff:
            print ('when lambda = {0}, Pegasos converge in {1} iteration\n'.format(lambda_reg, i))
            break
        if(i == num_iter - 1):
            print ('don\'t converge')
    end = time.time()
    print ('the time to run {0} iteration is {1}'.format(num_iter, end - start))
    return w_small, loss_hist[i+1]




def plot_Pegasos_lambda(X, y, X_t, y_t, start_log_lamda, end_log_lambda, interval, num_iter = 400):
    loss_record = []
    lambda_range = np.arange(start_log_lamda, end_log_lambda, interval)
    for i in lambda_range:
        lambda_reg = 10.0 ** i
        w68, loss_68 = Pegasos_65(X, y, X_t, y_t, lambda_reg = lambda_reg, start_t = 2, num_iter = num_iter, stop_diff = 10**-8, loss_function = percent_loss_func)
        loss_record.append(loss_68)
        print('when lambda = {0}, percentage loss is {1}'.format(lambda_reg, loss_68))
    plt.figure(figsize=(10, 5))
    plt.plot(lambda_range,loss_record)
    plt.xlabel("log(lambda_reg)")
    plt.ylabel("percentage loss")
    plt.legend()
    #    plt.savefig('2571.png')
    plt.show()
    
def Pegasos_610(X, y, X_t, y_t, lambda_reg = 1, start_t = 2, num_iter = 100, stop_diff = 10**-9, loss_function = percent_loss_func, diff_1 = 10 ** -8):
    start = time.time()
    w={}
    w_test = {}
    t = start_t
    num_instances = len(X)
    loss_hist = np.zeros(num_iter + 1) #initialize loss_hist
    loss_hist[0] = loss_func(X_t, y_t, w, lambda_reg)
    index = np.arange(num_instances)
    s = 1
    w_small = {}
    for i in range(num_iter):
        start = time.time()
        #np.random.shuffle(index)
        
        for j in index:
            t += 1
            eta = 1/(t*lambda_reg)
            s *= (1 - eta * lambda_reg)
            if y[j]*dotProduct(X[j], w) == 1/s:
                print('not differentiable')
            if abs(y[j]*dotProduct(X[j], w) - 1/s) < diff_1:
                print('almost not differentiable')
            if y[j]*dotProduct(X[j], w) < 1/s:
                increment(w, 1/s*eta*y[j], X[j])
        w_small = {}
        increment(w_small, s, w)
        loss_hist[i+1] = loss_function(X_t, y_t, w_small, lambda_reg)
        if abs(loss_hist[i+1]-loss_hist[i]) < stop_diff:
            print ('when lambda = {0}, Pegasos converge in {1} iteration\n'.format(lambda_reg, i))
            break
        if(i == num_iter - 1):
            print ('don\'t converge')
    end = time.time()
    print ('the time to run {0} iteration is {1}'.format(num_iter, end - start))
    return w_small, loss_hist[i+1]    

'''
main
'''

data_words = shuffle_data()
Y = []
for i in range(len(data_words)):
    Y.append(data_words[i][-1])
    del data_words[i][-1]
data = []
for entry in data_words:
    data.append(bow_rep(entry))

#import pickle
#with open('review.pkl', 'wb') as f:
#    pickle.dump(data_words, f)


train_X, test_X, train_y, test_y = train_test_split(data, Y, test_size = 0.25)

#6.6
w64, loss_64 = Pegasos_64(train_X, train_y, test_X, test_y, lambda_reg = 1, num_iter = 30)
w65, loss_65 = Pegasos_65(train_X, train_y, test_X, test_y, lambda_reg = 1, start_t = 2, num_iter = 30, stop_diff = 10**-8, loss_function = loss_func)
#6.8
plot_Pegasos_lambda(train_X, train_y, test_X, test_y, -4, 6, 1, num_iter = 400)
plot_Pegasos_lambda(train_X, train_y, test_X, test_y, -3.5, 0, 0.3, num_iter = 400)

#6.9
lambda_reg_best = 0.04
w69, loss69 = Pegasos_65(train_X, train_y, test_X, test_y, lambda_reg = lambda_reg_best, start_t = 2, num_iter = 400, stop_diff = 10**-8, loss_function = percent_loss_func)
wxs69 = []
for i in range(len(test_y)):
    flag = 0
    pred = dotProduct(test_X[i], w69)
    if(pred * test_y[i] < 0):
        #misclassified
        flag = 1
    wxs69.append((abs(pred),flag))
wxs69_sorted = sorted(wxs69, key=lambda tup: tup[0], reverse = True)
fold = 5
num_each_fold =len(test_y) / fold
for i in range(fold):
    start_index = 100 * i
    end_index = 100 * i +100
    percentage_error = np.average([flag69 for flag69 in [tup[1] for tup in wxs69_sorted[start_index : end_index]]])
    print('for fold {}, the percentage error is {}'.format(i+1, percentage_error))
    
#6.10
lambda_reg_best = 0.04
w10, loss10 = Pegasos_610(train_X, train_y, test_X, test_y, lambda_reg = lambda_reg_best, start_t = 2, num_iter = 400, stop_diff = 10**-8, loss_function = percent_loss_func, diff_1 = 10 ** -9)

#7
lambda_reg_best = 0.04
w7, loss7 = Pegasos_65(train_X, train_y, test_X, test_y, lambda_reg = lambda_reg_best, start_t = 2, num_iter = 400, stop_diff = 10**-8, loss_function = percent_loss_func)

for i in range(300):
    pred = dotProduct(test_X[i], w7)
    if(pred * test_y[i] < 0):
        print(i)
        
i_index = [266, 273]
for i in range(2):
    index = i_index[i]
    print('The result is as follows, the class for this review should be {0}, however, it has been misclassfied.'.format(test_y[index]))
    wx = {}
    entry = test_X[index]
    for key in entry:
        if key in w7:
            wx[key] = abs(w7[key] * entry[key])
            
    from operator import itemgetter
    wx = sorted(wx.items(), key=itemgetter(1), reverse = True)
    print('key      abs(wx)      w       x       xw')
    for i in range(10):
        tup = wx[i]
        key = tup[0]
        w = w7[key]
        x = entry[key]
        xw = w7[key] * entry[key]
        print('{0:8s}: {1}  {2}  {3}  {4}'.format(tup[0],round(tup[1],5),round(w,5),round(x,5),round(xw,5)))
#8

data_words = shuffle_data()
Y = []
for i in range(len(data_words)):
    Y.append(data_words[i][-1])
    del data_words[i][-1]
data = []
for entry in data_words:
    data.append(bow_rep(entry))
data_tfidf = []
key_list = []
for entry in data:
    for key in entry:
        if key not in key_list:
            key_list.append(key)
for entry in data:
    data_tfidf.append(entry)
#data_tfidf = [{'1':1, '2':2}, {'1':4, '2':6, '3':8}]
for entry in data_tfidf:
    total_num_words = len(entry)
    for key in entry:
        entry[key] /= total_num_words
inv_log_num_time_words = {}
for key in key_list:
    num = 0.0
    for entry in data:
        if entry[key] != 0:
            num += 1
    if num == 1:
        DF = 1/2
    else:
        DF = np.log10(num)  
    inv_log_num_time_words[key] = 1/DF

for entry in data_tfidf:
    for key in entry:
        entry[key] *= inv_log_num_time_words[key]
        
train_X_tfidf, test_X_tfidf, train_y_tfidf, test_y_tfidf = train_test_split(data_tfidf, Y, test_size = 0.25)

plot_Pegasos_lambda(train_X_tfidf, train_y_tfidf, test_X_tfidf, test_y_tfidf, -9, -3, 1, num_iter = 400)
plot_Pegasos_lambda(train_X_tfidf, train_y_tfidf, test_X_tfidf, test_y_tfidf, -7, -6, 0.1, num_iter = 400)


    

"""
draft
"""
'''
testing
'''



X = train_X
y = train_y
j = 0
lambda_reg = 1
num_iter = 100
'''
func draft
'''





def lasso_cood_descent_32(X_q, y, X_t, y_t, lambda_reg = 1, stop_diff = 10.0**-8, num_iter = 1000, Murphy = 1, cyclic = 1):
    """
    at each step we optimize over one component of the unknown parameter vector, 
    ﬁxing all other components. The descent path so obtained is a sequence of steps, 
    each of which is parallel to a coordinate axis in R^d , hence the name. 
    It turns out that for the Lasso optimization problem, we can ﬁnd a closed form 
    solution for optimization over a single component ﬁxing all other components. 
    This gives us the following algorithm, known as the shooting algorithm

    Args:
        X_q - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        X_t - the feature vector for tesing, 2D numpy array of size (num_instances, num_features)
        y_t - the label vector for tesing, 1D numpy array of size (num_instances)
        lambda_reg - the regulation parameter
        num_iter - number of iterations to run
        stop_diff - stop criteria: if the difference between two iteration is smaller than stop, then break and return
        Murphy - if Murphy = 1, start at the ridge regression solution suggested by Murphy, else start at 0
        cyclic - if cyclic = 1, do cyclic coordinate descent, else do randomized coordinate descent

    """
    
    #delete columns of all zeros, as these feature are of no use
    X, delete_ind = delete_all_zero(X_q)
    X_td = np.delete(X_t, delete_ind, 1)
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #initialize loss_hist
    
    if Murphy == 1:
        #start at the ridge regression solution suggested by Murphy
        w = inv(X.T.dot(X) + lambda_reg * np.identity(num_features)).dot(X.T).dot(y)
        label_M = 'start at the solution suggested by Murphy'
    else:
        w = np.zeros(num_features)
        label_M = 'starting at 0'
    
    theta_hist[0] = w
    loss_hist[0] = ridge_obj_loss(X, y, w, l1reg = lambda_reg)
    
    for i in range(num_iter):
        if cyclic == 1:
            cyclic_label = 'cyclic'
            index_list = np.arange(num_features)
        else:
            cyclic_label = 'randomized'
            index_list = np.arange(num_features)
            np.random.shuffle(index_list)
        # coodinate descent
        for j in index_list:
            a = 2 * X[:,j].dot(X[:,j])
            c = 2 * X[:,j].dot(y - X.dot(w) + w[j] * X[:,j])
            if c < -lambda_reg:
                w[j] = (c + lambda_reg) / a
            elif c > lambda_reg:
                w[j] = (c - lambda_reg) / a
            else:
                w[j] = 0
        theta_hist[i+1] = w
        loss_hist[i+1] = ridge_obj_loss(X, y, w, lambda_reg)
        if abs(loss_hist[i+1]-loss_hist[i]) < stop_diff:
            print ('when lambda = {0}, coodinate descent converge in {1} iteration\n'.format(lambda_reg, i))
            print('for {2} coodinate descent, the test squared loss is {0} using the solution {1}\n'.format(loss_func(X_td, y_t, w), label_M, cyclic_label))
            break
        if(i == num_iter - 1):
            print ('for {1} coodinate descent, the coodinate descent don\'t converge using the solution {0}'.format(label_M, cyclic_label))

        
        
#loss_record = []
#lambda_range = range(-4, 6)
#lambda_range = np.arange(-3.5, 0, 0.3)
#for i in lambda_range:
#    print(i)
#    lambda_reg = 10** i
#    loss = Pegasos_65(train_X, train_y, test_X, test_y, lambda_reg = lambda_reg, num_iter = 200)
#    loss_record.append(loss)
#plt.figure(figsize=(10, 5))
#plt.plot(lambda_range,loss_record)
#plt.xlabel("log(lambda_reg)")
#plt.ylabel("percentage loss")
#plt.legend()
##    plt.savefig('2571.png')
#plt.show()        

        