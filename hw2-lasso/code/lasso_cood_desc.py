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


from setup_problem import load_problem
from ridge_regression import RidgeRegression

#PLOT CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
import itertools

#matrix inverse
from numpy.linalg import inv


def delete_all_zero(X):
    delete_index = []
    num_instances, num_features = X.shape[0], X.shape[1]
    for i in range(num_features):
        if all(v == 0 for v in X[:, i]):
            delete_index.append(i)
    X_nz = np.delete(X, delete_index, 1)
    return X_nz, delete_index

def ridge_obj_loss(X, y, w, l1reg):
        num_instances, num_features = X.shape[0], X.shape[1]
        predictions = np.dot(X, w)
        residual = y - predictions
        empirical_risk = np.sum(residual**2) / num_instances
        l1_norm_squared = np.sum(w)
        objective = empirical_risk + l1reg * l1_norm_squared
        return objective
def loss_func(X, y, w):
        num_instances, num_features = X.shape[0], X.shape[1]
        predictions = np.dot(X, w)
        residual = y - predictions
        loss = np.sum(residual**2) / num_instances
        return loss       
    
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

def lasso_cood_descent_32_helper(X_q, y, X_t, y_t, lambda_reg = 1, stop_diff = 10.0**-8, num_iter = 1000):
    #do two for loop for different stating point and different styles of coodinate descent(cyclic or randomized)
    for m in [0, 1]:
        for c in [0, 1]:
            lasso_cood_descent_32(X_q, y, X_t, y_t, lambda_reg = lambda_reg, stop_diff = stop_diff, num_iter = num_iter, Murphy = m, cyclic = c)    

def lasso_cood_descent_33(X_q, y, x_t, y_t, stop_diff = 10.0**-8, num_iter = 1000, homotopy = 0, homotopy_para = 0.8, y_centered = 0):
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
        homotopy - if homotopy!=0, use homotopy method
        homotopy_para - reduce partmeter for homotopy method
        y_centered - if y_centered != 0, y is centered
    """
    
    
    if y_centered != 0:
        #center y
        y_ = (y-np.mean(y)) / np.std(y)
        y_t = (y_t-np.mean(y)) / np.std(y)
        y = y_
    #delete columns of all zeros, as these feature are of no use
    X, delete_ind = delete_all_zero(X_q)
    #x_t = np.sort(x_t)
    X_t = featurize(x_t)
    X_td = np.delete(X_t, delete_ind, 1)
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #initialize loss_hist
    
    index_list = np.arange(num_features)
    #record the validation loss for each lambda
    loss_record = []
    #record the prediction function for each lambda
    pred_fns = []
    if homotopy == 0:
        lambda_reg_list = 10.**np.arange(-6, 2)
    else:
        #homotopy method: start lambda from lambda_max, then reduce it repeatly. 
        #And the optimization problem is solved using the previous optimal 
        # point as the starting point.
        lambda_reg_list = 10.**np.arange(-6, 2)
        lambda_reg_list = []
        lambda_reg_max = np.max(2*X.T.dot(y))
        lambda_reg_r = lambda_reg_max
        for i in range(35):
            lambda_reg_list.append(lambda_reg_r)
            lambda_reg_r *= homotopy_para
    w = np.zeros(num_features)
    for lambda_reg in lambda_reg_list:
        if homotopy == 0:
            w = inv(X.T.dot(X) + lambda_reg * np.identity(num_features)).dot(X.T).dot(y)
            
        #w = np.zeros(num_features)
        theta_hist[0] = w
        loss_hist[0] = ridge_obj_loss(X, y, w, l1reg = lambda_reg)
        # coodinate descent
        for i in range(num_iter):
            np.random.shuffle(index_list)
            for j in range(num_features):
                a = 2 * X[:,j].dot(X[:,j])
                c = 2 * X[:,j].dot(y - X.dot(w) + w[j] * X[:,j])
                if c < -lambda_reg:
                    w[j] = (c + lambda_reg) / a
                elif c > lambda_reg:
                    w[j] = (c - lambda_reg) / a
                else:
                    w[j] = 0
    
                #return theta_hist, loss_hist
            theta_hist[i+1] = w
            loss_hist[i+1] = ridge_obj_loss(X, y, w, l1reg = lambda_reg)
            if abs(loss_hist[i+1]-loss_hist[i]) < stop_diff:
                loss_record.append(loss_func(X_td, y_t, w))

                print ('when lambda = 10^{0}, coodinate descent converge in {1} iteration)\n'.format(np.log10(lambda_reg), i))
                break
            if(i == num_iter - 1):
                loss_record.append(-1)
                print ('don\'t converge')
         
        # append the prediction function
        name = 'lambda = {}'.format(lambda_reg)
        
        x_ts = np.sort(x_t)
        X_ts = featurize(x_ts)
        X_tsd = np.delete(X_ts, delete_ind, 1)
    
    
        pred_fns.append({"name": name,
                 "coefs": w,
                 "preds": X_tsd.dot(w)})
    ii = 0
    for record in loss_record:
        print ('lambda = 10^{0}, loss = {1}'.format(np.log10(lambda_reg_list[ii]), record))
        ii += 1
    
    plt.figure(figsize=(8, 5))
    plt.plot(lambda_reg_list, loss_record)
    plt.xlabel("log10(lambda)")
    plt.ylabel('test square loss')
    plt.legend()
    params = {'legend.fontsize': 5,
          'legend.handlelength': 2}
    plt.rcParams.update(params)
    plt.tight_layout()
    #plt.savefig()
    plt.show()
    return pred_fns

#np.log10(lambda_reg_list[ii])
    
    
    
def plot_prediction_functions(x, pred_fns, x_train, y_train, legend_loc="bottom left"):
    # Assumes pred_fns is a list of dicts, and each dict has a "name" key and a
    # "preds" key. The value corresponding to the "preds" key is an array of
    # predictions corresponding to the input vector x. x_train and y_train are
    # the input and output values for the training data
    fig, ax = plt.subplots()
    ax.set_xlabel('Input Space: [0,1)')
    ax.set_ylabel('Action/Outcome Space')
    ax.set_title("Prediction Functions")
    plt.scatter(x_train, y_train, label='Training data')
    for i in range(len(pred_fns)):
        ax.plot(x, pred_fns[i]["preds"], label=pred_fns[i]["name"])
    legend = ax.legend(loc=legend_loc, shadow=True)
    return fig


def compare_parameter_vectors(pred_fns):
    # Assumes pred_fns is a list of dicts, and each dict has a "name" key and a
    # "coefs" key
    fig, axs = plt.subplots(len(pred_fns),1, sharex=True)
    num_ftrs = len(pred_fns[0]["coefs"])
    for i in range(len(pred_fns)):
        title = pred_fns[i]["name"]
        coef_vals = pred_fns[i]["coefs"]
        axs[i].bar(range(num_ftrs), coef_vals)
        axs[i].set_xlabel('Feature Index')
        axs[i].set_ylabel('Parameter Value')
        axs[i].set_title(title)
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.3)
    return fig



#lambda_reg_list = []
#lambda_reg_max = np.max(2*X.T.dot(y))
#lambda_reg_r = lambda_reg_max
#for i in range(35):
#    lambda_reg_list.append(lambda_reg_r)
#    lambda_reg_r *= homotopy_para
#X = X_train
##y_t = y_val
##X_q = X_train
#y = y_train
##num_iter = 1000
##stop_diff = 10.0**-8
#homotopy_para = 0.8


lasso_data_fname = "/Users/hp/GitHub/ML_HW/hw2-lasso/code/lasso_data.pickle"
x_train, y_train, x_val, y_val, target_fn, coefs_true, featurize = load_problem(lasso_data_fname)

X_train = featurize(x_train)
X_val = featurize(x_val)

#3.2
lasso_cood_descent_32_helper(X_train, y_train, X_val, y_val, lambda_reg = 1, stop_diff = 10.0**-8, num_iter = 1000)
#3.3
pred_fns = lasso_cood_descent_33(X_train, y_train, x_val, y_val, stop_diff = 10.0**-8, num_iter = 1000, homotopy = 0)
f332 = compare_parameter_vectors(pred_fns)
f332.show()
pred_fns.append({"name":"Bayes Optimal", "coefs":coefs_true, "preds": target_fn(np.sort(x_val)) })

f331 = plot_prediction_functions(np.sort(x_val), pred_fns, x_train, y_train, legend_loc="bottom left")
f331.show()

#lambda  avergafe

#3.4
pred_fns = lasso_cood_descent_33(X_train, y_train, x_val, y_val, stop_diff = 10.0**-8, num_iter = 1000, homotopy = 1)

#3.5
pred_fns = lasso_cood_descent_33(X_train, y_train, x_val, y_val, stop_diff = 10.0**-8, num_iter = 1000, homotopy = 0, y_centered = 1)


def projected_SGD (X_q, y, x_t, y_t, stop_diff = 10.0**-8, num_iter = 1000):

    X, delete_ind = delete_all_zero(X_q)
    X_td = np.delete(X_t, delete_ind, 1)
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #initialize loss_hist
    
    index_list = np.arange(num_features)
    loss_record = []
    lambda_reg_list = 10.**np.arange(-6, 2)

    for lambda_reg in lambda_reg_list:
        w = inv(X.T.dot(X) + lambda_reg * np.identity(num_features)).dot(X.T).dot(y)
        theta_hist[0] = w
        loss_hist[0] = ridge_obj_loss(X, y, w, l1reg = lambda_reg)
        for i in range(num_iter):
            np.random.shuffle(index_list)
            for j in range(num_features):
                w[:, j] += np.max(-w[:, j], gradient)
                #return theta_hist, loss_hist
            theta_hist[i+1] = w
            loss_hist[i+1] = ridge_obj_loss(X, y, w, l1reg = lambda_reg)
            if abs(loss_hist[i+1]-loss_hist[i]) < stop_diff:
                loss_record.append(loss_func(X_td, y_t, w))

                print ('when lambda = {0}, coodinate descent converge in {1} iteration)\n'.format(lambda_reg, i))
                break
            if(i == num_iter - 1):
                loss_record.append(-1)
                print ('don\'t converge')
                
    ii = 0
    for record in loss_record:
        print ('lambda = 10^{0}, loss = {1}'.format(np.log10(lambda_reg_list[ii]), record))
        ii += 1
    
    plt.figure(figsize=(8, 5))
    plt.plot(np.log10(lambda_reg_list), loss_record)
    plt.xlabel("log lambda")
    plt.ylabel('test square loss')
    plt.legend()
    params = {'legend.fontsize': 5,
          'legend.handlelength': 2}
    plt.rcParams.update(params)
    plt.tight_layout()
    #plt.savefig()
    
    plt.show()
    #w_best = np.hstack((w_best, np.zeros(len(delete_ind))))
    return pred_fns









lasso_data_fname = "/Users/hp/GitHub/ML_HW/hw2-lasso/code/lasso_data.pickle"
x_train, y_train, x_val, y_val, target_fn, coefs_true, featurize = load_problem(lasso_data_fname)

X_train = featurize(x_train)
X_val = featurize(x_val)


#6.1
#projected_SGD (X_q, y, x_t, y_t, stop_diff = 10.0**-8, num_iter = 1000)
