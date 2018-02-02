import pandas as pd
import logging
import numpy as np
import sys
import matplotlib.pyplot as plt
import time
from sklearn.cross_validation import train_test_split
from random import shuffle

def feature_normalization(train, test):
    """Rescale the data so that each feature in the training set is in
    the interval [0,1], and apply the same transformations to the test
    set, using the statistics computed on the training set.

    Args:
        train - training set, a 2D numpy array of size (num_instances, num_features)
        test  - test set, a 2D numpy array of size (num_instances, num_features)
    Returns:
        train_normalized - training set after normalization
        test_normalized  - test set after normalization

    """
    train_normalized = (train - np.amin(train,axis = 0)) / (np.amax(train,axis = 0) - np.amin(train,axis = 0))
    test_normalized = (test - np.amin(train,axis = 0)) /  (np.amax(train,axis = 0) - np.amin(train,axis = 0))
    return (train_normalized, test_normalized)

def compute_square_loss(X, y, theta):
    """
    Given a set of X, y, theta, compute the square loss for predicting y with X*theta

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D array of size (num_features)

    Returns:
        loss - the square loss, scalar
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    loss = 0
    l = y - X.dot(theta)
    loss = sum(i*i for i in l)/num_instances
    return loss
#test
#X = np.array([[1,2],[4,5],[8,9]])
#Y = np.array([3,11,18])
#theta = np.array([1.0,1.0])

def compute_regularized_loss(theta, lambda_reg):

    loss = lambda_reg * sum(i*i for i in theta)
    return loss

def compute_total_loss(X, y, theta, lambda_reg):
    '''
    compute squared loss + regularized loss
    '''
    num_instances, num_features = X.shape[0], X.shape[1]
    loss = 0
    l = y - X.dot(theta)
    loss = sum(i*i for i in l)/num_instances + lambda_reg * theta.dot(theta)
    return loss


def compute_square_loss_gradient(X, y, theta):
    """
    Compute gradient of the square loss (as defined in compute_square_loss), at the point theta.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)

    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    grad = -2*(y-X.dot(theta)).dot(X)/num_instances
    return grad

   
def grad_checker(X, y, theta, epsilon=0.01, tolerance=1e-4):
     """Implement Gradient Checker
    Check that the function compute_square_loss_gradient returns the
    correct gradient for the given X, y, and theta.

    Let d be the number of features. Here we numerically estimate the
    gradient by approximating the directional derivative in each of
    the d coordinate directions:
    (e_1 = (1,0,0,...,0), e_2 = (0,1,0,...,0), ..., e_d = (0,...,0,1)

    The approximation for the directional derivative of J at the point
    theta in the direction e_i is given by:
    ( J(theta + epsilon * e_i) - J(theta - epsilon * e_i) ) / (2*epsilon).

    We then look at the Euclidean distance between the gradient
    computed using this approximation and the gradient computed by
    compute_square_loss_gradient(X, y, theta).  If the Euclidean
    distance exceeds tolerance, we say the gradient is incorrect.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        epsilon - the epsilon used in approximation
        tolerance - the tolerance error

    Return:
        A boolean value indicate whether the gradient is correct or not

    """
    true_gradient = compute_square_loss_gradient(X, y, theta) #the true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features)
    
    for i in range(num_features):
        theta_plus = np.copy(theta)
        theta_minus = np.copy(theta)
        theta_plus[i] = theta_plus[i] + epsilon
        theta_minus[i] -= epsilon
        ag = (compute_square_loss(X, y, theta_plus) - 
        compute_square_loss(X, y, theta_minus))/(2*epsilon)
        approx_grad[i] = ag
        if abs(approx_grad[i] - true_gradient[i]) > tolerance:
            return False
    
    return True


def generic_gradient_checker(X, y, theta, objective_func, 
gradient_func, epsilon=0.01, tolerance=1e-4):
    generic_gradient_checker(X, y, theta, objective_func, gradient_func, epsilon=0.01, tolerance=1e-4):
    """
    The functions takes objective_func and gradient_func as parameters. And check whether gradient_func(X, y, theta) returned
    the true gradient for objective_func(X, y, theta).
    Eg: In LSR, the objective_func = compute_square_loss, and gradient_func = compute_square_loss_gradient
    """
    true_gradient = gradient_func(X, y, theta) #the true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features)
    
    for i in range(num_features):
        theta_plus = np.copy(theta)
        theta_minus = np.copy(theta)
        theta_plus[i] = theta_plus[i] + epsilon
        theta_minus[i] -= epsilon
        ag = (objective_func(X, y, theta_plus) - objective_func(X, y, theta_minus))/(2*epsilon)
        approx_grad[i] = ag
        if abs(approx_grad[i] - true_gradient[i]) > tolerance:
            return False
    
    return True

def batch_grad_descent(X, y, alpha=0.1, num_iter=1000, check_gradient=False, stop_if = 0, stop = 0.01):
    """
    In this question you will implement batch gradient descent to
    minimize the square loss objective

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        num_iter - number of iterations to run
        check_gradient - a boolean value indicating whether checking the gradient when updating
        stop_if - whether to use stop criteria
        stop - stop criteria: if the difference between two iteration is smaller than stop, then break and return

    Returns:
        theta_hist - store the the history of parameter vector in iteration, 2D numpy array of size (num_iter+1, num_features)
                    for instance, theta in iteration 0 should be theta_hist[0], theta in ieration (num_iter) is theta_hist[-1]
        loss_hist - the history of objective function vector, 1D numpy array of size (num_iter+1)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #initialize loss_hist
    theta = np.zeros(num_features) #initialize theta
    theta_hist[0] = theta
    loss_hist[0] = compute_square_loss(X, y, theta)
    if check_gradient == True & grad_checker(X, y, theta, epsilon=0.01, tolerance=1e-4) == False:
        return False
    for i in range(num_iter):
        if check_gradient == True and not grad_checker(X, y, theta, epsilon=0.01, tolerance=1e-4):
            return False
        gradient = compute_square_loss_gradient(X, y, theta)
        theta = theta - alpha * gradient
        theta_hist[i+1] = theta
        loss_hist[i+1] = compute_square_loss(X, y, theta)
        if stop_if == 1:
            if loss_hist[i+1]-loss_hist[i-1] < stop:
                break
    return theta_hist, loss_hist

#compute the loss_hist using batch_grad_descent, and plot the graph of squared loss against steps using alpha = 0.05 and 0.01.
theta_hist, loss_hist1 = batch_grad_descent(X_train, y_train, alpha=0.05, num_iter=1000, check_gradient=True)
theta_hist, loss_hist2 = batch_grad_descent(X_train, y_train, alpha=0.01, num_iter=1000, check_gradient=True)

plt.figure(figsize=(10, 5))
plt.plot(loss_hist1,label='alpha=0.05')
plt.plot(loss_hist2,label='alpha=0.01')
plt.xlabel("n(lambda = 10^n)")
plt.ylabel("squared loss")
plt.legend()
plt.savefig('242.png')
plt.show()


#check the compute time for batch_grad_descent using alpha = 0.05 and 0.01.
speed = []
for alpha in [0.05, 0.01]:
    start = time.time()
    theta_hist, loss_hist1 = batch_grad_descent(X_train, y_train, alpha=alpha, num_iter=1000, check_gradient=False)
    end = time.time()
    speed.append(end - start)
print(speed)


def batch_grad_descent_back(X, y, beta = 0.4, alpha=0.1, num_iter=1000, check_gradient=False, stop_if = 0, stop = 0.01):
    '''
    Implement backtracking line search to do batch gradient descent to
    minimize the square loss objective
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        num_iter - number of iterations to run
        check_gradient - a boolean value indicating whether checking the gradient when updating
        stop_if - whether to use stop criteria
        stop - stop criteria: if the difference between two iteration is smaller than stop, then break and return
    Returns:
        theta_hist - store the the history of parameter vector in iteration, 2D numpy array of size (num_iter+1, num_features)
                    for instance, theta in iteration 0 should be theta_hist[0], theta in ieration (num_iter) is theta_hist[-1]
        loss_hist - the history of objective function vector, 1D numpy array of size (num_iter+1)
    """
    '''
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #initialize loss_hist
    theta = np.zeros(num_features) #initialize theta
    theta_hist[0] = theta
    loss_hist[0] = compute_square_loss(X, y, theta)
    temp = alpha
    for i in range(num_iter):
        alpha = temp
        gradient = compute_square_loss_gradient(X, y, theta)
        while (compute_square_loss(X, y, theta - alpha * gradient) > compute_square_loss(X, y, theta) - alpha / 2 * np.sum([gr**2 for gr in gradient])):
            alpha *= beta
        theta = theta - alpha * gradient
        theta_hist[i+1] = theta
        loss_hist[i+1] = compute_square_loss(X, y, theta)
        if stop_if == 1:
            if loss_hist[i+1]-loss_hist[i-1] < stop:
                break
    return theta_hist, loss_hist

#check the compute time for batch_grad_descent using alpha = 0.05 and 0.01.

speed = []
for beta in [0.8, 0.6, 0.4, 0.2]:
    start = time.time()
    theta_hist, loss_hist1 = batch_grad_descent_back(X_train, y_train, beta = beta, alpha=0.5, num_iter=1000, check_gradient=True, stop_if = 1)
    end = time.time()
    speed.append(end - start)
print(speed)

#compute the loss_hist using batch_grad_descent_back, and plot the graph of squared loss against steps using beta = 0.8, 0.6, 0.4, 0.2.

theta_hist, loss_hist1 = batch_grad_descent(X_train, y_train, beta = 0.8, alpha=1, num_iter=1000, check_gradient=True)
theta_hist, loss_hist2 = batch_grad_descent(X_train, y_train, beta = 0.6, alpha=1, num_iter=1000, check_gradient=True)
theta_hist, loss_hist3 = batch_grad_descent(X_train, y_train, beta = 0.4, alpha=1, num_iter=1000, check_gradient=True)
theta_hist, loss_hist4 = batch_grad_descent(X_train, y_train, beta = 0.2, alpha=1, num_iter=1000, check_gradient=True)

plt.figure(figsize=(8, 5))
plt.plot(loss_hist4,label='beta = 0.2')
plt.plot(loss_hist3,label='beta = 0.4')
plt.plot(loss_hist1,label='beta = 0.8')
plt.plot(loss_hist2,label='beta = 0.6')
plt.xlabel("iter")
plt.ylabel("squared loss")
plt.legend()
plt.savefig('243.png')
plt.show()




def compute_regularized_square_loss_gradient(X, y, theta, lambda_reg):
    """
    Compute the gradient of L2-regularized square loss function given X, y and theta

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        lambda_reg - the regularization coefficient

    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    grad = -2*((y-X.dot(theta)).dot(X))/num_instances+2*lambda_reg * theta
    return grad

def regularized_grad_descent(X, y, alpha=0.1, lambda_reg=1, num_iter=1000):
    """
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        lambda_reg - the regularization coefficient
        numIter - number of iterations to run

    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size (num_iter+1, num_features)
        loss_hist - the history of loss function without the regularization term, 1D numpy array.
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #initialize loss_hist
    theta = np.zeros(num_features) #initialize theta
    theta_hist[0] = theta
    loss_hist[0] = compute_square_loss(X, y, theta)
    for i in range(num_iter):
        gradient = compute_regularized_square_loss_gradient(X, y, theta, lambda_reg)
        theta = theta - alpha * gradient
        theta_hist[i+1] = theta
        loss_hist[i+1] = compute_square_loss(X, y, theta)
    return theta_hist, loss_hist

'''
For regression problems, we may prefer to leave the bias term unregularized. 
One approach is to change J(θ) so that the bias is separated out from the other parameters and left unregularized. 
Another approach that can achieve approximately the same thing is to use a very large number B, rather than 1, 
for the extra bias dimension.
'''
def regularized_grad_descent_plot_b(X, y, X_test, y_test, bs, alpha=0.01, num_iter=1000):
    """
    Plot the loss vs iteration graph using different b.
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        bs - the list of b that is used to minimize loss
        alpha - step size in gradient descent
        lambda_reg - the regularization coefficient
        numIter - number of iterations to run
    """
    lambda_reg = 0.2
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #initialize loss_hist
    loss_regularize = np.zeros(len(bs))
    loss_test = np.zeros(len(bs))
    for i, b in enumerate(bs): 
        X_b = np.copy(X)
        X_b[:,-1] *= b
        X_test_b = np.copy(X)
        X_test_b[:,-1] *= b
        num_instances, num_features = X.shape[0], X.shape[1]
        num_iter = int(1000*b)
        theta_hist, loss_hist = regularized_grad_descent(X_b, y, alpha=alpha/np.sqrt(b), lambda_reg = lambda_reg, num_iter=num_iter)
        theta = theta_hist[num_iter]
        loss_regularize[i] = compute_regularized_loss(theta, lambda_reg)
        loss_test[i] = compute_square_loss(X_test, y_test, theta)
        
    plt.figure(figsize=(8, 4))
    plt.plot(bs,loss_test,label='test loss')
    plt.xlabel("b")
    plt.ylabel("squared loss")
    plt.xlim(1,10)
    plt.legend()
    plt.savefig('2561.png')
    plt.show()
    plt.plot(bs,loss_regularize,label='regularized loss')
    plt.xlabel("b")
    plt.ylabel("squared loss")
    plt.legend()
    plt.savefig('2562.png')
    plt.show()
    return loss_regularize, loss_test
# try to find out the best b that minimize the total loss.
bs=[1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.5, 3, 4, 10]
regularized_grad_descent_plot_b(X_train, y_train, X_test, y_test, bs, num_iter=1000)



def regularized_grad_descent_plot_l(X_train, y_train, X_test, y_test, expos, alpha=0.025, num_iter=1000):
    """
    Plot the loss vs iteration graph using different lambda.
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        bs - the list of b that is used to minimize loss
        alpha - step size in gradient descent
        expos- lambda_reg = 10 ** expos
        numIter - number of iterations to run

    """
    num_instances, num_features = X_train.shape[0], X_train.shape[1]
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #initialize loss_hist
    losses_train = np.zeros(len(expos))
    losses_test = np.zeros(len(expos))
    lambda_reg=[pow(10, expo) for expo in expos]
    for i, expo in enumerate(expos): 
        num_instances, num_features = X_train.shape[0], X_train.shape[1]
        theta_hist, loss_hist = regularized_grad_descent(X_train, y_train, alpha=alpha, lambda_reg = lambda_reg[i], num_iter=num_iter)
        theta = theta_hist[num_iter]
        losses_train[i] = compute_square_loss(X_train, y_train, theta)
        losses_test[i] = compute_square_loss(X_test, y_test, theta)
    plt.figure(figsize=(10, 5))
    plt.plot(np.log10(lambda_reg),losses_train,label='Train')
    plt.plot(np.log10(lambda_reg),losses_test,label='Test')
    plt.xlabel("log(lambda_reg)")
    plt.ylabel("squared loss")
    plt.legend()
    plt.savefig('2571.png')
    plt.show()
    return losses_train, losses_test
# try to find out the best lambda that minimize the total loss.

expos = [-7, -5, -3, -1, 0, 1, 2]
#expos = [-3, -2.5, -2, -1.75, -1.5, -1.25, -1, -0.5]
losses_train, losses_test = regularized_grad_descent_plot_l(X_train, y_train, X_test,
                                                        y_test, expos, alpha=0.025, num_iter=1000)



def stochastic_grad_descent_wr(X_, y, alpha=0.025, b=2.5, lambda_reg = pow(10,-1.75), num_iter=1000):
    """
    In this question you will implement stochastic gradient descent with a regularization term
    This implementation of SGD is with replacement
    Args:
        X_ - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - string or float. step size in gradient descent
                NOTE: In SGD, it's not always a good idea to use a fixed step size. Usually it's set to 1/sqrt(t) or 1/t
                if alpha is a float, then the step size in every iteration is alpha.
                if alpha == "1/sqrt(t)", alpha = 1/sqrt(t)
                if alpha == "1/t", alpha = 1/t
        lambda_reg - the regularization coefficient
        num_iter - number of epochs (i.e number of times) to go through the whole training set

    Returns:
        theta_hist - the history of parameter vector, 3D numpy array of size (num_iter, num_instances, num_features)
        loss hist - the history of regularized loss function vector, 2D numpy array of size(num_iter, num_instances)
    """
    X = np.copy(X_)
    X[:,-1] *= b
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.ones(num_features) #Initialize theta
    theta_hist = np.zeros((num_iter, num_instances, num_features))  #Initialize theta_hist
    loss_hist = np.zeros((num_iter, num_instances)) #Initialize loss_hist
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #initialize loss_hist
    theta_hist[0] = theta
    loss_hist[0] = compute_square_loss(X, y, theta)
    for i in range(num_iter):
        random_ind = np.random.choice(num_instances, replace=True)
        gradient = compute_regularized_square_loss_gradient(X[random_ind, :].reshape(1,X[random_ind, :].shape[0]), y[random_ind], theta, lambda_reg)
        theta = theta - alpha * gradient
        theta_hist[i+1] = theta
        loss_hist[i+1] = compute_square_loss(X, y, theta)
    return theta_hist, loss_hist


def stochastic_grad_descent(X_, y, alpha_=0.025, b=1.6, lambda_reg = pow(10,-1.75), num_iter=2000):
    """
    In this question you will implement stochastic gradient descent with a regularization term
    This implementation of SGD is without replacement. One iteration is one epoch (will go over all the points in X_)

    Args:
        X_ - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - string or float. step size in gradient descent
                NOTE: In SGD, it's not always a good idea to use a fixed step size. Usually it's set to 1/sqrt(t) or 1/t
                if alpha is a float, then the step size in every iteration is alpha.
                if alpha == "1/sqrt(t)", alpha = 1/sqrt(t)
                if alpha == "1/t", alpha = 1/t
        lambda_reg - the regularization coefficient
        num_iter - number of epochs (i.e number of times) to go through the whole training set

    Returns:
        theta_hist - the history of parameter vector, 3D numpy array of size (num_iter, num_instances, num_features)
        loss hist - the history of regularized loss function vector, 2D numpy array of size(num_iter, num_instances)
    """
    X = np.copy(X_)
    X[:,-1] *= b
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.ones(num_features) #Initialize theta
    theta_hist = np.zeros((num_iter, num_instances, num_features))  #Initialize theta_hist
    loss_hist = np.zeros((num_iter, num_instances)) #Initialize loss_hist
    losses = np.zeros(num_iter)
    temp_index = 0
    start = 0
    if not isinstance(alpha_,float):
        start = 100
    for i in range(start,num_iter):
        if not isinstance(alpha_,float):
            if (alpha_ == "1/t"):
                alpha = 1/i
            elif (alpha_ == "1/sqrt(t)"):
                alpha =  1/np.sqrt(i)
            elif (alpha_ == "mode3"):
                alpha = 0.01
                alpha =  alpha/(1+alpha*lambda_reg*i)
            else:
                print ("erro")
                return None, None
        else:
            alpha = alpha_
        index = np.array(range(num_instances))
        np.random.shuffle(index)
        for k, j in enumerate(index):
            gradient = compute_regularized_square_loss_gradient(X[j, :].reshape(1,X[j, :].shape[0]), y[j], theta, lambda_reg)
            theta = theta - alpha * gradient
            theta_hist[i-start, j] = theta
            loss = compute_square_loss(X[j, :].reshape(1,X[j, :].shape[0]), y[j], theta)
            if loss > 999999999999:
                print( '{} overflow'.format(alpha_))
                return None, None 
            loss_hist[i-start, j] = loss
    return theta_hist, loss_hist

for i, alpha in enumerate([0.005, "1/t", "mode3"]):
    theta_hist, loss_hist = stochastic_grad_descent(X_train, y_train, alpha, lambda_reg=lambda_reg, num_iter=200)
    losses = [np.average(loss_hist[i, :])for i in range(len(loss_hist))]
    plt.figure(figsize=(10, 5))
    plt.plot(np.log10(losses), label = str(alpha))
    plt.xlabel("epoch")
    plt.ylabel("log squared loss")
    plt.legend()
    plt.savefig(str(i) +'.png'.format(alpha))
    plt.show()

for i, alpha in enumerate([0.005, "1/t", "mode3"]):
    theta_hist, loss_hist = stochastic_grad_descent(X_train, y_train, alpha, lambda_reg=lambda_reg, num_iter=1000)
    losses = loss_hist.flat[::100]
    plt.figure(figsize=(8, 4))
    plt.plot(np.log10(losses), label = str(alpha))
    plt.xlabel("epoch")
    plt.ylabel("log squared loss")
    plt.legend()
    plt.savefig(str(i+5) +'.png'.format(alpha))
    plt.show()


lambda_reg = pow(10,-1.75)
alpha = 0.001
theta_hist, loss_hist, losses = stochastic_grad_descent(X_train, y_train, 0.005, lambda_reg=lambda_reg, num_iter=1000)
plt.figure(figsize=(10, 5))
plt.plot(np.log10(losses))
plt.xlabel("iteration)")
plt.ylabel("squared loss")

plt.legend()
#plt.savefig('2571.png')
plt.show()


print(loss_hist[-1])
print(theta[-1,:])
alpha = 0.001
theta_hist, loss_hist = stochastic_grad_descent(X_train, y_train, alpha=alpha, lambda_reg=lambda_reg, num_iter=1000)
print(loss_hist[-1])
print(theta_hist[-1,:])

def stochastic_grad_descent_decreasing_steps(X_, y, alpha_=0.025, b=2.5, lambda_reg = pow(10,-1.75), power = 1, option = 0, num_iter=1000):
    """
    This method is SGD with replacement
    In this question you will implement stochastic gradient descent with a regularization term

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - string or float. step size in gradient descent
                NOTE: In SGD, it's not always a good idea to use a fixed step size. Usually it's set to 1/sqrt(t) or 1/t
                if alpha is a float, then the step size in every iteration is alpha.
                if alpha == "1/sqrt(t)", alpha = 1/sqrt(t)
                if alpha == "1/t", alpha = 1/t
        lambda_reg - the regularization coefficient
        num_iter - number of epochs (i.e number of times) to go through the whole training set

    Returns:
        theta_hist - the history of parameter vector, 3D numpy array of size (num_iter, num_instances, num_features)
        loss hist - the history of regularized loss function vector, 2D numpy array of size(num_iter, num_instances)
    """
    X = np.copy(X_)
    X[:,-1] *= b
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.ones(num_features) #Initialize theta
    theta_hist = np.zeros((num_iter, num_instances, num_features))  #Initialize theta_hist
    loss_hist = np.zeros((num_iter, num_instances)) #Initialize loss_hist
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #initialize loss_hist
    theta_hist[0] = theta
    loss_hist[0] = compute_total_loss(X, y, theta, lambda_reg)
    start_step = 100
    for i in range(start_step, num_iter):
        if option == 0:
            alpha = 1/pow(i,power)
        else:
            alpha = alpha_/(1+alpha_*lambda_reg*i)
        random_ind = np.random.choice(num_instances, replace=True)
        gradient = compute_regularized_square_loss_gradient(X[random_ind, :].reshape(1,X[random_ind, :].shape[0]), y[random_ind], theta, lambda_reg)
        theta = theta - alpha * gradient
        theta_hist[i+1] = theta
        loss_hist[i+1] = compute_total_loss(X, y, theta, lambda_reg)
    plt.figure(figsize=(10, 5))
    plt.plot(range(start_step+1, num_iter+1),np.log10(loss_hist[start_step+1:]),label='Train')
    plt.xlabel("num_iter")
    plt.ylabel("squared loss")
    plt.legend()
    plt.show()

    return theta_hist, loss_hist

theta_hist, loss_hist = stochastic_grad_descent_decreasing_steps(X_train, y_train, alpha_= 0.01, power = 1, option = 0, num_iter=50000)
print(loss_hist[-1])

print(theta_hist[-1,:])



#read data and build trainning and test set
df = pd.read_csv('/Users/hp/Desktop/Mahine Learning/1 Statiistical Learning Theory and Stochastic Gradient Descent /hw1-sgd/data.csv', delimiter=',')
X = df.values[:,:-1]
y = df.values[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =100, random_state=10)

X_train, X_test = feature_normalization(X_train, X_test)
X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # Add bias term
X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1)))) # Add bias term



