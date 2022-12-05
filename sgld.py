# importing libraries 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.special import expit
from scipy.linalg import eigvals
from tqdm import tqdm
import time

""" This module presents useful function used for the Bayesain logistic regression 
    model with a Gaussian prior."""


def log_posterior(X, y, theta, sigma):
    """Computes the log of the (unnormalized) posterior over the dataset (X,y)

    Parameters
    ----------
    X : np.array
        training samples 
    y : np.array
        labels
    theta : np.array
        parameters of the Bayesian logistic regression model
    sigma : float
        standard deviation of the prior over theta
        
    Returns
    -------
    float
       log-(unnormalized) posterior in theta 
    """
    log_lkl = np.sum(np.log(expit(y * np.dot(X, theta))), axis=0) # log-likelihood
    log_prior = - np.linalg.norm(theta, ord=2)**2/(2*sigma**2)    # log-prior
    return log_lkl + log_prior


def gradient_posterior(X_batch, y_batch, theta, N, sigma, compute_Vs=False):
    """ Computes the gradient of the posterior on a mini-batch

    Parameters
    ----------
    X_batch : np.array
        batch training samples
    y_batch : np.array
        labels of the batch training samples
    theta : np.array
        parameters of the Bayesian log.-reg. model
    N : int
        total number of training samples
    sigma : float
        std. deviation of the theta-prior
    compute_Vs : boolean
        whether to compute the covariance matrix of the scores s_ti (c.f. the paper)

    Returns
    -------
    float
        mini-batch log-posterior gradient
    """
    n_batch = len(X_batch)  
    scores_ = ((1 - expit(y_batch*np.dot(X_batch, theta))) * y_batch)[:, None] * X_batch
    grad_batch = (N/n_batch) * np.sum(scores_, axis=0) - theta/(sigma**2)   

    if compute_Vs:
        scores = scores_ - (1/N) * theta/(sigma**2) 
        Vs = (1/n_batch) * (scores - np.mean(scores, axis=0)).T.dot(scores - np.mean(scores, axis=0))
        return grad_batch, Vs
    return grad_batch


def sgld_logreg(X, y, theta_0, sigma=1., gamma=.55, a=1, b=10, batch_size=1, n_iter=5000, compute_Vs=False):
    """Runs SGLD logistic regression n_iter on the provided training data (X, y).

    Parameters
    ----------
    X : np.array
        training samples
    y : np.array
        labels of the training samples
    theta_0 : np.array
        intial theta valie
    sigma : float, optional
        std. dev. of the prior on theta, by default 1
    gamma : float, optional
        exponent of the step size, by default .55
    a : int, optional
        param1 of the step size, by default 1
    b : int, optional
        param2 of the step size, by default 10
    batch_size : int, optional
        size of the batch at each iteration, by default 1
    n_iter : int, optional
        total number of iterations, by default 5000
    compute_Vs : bool, optional
        whether to comute the covariance matrix of the scores, by default False

    Returns
    -------
    thetas : np.array
        theta values along iterations
    step_sizes : np.array
        step size along iterations
    log_posts : np.array
        log-posterior along iterations
    alphas : np.array
        samples transition threshold (optim. -> sampling)
        
    """
    start = time.time()
    theta = theta_0
    N, p = X.shape
    np.random.seed(1)

    thetas = np.empty((n_iter+1, p))   
    thetas[0] = theta_0  
    step_sizes = np.empty((n_iter))      
    log_posts = np.empty(n_iter)   # log posterior pdf along iterations
    alphas = np.empty(n_iter)     # sample threshold (transition optim. --> sampling)

    for i in tqdm(range(n_iter)):
        idx = np.random.choice(N, batch_size, replace=False)   # indexes of the batch samples 
        X_batch = X[idx]
        y_batch = y[idx]

        # log-posterior
        log_posts[i] = log_posterior(X, y, theta, sigma)

        # step size
        step_size = a*(b+i)**(-gamma)
        step_sizes[i] = step_size
        
        # added gaussian noise
        eta = np.random.multivariate_normal(mean=np.zeros(p), cov=step_size*np.identity(p))

        # computing the gradient 
        if compute_Vs:
            grad, Vs = gradient_posterior(X_batch, y_batch, theta, N=N, sigma=sigma, compute_Vs=compute_Vs)
            alphas[i] = ((step_size * N**2) / (4 * batch_size)) * np.max(np.real(eigvals(Vs)))
        else:   
            grad = gradient_posterior(X_batch, y_batch, theta, N=N, sigma=sigma, compute_Vs=compute_Vs)
        
        # updating theta
        theta = theta + (step_size/2) * grad + eta
        thetas[i+1] = theta
    end = time.time()
    print(f"Finished within {round(end-start, 2)} s.")
    return thetas, step_sizes, log_posts, alphas


def make_predictions(X_test, post_samples, t_burn_in=1, step_sizes=None):
    """computes the predictive posterior distribution of the test set X_test.

    Parameters
    ----------
    X_test : np.array
        test samples
    post_samples : np.array
        posterior samples
    t_burn_in : int, optional
        end of the burn-in phase, by default 1
    step_sizes : np.array, optional
        step sizes along iterations, by default None

    Returns
    -------
    np.array
        predictive posterior  p(y|X, Y, x) of the samples in X_test
    """
    if np.all(step_sizes) == None:
        # expectation of the pred. posterior without weighting
        return np.mean(expit(X_test.dot(post_samples[t_burn_in+1:].T)), axis=1)
    else:
        # expectation of the predictive posterior weighted by the step sizes
        return np.sum(step_sizes[t_burn_in:].T * expit(X_test.dot(post_samples[t_burn_in+1:].T)), axis=1) / (np.sum(step_sizes[t_burn_in:]))


def compute_accuracy(X, y, theta_map, t_burn_in=1):
    """computes the accuracy of sgld_logreg over the set (X, y).

    Parameters
    ----------
    X : np.array
        data samples
    y : np.array
        labels of the samples
    theta_map : np.array
        posterior samples
    t_burn_in : int, optional
        end of the burn-in phase, by default 1

    Returns
    -------
    float
        accuracy of the sgld model
    """
    well_class = np.sign(X.dot(theta_map[t_burn_in+1:].T)) == y[:, None]  # well classified samples
    return np.mean(well_class, axis=0)
