import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.special import expit
from scipy.linalg import eigvals
from sgld import *
from tqdm import tqdm
import time


def one_leapfrog(X_batch, y_batch, q, p, h_step, M_mass, N, sigma=1.):
    """Perfoms one leapfrog step of size h_step

    Parameters
    ----------
    X_batch : np.array
        batch samples
    y_batch : np.array
        batch labels
    q : np.array
        parameter vector (theta)
    p : np.array
        momentum vector
    h_step : float
        size of the leapfrog step
    M_mass : np.array
        mass matrix
    N : int
        total number of samples
    sigma : float, optional
        std. dev. of the prior on theta (q), by default 1.

    Returns
    -------
    q_new : np.array 
        new value of the parameter (theta)
    p_new : np.array
        new value of the momentum
    """
    q_grad_1 = gradient_posterior(X_batch, y_batch, theta=q, N=N, sigma=sigma)   # log(U)
    p_half = p + h_step/2 * q_grad_1  
    q_new = q + h_step * M_mass.dot(p_half)
    q_grad_2 = gradient_posterior(X_batch, y_batch, theta=q_new, N=N, sigma=sigma)
    p_new = p_half + h_step/2 * q_grad_2
    return q_new, p_new

def hamiltonian_value(X_batch, y_batch, q, p, M_mass, sigma=1):
    """Computes the value of the Hamiltonian function H(q,p).

    Parameters
    ----------
    X_batch : np.array
        batch samples
    y_batch : np.array
        batch labels
    q : np.array
        parameter vector (theta)
    p : np.array
        momentum vector
    h_step : float
        size of the leapfrog step
    M_mass : np.array
        mass matrix
    sigma : int, optional
        std. dev. of the prior on theta (q), by default 1.

    Returns
    -------
    float
        Value of H(q,p)
    """
    pot_energy = - log_posterior(X_batch, y_batch, q, sigma)
    kin_energy = p.T.dot(M_mass.dot(p))
    return pot_energy + kin_energy
    

def hamiltonian_dynamic(X_batch, y_batch, q, p, h_step, M_mass, n_steps, N, sigma=1.):
    """Performs one step of the Hamiltonian dynamic.

    Parameters
    ----------
    X_batch : np.array
        batch samples
    y_batch : np.array
        batch labels
    q : np.array
        parameter vector (theta)
    p : np.array
        momentum vector
    h_step : float
        size of the leapfrog step
    M_mass : np.array
        mass matrix
    n_steps : int
        number of steps of the hamiltonian process
    N : int
        total number of samples
    sigma : float, optional
        std. dev. of the prior on theta (q), by default 1.

    Returns
    -------
    q_new : np.array 
        new value of the parameter (theta)
    p_new : np.array
        new value of the momentum
    """
    q_new, p_new = q, p
    for i in range(n_steps):
        q_new, p_new = one_leapfrog(X_batch, y_batch, q_new, p_new, h_step, M_mass, N, sigma)
    return q_new, p_new


def sghd_logreg(X, y, theta_0, h_step, n_steps, M_mass, sigma=1., batch_size=1, n_iter=5000):    
    """Runs SGHD logistic regression for n_iter iterations on the provided training data (X, y).

    Parameters
    ----------
    X : np.array
        training samples
    y : np.array
        labels of the training samples
    theta_0 : np.array
        intial theta valie
    h_step : float
        size of the leapfrog step
    n_steps : int
        number of steps of the hamiltonian process
    M_mass : np.array
        amxx matrix
    sigma : float, optional
        std. dev. of the prior on theta, by default 1
    batch_size : int, optional
        size of the batch at each iteration, by default 1
    n_iter : int, optional
        total number of iterations, by default 5000

    Returns
    -------
    thetas : np.array
        values of theta along iterations
    log_posts : np.array
        value of the log-posterioro along iterations
    accept_ratio : float
        acceptance ration
    accept_probas : np.array   
        probabilities of acceptance along iterations
    """
    start = time.time()
    np.random.seed(0)
    N, d = X.shape

    # storing values of theta 
    theta = theta_0
    thetas = np.empty((n_iter+1, d))
    thetas[0] = theta_0

    # initialsing momentum from N(0, M_mass)
    p = np.random.multivariate_normal(mean=np.zeros(d), cov=M_mass)
    
    accept_rejects = np.zeros(n_iter) # accept/reject along iterations
    accept_probas = np.zeros(n_iter)  # acceptance rate (alpha) values
    log_posts = np.empty(n_iter+1)    # log-posterior along iterations
    log_posts[0] = log_posterior(X, y, theta, sigma)
        
    for i in tqdm(range(n_iter)):
                
        # select a batch of size batch_size
        idx = np.random.choice(N, size=batch_size)
        X_batch = X[idx]
        y_batch = y[idx]
        
        # initialise momentum
        p = np.random.multivariate_normal(mean=np.zeros(d), cov=M_mass)
        
        # one hamiltoninan dynamic step
        theta_new, p_new = hamiltonian_dynamic(X_batch, y_batch, theta, p, h_step, M_mass, n_steps, N, sigma)
            
        # Negate momentum at end of trajectory to make the proposal symmetric
        p_new = - p_new 
        
        # compute the values of the old and the new hamiltonian
        Hamlt_old = hamiltonian_value(X_batch, y_batch, theta, p, M_mass, sigma)
        Hamlt_new = hamiltonian_value(X_batch, y_batch, theta_new, p_new, M_mass, sigma)

        # MH: compute the acceptance rate +  decision (1 == accept, 0 == reject)
        log_proba_accept = min(0, Hamlt_old - Hamlt_new)
        accept_rejects[i] = np.log(np.random.rand()) < log_proba_accept
        accept_probas[i] = np.exp(log_proba_accept)

        if accept_rejects[i] or batch_size < N:
            theta, p = theta_new, p_new
        thetas[i+1] = theta
        log_posts[i+1] = log_posterior(X, y, theta, sigma)
            
    accept_ratio = np.mean(accept_rejects.astype('int'))
    end = time.time()
    print(f"Finished within {round(end-start, 2)} s.")
    return thetas, log_posts, accept_ratio, accept_probas
    