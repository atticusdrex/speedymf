import jax 
import jax.numpy as jnp 
from jax import vmap, random 
from jax.scipy.linalg import cho_solve
from jax.numpy.linalg import cholesky
from tqdm import tqdm 
from copy import copy, deepcopy 
import numpy as np 

# For performing kernel matrix operations 
def K(X1, X2, kernel_func, kernel_params):
    return vmap(lambda x: vmap(lambda y: kernel_func(x, y, kernel_params))(X2))(X1)

# Function for greedily choosing the number of inducing inputs 
def greedy_k_center(X, k):
    N = X.shape[0]
    selected_indices = []
    idx = np.random.randint(N)
    selected_indices.append(idx)

    distances = np.linalg.norm(X - X[idx], axis=1)

    for _ in range(1, k):
        idx = np.argmax(distances)
        selected_indices.append(idx)
        new_distances = np.linalg.norm(X - X[idx], axis=1)
        distances = np.minimum(distances, new_distances)

    return X[np.array(selected_indices)], selected_indices

# Log-normal distribution for uncertain kernel hyperparameters
def log_normal(key, mean, cov, size=1):
    # Cholesky decomposition of covariance matrix
    L = cholesky(cov)

    # Sample standard normal variables
    z = jax.random.normal(key, shape=(mean.shape[0], size))

    # Reparameterization trick
    normal_samples = mean.reshape(-1,1) + L @ z

    # Take the exponent to get log-normal samples
    lognormal_samples = jnp.exp(normal_samples)

    return lognormal_samples

# Special KL-divergence function that computes KL(q(mu, L)||p(0, L)) where p has zero-mean 
def KL_div(mu_q, L_q, L_p):
    # we assume the mean function is zero 
    k = mu_q.shape[0]

    # Trace of the q(u) covariance matrix 
    Tr_q = jnp.sum(jnp.diag(L_q)**2)

    # Log-determinant of p(u) covariance/kernel matrix 
    Logdet_p = 2.0 * jnp.sum(jnp.log(jnp.diag(L_p)))

    return 0.5 * (Tr_q- k - Logdet_p)

# For batching the training data
def create_batches(X, Y, batch_size, shuffle=True):
    n_samples = X.shape[0]
    
    if shuffle:
        # Shuffle indices
        indices = np.random.permutation(n_samples)
        X = X[indices]
        Y = Y[indices]
    
    # Yield batches
    for i in range(0, n_samples, batch_size):
        X_batch = X[i:i + batch_size, :]
        Y_batch = Y[i:i + batch_size]
        yield X_batch, Y_batch

# An anonymous GP function for predicting outputs from specific training data
def predict(p, Xtest, X, Y, kernel):
    Ktrain = K(X, X, kernel, p['k_param']) + p['noise_var'] * jnp.eye(X.shape[0]) 
    Ktest = K(Xtest, X, kernel, p['k_param'])
    Ktestvar = K(Xtest, Xtest, kernel, p['k_param'])
    L = jnp.linalg.cholesky(Ktrain)
    Yhat = Ktest @ jax.scipy.linalg.cho_solve((L, True), Y)
    Yvar = Ktestvar - Ktest @ jax.scipy.linalg.cho_solve((L, True), Ktest.T)
    return Yhat, Yvar

"""
Kernel Covariance Functions 
---------------------------
For the gp-related classes 
"""

# Deep neural net kernel (single-layer)
def deep_rbf(x,y,kernel_params, activation = jnp.tanh, h_size = 10):
    W1 = kernel_params[1:1+h_size].reshape(h_size,1)
    W2 = kernel_params[1+h_size:1+h_size+h_size**2].reshape(h_size,h_size)
    x, y = W2 @ activation(W1 @ x.reshape(-1,1)), W2 @ activation(W1 @ y.reshape(-1,1))
    h = (x-y).ravel()
    return kernel_params[0]*jnp.exp(-jnp.inner(h,h))


# Radial basis function kernel 
def rbf(x,y,kernel_params, epsilon = 1e-8):
    assert x.shape[0] == y.shape[0], 'Input vectors have mismatched dimensions!'
    assert kernel_params.shape[0] == x.shape[0]+1, 'Kernel parameters are wrong dimension! '
    h = (x-y).ravel()
    return kernel_params[0]*jnp.exp(-jnp.sum(h**2 / (jnp.abs(kernel_params[1:])+epsilon)))

# Laplace kernel 
def laplace(x,y,kernel_params):
    h = (x-y).ravel()
    return kernel_params[0]*jnp.exp(-jnp.sum(jnp.abs(h) / kernel_params[1:]))

# Nonlinear Auto-Regressive RBF Kernel
def nargp_rbf(x1, x2, kernel_params):
    assert x1.shape[0] == x2.shape[0], "Input vectors are different dimensions!"
    assert kernel_params.shape[0] == 2*(x1.shape[0])+2, "Kernel params wrong dimensions!"

    y, x = x1[-1], x1[:-1]
    yp, xp = x2[-1], x2[:-1]
    d = len(x)
    kx, ky, kd = kernel_params[:d+1], kernel_params[d+1:d+3], kernel_params[d+3:]


    return rbf(x, xp, kx) * rbf(y.reshape(-1,1), yp.reshape(-1,1), ky) + rbf(x, xp, kd)

"""
Activation Functions for Neural Nets 
------------------------------------

Tanh, Sigmoid, ReLU at the moment

"""

# Hyperbolic tangent
def tanh(x):
    return jnp.tanh(x)

# Sigmoid 
def sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))

# Rectified Linear Units 
def relu(x):
    return jnp.maximum(0, x)

'''
ADAM Optimization Routine
------------------------------------

I do quite a bit of optimizing in these gosh-darn ml scripts and it would be nice to have an encapsulated script for unconstrained ADAM optimization which I could plug into the whole thing instead of rewriting it each time.
'''
def ADAM(
    loss_func, p,
    keys_to_optimize,
    X, Y,
    constr={},
    batch_size=250,
    epochs=100,
    lr=1e-8,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    shuffle=False,
    max_backoff=50
):
    def contains_nan(val_dict):
        return any(jnp.isnan(x).any() for x in val_dict.values())

    def adam_step(m, v, p, grad, lr, t):
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        update = lr * m_hat / (jnp.sqrt(v_hat) + epsilon)
        p = p - update
        return m, v, p

    def try_adam_step(grad_func, p, m, v, lr, t):
        new_p, new_m, new_v = deepcopy(p), {}, {}
        for key in keys_to_optimize:
            m_k, v_k, p_k = adam_step(m[key], v[key], p[key], grad[key], lr, t)
            if key in constr:
                p_k = constr[key](p_k)
            new_p[key], new_m[key], new_v[key] = p_k, m_k, v_k

        # Keep batch inputs
        new_p['X'], new_p['Y'] = p['X'], p['Y']
        loss, grad_new = grad_func(new_p)
        return loss, grad_new, new_p, new_m, new_v

    # Initialize optimizer states
    m = {key: jnp.zeros_like(p[key]) for key in keys_to_optimize}
    v = {key: jnp.zeros_like(p[key]) for key in keys_to_optimize}

    grad_func = jax.value_and_grad(loss_func)

    best_loss = jnp.inf
    best_p = deepcopy(p)

    # Breaking up the training data into batches and storing it in the parameters
    p['X'], p['Y'] = X[:batch_size, :], Y[:batch_size]
    _, grad = grad_func(p)

    iterator = tqdm(range(epochs))

    for epoch in iterator:
        for Xbatch, Ybatch in create_batches(X, Y, batch_size, shuffle=shuffle):
            # Setting the X batches 
            p['X'], p['Y'] = Xbatch, Ybatch

            # Making a trial learning rate 
            trial_lr = lr

            # Backing off learning rate in the case of NaNs found 
            for _ in range(max_backoff):
                loss, grad, trial_p, trial_m, trial_v = try_adam_step(
                    grad_func, p, m, v, trial_lr, epoch+1
                )

                if not (jnp.isnan(loss) or contains_nan(grad)):
                    break  # successful step
                trial_lr *= 0.5
            else:
                print("Too many NaNs. Stopping optimization.")
                return best_p  # return best found so far

            if loss < best_loss:
                best_loss, best_p = loss, deepcopy(trial_p)

            p, m, v = trial_p, trial_m, trial_v

            iterator.set_postfix_str(f"Loss: {loss:.5f}, LR: {trial_lr:.2e}")

    return best_p