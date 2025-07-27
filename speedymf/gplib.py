"""
GPLib - Gaussian Process Library 
-------------

This is a document for a lightweight jax-based Gaussian Process Regression/with gradient-based hyperparameter optimization. 

Atticus Rex, 2025
"""
from util import * 

# Configure jax to use 64-bit floating point arithmetic 
try:
    jax.config.update("jax_enable_x64", True)
except:
    print("64-bit Jax Computation is not available on your CPU.")

'''
SimpleGP: This is the base Gaussian Process class with all the functionality to train, predict, and optimize the hyperparameters of a Gaussian Process model. 
'''

class SimpleGP:
    def __init__(self, X, Y, kernel, kernel_dim, noise_var = 1e-6, jitter=1e-6):
        # Storing the training data matrices 
        self.X, self.Y = copy(X), copy(Y) 
        self.kernel = kernel 
        self.kernel_dim = kernel_dim 
        # Storing the parameter dictionary object
        self.p = {
            'X':copy(X), 'Y':copy(Y),
            'k_param':jnp.ones(kernel_dim)*0.1, 
            'noise_var':noise_var
        }
        # Storing jitter
        self.jitter = jitter

    # Function for predicting outputs at new inputs 
    def predict(self, Xtest):
        # Unpack necessary parameters 
        noise_var, k_param = self.p['noise_var'], self.p['k_param']
        # Compute kernel matrices
        Ktrain = K(self.X, self.X, self.kernel, k_param) + noise_var * jnp.eye(self.X.shape[0])
        Ktest = K(Xtest, self.X, self.kernel, k_param)
        Ktestvar = K(Xtest, Xtest, self.kernel, k_param)
        # Cholesky and GP predictive mean and covariance
        L = cholesky(Ktrain)
        # Compute posterior mean and variance 
        Ymu = Ktest @ cho_solve((L, True), self.Y)
        Ycov = Ktestvar - Ktest @ cho_solve((L, True), Ktest.T)
        return Ymu, Ycov
    
    # For computing the log-evidence term of a single GP 
    def objective(self, p):
        # Form training kernel matrix 
        Ktrain = K(p['X'], p['X'], self.kernel, p['k_param']) + p['noise_var']* jnp.eye(p['X'].shape[0])
        # Take cholesky factorization 
        L = cholesky(Ktrain)
        # Compute log-determinant of Ktrain 
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        # Compute quadratic term of Ktrain Y.T @ Ktrain^{-1} Y 
        quadratic_term = p['Y'].T @ jax.scipy.linalg.cho_solve((L, True), p['Y']) 
        # Add total loss back out
        return (quadratic_term + logdet).squeeze()
    
    def optimize(self, params_to_optimize = ['k_param'], constr={}, lr = 1e-5, epochs = 1000, beta1 = 0.9, beta2 = 0.999, batch_size = 250, shuffle = False):

        # Optimizing with ADAM 
        self.p = deepcopy(ADAM(
            lambda p: self.objective(p), 
            self.p, params_to_optimize, 
            self.X, self.Y, 
            constr=constr,
            batch_size = batch_size,
            epochs=epochs, 
            lr = lr, 
            beta1 = beta1, 
            beta2=beta2,
            shuffle = shuffle
        ))

class LogNormalGP(SimpleGP):
    # Initializing with a mu and cov variable
    def __init__(self, *args, mu = 0.0, cov = 1.0, N_mc = 10, **kwargs):
        super().__init__(*args, **kwargs)
        # Storing the number of monte-carlo samples 
        self.N_mc = N_mc 
        # Storing the kernel mean and covariance cholesky factor
        self.p['k_mu'] = mu*jnp.ones(self.kernel_dim)
        self.p['k_L'] = jnp.sqrt(cov)*jnp.eye(self.kernel_dim)

    def predict(self, Xtest, N_mc=20, seed = 42):
        # Setup rng keys 
        key = random.PRNGKey(seed)
        keys = random.split(key, N_mc)

        # Extracting the prior distributional parameters
        mu, k_L = self.p['k_mu'], self.p['k_L']

        # Helper function for one Monte Carlo sample
        def single_sample(key):
            # Sample kernel parameters
            k_param = log_normal(key, mu, k_L @ k_L.T + jnp.eye(mu.shape[0])*self.jitter, size=1).ravel()

            # Compute kernel matrices
            Ktrain = K(self.X, self.X, self.kernel, k_param) + self.p['noise_var'] * jnp.eye(self.X.shape[0])
            Ktest = K(Xtest, self.X, self.kernel, k_param)
            Ktestvar = K(Xtest, Xtest, self.kernel, k_param)

            # Cholesky and GP predictive mean and covariance
            L = cholesky(Ktrain)
            Ymu = Ktest @ cho_solve((L, True), self.Y)
            Ycov = Ktestvar - Ktest @ cho_solve((L, True), Ktest.T)

            # Sample from predictive distribution
            L = cholesky(Ycov+ self.jitter*jnp.eye(Ymu.shape[0])) 
            sample = Ymu.reshape(-1,1) +  L @ random.normal(key, shape = (Ymu.shape[0], 1))
            return sample.ravel() 

        # Vectorize over keys
        Yhat = (vmap(single_sample)(keys)).T  # shape (N_mc, n_test)

        # Return mean and covariance
        return jnp.mean(Yhat, axis=1), jnp.cov(Yhat)
    
    # For computing the log-evidence term of a single GP 
    def objective(self, p):
        # Setup keys
        key = random.PRNGKey(42)
        keys = random.split(key, self.N_mc)

        # Evaluating the loss for a single sample 
        def single_loss(key):
            k_param = log_normal(key, p['k_mu'], p['k_L'] @ p['k_L'].T + jnp.eye(p['k_mu'].shape[0])*self.jitter, size=1).ravel()
            noise_var = p['noise_var']
            Ktrain = K(p['X'], p['X'], self.kernel, k_param) + noise_var* jnp.eye(p['X'].shape[0])
            L = cholesky(Ktrain)
            logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
            quadratic_term = p['Y'].T @ jax.scipy.linalg.cho_solve((L, True), p['Y'])
            return (quadratic_term + logdet).squeeze()
        
        # Casting the single-sample losses over N_mc losses 
        losses = vmap(single_loss)(keys)  # shape (N_mc)

        # Taking the average loss over the samples 
        return jnp.mean(losses)

class SVGP(SimpleGP):
    def __init__(self, M, *args, N_mc = 10, xmin=0.0, xmax=1.0, seed = 42, **kwargs):
        # Calling the parent class with the usual parameters 
        super().__init__(*args, **kwargs)

        # Storing the monte-carlo iterations and input bounds
        self.N_mc, self.xmin, self.xmax, self.seed = N_mc, xmin, xmax, seed

        # Making sure the number of inducing points is smaller than training data
        assert M < self.X.shape[0], "Number of inducing points is larger than number of training data points!"
        self.M = M

        # Initializing inducing points with greedy k-center
        Z, inds = greedy_k_center(self.X, M)

        # Storing model parameters in a PyTree structure
        self.p = {
            'Z':Z, 
            'q_mu':self.Y[inds],
            'q_L':kwargs['noise_var']*jnp.eye(M),
            'k_param':jnp.ones(self.kernel_dim)*0.1, 
            'noise_var':kwargs['noise_var']
        }

    def predict(self, Xtest, N_mc=25, seed=42):
        # Setup keys
        key = random.PRNGKey(seed)
        keys = random.split(key, N_mc)

        # Extracting parameters for easy access
        k_param = self.p['k_param']
        noise_var = self.p['noise_var']
        Z = self.p['Z']
        q_mu = self.p['q_mu'].reshape(-1,1)
        q_L = self.p['q_L']
        M = self.M 

        # Defining a single-MC sample prediction 
        def single_prediction(key):
            keys = random.split(key, 2)
            # Sample u from variational distribution 
            u = q_mu.reshape(-1,1) + q_L @ random.normal(keys[0], shape=(M,1))
            # Compute kernel matrices
            Ktrain = K(Z, Z, self.kernel, k_param) + noise_var * jnp.eye(M)
            Ktest = K(Xtest, Z, self.kernel, k_param)
            Ktestvar = K(Xtest, Xtest, self.kernel, k_param)

            # Cholesky and GP predictive mean and covariance
            L = cholesky(Ktrain)
            Ymu = Ktest @ cho_solve((L, True), u)
            Ycov = Ktestvar - Ktest @ cho_solve((L, True), Ktest.T)

            # Sample from predictive distribution
            L = cholesky(Ycov+ self.jitter*jnp.eye(Ymu.shape[0])) 
            sample = Ymu.reshape(-1,1) +  L @ random.normal(keys[1], shape = (Ymu.shape[0], 1))
            return sample.ravel() 
        
        # Mapping the single prediction over the RNG keys 
        Yhat = (vmap(single_prediction)(keys)).T 

        # Return mean and covariance
        return jnp.mean(Yhat, axis=1), jnp.cov(Yhat)
    
    # Evidence Lower Bound (ELBO) Objective Function
    def objective(self,p):
        # Setup keys
        key = random.PRNGKey(self.seed)
        keys = random.split(key, self.N_mc)

        # Extracting parameters for easy access
        k_param = p['k_param']
        noise_var = p['noise_var']
        Z = p['Z']
        q_mu = p['q_mu'].reshape(-1,1)
        q_L = p['q_L']
        M = self.M 
        X, Y = p['X'], p['Y']
        N = X.shape[0]

        # Defining a single-MC sample prediction 
        def squared_error(key):
            keys = random.split(key, 2)
            # Sample u from variational distribution 
            u = q_mu.reshape(-1,1) + q_L @ random.normal(keys[0], shape=(M,1))
            # Compute kernel matrices
            Ktrain = K(Z, Z, self.kernel, k_param) + noise_var * jnp.eye(M)
            Ktest = K(X, Z, self.kernel, k_param)
            Ktestvar = K(X, X, self.kernel, k_param)

            # Cholesky and GP predictive mean and covariance
            L = cholesky(Ktrain)
            Ymu = Ktest @ cho_solve((L, True), u)
            Ycov = Ktestvar - Ktest @ cho_solve((L, True), Ktest.T)

            # Sample from predictive distribution
            L = cholesky(Ycov+ self.jitter*jnp.eye(Ymu.shape[0])) 
            pred = Ymu.reshape(-1,1) +  L @ random.normal(keys[1], shape = (Ymu.shape[0], 1))
            # Returning squared error of model prediction and true outputs 
            return jnp.sum((Y - pred.ravel())**2)
        
        preds = (vmap(squared_error)(keys)).T  # shape (N_train, N_mc)

        # Computing likelihood term 
        likelihood = preds.mean() / (2 * noise_var) + N * jnp.log(2*jnp.pi*noise_var) / 2
        p_L = cholesky(K(Z, Z, self.kernel, k_param)+self.jitter*jnp.eye(M))
        kl_divergence = KL_div(q_mu, q_L, p_L)

        # Return mean and covariance
        return (likelihood + kl_divergence).squeeze()

class LogNormalSVGP(SVGP):
    def __init__(self, *args, k_mu = 0.0, k_cov = 1.0, **kwargs):
        # Calling SVGP function 
        super().__init__(*args, **kwargs)
        # Storing prior distributional parameters 
        self.p['k_mu'] = k_mu * jnp.ones(self.kernel_dim)
        self.p['k_L'] = jnp.sqrt(k_cov)*jnp.eye(self.kernel_dim)
        # Removing the kernel param index (for storage)
        del self.p['k_param']
    
    def predict(self, Xtest, N_mc=25, seed=42):
        # Setup keys
        key = random.PRNGKey(seed)
        keys = random.split(key, N_mc)

        noise_var = self.p['noise_var']
        Z = self.p['Z']
        q_mu = self.p['q_mu'].reshape(-1,1)
        q_L = self.p['q_L']
        M = self.M 

        # Defining a single-MC sample prediction 
        def single_prediction(key):
            # Sampling the kernel hyperparameters
            keys = random.split(key, 3)
            k_param = log_normal(keys[2], self.p['k_mu'], self.p['k_L'] @ self.p['k_L'].T).ravel()

            # Sample u from variational distribution 
            u = q_mu.reshape(-1,1) + q_L @ random.normal(keys[0], shape=(M,1))
            # Compute kernel matrices
            Ktrain = K(Z, Z, self.kernel, k_param) + noise_var * jnp.eye(M)
            Ktest = K(Xtest, Z, self.kernel, k_param)
            Ktestvar = K(Xtest, Xtest, self.kernel, k_param)

            # Cholesky and GP predictive mean and covariance
            L = cholesky(Ktrain)
            Ymu = Ktest @ cho_solve((L, True), u)
            Ycov = Ktestvar - Ktest @ cho_solve((L, True), Ktest.T)

            # Sample from predictive distribution
            L = cholesky(Ycov+ self.jitter*jnp.eye(Ymu.shape[0])) 
            sample = Ymu.reshape(-1,1) +  L @ random.normal(keys[1], shape = (Ymu.shape[0], 1))
            return sample.ravel() 
        
        # Predictions
        Yhat = (vmap(single_prediction)(keys)).T  # shape (N_mc)

        # Return mean and covariance
        return jnp.mean(Yhat, axis=1), jnp.cov(Yhat)
    
    def objective(self,p):
        # Setup keys
        key = random.PRNGKey(self.seed)
        keys = random.split(key, self.N_mc)

        # Extracting parameters for easy access
        noise_var = p['noise_var']
        Z = p['Z']
        q_mu = p['q_mu'].reshape(-1,1)
        q_L = p['q_L']
        M = self.M 
        X, Y = p['X'], p['Y']
        N = X.shape[0]

        # Defining a single-MC sample prediction 
        def squared_error(key):
            # Sampling the kernel hyperparameters
            keys = random.split(key, 3)
            k_param = log_normal(keys[2], self.p['k_mu'], self.p['k_L'] @ self.p['k_L'].T).ravel()

            # Sample u from variational distribution 
            u = q_mu.reshape(-1,1) + q_L @ random.normal(keys[0], shape=(M,1))
            # Compute kernel matrices
            Ktrain = K(Z, Z, self.kernel, k_param) + noise_var * jnp.eye(M)
            Ktest = K(X, Z, self.kernel, k_param)
            Ktestvar = K(X, X, self.kernel, k_param)

            # Cholesky and GP predictive mean and covariance
            L = cholesky(Ktrain)
            Ymu = Ktest @ cho_solve((L, True), u)
            Ycov = Ktestvar - Ktest @ cho_solve((L, True), Ktest.T)

            # Sample from predictive distribution
            L = cholesky(Ycov+ self.jitter*jnp.eye(Ymu.shape[0])) 
            pred = Ymu.reshape(-1,1) +  L @ random.normal(keys[1], shape = (Ymu.shape[0], 1))

            # Computing kl-divergence term and averaging over the kernel hyperparameters 
            p_L = cholesky(K(Z, Z, self.kernel, k_param)+self.jitter*jnp.eye(M))
            kl_divergence = KL_div(q_mu, q_L, p_L)

            # Returning squared error of model prediction and true outputs 
            return jnp.sum((Y - pred.ravel())**2) / (2 * noise_var) + kl_divergence
        
        # Computing single predictions at each rng key 
        preds = (vmap(squared_error)(keys)).T 

        # Computing likelihood term 
        return preds.mean() + N * jnp.log(2*jnp.pi*noise_var) / 2

# GP Class for predicting the difference between two functions 
class DeltaGP(SimpleGP):
    def __init__(self,X, Y1, Y2, kernel,kernel_dim, rho=1.0, **kwargs):
        # Calling the parent class using the arguments and keyword arguments
        super().__init__(X, np.ones((1,1)), kernel, kernel_dim, **kwargs)
        # Storing the Y1 and Y2 variables 
        self.Y1, self.Y2 = Y1, Y2
        # Storing the parameter rho in the parameter dictionary
        self.p['rho'] = rho

    def predict(self, Xtest):
        # Sample kernel parameters
        noise_var = self.p['noise_var']
        k_param = self.p['k_param']
        rho = self.p['rho']
        # Compute kernel matrices
        Ktrain = K(self.X, self.X, self.kernel, k_param) + noise_var * jnp.eye(self.X.shape[0])
        Ktest = K(Xtest, self.X, self.kernel, k_param)
        Ktestvar = K(Xtest, Xtest, self.kernel, k_param)
        # Cholesky and GP predictive mean and covariance
        L = cholesky(Ktrain)
        # Compute & return posterior mean and variance 
        Ymu = Ktest @ cho_solve((L, True), self.Y1 - rho * self.Y2)
        Ycov = Ktestvar - Ktest @ cho_solve((L, True), Ktest.T)
        return Ymu, Ycov

    # For computing the log-evidence term of a single GP 
    def objective(self, p):
        # Form training kernel matrix 
        Ktrain = K(self.X, self.X, self.kernel, p['k_param']) + p['noise_var']* jnp.eye(self.X.shape[0])
        # Take cholesky factorization 
        L = cholesky(Ktrain)
        # Compute log-determinant of Ktrain 
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        # Compute quadratic term of Ktrain Y.T @ Ktrain^{-1} Y 
        delta = (self.Y1 - p['rho']*self.Y2)
        quadratic_term = delta.T @ jax.scipy.linalg.cho_solve((L, True), delta) 
        # Add total loss back out
        return (quadratic_term + logdet).squeeze()

