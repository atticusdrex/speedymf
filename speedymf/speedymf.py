"""
One Speedy MF (Multi-Fidelity library, that is)
-------------

This is a document for a lightweight jax-based Gaussian Process Regression/Neural Network implementation with gradient-based hyperparameter optimization. 

Atticus Rex, 2025
"""
from gplib import * 

# Creating a parent class for the multi-fidelity regressor objects 
class MFRegressor:
    """
    Initialize a multi-fidelity object. 

    Parameters
    ----------
    data_dict : dict
        The multi-fidelity data dictionary
    kernel : function
        A kernel covariance function specifying the Gaussian Process prior functional distribution. 
    kernel_dim : int 
        An integer specifying the dimension of the kernel parameters needing to be fed into the kernel function. 
    jitter : float
        A floating point number defining how much to regularize the solution with to avoid numerical instability (default = 1e-6).
    """
    def __init__(self, data_dict, kernel, kernel_dim, jitter = 1e-6):
        # Storing data dictionary, kernel and kernel base dimension
        self.d, self.kernel, self.kernel_dim = data_dict, kernel, kernel_dim 
        # Storing keyword arguments 
        self.jitter = jitter
        # Number of levels of fidelity
        self.K = len(self.d) 

class CoKriging(MFRegressor):
    """
    Initialize a multi-fidelity object. 

    Parameters
    ----------
    data_dict : dict
        The multi-fidelity data dictionary
    kernel : function
        A kernel covariance function specifying the Gaussian Process prior functional distribution. 
    kernel_dim : int 
        An integer specifying the dimension of the kernel parameters needing to be fed into the kernel function. 
    xbounds : tuple 
        A tuple specifying the floating point bounds for the inputs. It should be in the form (min, max) (default = (0.0, 1.0)). 
    jitter : float
        A floating point number defining how much to regularize the solution with to avoid numerical instability (default = 1e-6).
    """
    def __init__(self, *args, xbounds = (0.0, 1.0), **kwargs):
        # Storing the initial arguments 
        super().__init__(*args, **kwargs)
        # Storing the input bounds
        self.xmin, self.xmax = xbounds

        # Initializing the parameters 
        p = {} 
        for level1 in range(self.K):
            for level2 in range(self.K):
                if level1 == level2:
                    p['%d_%d_k_param' % (level1, level2)] = jnp.ones(self.kernel_dim)*0.1 
                elif level1 < level2:
                    k_param = np.ones(self.kernel_dim)*0.1
                    k_param[0] = 0 
                    p['%d_%d_k_param' % (level1, level2)] = jnp.array(k_param)
            
            p['%d_noise_var' % level1] = self.d[level1]['noise_var']
        
        self.p = p 

        # Forming the Y matrix 
        self.Y = jnp.vstack([self.d[level]['Y'].reshape(-1, 1) for level in range(self.K)[::-1]]).ravel()

    
    # Forming the training kernel matrix
    def Ktrain(self, p):
        rows = [] 
        for level1 in self.d.keys():
            cols = []
            for level2 in self.d.keys():
                # Switching to make the references upper-triangular
                if level2 < level1:
                    inds = (level2, level1)
                else:
                    inds = (level1, level2)

                # Retreiving the kernel parameters 
                k_param = p['%d_%d_k_param' % inds]

                # Forming the kernel matrix 
                Kmat = K(self.d[level1]['X'], self.d[level2]['X'], self.kernel, k_param)

                # Adding noise if necessary 
                if level1 == level2:
                    Kmat += jnp.eye(Kmat.shape[0])*(p['%d_noise_var' % level1])

                # Appending Kmat to col 
                cols.append(Kmat)

            # Appending row of cols to rows
            rows.append(cols)
        
        # Forming a block matrix from the composite matrix 
        return jnp.block(rows)


    def Ktest(self, Xtest, level1, p):
        cols = [] 

        for level2 in self.d.keys():
            # Switching to make the references upper-triangular
            if level2 < level1:
                inds = (level2, level1)
            else:
                inds = (level1, level2)

            # Retreiving the kernel parameters 
            k_param = p['%d_%d_k_param' % inds]

            # Forming the kernel matrix 
            Kmat = K(Xtest, self.d[level2]['X'], self.kernel, k_param)

            # Appending Kmat to col 
            cols.append(Kmat)
        
        # Forming Block Matrix
        return jnp.block([cols])

    def predict(self, Xtest, level):
        # Constructing the training and testing matrices 
        K_test = self.Ktest(Xtest, level, self.p)
        K_train = self.Ktrain(self.p)
        Ktestvar = K(Xtest, Xtest, self.kernel, self.p['%d_%d_k_param' % (level, level)])
        # Cholesky factorization
        L = cholesky(K_train+jnp.eye(K_train.shape[0])*self.jitter)
        # Compute posterior mean and variance 
        Ymu = K_test @ cho_solve((L, True), self.Y)
        Ycov = Ktestvar - K_test @ cho_solve((L, True), K_test.T)
        return Ymu, Ycov
    
    def predict_var(self, Xtest, level, K_train, K_test):
        Ktestvar = K(Xtest, Xtest, self.kernel, self.p['%d_%d_k_param' % (level, level)])
        # Cholesky factorization
        L = cholesky(K_train+jnp.eye(K_train.shape[0])*self.jitter)
        # Compute posterior mean and variance 
        return Ktestvar - K_test @ cho_solve((L, True), K_test.T)
    
    # For computing the log-evidence term of a single GP 
    def log_evidence(self, p):
        # Form training kernel matrix 
        K_train = self.Ktrain(p)
        # Take cholesky factorization 
        L = cholesky(K_train+jnp.eye(K_train.shape[0])*self.jitter)
        # Compute log-determinant of Ktrain 
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        # Compute quadratic term of Ktrain Y.T @ Ktrain^{-1} Y 
        quadratic_term = self.Y.T @ jax.scipy.linalg.cho_solve((L, True), self.Y) 
        # Add total loss back out
        return (quadratic_term + logdet).squeeze()
    
    def optimize(self, params_to_optimize=['k_param'], lr = 1e-5, epochs = 1000, batch_size = 250, beta1 = 0.9, beta2=0.999, shuffle = False):
        # Initializing parameters 
        p = deepcopy(self.p)

        # Adding the appropriate keys to optimize
        keys_to_optimize = []
        for key in p.keys():
            if ('k_param' in key) and ('k_param' in params_to_optimize):
                keys_to_optimize.append(key)
            if ('noise_var' in key) and ('noise_var' in params_to_optimize):
                keys_to_optimize.append(key)
        
        # Optimizing with ADAM 
        self.p = deepcopy(ADAM(
            lambda p: self.log_evidence(p), 
            self.p, 
            keys_to_optimize, 
            self.d[0]['X'], self.d[0]['Y'], 
            batch_size = batch_size,
            epochs=epochs, 
            lr = lr, 
            beta1 = beta1, 
            beta2=beta2,
            shuffle = shuffle
        ))

class Hyperkriging:
    def __init__(self, d, kernel, kernel_dim, jitter = 1e-6):
        # Obtaining constructor parameters 
        self.d, self.kernel, self.jitter = d, kernel, jitter 
        self.K = len(d) 
        self.kernel_dim = kernel_dim

        # Initializing low-fidelity model 
        self.d[0]['model'] = SimpleGP(
                d[0]['X'],
                d[0]['Y'],
                kernel, kernel_dim,
                noise_var = d[0]['noise_var'],
            )

        # Initializing models 
        for level in range(1, self.K):
            # Initializing the features 
            features = d[level]['X']
            for sublevel in range(level):
                # Getting the mean function from the sublevel immediately under
                mean, _ = self.d[sublevel]['model'].predict(features)

                # Horizontally concatenating the mean function to the existing features 
                features = jnp.hstack((features, mean.reshape(-1,1)))
            
            # Creating a model trained on this set of features 
            self.d[level]['model'] = SimpleGP(
                features,
                d[level]['Y'],
                kernel, self.kernel_dim + level,
                noise_var = d[level]['noise_var'],
            )

    def predict(self, Xtest, level):
        test_features = Xtest
        # Initializing the features 
        for sublevel in range(level):
            # Getting the mean function from the sublevel immediately under
            mean, _ = self.d[sublevel]['model'].predict(test_features)

            # Horizontally concatenating the mean function to the existing features 
            test_features = jnp.hstack((test_features, mean.reshape(-1,1)))
        
        return self.d[level]['model'].predict(test_features)
    
    def optimize(self, level, params_to_optimize = ['k_param'], lr = 1e-5, epochs = 1000, beta1 = 0.9, beta2 = 0.999, batch_size = 250, shuffle = False):
        # Optimizing lowest-fidelity model 
        if level == 0:
            self.d[0]['model'].optimize(params_to_optimize=params_to_optimize, lr = lr, epochs = epochs, beta1 = beta1, beta2 = beta2, batch_size = batch_size, shuffle = shuffle)
        else:
            features = self.d[level]['X']
            for sublevel in range(level):
                # Getting the mean function from the sublevel immediately under
                mean, _ = self.d[sublevel]['model'].predict(features)

                # Horizontally concatenating the mean function to the existing features 
                features = jnp.hstack((features, mean.reshape(-1,1)))

            # Updating the features at this fidelity-level 
            self.d[level]['model'].X = copy(features)
            
            # Creating a model trained on this set of features 
            self.d[level]['model'].optimize(params_to_optimize =params_to_optimize, lr = lr, epochs = epochs, beta1 = beta1, beta2 = beta2, batch_size = batch_size, shuffle = shuffle)
            
class StochasticHyperkriging:
    def __init__(self, d, kernel, base_kernel_dim, jitter = 1e-6, mu=0.0, cov = 1.0, N_mc = 25):
        # Obtaining constructor parameters 
        self.d, self.kernel, self.jitter = d, kernel, jitter 
        self.K = len(d) 
        self.N_mc = N_mc 

        # Storing sum of all kernel hyperparameters 
        self.base_dim = base_kernel_dim
        self.full_kernel_dim = self.base_dim*self.K + int(self.K/2*(self.K-1))

        # Initializing kernel hyperparameters
        self.p = {
            'k_mu':jnp.ones((self.base_dim + self.K-1, 1))*mu, 
            'k_L':cholesky(jnp.eye(self.base_dim + self.K-1)*cov)
        }

        for level in range(self.K):
            self.p['%d_k_param' % level] = jnp.ones(self.base_dim+level)*0.1

        # Updating the features at each fidelity-level
        self.update_features()

    def mean_prediction(self, k_param, noise_var, Xtest, X, Y):
        # Creating Kernel matrices 
        Ktest = K(Xtest, X, self.kernel, k_param)
        Ktrain = K(X, X, self.kernel, k_param) + noise_var * jnp.eye(X.shape[0])
        L = cholesky(Ktrain)

        # Returning the mean function
        return (Ktest @ cho_solve((L, True), Y)).reshape(-1,1)
    
    def full_prediction(self, k_param, noise_var, Xtest, X, Y):
        # Creating Kernel matrices 
        Ktest = K(Xtest, X, self.kernel, k_param)
        Ktrain = K(X, X, self.kernel, k_param) + (noise_var + self.jitter) * jnp.eye(X.shape[0])
        L = cholesky(Ktrain)

        # Returning the mean function
        mean = (Ktest @ cho_solve((L, True), Y)).reshape(-1,1)
        cov = K(
            Xtest, Xtest, self.kernel, k_param
        ) - Ktest @ cho_solve((L, True), Ktest.T)

        return mean, cov

    # High-Fidelity Loss Function
    def log_evidence(self, p):
        # Setup keys
        key = random.PRNGKey(12)
        keys = random.split(key, self.N_mc)

        features = self.d[self.K-1]['X']
        # Initializing the features 
        for level in range(self.K-1):
            # Getting kernel hyperparameters
            k_param = p['%d_k_param' % level] 

            # Getting the mean function from the sublevel immediately under
            mean = self.mean_prediction(
                k_param, self.d[level]['noise_var'],
                features, self.d[level]['F'], self.d[level]['Y']
            )
            # Horizontally concatenating the mean function to the existing features 
            features = jnp.hstack((features, mean))

        def single_eval(key):
            # Taking the last kernel parameters
            k_param = log_normal(key, p['k_mu'], p['k_L'] @ p['k_L'].T).ravel()
            
            # Making the training kernel matrix and taking cholesky factor
            Ktrain = K(features, features, self.kernel, k_param) + (self.d[self.K-1]['noise_var']+self.jitter)*jnp.eye(features.shape[0]) 
            y = self.d[self.K-1]['Y'] 
            L = cholesky(Ktrain)

            # Compute log-determinant of Ktrain 
            logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
            # Compute quadratic term of Ktrain Y.T @ Ktrain^{-1} Y 
            quadratic_term = y.T @ jax.scipy.linalg.cho_solve((L, True), y) 
            # Add total loss back out
            return (quadratic_term + logdet).squeeze()

        losses = jax.vmap(single_eval)(keys)

        return jnp.mean(losses)
    
    # Updating features stored at each level 
    def update_features(self):
        for level in range(self.K):
            # Initializing the features 
            features = self.d[level]['X']

            for sublevel in range(level):
                # Getting kernel hyperparameters
                k_param = self.p['%d_k_param' % sublevel]

                # Getting the mean function from the sublevel immediately under
                mean = self.mean_prediction(
                    k_param, self.d[sublevel]['noise_var'],
                    features, self.d[sublevel]['F'], self.d[sublevel]['Y']
                )
                # Horizontally concatenating the mean function to the existing features 
                features = jnp.hstack((features, mean))
                
            # Taking the average of the sample for this level and saving it
            self.d[level]['F'] = copy(features)
            
    # Updating features stored at each level 
    def predict(self, Xtest, N_mc = 25):
        # Setup keys
        key = random.PRNGKey(42)
        keys = random.split(key, N_mc)

        features = Xtest

        for sublevel in range(self.K-1):
            # Getting kernel hyperparameters
            k_param = self.p['%d_k_param' % sublevel]

            # Getting the mean function from the sublevel immediately under
            mean = self.mean_prediction(
                k_param, self.d[sublevel]['noise_var'],
                features, self.d[sublevel]['F'], self.d[sublevel]['Y']
            )
            # Horizontally concatenating the mean function to the existing features 
            features = jnp.hstack((features, mean))

        def single_eval(key):
            k_param = log_normal(key, self.p['k_mu'], self.p['k_L'] @ self.p['k_L'].T).ravel()

            # Getting the posterior 
            mean, cov = self.full_prediction(
                k_param,
                self.d[self.K-1]['noise_var'],
                features, 
                self.d[self.K-1]['F'],
                self.d[self.K-1]['Y']
            )

            cov += self.jitter*jnp.eye(mean.shape[0])
            pred = random.multivariate_normal(key, mean.ravel(), cov)

            return pred.ravel()
        

        # Computing monte-carlo features for this level 
        preds = jax.vmap(lambda key: single_eval(key))(keys)
        return preds.mean(axis=0), jnp.cov(preds.T)
    
    def optimize(self, params_to_optimize = ['k_mu','k_L', 'k_param'], epochs = 100, batch_size = 250, lr = 1e-8, beta1 = 0.9, beta2 = 0.999, seed = 42, shuffle = False):
        # Adding kernel parameters to parameters to optimize
        if 'k_param' in params_to_optimize:
            params_to_optimize.remove('k_param')
            for level in range(self.K):
                params_to_optimize.append('%d_k_param' % level)
        
        constr = {
            'k_L':lambda L: jnp.tril(L)
        }
        self.p = deepcopy(ADAM(
            lambda p: self.log_evidence(p),
            self.p,
            params_to_optimize,
            jnp.ones((1,1)), jnp.ones((1,1)),
            constr = constr,
            epochs = epochs, 
            batch_size = batch_size, 
            lr = lr, 
            beta1 = beta1, 
            beta2 = beta2, 
            shuffle = shuffle
        ))



'''
Kennedy O'Hagan Co-Kriging 
---------------------------------
NOTE: Requires training data to be nested! 
'''
class KennedyOHagan:
    def __init__(self, d, kernel, jitter=1e-6):
        # Initializing constructor parameters 
        self.d, self.kernel, self.jitter = d, kernel, jitter 
        self.K = len(d) 

        kernel_dim = self.d[0]['X'].shape[1]+1

        # Initializing level zero as a simple GP 
        self.d[0]['model'] = SimpleGP(
            self.d[0]['X'], 
            self.d[0]['Y'],
            self.d[0]['noise_var'],
            kernel, np.ones(kernel_dim),
            jitter = jitter
        )

        # Iterating through the levels of fidelity
        for level in range(1, self.K):
            n_points = self.d[level]['X'].shape[0]
            self.d[level]['model'] = DeltaGP(
                self.d[level]['X'], 
                self.d[level]['Y'],
                self.d[level-1]['Y'][:n_points],
                self.d[level]['noise_var'],
                kernel, 
                jnp.ones(kernel_dim)*0.1, 
                rho = 1.0, 
                jitter=jitter
            )
    
    def predict(self, Xtest, level):
        # Predicting lowest level of fidelity 
        Ymean, Ycov = self.d[0]['model'].predict(Xtest)

        # Predicting up to the specified level of fidelity
        for sublevel in range(1, level+1):
            # Getting rho 
            rho = self.d[sublevel]['model'].rho 
            # Getting the delta predictions
            delta_mean, delta_cov = self.d[sublevel]['model'].predict(Xtest)

            # Getting this level's mean and variance 
            Ymean = rho * Ymean + delta_mean 
            Ycov = rho**2 * Ycov + delta_cov 

        return Ymean, Ycov 
    
    def optimize(self, level, k_param = True, rho = True, lr = 1e-5, max_iter = 1000, momentum = 0.9):
        # The level-0 surrogate is a SimpleGP model so we optimize it differently
        if level == 0:
            self.d[0]['model'].optimize(k_param = True, noise_var = False, lr = lr, max_iter = max_iter, momentum = momentum)
        else:
            self.d[level]['model'].optimize(k_param = k_param, rho = rho, lr = lr, max_iter = max_iter, momentum = momentum)
        
class NARGP:
    def __init__(self, d, kernel, jitter = 1e-6):
        # Obtaining constructor parameters 
        self.d, self.kernel, self.jitter = copy(d), kernel, jitter 
        self.K = len(d) 

        self.kernel_dim = 2*d[0]['X'].shape[1]+4

        # Initializing low-fidelity model 
        self.d[0]['model'] = SimpleGP(
                d[0]['X'],
                d[0]['Y'],
                noise_var = d[0]['noise_var'],
                kernel = rbf, 
                k_param = jnp.ones(d[0]['X'].shape[1]+1)*0.1, 
                jitter=jitter
            )
        
        # Initializing models 
        for level in range(1, self.K):
            # Initializing the features 
            features = d[level]['X']
            n_points = features.shape[0]
            for sublevel in range(level):
                # Getting the mean function from the sublevel immediately under
                #mean, _ = self.d[sublevel]['model'].predict(features)

                # Horizontally concatenating the mean function to the existing features 
                features = jnp.hstack((d[level]['X'], d[sublevel]['Y'][:n_points].reshape(-1,1)))
            
            # Creating a model trained on this set of features 
            self.d[level]['model'] = SimpleGP(
                features,
                d[level]['Y'],
                noise_var = d[level]['noise_var'],
                kernel = kernel, 
                k_param = jnp.ones(self.kernel_dim)*0.1,
                jitter = jitter
            )

    def predict(self, Xtest, level):
        test_features = Xtest
        # Initializing the features 
        for sublevel in range(level):
            # Getting the mean function from the sublevel immediately under
            mean, _ = self.d[sublevel]['model'].predict(test_features)

            # Horizontally concatenating the mean function to the existing features 
            test_features = jnp.hstack((Xtest, mean.reshape(-1,1)))
        
        return self.d[level]['model'].predict(test_features)
    
    def optimize(self, level, lr = 1e-8, epochs = 250, beta1 = 0.9, beta2 = 0.999, batch_size = 250, shuffle = False):
        # Optimizing lowest-fidelity model 
        if level == 0:
            self.d[0]['model'].optimize(params_to_optimize = ['k_param'], lr = lr, epochs = epochs, beta1 = beta1, beta2 = beta2, batch_size = batch_size, shuffle = shuffle)
        else:
            features = self.d[level]['X']
            for sublevel in range(level):
                # Getting the mean function from the sublevel immediately under
                mean, _ = self.d[sublevel]['model'].predict(features)

                # Horizontally concatenating the mean function to the existing features 
                features = jnp.hstack((self.d[level]['X'], mean.reshape(-1,1)))

            # Updating the features at this fidelity-level 
            self.d[level]['model'].X = copy(features)
            
            # Creating a model trained on this set of features 
            self.d[level]['model'].optimize(params_to_optimize = ['k_param'], lr = lr, epochs = epochs, beta1 = beta1, beta2 = beta2, batch_size = batch_size, shuffle = shuffle)



class MFNest:
    def __init__(self, d, kernel, xmin, xmax, M, jitter = 1e-6):
        # Storing the input bounds 
        self.xmin, self.xmax = xmin, xmax 

        # Storing the jitter value (for matrix inversion)
        self.jitter = jitter 

        # Storing the input dimension 
        self.input_dim = d[0]['X'].shape[1] 

        # Saving the data dictionary 
        self.d = d

        # Getting the number of fidelity-levels 
        self.K = len(d)

        # Storing the kernel and initial kernel parameters 
        self.kernel = kernel 

        # Storing the number of inducing points 
        self.M = M 

        # Initializing the kernel params 
        for level in range(self.K):
            # The total input dimension including the dimensionality of X and the number of features is the level of the predictor 
            dim = self.input_dim + level  

            # Clips the number of training samples by the number of inducing points specified for sparsity 
            if d[level]['X'].shape[0] > self.M:
                Zinit, inds = greedy_k_center(d[level]['X'], self.M)
                mu_init = d[level]['Y'][inds]
            else:
                Zinit = d[level]['X']
                mu_init = d[level]['Y']

            # Initializing a parameter dictionary at each level 
            self.d[level]['p'] = {
                'k_param' : jnp.concat([jnp.array([self.d[level]['var']]), jnp.ones(dim)]),
                'noise_var' : self.d[level]['noise_var'],
                'q_mu':mu_init,
                'q_L':jnp.tril(jnp.eye(Zinit.shape[0])*1e-6),
                'Z':Zinit
            }
    
    def predict(self, level, Xtest, N_mc=25, seed=42):
        # Generate all random keys up front and split into groups of 3
        keys = random.split(random.PRNGKey(seed), N_mc * 3).reshape(N_mc, 3, 2)

        # Getting the parameters for this level 
        p = copy(self.d[level]['p'])

        def single_sample(key_triplet):
            key1, key2, key3 = key_triplet

            # Compute test and training features
            psi_test = self.psi(key1, Xtest, level - 1, self.d)
            psi_train = self.psi(key2, self.d[level]['p']['Z'], level - 1, self.d)

            # Sample u from variational distribution 
            u = p['q_mu'] + p['q_L'] @ random.multivariate_normal(key3, jnp.zeros_like(p['q_mu']), jnp.eye(p['q_mu'].shape[0]))

            # Predictive mean and covariance
            mean, cov = predict(self.d[level]['p'], psi_test, psi_train, u, self.kernel)
            cov += jnp.eye(cov.shape[0]) * self.jitter

            # Draw sample from predictive distribution
            return random.multivariate_normal(key3, mean, cov)

        # Use vmap to run single_sample over all triplets of keys
        Yhat = vmap(single_sample)(keys)

        # Rearrange to shape (Xtest.shape[0], N_mc)
        return Yhat.T


    def loss(self, p, level, N_mc=25, seed=42):
        noise_var = p['noise_var']
        Y = self.d[level]['Y']
        N = len(Y)

        # Copying the parameters of the network (might be slow, but oh well)
        d = copy(self.d)
        d[level]['p'] = p

        # Generate all random keys and reshape into triplets
        keys = random.split(random.PRNGKey(seed), N_mc * 3).reshape(N_mc, 3, 2)

        def single_sample(key_triplet):
            key1, key2, key3 = key_triplet

            # Compute test and training features
            psi_test = self.psi(key1, self.d[level]['X'], level - 1, d)
            psi_train = self.psi(key2, p['Z'], level - 1, d)

            # Sample u from variational distribution 
            u = p['q_mu'] + p['q_L'] @ random.multivariate_normal(key3, jnp.zeros_like(p['q_mu']), jnp.eye(p['q_mu'].shape[0]))

            # Predict mean and covariance
            mean, cov = predict(p, psi_test, psi_train, u, self.kernel)
            cov += jnp.eye(cov.shape[0]) * self.jitter

            # Sample from the predictive distribution
            L = jnp.linalg.cholesky(cov)
            Yhat = mean + L @ random.multivariate_normal(key3, jnp.zeros_like(mean), jnp.eye(mean.shape[0])).ravel()

            # Likelihood term
            likelihood = jnp.sum((Y - Yhat) ** 2) / (2 * noise_var) + N * jnp.log(2 * jnp.pi * noise_var) / 2

            # KL divergence term (same for all samples, but we compute it inside for JAX compatibility)
            Kp = K(psi_train, psi_train, self.kernel, p['k_param']) + self.jitter * jnp.eye(p['q_mu'].shape[0])
            Lp = jnp.linalg.cholesky(Kp)
            kl_divergence = KL_div(p['q_mu'], p['q_L'], Lp)

            return likelihood + kl_divergence

        # Vectorized application
        losses = vmap(single_sample)(keys)

        # Average over Monte Carlo samples
        return jnp.mean(losses)

    def optimize(self, level, params_to_optimize = ['k_param', 'Z', 'q_L', 'q_mu'], lr = 1e-6, max_iter = 100, N_mc=25, seed = 42, beta1 = 0.9, beta2=0.999):
        # Copying the parameters from the global data dictionary 
        p = copy(self.d[level]['p'])

        constr = {
            'Z':lambda x: jnp.clip(x, self.xmin, self.xmax),
            'q_L':lambda x: jnp.tril(x),
            'k_param':lambda x: jnp.clip(x, 0.0, 100.0)
        }

        self.d[level]['p'] = ADAM(
            lambda p: self.loss(p, level, N_mc = N_mc, seed = seed), 
            p, 
            params_to_optimize,
            constr=constr, 
            max_iter=max_iter, 
            lr = lr, 
            beta1 = beta1, 
            beta2=beta2
        )

    def psi(self, rng_key, X, level, d):
        # Initializing the features as the prediction X 
        features = X

        # Iterate through the levels starting from zero 
        for l in range(level + 1):
            # Splitting the random key into a subkey 
            rng_key, key1, key2 = random.split(rng_key, 3)

            # Extracting the variational parameters 
            q_mu = d[l]['p']['q_mu']
            q_L = d[l]['p']['q_L']

            # Using the parameterized cholesky factor to form the covariance matrix with jitter so that the matrix stays numerically SPD 
            u = q_mu + q_L @ random.multivariate_normal(key1, jnp.zeros_like(q_mu), jnp.eye(q_mu.shape[0])).ravel() 

            # Base case: the lowest-fidelity level in which case the features are simply just the inputs 
            if l == 0:
                mean, cov = predict(d[0]['p'], X, d[0]['p']['Z'], u, self.kernel)
            else:
                # Recursive case: we make the training-features based on the lower-fidelity evaluations 
                train_features = self.psi(rng_key, d[l]['p']['Z'], l - 1, d)
                mean, cov = predict(d[l]['p'], features, train_features, u, self.kernel)

            # Adding jitter to the covariance matrix to ensure SPD 
            cov += jnp.eye(cov.shape[0])*self.jitter
            L = jnp.linalg.cholesky(cov)

            # Taking a random sample of the predictive at this fidelity-level 
            sample = mean + L @ random.multivariate_normal(key2, jnp.zeros_like(mean), jnp.eye(mean.shape[0]))
            
            # Adding this sample to the growing features matrix 
            features = jnp.hstack((features, sample.reshape(-1, 1)))

        return features
    
class NNRegressor:
    def __init__(self, input_dim, output_dim, hidden_sizes, activation = relu, l2_reg = 1e-6, seed = 42, activate_last_layer = False):
        # Initializing class parameters 
        self.hidden_sizes = [input_dim] + list(hidden_sizes) + [output_dim]
        self.activation = activation
        self.l2_reg = l2_reg
        self.activate_last_layer = activate_last_layer

        # Initializing parameters
        p = {} 

        # Initializing Random state 
        key = random.PRNGKey(seed)
        keys = random.split(key, len(hidden_sizes))

        # Iterating through hidden layers and initializing parameters
        for layer in range(1, len(self.hidden_sizes)):
            Wshape = (self.hidden_sizes[layer-1], self.hidden_sizes[layer]) 
            bshape = (self.hidden_sizes[layer], 1)
            p['%d_W' % layer] = random.normal(keys[layer], Wshape)
            p['%d_b' % layer] = random.normal(keys[layer], bshape).reshape(-1,1)
        
        # Storing the parameters to the object
        self.p = p
    
    def predict(self, X):
        for layer in range(1, len(self.hidden_sizes)):
            # Linear Affine Transformation
            X = X @ self.p['%d_W' % layer] + self.p['%d_b' % layer].T 

            # Nonlinear Activation 
            if (layer == len(self.hidden_sizes)-1 and self.activate_last_layer) or (layer < len(self.hidden_sizes) - 1):
                X = self.activation(X)
        return X
    
    # Defining the MSE loss function
    def mse_loss(self,p,X, Y):

        # Initializing a tally of l2-regularization (excluding biases, of course)
        l2_loss = 0.0 

        X, y = p['X'], p['Y']

        for layer in range(1, len(self.hidden_sizes)):
            # Linear Affine Transformation
            X = X @ p['%d_W' % layer] + p['%d_b' % layer].T 

            # Nonlinear Activation 
            if (layer == len(self.hidden_sizes)-1 and self.activate_last_layer) or (layer < len(self.hidden_sizes) - 1):
                X = self.activation(X)

            # Regularization loss 
            l2_loss += jnp.linalg.norm(p['%d_W' % layer], 'fro')
        
        # Returning the Frobenius Norm 
        return jnp.linalg.norm(X-y.reshape(-1,1), 'fro') + l2_loss * self.l2_reg
    
    def fit(self, X, Y, epochs = 50, batch_size = 25, lr = 1e-6, momentum = 0.9):
        # Copying the parameters 
        p = copy(self.p)

        # Creating the gradient function 
        grad_func = jax.value_and_grad(lambda p: self.mse_loss(p, X, Y))

        # Copy parameters 
        p = copy(self.p)

        # Initializing momentum arrays
        v = copy(p)
        for key in v.keys():
            v[key] = jnp.zeros_like(v[key])

        p['X'], p['Y'] = X, Y

        # Printing the initial loss 
        print("Initial Loss: %.5f" % (self.mse_loss(p, X, Y)))

        # Constructing the tqdm iterator 
        iterator = tqdm(range(epochs))

        # Iteratively updating the kernel parameters 
        for _ in iterator:
            for Xbatch, Ybatch in create_batches(X, Y, batch_size=batch_size):
                # Setting batch inputs 
                p['X'] = Xbatch
                p['Y'] = Ybatch

                # Computing the loss and gradient 
                loss, grad = grad_func(p)

                # Displaying the loss and gradient at each iteration 
                iterator.set_postfix_str("Loss: %.5f" % (loss))

                # Updating weights and biases
                for key in p.keys():
                    if key not in ['X', 'Y']:
                        v[key] = momentum * v[key] - lr * grad[key]
                        p[key] += v[key]

        
        # Store the parameter values back to the object
        self.p = copy(p)