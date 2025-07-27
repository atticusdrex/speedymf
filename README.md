# gaussianprocess

This is a simple library designed for producing fast, efficient Gaussian Process regression models with custom kernels ad hyperparameter optimization. This was largely designed to function like the popular ```scikit-learn``` package does, with a ```GaussianProcess()``` constructor, a ```fit()``` function and a ```predict()``` function. The optimization of the hyperparameters assumes a Gaussian log-likelihood and uses ADAM to optimize the parameters. 
