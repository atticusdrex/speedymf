# speedygp/__init__.py

from .speedymf import rbf, laplace_kernel, nargp_kernel, K, GaussianProcess 

__all__ = [
    "rbf", "laplace_kernel", "nargp_kernel", "K", "GaussianProcess" 
]