import contextlib
import numpy as np
from scipy.stats import norm, laplace


class norm_distr(object):
    def __init__(self, mu, sigma=1):
        self.mu = mu
        self.sigma = sigma
        self.distribution = norm(loc=mu, scale=sigma)

    def rvs(self):
        '''sample'''
        return self.distribution.rvs()

    def pdf(self, x):
        return self.distribution.pdf(x)

    def logpdf(self, x):
        return self.distribution.logpdf(x)

    def logdistr_grad(self, x):
        return (self.mu-x)/(self.sigma**2)


class laplace_distr(object):
    def __init__(self, mu, b=1):
        self.mu = mu
        self.b = b
        self.distribution = laplace(loc=mu, scale=b)

    def rvs(self):
        '''sample'''
        return self.distribution.rvs()

    def pdf(self, x):
        return self.distribution.pdf(x)

    def logpdf(self, x):
        return self.distribution.logpdf(x)

    def logdistr_grad(self, x):
        return (self.mu-x)/(np.fabs(x-self.mu)*self.b)


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)
