import numpy as np


class GaussianRFF():

    def __init__(self,d,D,kernel_var=1,seed=True):
        if seed == True:
            np.random.seed(0)
        self.A = np.random.normal(loc=0,scale=kernel_var,size=(d,D))
        self.b = np.random.uniform(low=0,high=2*np.pi,size=(D,1))
        self.D = D

    def transform(self,x):
        """
        x - feature vector (d x 1)
        z - rff feature vector (D x 1)
        """
        temp = (self.A.T @ x + self.b)
        z = np.sqrt(2/self.D) * np.cos(temp)
        return z