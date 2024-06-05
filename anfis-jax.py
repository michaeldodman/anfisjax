import numpy as np
import jax.numpy as jnp

class ANFIS:

    def __init__(self, X, y, membershipFn, epochs):
        pass


    def LSE(self, A, B):
        x, _, _, _ = jnp.linalg.lstsq(A, B)
        return x

    def training(self):
        pass

    def forward_pass(self):
        pass


    def backprop(self):
        pass

    def pred(self):
        pass