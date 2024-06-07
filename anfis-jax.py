import numpy as np
import jax.numpy as jnp
import jax
import skfuzzy as fuzz

class ANFIS:

    def __init__(self, X, y, membershipFn, epochs):
        self.X
        self.y
        self.epochs
        self.membershipFunction


    @jax.jit
    def LSE(A, B, initialGamma=1000.):
        coeffMat = A
        rhsMat = B
        S = jnp.eye(coeffMat.shape[1]) * initialGamma
        x = jnp.zeros((coeffMat.shape[1], 1))

        def body_fn(i, val):
            S, x = val
            a = coeffMat[i, :]
            b = rhsMat[i]
            a_T = jnp.transpose(a)
            S = S - (jnp.dot(jnp.dot(jnp.dot(S, a_T), a), S)) / (1 + jnp.dot(jnp.dot(S, a), a))
            x = x + jnp.dot(S, jnp.dot(a_T, b - jnp.dot(a, x)))
            return S, x

        S, x = jax.lax.fori_loop(0, len(coeffMat[:, 0]), body_fn, (S, x))
        return x
    
    def membershipFunction(self, func_name, x, *args):
        func_dict = {'gaussmf':fuzz.gaussmf,
                    'gbellmf':fuzz.gbellmf,
                    'trapmf':fuzz.trapmf,
                    'trimf':fuzz.trimf,
                    'sigmf':fuzz.sigmf}
        
        if func_name in func_dict:
            func = func_dict[func_name]
            return func(x, *args)
        else:
            raise ValueError(f"Unknown membership function: {func_name}")
    
    def training(self):
        pass

    def forwardPass(self):
        pass


    def backProp(self):
        pass

    def pred(self, x):
        pass


if __name__ == "__main__":
    pass 