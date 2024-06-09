import skfuzzy as fuzz
import jax.numpy as jnp

# Each instance of any of these must contain all the functions for one input -> array of parameters

#TO DO
# make sure the instances are for all the functions for one input i.e. as an array
# some can stay as single numbers such as sigma if it is the same sigma for all - but in jax arrays
# AKSHHULLY these should be as array of the same number since they get updated in training

class gaussian:
    def __init__(self, lower_bound, upper_bound, n, sigma) -> None:
       self.m  = gaussian.calculate_m(lower_bound, upper_bound, n)
       self.sigma = jnp.ones(n) * sigma

    def calculate_m(lower_bound, upper_bound, n):
        if n == 1:
            return (upper_bound - lower_bound) / 2
        elif n == 2:
            d = (upper_bound - lower_bound) / 3
            return  jnp.array([lower_bound + d, upper_bound - d])
        else:
            return jnp.linspace(start = lower_bound, stop = upper_bound, num = n)

class gbell:
    def __init__(self, domain, n) -> None:
        self.a
        self.b
        self.c

class trapezoidal:
    def __init__(self, domain, n) -> None:
        self.a
        self.b
        self.c
        self.d

class triangular:
    def __init__(self, domain, n) -> None:
        self.a
        self.b
        self.c

class sigmoid:
    def __init__(self, domain, n) -> None:
        self.a
        self.c
