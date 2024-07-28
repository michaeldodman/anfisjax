import jax.numpy as jnp
import jax
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import equinox as eqx

class FuzzificationLayer(eqx.Module):
    membership_functions: list # this will be a list of membership function instances
    
    def __init__(self, membership_functions):
        pass

    def __call__(self, x):
        pass

class RuleLayer(eqx.Module):
    def __call__(self, fuzzified_input):
        pass

class NormalizationLayer(eqx.Module):
    def __call__(self, firing_strengths):
        pass

class ConsequentLayer(eqx.Module):
    coefficients: jnp.ndarray

    def __init__(self, num_rules, num_features):
        pass

    def __call__(self):
        pass

class DefuzzificationLayer(eqx.Module):
    def __call__self():
        pass

class ANFIS(eqx.Module):
    def __init__(self) -> None:
        pass