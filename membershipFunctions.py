import jax.numpy as jnp

# Each instance of any of these must contain all the functions for one input -> array of parameters

# TO DO
# make sure the instances are for all the functions for one input i.e. as an array
# some can stay as single numbers such as sigma if it is the same sigma for all - but in jax arrays
# AKSHHULLY these should be as array of the same number since they get updated in training


class membershipFunction:
    def __init__(self, X, y):
        self.max
        self.min
        self.sigma


class gaussian:
    def __init__(self, lower_bound, upper_bound, n, sigma) -> None:
        self.m = gaussian.calculate_m(lower_bound, upper_bound, n)
        self.sigma = jnp.ones(n) * sigma

    def calculate_m(lower_bound, upper_bound, n):
        if n == 1:
            return jnp.array([(upper_bound - lower_bound) / 2])
        elif n == 2:
            d = (upper_bound - lower_bound) / 3
            return jnp.array([lower_bound + d, upper_bound - d])
        else:
            return jnp.linspace(start=lower_bound, stop=upper_bound, num=n)


class gbell:
    def __init__(self, lower_bound, upper_bound, n) -> None:
        self.c = gbell.calculate_c(lower_bound, upper_bound, n)
        self.a
        self.b

    def calculate_c(lower_bound, upper_bound, n):
        if n == 1:
            return (upper_bound - lower_bound) / 2
        elif n == 2:
            d = (upper_bound - lower_bound) / 3
            return jnp.array([lower_bound + d, upper_bound - d])
        else:
            return jnp.linspace(start=lower_bound, stop=upper_bound, num=n)

    def calculate_a(c):
        # num_functions = len(c)

        pass


class trapezoidal:
    def __init__(self, lower_bound, upper_bound, n) -> None:
        self.a, self.b, self.c, self.d = trapezoidal.calculate_attributes(lower_bound, upper_bound, n)

    def calculate_attributes(lower_bound, upper_bound, n):
        if n == 1:
            return (
                jnp.array([lower_bound]),
                jnp.array([lower_bound * 4 / 3]),
                jnp.array([lower_bound * 5 / 3]),
                jnp.array([upper_bound]),
            )
        elif n == 2:
            step = (upper_bound - lower_bound) / 5
            return (
                jnp.array([lower_bound, lower_bound + (2 * step)]),
                jnp.array([lower_bound + step, lower_bound + (3 * step)]),
                jnp.array([lower_bound + (2 * step), lower_bound + (4 * step)]),
                jnp.array([lower_bound + (3 * step), upper_bound]),
            )
        else:
            domain = upper_bound - lower_bound
            leftmost_parameter = lower_bound - (domain / (2 * (n - 1)))
            rightmost_parameter = upper_bound + (domain / (2 * (n - 1)))
            width = domain / (n - 1)
            return (
                jnp.linspace(
                    start=leftmost_parameter, stop=(upper_bound - (width / 2)), num=n
                ),
                jnp.linspace(
                    start=leftmost_parameter + (width / 3),
                    stop=rightmost_parameter - (width * (2 / 3)),
                    num=n,
                ),
                jnp.linspace(
                    start=leftmost_parameter + ((2 * width) / 3),
                    stop=rightmost_parameter - (width / 3),
                    num=n,
                ),
                jnp.linspace(
                    start=(lower_bound + (width / 2)), stop=rightmost_parameter, num=n
                ),
            )


class triangular:
    def __init__(self, domain, n) -> None:
        self.a
        self.b
        self.c


class sigmoid:
    def __init__(self, domain, n) -> None:
        self.a
        self.c


if __name__ == "__main__":
    val = trapezoidal(0, 10, 3)
    print(val.a)
    print(val.b)
    print(val.c)
    print(val.d)
