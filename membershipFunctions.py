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
        self.m = gaussian.initialize_m(lower_bound, upper_bound, n)
        self.sigma = jnp.ones(n) * sigma

    def initialize_m(lower_bound, upper_bound, n):
        if n == 1:
            return jnp.array([lower_bound + ((upper_bound - lower_bound) / 2)])
        elif n == 2:
            d = (upper_bound - lower_bound) / 3
            return jnp.array([lower_bound + d, upper_bound - d])
        else:
            return jnp.linspace(start=lower_bound, stop=upper_bound, num=n)


class gbell:
    def __init__(self, lower_bound, upper_bound, n) -> None:
        self.c = gbell.initialize_c(lower_bound, upper_bound, n)
        self.a
        self.b

    def initialize_c(lower_bound, upper_bound, n):
        if n == 1:
            return (upper_bound - lower_bound) / 2
        elif n == 2:
            d = (upper_bound - lower_bound) / 3
            return jnp.array([lower_bound + d, upper_bound - d])
        else:
            return jnp.linspace(start=lower_bound, stop=upper_bound, num=n)

    def initialize_a(c):
        # num_functions = len(c)

        pass


class trapezoidal:
    def __init__(self, lower_bound, upper_bound, n) -> None:
        self.parameters = trapezoidal.initialize_parameters(lower_bound, upper_bound, n)

    def initialize_parameters(lower_bound, upper_bound, n):
        if n == 1:
            return jnp.array(
                [[lower_bound, lower_bound * 4 / 3, lower_bound * 5 / 3, upper_bound]]
            )
        elif n == 2:
            step = (upper_bound - lower_bound) / 5
            return jnp.array(
                [
                    [
                        lower_bound,
                        lower_bound + step,
                        lower_bound + (2 * step),
                        lower_bound + (3 * step),
                    ],
                    [
                        lower_bound + (2 * step),
                        lower_bound + (3 * step),
                        lower_bound + (4 * step),
                        upper_bound,
                    ],
                ]
            )
        else:
            domain = upper_bound - lower_bound
            width = domain * (n / (2 * n - 2))
            leftmost_parameter = lower_bound - width / 2
            first = jnp.array(
                [
                    leftmost_parameter,
                    leftmost_parameter + (width / 3),
                    leftmost_parameter + width * (2 / 3),
                    leftmost_parameter + width,
                ]
            )
            parameters = jnp.zeros((n, 4))
            parameters = parameters.at[0, :].set(first)

            for i in range(1, n):
                parameters = parameters.at[i, :].set(
                    width * jnp.array([0, 1 / 3, 2 / 3, 1]) + parameters[i - 1, 2]
                )

            return parameters


class triangular:
    def __init__(self, lower_bound, upper_bound, n) -> None:
        self.parameters = triangular.initialize_parameters(lower_bound, upper_bound, n)

    def initialize_parameters(lower_bound, upper_bound, n):
        if n == 1:
            return jnp.array(
                [
                    [
                        lower_bound,
                        (lower_bound + ((upper_bound - lower_bound) / 2)),
                        upper_bound,
                    ]
                ]
            )
        elif n == 2:
            step = (upper_bound - lower_bound) / 4
            return jnp.array(
                [
                    [lower_bound, lower_bound + (step * 1), lower_bound + (step * 2)],
                    [lower_bound + (step * 2), lower_bound + (step * 3), upper_bound],
                ]
            )
        else:
            domain = upper_bound - lower_bound
            width = domain * ((2) / (n - 1))
            leftmost_parameter = lower_bound - (width * 0.5)
            first = jnp.array([leftmost_parameter, 0, leftmost_parameter + width])
            parameters = jnp.zeros((n, 3))
            parameters = parameters.at[0, :].set(first)

            for i in range(1, n):
                parameters = parameters.at[i, :].set(
                    width * jnp.array([0, 1 / 2, 1]) + parameters[i - 1, 1]
                )

            return parameters


class sigmoid:
    def __init__(self, lower_bound, upper_bound, n) -> None:
        self.parameters = sigmoid.initialize_parameters(lower_bound, upper_bound, n)

    def initialize_parameters(lower_bound, upper_bound, n):
        domain = upper_bound - lower_bound
        if n == 1:
            return jnp.array([[1, lower_bound + (domain / 2)]])
        elif n == 2:
            step = domain / 4
            return jnp.array([[1, lower_bound + step], [1, lower_bound + 3 * step]])
        else:
            c_arr = jnp.linspace(start=lower_bound, stop=upper_bound, num=n)
            return jnp.array([[1, c] for c in c_arr])


if __name__ == "__main__":
    val = sigmoid(0, 10, 4)
    print(val.parameters)
