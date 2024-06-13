import jax.numpy as jnp
# import skfuzzy as fuzz


class membershipFunctions:
    def __init__(self, X, y):
        self.max
        self.min
        self.sigma

    def plot(object):
        pass


class gaussian:
    def __init__(self, lower_bound, upper_bound, n, sigma) -> None:
        self.parameters = gaussian.initialize_parameters(
            lower_bound, upper_bound, n, sigma
        )

    @staticmethod
    def initialize_parameters(lower_bound, upper_bound, n, sigma):
        if n == 1:
            return jnp.array([[lower_bound + ((upper_bound - lower_bound) / 2), sigma]])
        elif n == 2:
            d = (upper_bound - lower_bound) / 3
            return jnp.array([[lower_bound + d, sigma], [upper_bound - d, sigma]])
        else:
            m = jnp.linspace(start=lower_bound, stop=upper_bound, num=n)
            sigma_arr = jnp.ones(n) * sigma
            return jnp.stack([m, sigma_arr], axis=1)


class gbell:
    def __init__(self, lower_bound, upper_bound, n) -> None:
        self.parameters = gbell.initialize_parameters(lower_bound, upper_bound, n)

    @staticmethod
    def initialize_parameters(lower_bound, upper_bound, n):
        domain = upper_bound - lower_bound
        if n == 1:
            return jnp.array([domain / 4, 1, lower_bound + domain / 2])
        elif n == 2:
            d = domain / 3
            return jnp.array(
                [
                    [(upper_bound + lower_bound) / 2, 1, lower_bound + d],
                    [(upper_bound + lower_bound) / 2, 1, upper_bound - d],
                ],
            )
        else:
            centers = jnp.linspace(start=lower_bound, stop=upper_bound, num=n)
            widths = jnp.full(n, domain / n)
            heights = jnp.ones(n)
            parameters = jnp.stack([widths, heights, centers], axis=1)
            return parameters


class trapezoidal:
    def __init__(self, lower_bound, upper_bound, n) -> None:
        self.parameters = trapezoidal.initialize_parameters(lower_bound, upper_bound, n)

    @staticmethod
    def initialize_parameters(lower_bound, upper_bound, n):
        if n == 1:
            domain = upper_bound - lower_bound
            return jnp.array(
                [
                    [
                        lower_bound,
                        lower_bound + (domain * 1 / 3),
                        lower_bound + (domain * 2 / 3),
                        upper_bound,
                    ]
                ]
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
            centers = jnp.linspace(start=lower_bound, stop=upper_bound, num=n)
            left_points = centers - width * 1 / 2
            mid_left_points = centers - width / 6
            mid_right_points = centers + width / 6
            right_points = centers + width * 1 / 2
            parameters = jnp.stack(
                [left_points, mid_left_points, mid_right_points, right_points], axis=1
            )
            return parameters


class triangular:
    def __init__(self, lower_bound, upper_bound, n) -> None:
        self.parameters = triangular.initialize_parameters(lower_bound, upper_bound, n)

    @staticmethod
    def initialize_parameters(lower_bound, upper_bound, n):
        if n == 1:
            return jnp.array(
                [[lower_bound, (lower_bound + upper_bound) / 2, upper_bound]]
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
            width = domain * (2 / (n - 1))
            centers = jnp.linspace(start=lower_bound, stop=upper_bound, num=n)
            left_points = centers - width / 2
            right_points = centers + width / 2
            parameters = jnp.stack([left_points, centers, right_points], axis=1)
            return parameters


class sigmoid:
    def __init__(self, lower_bound, upper_bound, n) -> None:
        self.parameters = sigmoid.initialize_parameters(lower_bound, upper_bound, n)

    @staticmethod
    def initialize_parameters(lower_bound, upper_bound, n):
        if n == 1:
            return jnp.array([[1, (lower_bound + upper_bound) / 2]])
        elif n == 2:
            return jnp.array([[1, lower_bound + ((upper_bound - lower_bound) / 4)], [1, lower_bound + 3 * ((upper_bound - lower_bound) / 4)]])
        else:
            c_arr = jnp.linspace(start=lower_bound, stop=upper_bound, num=n)
            return jnp.stack([jnp.ones(n), c_arr], axis=1)


if __name__ == "__main__":
    val = gbell(0, 10, 2)
    print(val.parameters)
