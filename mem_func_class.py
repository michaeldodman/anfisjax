import jax.numpy as jnp
import numpy as np

class membershipFunctions:
    def __init__(self, X, **kwargs):

        if isinstance(X, list):
            X = jnp.array(X)
        elif isinstance(X, jnp.ndarray):
            pass
        else:
            raise ValueError("Invalid input type for X. Expected list or jnp.ndarray.")

        #self.X = X
        input_columns = X.shape[0]
        lower_bound = jnp.max(X, axis = 1)
        upper_bound = jnp.min(X, axis = 1)
        sigma = jnp.std(X, axis = 1)
        
        self.membership_functions = dict() # dict with input name and object
        #self.names = []

        mf_dict = {
            "gaussian": gaussian,
            "gbell": gbell,
            "trapezoidal": trapezoidal,
            "triangular": triangular,
            "sigmoid": sigmoid,
        }

        if "type" in kwargs and "num" in kwargs:
            mf_type = kwargs["type"]
            mf_num = kwargs["num"]
            mf_specs = [(mf_type, mf_num)] * input_columns
        elif "mf_specs" in kwargs:
            mf_specs = kwargs["mf_specs"]
            if not isinstance(mf_specs, list) or not all(
                isinstance(spec, tuple) and len(spec) == 2 for spec in mf_specs
            ):
                raise ValueError("Invalid membership function specification")
            if len(mf_specs) != input_columns:
                raise ValueError(
                    "Number of membership function specifications must match the number of inputs"
                )
        else:
            raise ValueError(
                "Invalid input format. Please provide either 'type' and 'num' or 'mf_specs'."
            )

        for i in range(input_columns):
            mf_type, mf_num = mf_specs[i]

            if mf_type in mf_dict:
                mf_class = mf_dict[mf_type]
                if mf_class is gaussian:
                    mf = mf_class(lower_bound[i], upper_bound[i], mf_num, sigma[i])
                else:
                    mf = mf_class(lower_bound[i], upper_bound[i], mf_num)
                #self.names.append(f"{mf_type}_{i+1}")
            else:
                raise ValueError(f"Invalid membership function type: {mf_type}")
            if "names" in kwargs:
                self.membership_functions[kwargs["names"][i]] = mf
            else:
                self.membership_functions[f"{mf_type}_{mf_num}_{i+1}"] = mf

    def plot(object):
        pass


class gaussian(membershipFunctions):
    def __init__(self, lower_bound, upper_bound, n, sigma):
        self.parameters = gaussian.initialize_parameters(lower_bound, upper_bound, n, sigma)

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


class gbell(membershipFunctions):
    def __init__(self, lower_bound, upper_bound, n):
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


class trapezoidal(membershipFunctions):
    def __init__(self, lower_bound, upper_bound, n):
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


class triangular(membershipFunctions):
    def __init__(self, lower_bound, upper_bound, n):
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


class sigmoid(membershipFunctions):
    def __init__(self, lower_bound, upper_bound, n):
        self.parameters = sigmoid.initialize_parameters(lower_bound, upper_bound, n)

    @staticmethod
    def initialize_parameters(lower_bound, upper_bound, n):
        if n == 1:
            return jnp.array([[1, (lower_bound + upper_bound) / 2]])
        elif n == 2:
            return jnp.array(
                [
                    [1, lower_bound + ((upper_bound - lower_bound) / 4)],
                    [1, lower_bound + 3 * ((upper_bound - lower_bound) / 4)],
                ]
            )
        else:
            c_arr = jnp.linspace(start=lower_bound, stop=upper_bound, num=n)
            return jnp.stack([jnp.ones(n), c_arr], axis=1)


if __name__ == "__main__":

    # Generate a random dataset with 5 nested arrays, each containing 10 values
    # Set a random seed for reproducibility
    np.random.seed(42)

    # Generate a random dataset with 5 nested arrays, each containing 10 values
    X = np.array([
        np.random.normal(size=10),
        np.random.normal(size=10),
        np.random.normal(size=10),
        np.random.normal(size=10),
        np.random.normal(size=10)
    ])
    # Convert the NumPy array to a JAX array
    X_jax = jnp.array(X)

    mfs_default = membershipFunctions(X_jax, type="trapezoidal", num=3)  # for all the same
    from pprint import pprint
    pprint(mfs_default.membership_functions['trapezoidal_3_1'].parameters)

  
    mf_specs = [
        ("gaussian", 3),
        ("gbell", 2),
        ("gaussian", 4),
        ("trapezoidal", 5),
        ("sigmoid", 3),
    ]
    mfs_standard = membershipFunctions(X_jax, mf_specs = mf_specs)