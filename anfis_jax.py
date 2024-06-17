import jax.numpy as jnp
import jax
import skfuzzy as fuzz


class ANFIS:
    def __init__(self, X, y, membershipFn, epochs):
        self.X
        self.y
        self.membershipFunction
        self.epochs
        self.lower_bound
        self.upper_bound
        self.sigma

    def initializeMembershipFunctions(self):
        pass

    @jax.jit
    def LSE(A, B, initialGamma=1000.0):
        coeffMat = A
        rhsMat = B
        S = jnp.eye(coeffMat.shape[1]) * initialGamma
        x = jnp.zeros((coeffMat.shape[1], 1))

        def body_fn(i, val):
            S, x = val
            a = coeffMat[i, :]
            b = rhsMat[i]
            a_T = jnp.transpose(a)
            S = S - (jnp.dot(jnp.dot(jnp.dot(S, a_T), a), S)) / (
                1 + jnp.dot(jnp.dot(S, a), a)
            )
            x = x + jnp.dot(S, jnp.dot(a_T, b - jnp.dot(a, x)))
            return S, x

        S, x = jax.lax.fori_loop(0, len(coeffMat[:, 0]), body_fn, (S, x))
        return x

    def membershipFunction(self, func_name, x, *args):
        func_dict = {
            "gaussmf": fuzz.gaussmf,
            "gbellmf": fuzz.gbellmf,
            "trapmf": fuzz.trapmf,
            "trimf": fuzz.trimf,
            "sigmf": fuzz.sigmf,
        }

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

class membershipFunctions:
    def __init__(self, X, **kwargs):
        if isinstance(X, list):
            X = jnp.array(X)
        elif isinstance(X, jnp.ndarray):
            pass
        else:
            raise ValueError("Invalid input type for X. Expected list or jnp.ndarray.")

        input_columns = X.shape[0]
        lower_bound = jnp.min(X, axis=1)
        upper_bound = jnp.max(X, axis=1)
        sigma = jnp.std(X, axis=1)

        self.membership_functions = dict()

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
            else:
                raise ValueError(f"Invalid membership function type: {mf_type}")
            if "names" in kwargs:
                self.membership_functions[kwargs["names"][i]] = mf
            else:
                self.membership_functions[f"{mf_type}x{mf_num}_{i+1}"] = mf

    def plot(object):
        pass


class gaussian(membershipFunctions):
    def __init__(self, lower_bound, upper_bound, n, sigma):
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
    pass
