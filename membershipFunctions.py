import jax.numpy as jnp
import skfuzzy as fuzz
import matplotlib.pyplot as plt


class membershipFunctions:
    def __init__(self, X, **kwargs):
        if isinstance(X, list):
            X = jnp.array(X)
        elif isinstance(X, jnp.ndarray):
            pass
        else:
            raise ValueError("Invalid input type for X. Expected list or jnp.ndarray.")

        input_columns = X.shape[0]
        sigma = jnp.std(X, axis=1)

        self.membership_functions = dict()
        self.lower_bound = jnp.min(X, axis=1)
        self.upper_bound = jnp.max(X, axis=1)
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
                    mf = mf_class(
                        self.lower_bound[i], self.upper_bound[i], mf_num, sigma[i]
                    )
                else:
                    mf = mf_class(self.lower_bound[i], self.upper_bound[i], mf_num)
            else:
                raise ValueError(f"Invalid membership function type: {mf_type}")
            if "names" in kwargs:
                self.membership_functions[kwargs["names"][i]] = mf
            else:
                self.membership_functions[f"{mf_type}x{mf_num}_{i+1}"] = mf

    def plot(self):
        skfuzzy_functions = []
        titles = []
        mf_dict = {
            "gaussian": fuzz.gaussmf,
            "gbell": fuzz.gbellmf,
            "trapezoidal": fuzz.trapmf,
            "triangular": fuzz.trimf,
            "sigmoid": fuzz.sigmf,
        }

        for name, obj in self.membership_functions.items():

            titles.append(name)
            
            x = jnp.linspace(
                start=obj.lower_bound,
                stop=obj.upper_bound,
                num=round(obj.upper_bound - obj.lower_bound),
            )

            class_name = str(obj.__class__).split(".")[-1]
            mf_type = class_name.split("'")[0]

            for i, params in enumerate(obj.parameters):
                skfuzzy_func = mf_dict[mf_type]
                mem_func = skfuzzy_func(x, params)
                print(mem_func)
                skfuzzy_functions.append(mem_func)

       # print(skfuzzy_functions)

    def plot_old(self):
        # jnp.arrange to get linearly spaced values
        x_values = jnp.stack(
            [self.lower_bound, self.upper_bound, jnp.ones_like(self.lower_bound)],
            axis=1,
        )
        x_arrays = []
        for start, stop, step in x_values:
            x_arrays.append(jnp.arange(start=start, stop=(stop + step), step=step))

        # generate fuzzy membership

        types = []
        for item in self.membership_functions.items():
            class_name = str(item[1].__class__).split(".")[-1]
            type = class_name.split("'")[0]
            types.append(type)

        mf_dict = {
            "gaussian": fuzz.gaussmf,
            "gbell": fuzz.gbellmf,
            "trapezoidal": fuzz.trapmf,
            "triangular": fuzz.trimf,
            "sigmoid": fuzz.sigmf,
        }

        functions = []
        for i, (mf_name, mf_obj) in enumerate(self.membership_functions.items()):
            mf_type = types[i]
            mf_func = mf_dict[mf_type]
            # Call the function with the appropriate arguments
            result = mf_func(x_arrays[i], mf_obj.parameters)
            functions.append(result)

        for name, obj in self.membership_functions.items():
            print(f"{name}:")
            for i, params in enumerate(obj.parameters):
                print(f"  Nested Array {i+1}: {params}")

        print(functions)
        # plot these as x, y


class gaussian(membershipFunctions):
    def __init__(self, lower_bound, upper_bound, n, sigma):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
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
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
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
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
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
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
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
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
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
