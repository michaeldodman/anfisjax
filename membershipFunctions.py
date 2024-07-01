import jax.numpy as jnp
import skfuzzy as fuzz
import matplotlib.pyplot as plt


class MembershipFunctions:
    def __init__(self, X, **kwargs):
        self.X = self._validate_input(X)
        self.input_columns = self.X.shape[0]
        self.sigma = jnp.std(self.X, axis=1)
        self.lower_bound = jnp.min(self.X, axis=1)
        self.upper_bound = jnp.max(self.X, axis=1)

        mf_specs = self._parse_kwargs(kwargs)
        self.membership_functions = self._initialize_membership_functions(
            mf_specs, kwargs
        )

    def __str__(self):
        output = "Membership Functions:\n"
        for name, mf in self.membership_functions.items():
            output += f"  {name}:\n"
            output += f"    Type: {mf['type']}\n"
            output += f"    Lower Bound: {mf['lower_bound']}\n"
            output += f"    Upper Bound: {mf['upper_bound']}\n"
            output += "    Parameters:\n"
            for i, params in enumerate(mf['params']):
                output += f"      MF {i+1}: {params}\n"
            output += "\n"
        return output

    def _validate_input(self, X):
        if isinstance(X, list):
            return jnp.array(X)
        elif isinstance(X, jnp.ndarray):
            return X
        else:
            raise ValueError("Invalid input type for X. Expected list or jnp.ndarray.")

    def _parse_kwargs(self, kwargs):
        if "type" in kwargs and "num" in kwargs:
            mf_type = kwargs["type"]
            mf_num = kwargs["num"]
            return [(mf_type, mf_num)] * self.input_columns
        elif "mf_specs" in kwargs:
            mf_specs = kwargs["mf_specs"]
            if not isinstance(mf_specs, list) or not all(
                isinstance(spec, tuple) and len(spec) == 2 for spec in mf_specs
            ):
                raise ValueError("Invalid membership function specification")
            if len(mf_specs) != self.input_columns:
                raise ValueError(
                    "Number of membership function specifications must match the number of inputs"
                )
            return mf_specs
        else:
            raise ValueError(
                "Invalid input format. Please provide either 'type' and 'num' or 'mf_specs'."
            )

    def _initialize_membership_functions(self, mf_specs, kwargs):
        membership_functions = {}
        for i, (mf_type, mf_num) in enumerate(mf_specs):
            if mf_type in MEMBERSHIP_FUNCTION_DICT:
                mf_func = MEMBERSHIP_FUNCTION_DICT[mf_type]
                if mf_func is MembershipFunctions.gaussian:
                    params = mf_func(
                        self.lower_bound[i], self.upper_bound[i], mf_num, self.sigma[i]
                    )
                else:
                    params = mf_func(self.lower_bound[i], self.upper_bound[i], mf_num)
                
                if "names" in kwargs:
                    name = kwargs["names"][i]
                else:
                    name = f"{mf_type}x{mf_num}_{i+1}"
                
                membership_functions[name] = {
                    "type": mf_type,
                    "params": params,
                    "lower_bound": self.lower_bound[i],
                    "upper_bound": self.upper_bound[i]
                }
            else:
                raise ValueError(f"Invalid membership function type: {mf_type}")

        return membership_functions

    @staticmethod
    def gaussian(lower_bound, upper_bound, n, sigma):
        if n == 1:
            return jnp.array([[lower_bound + ((upper_bound - lower_bound) / 2), sigma]])
        elif n == 2:
            d = (upper_bound - lower_bound) / 3
            return jnp.array(
                [[lower_bound + d, sigma / n], [upper_bound - d, sigma / n]]
            )
        else:
            m = jnp.linspace(start=lower_bound, stop=upper_bound, num=n)
            sigma_arr = jnp.ones(n) * (sigma / (n - 1))
            return jnp.stack([m, sigma_arr], axis=1)

    @staticmethod
    def gbell(lower_bound, upper_bound, n):
        domain = upper_bound - lower_bound
        if n == 1:
            return jnp.array([domain / 4, 1, lower_bound + domain / 2])
        elif n == 2:
            d = domain / 3
            return jnp.array(
                [
                    [(upper_bound + lower_bound) / (n + 2), 1, lower_bound + d],
                    [(upper_bound + lower_bound) / (n + 2), 1, upper_bound - d],
                ],
            )
        else:
            centers = jnp.linspace(start=lower_bound, stop=upper_bound, num=n)
            widths = jnp.full(n, domain / n)
            heights = jnp.ones(n)
            parameters = jnp.stack([widths, heights, centers], axis=1)
            return parameters

    @staticmethod
    def trapezoidal(lower_bound, upper_bound, n):
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

    @staticmethod
    def triangular(lower_bound, upper_bound, n):
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

    @staticmethod
    def sigmoid(lower_bound, upper_bound, n):
        if n == 1:
            return jnp.array([[(lower_bound + upper_bound) / 2, 1]])
        elif n == 2:
            return jnp.array(
                [
                    [lower_bound + ((upper_bound - lower_bound) / 4), 2],
                    [lower_bound + 3 * ((upper_bound - lower_bound) / 4), 2],
                ]
            )
        else:
            b_arr = jnp.linspace(start=lower_bound, stop=upper_bound, num=n)
            return jnp.stack([b_arr, jnp.ones(n) * (0.5 * n + 0.5)], axis=1)

    def plot(self):
        mf_dict = {
            "gaussian": fuzz.gaussmf,
            "gbell": fuzz.gbellmf,
            "trapezoidal": fuzz.trapmf,
            "triangular": fuzz.trimf,
            "sigmoid": fuzz.sigmf,
        }

        num_subplots = len(self.membership_functions)
        num_cols = int(num_subplots**0.5)
        num_rows = (num_subplots + num_cols - 1) // num_cols

        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(8, 8))
        axes = axes.flatten()

        linewidth = 2.0
        ylim_margin = linewidth / (2 * 72)

        for subplot_index, (name, mf) in enumerate(self.membership_functions.items()):
            x = jnp.linspace(
                start=mf['lower_bound'],
                stop=mf['upper_bound'],
                num=round(mf['upper_bound'] - mf['lower_bound']) * 100,
            )
            mf_type = mf['type']

            skfuzzy_func = mf_dict[mf_type]
            if mf_type in ["gaussian", "gbell", "sigmoid"]:
                for params in mf['params']:
                    mem_func = skfuzzy_func(x, *params)
                    axes[subplot_index].plot(x, mem_func)
            else:
                for params in mf['params']:
                    mem_func = skfuzzy_func(x, params)
                    axes[subplot_index].plot(x, mem_func)

            axes[subplot_index].set_title(name)
            axes[subplot_index].set_xlabel("x")
            axes[subplot_index].set_ylabel("Membership")
            axes[subplot_index].set_xlim([mf['lower_bound'], mf['upper_bound']])
            axes[subplot_index].set_ylim([0, 1 + ylim_margin])

        for i in range(num_subplots, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.show()


MEMBERSHIP_FUNCTION_DICT = {
    "gaussian": MembershipFunctions.gaussian,
    "gbell": MembershipFunctions.gbell,
    "trapezoidal": MembershipFunctions.trapezoidal,
    "triangular": MembershipFunctions.triangular,
    "sigmoid": MembershipFunctions.sigmoid,
}

if __name__ == "__main__":
    pass
