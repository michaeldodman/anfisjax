import time
import jax.numpy as jnp
import numpy as np
from membershipFunctions import gaussian, gbell, trapezoidal, triangular, sigmoid

class membershipFunctions_elif:
    def __init__(self, X, mf_spec):
        self.input_columns = X.shape[1]
        self.membership_functions = []
        self.names = []
        
        if isinstance(mf_spec, tuple) and len(mf_spec) == 2:
            mf_type, mf_num = mf_spec
            mf_specs = [(mf_type, mf_num)] * self.input_columns
        elif isinstance(mf_spec, list) and all(isinstance(spec, tuple) and len(spec) == 2 for spec in mf_spec):
            mf_specs = mf_spec
        else:
            raise ValueError("Invalid membership function specification")
        
        for i in range(self.input_columns):
            mf_type, mf_num = mf_specs[i]
            
            if mf_type == 'gaussian':
                mf = gaussian(X[:, i].min(), X[:, i].max(), mf_num, 1.0)
                self.names.append(f'gaussian_{i+1}')
            elif mf_type == 'gbell':
                mf = gbell(X[:, i].min(), X[:, i].max(), mf_num)
                self.names.append(f'gbell_{i+1}')
            elif mf_type == 'trapezoidal':
                mf = trapezoidal(X[:, i].min(), X[:, i].max(), mf_num)
                self.names.append(f'trapezoidal_{i+1}')
            elif mf_type == 'triangular':
                mf = triangular(X[:, i].min(), X[:, i].max(), mf_num)
                self.names.append(f'triangular_{i+1}')
            elif mf_type == 'sigmoid':
                mf = sigmoid(X[:, i].min(), X[:, i].max(), mf_num)
                self.names.append(f'sigmoid_{i+1}')
            else:
                raise ValueError(f"Invalid membership function type: {mf_type}")
            
            self.membership_functions.append(mf)

class membershipFunctions_dict:
    def __init__(self, X, mf_spec):
        self.input_columns = X.shape[1]
        self.membership_functions = []
        self.names = []
        
        mf_dict = {
            'gaussian': gaussian,
            'gbell': gbell,
            'trapezoidal': trapezoidal,
            'triangular': triangular,
            'sigmoid': sigmoid
        }
        
        if isinstance(mf_spec, tuple) and len(mf_spec) == 2:
            mf_type, mf_num = mf_spec
            mf_specs = [(mf_type, mf_num)] * self.input_columns
        elif isinstance(mf_spec, list) and all(isinstance(spec, tuple) and len(spec) == 2 for spec in mf_spec):
            mf_specs = mf_spec
        else:
            raise ValueError("Invalid membership function specification")
        
        for i in range(self.input_columns):
            mf_type, mf_num = mf_specs[i]
            
            if mf_type in mf_dict:
                mf_class = mf_dict[mf_type]
                mf = mf_class(X[:, i].min(), X[:, i].max(), mf_num)
                self.names.append(f'{mf_type}_{i+1}')
            else:
                raise ValueError(f"Invalid membership function type: {mf_type}")
            
            self.membership_functions.append(mf)

# Generate sample input data
X = jnp.array(np.random.rand(100, 5))
mf_spec = ('gaussian', 3)

# Discard the first instance
_ = membershipFunctions_elif(X, mf_spec)
_ = membershipFunctions_dict(X, mf_spec)

# Time the execution of the elif version
elif_times = []
for _ in range(100):
    start_time = time.time()
    mf_instance_elif = membershipFunctions_elif(X, mf_spec)
    elif_times.append(time.time() - start_time)

# Time the execution of the dictionary version
dict_times = []
for _ in range(100):
    start_time = time.time()
    mf_instance_dict = membershipFunctions_dict(X, mf_spec)
    dict_times.append(time.time() - start_time)

# Calculate the average execution times
elif_avg_time = sum(elif_times) / len(elif_times)
dict_avg_time = sum(dict_times) / len(dict_times)

print(f"Average execution time (elif version): {elif_avg_time:.5f} seconds")
print(f"Average execution time (dictionary version): {dict_avg_time:.5f} seconds")