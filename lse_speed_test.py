import numpy as np
import jax.numpy as jnp
import jax.random as random
import time

def LSE(A, B, initialGamma = 1000.):
    coeffMat = A
    rhsMat = B
    S = np.eye(coeffMat.shape[1])*initialGamma
    x = np.zeros((coeffMat.shape[1],1)) # need to correct for multi-dim B
    for i in range(len(coeffMat[:,0])):
        a = coeffMat[i,:]
        b = rhsMat[i]
        S = S - (np.array(np.dot(np.dot(np.dot(S,np.matrix(a).transpose()),np.matrix(a)),S)))/(1+(np.dot(np.dot(S,a),a)))
        x = x + (np.dot(S,np.dot(np.matrix(a).transpose(),(np.matrix(b)-np.dot(np.matrix(a),x)))))
    return x

def LSE_jax(A, B):
    x, _, _, _ = jnp.linalg.lstsq(A, B)
    return x

if __name__ == "__main__":
    A = random.uniform(random.PRNGKey(0), shape=(10,1))
    B = random.uniform(random.PRNGKey(1), shape=(10,1))

    start_time = time.time()
    result1 = LSE(A, B, 1000)
    end_time = time.time()
    print("LSE result:")
    print(result1)
    print("Time taken by LSE:", end_time - start_time, "seconds")
    print()

    start_time = time.time()
    result2 = LSE_jax(A, B)
    end_time = time.time()
    print("LSE_jax result:")
    print(result2)
    print("Time taken by LSE_jax:", end_time - start_time, "seconds")
