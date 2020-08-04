import numpy as np


def sigmoid(z, d=False):
    return sigmoid(z) * (1 - sigmoid(z)) + 1e-12 if d else 1 / (1 + np.exp(-z))


def relu(z, d=False):
    return (z > 0) + 1e-12 if d else z * (z > 0)


# Adam updates
def adam(m, r, z, dz, i, adam_opts, epoch):
    lr, b1, b2 = adam_opts
    m[i] = b1 * m[i] + (1 - b1) * dz[i]
    r[i] = b2 * r[i] + (1 - b2) * dz[i] ** 2
    m_hat = m[i] / (1.0 - b1 ** epoch)
    r_hat = r[i] / (1.0 - b2 ** epoch)
    z[i] -= lr * m_hat / (r_hat ** 0.5 + 1e-12)
