import numpy as np

import utils


class AutoEncoder:
    def __init__(self, layers, adam_opts):
        self.layers = layers
        self.adam_opts = adam_opts

        # global properties
        self.layer_count, self.errors = len(layers), []

        self.rw, self.mw, self.rb, self.mb = {}, {}, {}, {}
        # layer properties
        self.layer_weights = {}
        self.layer_bias = {}
        self.activation_function = {}

        for i, layer in zip(range(1, self.layer_count + 1), layers):
            n_out, n_in = layer["shape"]

            self.activation_function[i] = layer["act"]
            # Xavier Initialization of weights
            self.layer_weights[i] = np.random.randn(n_out, n_in) / n_in ** 0.5

            self.layer_bias[i] = np.zeros((n_out, 1))
            self.rb[i] = np.zeros((n_out, 1))
            self.mb[i] = np.zeros((n_out, 1))

            self.rw[i] = np.zeros((n_out, n_in))
            self.mw[i] = np.zeros((n_out, n_in))

    def fit(self, x_train, x_test, epochs=30, batch_size=30):
        epoch_results = []

        for epoch in range(1, epochs + 1):
            # Train
            layer_result = []
            for batch in np.split(x_train, batch_size):
                # Forward pass
                layer_result.append(batch.T)
                for i in range(1, self.layer_count + 1):
                    layer_result.append(
                        self.activation_function[i](
                            (self.layer_weights[i] @ layer_result[i - 1])
                            + self.layer_bias[i]
                        )
                    )

                # Backpropagation
                dz, dw, db = {}, {}, {}
                for i in range(1, self.layer_count + 1)[::-1]:
                    d = (
                        self.layer_weights[i + 1].T @ dz[i + 1]
                        if self.layer_count - i
                        else 0.5 * (layer_result[self.layer_count] - layer_result[0])
                    )
                    dz[i] = d * self.activation_function[i](layer_result[i], d=1)
                    dw[i] = dz[i] @ layer_result[i - 1].T
                    db[i] = np.sum(dz[i], 1, keepdims=True)

                for i in range(1, self.layer_count + 1):
                    utils.adam(
                        self.mw,
                        self.rw,
                        self.layer_weights,
                        dw,
                        i,
                        self.adam_opts,
                        epoch,
                    )
                    utils.adam(
                        self.mb, self.rb, self.layer_bias, db, i, self.adam_opts, epoch
                    )

            # Validate

            epoch_results.append(self.predict(x_test))

            self.errors += [np.mean((epoch_results[-1] - layer_result[0]) ** 2)]
            print("Epoch: {e}\tVal loss {l}".format(e=epoch, l=self.errors[-1]))
            return epoch_results

    def predict(self, X):
        layer_result = [X.T]

        # forward pass
        for i in range(1, self.layer_count + 1):
            layer_result.append(
                self.activation_function[i](
                    self.layer_weights[i] @ layer_result[i - 1] + self.layer_bias[i]
                )
            )
        y_pred = layer_result[-1]

        return y_pred
