import keras

import utils
from model import AutoEncoder

if __name__ == "__main__":
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    x_train = (x_train / 255).reshape(-1, 784)
    x_test = (x_test / 255).reshape(-1, 784)

    layers = [
        # activation, shape:(out,in)
        # {"act": utils.relu, "shape": (1024, 784)},
        # {"act": utils.relu, "shape": (50, 1024)},
        # {"act": utils.relu, "shape": (1024, 50)},
        # {"act": utils.sigmoid, "shape": (784, 1024)},
        {"act": utils.relu, "shape": (50, 784)},
        {"act": utils.sigmoid, "shape": (784, 50)},
    ]

    # learning_rate, beta_1, beta_2
    adam_opts = [0.002, 0.9, 0.999]

    autoencoder = AutoEncoder(layers, adam_opts)

    epoch_results = autoencoder.fit(x_train, x_test, epochs=10)

    y_pred = autoencoder.predict(x_train[:20])

    import pylab as plt

    plt.figure(figsize=(20, 204))

    for i in range(20):
        plt.subplot(3, 20, i + 1)
        plt.imshow(x_train[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
        plt.grid(b=False)

    for k, epoch_result in enumerate(epoch_results):
        for i in range(20):
            plt.subplot(3, 20, i + 1 + 20 * (k + 1))
            plt.imshow(x_train[i].reshape(28, 28), cmap="gray")
            plt.axis("off")
            plt.grid(b=False)

    for i in range(20):
        plt.subplot(3, 20, i + 1 + 40)
        plt.imshow(y_pred.T[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
        plt.grid(b=False)

    plt.show()
