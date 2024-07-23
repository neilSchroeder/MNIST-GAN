import tensorflow as tf
import numpy as np
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Reshape,
    Conv2D,
    Conv2DTranspose,
    LeakyReLU,
)

def build_generator(latent_dim=100):
    """
    Builds a generator model for a DCGAN.
    """
    initial_kernel = RandomNormal(stddev=0.02)
    n_nodes = 128 * 7 * 7
    model = Sequential([
        Dense(n_nodes, kernel_initializer=initial_kernel, input_dim=latent_dim),
        LeakyReLU(alpha=0.2),
        Reshape((7, 7, 128)),
        Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=initial_kernel),
        LeakyReLU(alpha=0.2),
        Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=initial_kernel),
        LeakyReLU(alpha=0.2),
        Conv2D(1, (7, 7), activation='tanh', padding='same', kernel_initializer=initial_kernel)
    ])

    return model

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = np.random.randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = generator.predict(x_input)
	# create class labels
	y = np.zeros((n_samples, 1))
	return X, y

# select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = np.random.randint(0, dataset.shape[0], n_samples)
	# select images
	X = dataset[ix]
	# generate class labels
	y = np.ones((n_samples, 1))
	return X, y
        