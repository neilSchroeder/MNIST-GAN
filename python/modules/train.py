import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, Activation, Conv2D, LeakyReLU, Dropout

# Load MNIST data
data = np.load('data.npy', allow_pickle=True)
target = np.load('target.npy', allow_pickle=True)
data = data / 255.0  # Normalize to [0, 1]


# Define the generator model
output_shape = (28, 28)
def build_generator():
    # define the DCGAN generator
    model = Sequential([
        # Start with a dense layer that takes a 100-dimensional noise vector as input
        Dense(7*7*256, input_dim=100),
        BatchNormalization(),
        Activation('relu'),
        # Reshape into a 7x7 256-channel feature map
        Reshape((7, 7, 256)),
        # Upsample to 14x14
        Conv2DTranspose(128, kernel_size=5, strides=1, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        # Upsample to 28x28
        Conv2DTranspose(64, kernel_size=5, strides=2, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        # Final layer - output a 28x28x1 image
        Conv2DTranspose(1, kernel_size=5, strides=2, padding='same'),
        Activation('tanh')  # tanh is often used for the output activation
    ])
    return model

# Define the discriminator model
def build_discriminator():
    model = Sequential([
        # Input layer: 28x28x1 image
        Conv2D(64, kernel_size=5, strides=2, padding='same', input_shape=(28, 28, 1)),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        # Downsample to 14x14
        Conv2D(128, kernel_size=5, strides=2, padding='same'),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        # Downsample to 7x7
        Flatten(),
        Dense(1, activation='sigmoid')  # Output layer with a single neuron for binary classification
    ])
    return model

# Compile the GAN
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer='adam', loss='binary_crossentropy')
discriminator.trainable = False
gan_input = tf.keras.Input(shape=(100,))
fake_image = generator(gan_input)
gan_output = discriminator(fake_image)
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

# create a real-time plot of the generated images
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plot_generated_images(epoch, generator, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=(examples, 100))
    generated_images = generator.predict(noise)
    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'gan_generated_image_epoch_{epoch}.png')


# Define the training procedure
def train_gan(data, epochs=200, batch_size=128):
    for epoch in range(epochs):
        # Generate random noise as input
        noise = np.random.normal(0, 1, size=(batch_size, 100))
        # Generate fake images
        generated_images = generator.predict(noise)
        # Get a random set of real images
        real_images = data[np.random.randint(0, data.shape[0], batch_size)]
        real_images = real_images.reshape(-1, 28, 28)
        # Train the discriminator
        discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch(real_images, np.ones(batch_size))
        d_loss_fake = discriminator.train_on_batch(generated_images, np.zeros(batch_size))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        # Train the generator
        noise = np.random.normal(0, 1, size=(batch_size, 100))
        discriminator.trainable = False
        g_loss = gan.train_on_batch(noise, np.ones(batch_size))
        # Print the progress
        print(f"Epoch: {epoch} D Loss: {d_loss} G Loss: {g_loss}")
        # Plot the progress
        if epoch % 10 == 0:
            plot_generated_images(epoch, generator)

# Train the GAN
train_gan(data)
