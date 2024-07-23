import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, LeakyReLU, Flatten, Dense
)
from tensorflow.keras.optimizers import Adam

def build_discriminator(in_shape=(28, 28, 1)):
    """
    Builds a discriminator model for a DCGAN.
    """
    initial_kernel = tf.keras.initializers.RandomNormal(stddev=0.02)
    model = Sequential([
        Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=initial_kernel, input_shape=in_shape),
        LeakyReLU(alpha=0.2),
        Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=initial_kernel),
        LeakyReLU(alpha=0.2),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])

    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model