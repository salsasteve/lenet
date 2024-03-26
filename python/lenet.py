from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow as tf
from config import Config

# https://medium.com/analytics-vidhya/lenet-architecture-document-recognition-ed971ab2a23f
# http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf
"""
Building a CNN in my own words:
    First, we build the convolution steps to automatically create the features.
    These features are generated in the convolution step in the CNN. We initially size the
    filters to some window size also called a kernal. Then we also initialize the filter values
    to some random low number and then run them over the images in a convolution pattern.
    We then run them through ReLU to remove negatives. Next, we normalize by subtracting the batch
    mean and dividing by the batch standard deviation, and then perform max pooling to reduce the
    dimensions of the filters. We repeat this process a couple of times to obtain finer-grain output
    features. The output of this is then flattened and fed into a deep neural network, where we
    perform normal neural network learning. We train the neural network after we have created the
    features and feed those in with the label to train our model.

    Super Werid:
        The features are create simultaneously with the training of the neural network. Its part of the training process.
        The features get better as the neural network learns.
"""


def LeNet():
    # Input layer image size 28x28x1
    print(Config.IMAGE_SIZE.value, Config.COLOR_CHANNELS.value)
    input_layer = layers.Input(
        (
            Config.IMAGE_SIZE.value,
            Config.IMAGE_SIZE.value,
            int(Config.COLOR_CHANNELS.value),
        )
    )

    # Block 1
    x = layers.Conv2D(
        filters=6, kernel_size=5, strides=1, activation="sigmoid", padding="same" 
    )(input_layer) 


    # 28x28x1 + padding = 32x32x1 then 32x32x1 - 5x5x1x6 = 28x28x6 6 filters makes 6 feature maps
    x = layers.AvgPool2D(pool_size=2, strides=2)(x)
    # 28x28x6 -> 14x14x6 after pooling
    # Check dimensions
    if Config.DEBUG:
        print(x.shape)
    if x.shape[1] != 14 or x.shape[2] != 14:
        raise ValueError("Error in LeNet Block 1")

    # Block 2 input 14x14x6
    x = layers.Conv2D(
        filters=16, kernel_size=5, strides=1, activation="sigmoid", padding="valid"
    )(x)
    # 14x14x6 - 5x5x6x16 = 10x10x16
    x = layers.AvgPool2D(pool_size=2, strides=2)(x)
    # 10x10x16 -> 5x5x16 after pooling
    # Check dimensions'
    if Config.DEBUG:
        print(x.shape)
    if x.shape[1] != 5 or x.shape[2] != 5:
        raise ValueError("Error in LeNet Block 2")

    # Fully connected layers neural network
    x = layers.Flatten()(x) 
    print(x.shape)
    x = layers.Dense(120, activation="sigmoid")(x)
    print(x.shape)
    x = layers.Dense(84, activation="sigmoid")(x)
    x = layers.Dense(10, activation="softmax")(x)

    model = Model(inputs=input_layer, outputs=x)

    return model


LeNet().summary()
