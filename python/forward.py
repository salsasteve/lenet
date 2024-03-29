from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow as tf
from config import Config
import numpy as np



def LeNetForward():
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
        filters=6, kernel_size=5, strides=1, activation="tanh", padding="same" 
    )(input_layer) 

    # 28x28x1 + padding = 32x32x1 then 32x32x1 - 5x5x1x6 = 28x28x6 6 filters makes 6 feature maps
    x = layers.AvgPool2D(pool_size=2, strides=2)(x)
    # 28x28x6 -> 14x14x6 after pooling
    # Check dimensions
    if Config.DEBUG:
        print(x.shape)
    if x.shape[1] != 14 or x.shape[2] != 14:
        raise ValueError("Error in LeNet Block 1")

    

    # model[0]

   
    model = Model(inputs=input_layer, outputs=x)
    return model



model = LeNetForward()



 # load weights and biases from binary files
biases = np.fromfile('../read_model/parameters/conv2d_1_bias.bin', dtype=np.float32)
biases = biases.reshape(6)
weights = np.fromfile('../read_model/parameters/conv2d_1_weights.bin', dtype=np.float32)
weights = weights.reshape(5, 5, 1, 6)
model.layers[1].set_weights([weights, biases])
# do inference
# input data
input_data = np.fromfile('../mnist_data/mnist_x_test.bin', dtype=np.float32)

# reshape input data
input_data = input_data.reshape(10000, 28, 28, 1)

# do inference
output = model.predict(input_data)

# print first 10 outputs
print(output[0:10])
