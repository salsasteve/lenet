import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from config import Config
from lenet import LeNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import tf2onnx
import onnx


# def import_mnist():
#     mnist = tf.keras.datasets.mnist

#     (x_train, y_train), (x_test, y_test) = mnist.load_data()

#     x_merged = np.concatenate((x_train, x_test), axis=0)
#     y_merged = np.concatenate((y_train, y_test), axis=0)

#     # split the data into training and testing
#     x_train, x_test, y_train, y_test = train_test_split(
#         x_merged, y_merged, test_size=0.2, random_state=42
#     )

#     return x_train, x_test, y_train, y_test

def import_mnist():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    return x_train, x_test, y_train, y_test



def main():
    # Load the MNIST dataset
    x_train, x_test, y_train, y_test = import_mnist()

    # One-hot encode the labels
    y_train = to_categorical(y_train, num_classes=10)  # Assuming there are 10 classes
    y_test = to_categorical(y_test, num_classes=10)

    # Normalize the data
    x_train = x_train / 255.0 # Data is 0-255 so we normalize to 0-1
    x_test = x_test / 255.0

    # Reshape the data for CNN
    x_train = x_train.reshape(
        -1,
        Config.IMAGE_SIZE.value,
        Config.IMAGE_SIZE.value,
        int(Config.COLOR_CHANNELS.value),
    )
    x_test = x_test.reshape(
        -1,
        Config.IMAGE_SIZE.value,
        Config.IMAGE_SIZE.value,
        int(Config.COLOR_CHANNELS.value),
    )

    # Create the model
    model = LeNet()

    # Compile the model
    model.compile(
        loss=CategoricalCrossentropy(),
        optimizer=Adam(learning_rate=0.01),
        metrics=["accuracy"],
    )

    es = EarlyStopping(patience=5, monitor="loss")

    model.fit(
        x_train,
        y_train,
        batch_size=Config.BATCH_SIZE.value,
        epochs=Config.EPOCHS.value,
        validation_data=(x_test, y_test),
        callbacks=[es],
    )

    # model.save("attempt2.hdf5")

    input_signature = [
        tf.TensorSpec([None, 28, 28, 1], dtype=tf.float32)
    ]
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
    onnx.save(onnx_model, "attempt2.onnx")

    return 0


if __name__ == "__main__":
    main()
