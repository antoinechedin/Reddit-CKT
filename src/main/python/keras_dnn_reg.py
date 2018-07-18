import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def init_datasets(dataset):
    dataset = dataset.sample(frac=1)
    return dataset[features_name], dataset[label_name]

class TrainLog(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.start = datetime.now()

    def on_epoch_end(self, epoch, logs={}):
        if datetime.now() > self.start + timedelta(seconds=5):
            print(epoch)
            print(logs)
            self.start = datetime.now()


if __name__ == "__main__":
    features_name = [
        "selftextLength", "titleLength",
        "selftextARI", "titleARI",
        "selftextSpellingError", "titleSpellingError",
        "selftextSpellingErrorRatio", "titleSpellingErrorRatio",
        "selftextFOG", "titleFOG",
        "selftextFK", "titleFK",
        "selftextCLI", "titleCLI",
        "polaritySelftext", "polarityTitle",
        "polarity"
    ]
    label_name = "score"

    learning_rate = 0.001
    batch_size = 46659
    epochs = 5000

    dataset = pd.read_csv(filepath_or_buffer="../../../datasets/dataset.csv", index_col="id")
    train_dataset, train_labels = init_datasets(dataset)

    print("Training set {} {}".format(train_dataset.shape, train_labels.shape))

    model = keras.Sequential([
        keras.layers.Dense(32, input_shape=(len(features_name),), activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate),
        loss=keras.losses.mean_absolute_error,
        metrics=[keras.metrics.mean_squared_error]
    )
    print(model.summary())

    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

    history = model.fit(
        train_dataset, train_labels,
        epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0, callbacks=[TrainLog(), earlyStopping]
    )

    plt.plot(history.epoch, history.history["loss"], label="Train loss")
    plt.plot(history.epoch, history.history["val_loss"], label="Valid loss")
    plt.legend()
    plt.show()
