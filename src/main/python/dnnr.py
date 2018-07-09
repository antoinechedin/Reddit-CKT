import os
import shutil
import argparse
import math
from datetime import datetime

import pandas as pd
import tensorflow as tf
from sklearn import metrics
import matplotlib.pyplot as plt

# FILES VAR

# Features names
MODEL_DIR_PATH = "../../../model_dir/"
FEATURES_NAME_LIST = [
    "selftextLength", "titleLength",
    "selftextARI", "titleARI",
    "selftextSpellingError", "titleSpellingError",
    "selftextSpellingErrorRatio", "titleSpellingErrorRatio",
    "selftextFOG", "titleFOG",
    "selftextFK", "titleFK",
    "selftextCLI", "titleCLI",
    "polaritySelftext", "polarityTitle",
    "polarity"
]  # Deviation is not used, it has no meaning here
LABEL_NAME = "score"

# Hyper parameters
INPUT_RECORDS_NUM = None
BATCH_SIZE = 100
HIDDEN_UNITS = [25, 15, 5]
LEARNING_RATE = 0.3
PERIOD = 10
STEPS = 10000

FLAGS = None


def train_input_fn(features, labels, shuffle_buffer_size, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    return dataset.shuffle(shuffle_buffer_size).repeat().batch(batch_size).make_one_shot_iterator().get_next()


def eval_input_fn(features, labels=None, batch_size=None):
    if labels is None:
        # Use this for predictions
        inputs = dict(features)
    else:
        inputs = (dict(features), labels)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    return dataset.batch(batch_size)


def get_feature_columns(training_dataset):
    features_column_list = []
    for key in training_dataset.keys():
        features_column_list.append(tf.feature_column.numeric_column(key=key))
    return features_column_list


def evaluate_classifier(classifier, train_features, train_labels, train_rmse_list, validation_features,
                        validation_labels, validation_rmse_list):
    train_predictions = classifier.predict(
        input_fn=lambda: eval_input_fn(train_features, batch_size=BATCH_SIZE)
    )
    validation_predictions = classifier.predict(
        input_fn=lambda: eval_input_fn(validation_features, batch_size=BATCH_SIZE)
    )
    train_predictions = [item["predictions"][0] for item in train_predictions]
    validation_predictions = [item["predictions"][0] for item in validation_predictions]

    train_rmse = math.sqrt(metrics.mean_squared_error(train_labels, train_predictions))
    validation_rmse = math.sqrt(metrics.mean_squared_error(validation_labels, validation_predictions))

    train_rmse_list.append(train_rmse)
    validation_rmse_list.append(validation_rmse)

    return validation_predictions


def init_datasets(input_files_list, fold_index):
    # Load validation dataset
    validation_dataframe = pd.read_csv(filepath_or_buffer=input_files_list[fold_index], delimiter=",", header=0,
                                       index_col="id")

    # Merge and load train datasets
    train_dataframe = None
    is_first = True
    # For each dataset in the datasets dir
    for i, dataset_str in enumerate(input_files_list):
        if i is not fold_index:
            if is_first:
                # If first dataset found, just load it
                train_dataframe = pd.read_csv(filepath_or_buffer=dataset_str, delimiter=",", header=0,
                                              index_col="id")
                is_first = False
            else:
                # Else merge the dataset found
                train_dataframe = train_dataframe.append(
                    pd.read_csv(filepath_or_buffer=dataset_str, delimiter=",", header=0, index_col="id")
                )

    return train_dataframe[FEATURES_NAME_LIST], train_dataframe[LABEL_NAME], validation_dataframe[FEATURES_NAME_LIST], \
           validation_dataframe[LABEL_NAME]


def cross_val_predict(input_files_list, output_file, folds):
    predict_label_dataframe = None

    steps_by_period = int(STEPS / PERIOD)

    # Cross validation
    for fold_index in range(folds):
        # Load train and validation features/label
        train_features, train_labels, validation_features, validation_labels = init_datasets(input_files_list,
                                                                                             fold_index=fold_index)
        INPUT_RECORDS_NUM = train_features.shape[0]

        print("\n{}: Training records: {}".format(datetime.now(), train_features.shape[0]))
        print("{}: Validation records: {}".format(datetime.now(), validation_features.shape[0]))
        print("{}: Features number: {}".format(datetime.now(), train_features.shape[1]))
        # Remove previous classifier model directory
        if os.path.isdir(MODEL_DIR_PATH):
            print("{}: Remove model_directory: {}".format(datetime.now(), MODEL_DIR_PATH))
            shutil.rmtree(MODEL_DIR_PATH, ignore_errors=True)

        # Init classifier
        classifier = tf.estimator.DNNRegressor(
            hidden_units=HIDDEN_UNITS,
            feature_columns=get_feature_columns(train_features),
            model_dir=MODEL_DIR_PATH,
            label_dimension=1,
            optimizer=tf.train.AdagradOptimizer(learning_rate=LEARNING_RATE)
        )

        print("{}: BEGIN TRAINING FOLD {}/{}".format(datetime.now(), fold_index + 1, folds))
        train_rmse_list = []
        validation_rmse_list = []
        # train on step to initialise
        classifier.train(
            input_fn=lambda: train_input_fn(train_features, train_labels, train_features.shape[0], BATCH_SIZE),
            steps=1
        )
        evaluate_classifier(classifier, train_features, train_labels, train_rmse_list, validation_features,
                            validation_labels, validation_rmse_list)
        print("{}: Period {}/{}: train_rmse={}, validation_rmse={}".format(
            datetime.now(), 0, PERIOD, train_rmse_list[-1], validation_rmse_list[-1]
        ))

        validation_predict_list = None
        for i in range(PERIOD):
            classifier.train(
                input_fn=lambda: train_input_fn(train_features, train_labels, train_features.shape[0], BATCH_SIZE),
                steps=steps_by_period
            )

            validation_predict_list = evaluate_classifier(classifier, train_features, train_labels, train_rmse_list,
                                                          validation_features, validation_labels, validation_rmse_list)
            print("{}: Period {}/{}: train_rmse={}, validation_rmse={}".format(
                datetime.now(), i + 1, PERIOD, train_rmse_list[-1], validation_rmse_list[-1]
            ))

        # Get final predictions
        # validation_labels["karma_predict"] = validation_predict_list

        if predict_label_dataframe is None:
            predict_label_dataframe = pd.Series(validation_predict_list, index=validation_labels.index).to_frame(
                name="karma_predicted")
        else:
            predict_label_dataframe = predict_label_dataframe.append(
                pd.Series(validation_predict_list, index=validation_labels.index).to_frame(name="karma_predicted"), verify_integrity=True)

        fig, ax = plt.subplots()
        ax.set(xlabel="period", ylabel="rmse", title="rmse vs. Periods")
        ax.plot(train_rmse_list, label="training")
        ax.plot(validation_rmse_list, label="validation")
        ax.grid()
        ax.legend()

        print("{}: TRAINING FINISHED".format(datetime.now()))

        # plt.show()
    predict_label_dataframe.to_csv(output_file)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="path to the directory containing dataset for the the cross validation.")
    parser.add_argument("output_file", help="DNN regressor prediction output file.")
    FLAGS = parser.parse_args()

    # Gather flags
    input_dir = FLAGS.input_dir
    output_file_str = FLAGS.output_file

    # Get all dataset parts
    input_files_list = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if
                        os.path.isfile(os.path.join(input_dir, f))]
    num_fold = len(input_files_list)

    # Perform a cross-validation prediction on the whole dataset
    cross_val_predict(input_files_list, output_file_str, folds=num_fold)

    print("\n{}: DNNR Finished".format(datetime.now()))
