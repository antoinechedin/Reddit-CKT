from typing import Dict, List

from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import os
import math


def get_files_in(directory: str) -> List[str]:
    """Return a list of file path which are inside a certain directory

    :type directory: str
    :param directory: Directory path
    :rtype: List[str]
    :return: A list of file path
    """
    return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]


def init_datasets(input_dir: str, fold_index: int) -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    """Initialise the different dataset for the regression

    :param input_dir: Path to the directory that contains the dataset. It must be already sliced into multiple fold for the cross validation
    :param fold_index: the fold index of the validation dataset. For N folds, the index must be bound between 0 and N-1
    :return: 4 dataset: train_feature, train_target, valid_feature, and valid_target.
    """

    # Retrieve all dataset files in the input directory
    input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if
                   os.path.isfile(os.path.join(input_dir, f))]
    if fold_index >= len(input_files):
        raise ValueError("Not enough dataset file for the index {}. There's only {} files in this directory".format(
            fold_index, len(input_files)
        ))

    # Load validation dataset
    valid_dataset = pd.read_csv(filepath_or_buffer=input_files[fold_index], delimiter=",", header=0, index_col="id")

    # Merge and load train datasets
    train_dataset = None
    is_first = True
    # For each dataset in the datasets dir
    for i, dataset_str in enumerate(input_files):
        if i is not fold_index:
            if is_first:
                # If first dataset found, just load it
                train_dataset = pd.read_csv(filepath_or_buffer=dataset_str, delimiter=",", header=0, index_col="id")
                is_first = False
            else:
                # Else merge the dataset found
                train_dataset = train_dataset.append(
                    pd.read_csv(filepath_or_buffer=dataset_str, delimiter=",", header=0, index_col="id")
                )

    if karma_max is not None:
        train_dataset[target_name] = train_dataset[target_name].where(train_dataset[target_name] < karma_max, karma_max)
        valid_dataset[target_name] = valid_dataset[target_name].where(valid_dataset[target_name] < karma_max, karma_max)
        assert train_dataset[target_name].max() <= karma_max
        assert valid_dataset[target_name].max() <= karma_max

    return train_dataset[features_name], train_dataset[target_name], valid_dataset[features_name], \
           valid_dataset[target_name]


if __name__ == "__main__":
    # --- PARAMS ---
    features_name = [
        "selftextLength",
        "titleLength",
        "selftextARI",
        "titleARI",
        "selftextSpellingError",
        "titleSpellingError",
        "selftextSpellingErrorRatio",
        "titleSpellingErrorRatio",
        "selftextFOG",
        "titleFOG",
        "selftextFK",
        "titleFK",
        "selftextCLI",
        "titleCLI",
        "polaritySelftext",
        "polarityTitle"
        "polarity"
    ]
    target_name = "score"
    dataset_dir = "datasets/5_folds_test/"
    output_file = "results/5_folds_test/linear_reg_predictions.csv"
    karma_max = None
    # ------

    folds = len(get_files_in(dataset_dir)) # Get number of file into the dataset dir
    metrics: Dict[str, List[int]] = {"mae": [], "rmse": []}
    predict_dataset: pd.DataFrame = None

    print("Begin regression")
    for fold in range(folds):
        train_features, train_target, valid_features, valid_target = init_datasets(dataset_dir, fold)
        # Instantiate the linear regressor, normalise=True is used to reduce and center features
        regressor = linear_model.LinearRegression(normalize=True)
        regressor.fit(train_features, train_target)

        valid_pred = regressor.predict(X=valid_features)

        metrics["mae"].append(mean_absolute_error(valid_target, valid_pred))
        metrics["rmse"].append(math.sqrt(mean_squared_error(valid_target, valid_pred)))

        print("Fold {} validation[mae: {}, rmse: {}]".format(fold, metrics["mae"][-1], metrics["rmse"][-1]))

        valid_pred = pd.Series(valid_pred, index=valid_target.index, name="karma_predicted").to_frame()
        valid_pred["karma"] = valid_target
        for i in range(len(features_name)):
            print("    {}: {}".format(features_name[i], regressor.coef_[i]))

        if predict_dataset is None:
            predict_dataset = valid_pred
        else:
            predict_dataset = predict_dataset.append(valid_pred)

    print("Regression finished, saving prediction file...")
    predict_dataset[["karma", "karma_predicted"]].to_csv(output_file)
    
    print("END OF SCRIPT")
