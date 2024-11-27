# train_test_split.py

import numpy as np
from sklearn.model_selection import train_test_split
import yaml
import os
from cmflib import cmf
import click
from tensorflow.keras.utils import to_categorical


def load_and_split_data(input_path: str, out_put_path: str):
    """
    Loads the saved .npy files and splits the data into training and testing sets.

    Parameters:
    - image_numpy_file: Path to the saved images .npy file
    - label_numpy_file: Path to the saved labels .npy file
    - test_size: float, the proportion of the dataset to include in the test split
    - random_state: int, random seed for reproducibility

    Returns:
    - X_train, X_test, y_train, y_test: split data and labels
    """
    params = yaml.safe_load(open("params.yaml"))["parse"]
    ratio=params["split"]
    random_state=params["random_state"]
    print("random_state",random_state)
    data_path=os.path.join(input_path, "images.npy")
    label_path=os.path.join(input_path, "labels.npy")
    data = np.load(data_path)
    labels = np.load(label_path)


    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=ratio, random_state=random_state)


    class_totals = y_train.sum(axis=0)
    class_weight = class_totals.max() / class_totals

    train_images_path = os.path.join(out_put_path, "train_images.npy")
    val_images_path = os.path.join(out_put_path, "val_images.npy")
    train_labels_path = os.path.join(out_put_path, "train_labels.npy")
    val_labels_path = os.path.join(out_put_path, "val_labels.npy")
    classWeight_path = os.path.join(out_put_path, "classWeight.npy")

    np.save(train_images_path, X_train)
    np.save(val_images_path, X_test)
    np.save(train_labels_path, y_train)
    np.save(val_labels_path, y_test)
    np.save(classWeight_path, class_weight)

    #metadata artifect
    graph_env = os.getenv("NEO4J", "True")
    graph = True if graph_env == "True" or graph_env == "TRUE" else False
    metawriter = cmf.Cmf(filepath="cmf", pipeline_name="WILDFIRE", graph=graph)

    _ = metawriter.create_context(pipeline_stage="parse")
    _ = metawriter.create_execution(execution_type="train test split", custom_properties=params)

    _ = metawriter.log_dataset(data_path,'input')
    _ = metawriter.log_dataset(label_path,'input')
    _ = metawriter.log_dataset(train_images_path,'output', custom_properties={"split": "train", "num_images": len(X_train),"training_data_shape":X_train.shape})
    _ = metawriter.log_dataset(val_images_path,'output', custom_properties={"split": "validation", "num_images": len(X_test),"valuation_data_shape":X_test.shape})
    _ = metawriter.log_dataset(train_labels_path,'output')
    _ = metawriter.log_dataset(val_labels_path,'output')
    _ = metawriter.log_dataset(classWeight_path,'output')
    
    return X_train, X_test, y_train, y_test,class_weight


@click.command()
@click.argument('input_dir', required=True, type=str)
@click.argument('output_dir', required=True, type=str)
def train_test_split_cli(input_dir:str, output_dir: str) -> None:
    load_and_split_data(input_dir, output_dir)


if __name__ == '__main__':
    train_test_split_cli() 