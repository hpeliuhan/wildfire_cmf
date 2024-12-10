from cmflib import cmf
import tarfile
import numpy as np
from tensorflow.keras.utils import to_categorical
import cv2
import os
import regex as re
import click
from utils import *
import yaml
def load_dataset(config_file:str,dir_config_file:str):
    """ Load input file (images) and create images numpy files in output_dir directory.
    Args:
         
         output_dir: Path to a directory that will contain images.npy(normalized) and labels.npy files.

    Machine Learning Artifacts:
        Input: ${input_file}/all.tar.gz
        Output: ${output_dir}/images.npy, ${output_dir}/labels.npy
    """
    # grab the paths to all images in our dataset directory, then
    # initialize our lists of images
    config_file = yaml.safe_load(open(config_file))["data_parse"]
    normalizer=config_file["normalizer"]

    dir_config_file = yaml.safe_load(open(dir_config_file))["dir_config"]["data_parse"]
    inputPath = dir_config_file["input"]
    outputPath = dir_config_file["output"]
    os.makedirs(outputPath, exist_ok=True)

    metawriter = set_cmf_environment("cmf", "WILDFIRE")
    _ = metawriter.create_context(pipeline_stage="data_collection")
    _ = metawriter.create_execution(execution_type="data_parse", custom_properties={"normalizer":normalizer})
    datasetName="all.tar.gz"
    dataset_path = os.path.join(inputPath,datasetName)

    with tarfile.open(dataset_path, "r:gz") as tar:
            tar.extractall(path=inputPath)
    imagePaths = [d for d in os.listdir(inputPath) if os.path.isdir(os.path.join(inputPath, d))]

    _ = metawriter.log_dataset(dataset_path, "input", custom_properties={"image_name": dataset_path})  

    datasetList=[]
    labelList=[]
 
    # loop over the image paths
    for directories in imagePaths:
        print(directories)
        tempF= []
        tempNF = []
        counter=0
        for element in os.listdir(os.path.join(inputPath, directories)):
            if re.search(".jpg", element):
                counter+=1
                image_path=inputPath + "/"+ directories + "/" + element
                image = cv2.imread(image_path)
                image = cv2.resize(image, (128,128))
                if "+" in element:
                    tempF.append(image)
                else:
                    tempNF.append(image)
 
        tempF = np.array(tempF, dtype="float32")
        tempNF = np.array(tempNF,  dtype="float32")

        os.makedirs(outputPath, exist_ok=True)
        fireLabels = np.ones((tempF.shape[0],))
        nonFireLabels = np.zeros((tempNF.shape[0],))
        data = np.vstack([tempF, tempNF])
        labels = np.hstack([fireLabels, nonFireLabels])
        labels = to_categorical(labels, num_classes=2)    
        data /= normalizer
        #print(data.shape)
        #print(labels.shape)
        if counter !=0:
            datasetList.append(data)
            labelList.append(labels)
        print("there are ", counter, " images in ", directories)
        # Combine all data and labels into single arrays
    final_data = np.vstack(datasetList)
    final_labels = np.vstack(labelList)
    output_image_numpy_file= outputPath + "/images.npy"
    output_label_numpy_file= outputPath + "/labels.npy"
    np.save(output_image_numpy_file, final_data)
    np.save(output_label_numpy_file, final_labels)
    _ = metawriter.log_dataset(output_image_numpy_file, "output", custom_properties={"image_arrary_shape": data.shape})
    _ = metawriter.log_dataset(output_label_numpy_file, "output", custom_properties={"label": "labels"})




@click.command()
@click.argument('config_file', required=True, type=str)
@click.argument('dir_config_file', required=True, type=str)
def load_dataset_cli(config_file:str, dir_config_file: str) -> None:
    load_dataset(config_file, dir_config_file)


if __name__ == '__main__':
    load_dataset_cli()