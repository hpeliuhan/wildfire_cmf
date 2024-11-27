from cmflib import cmf
 
# import the necessary packages

# from imutils import paths
import tarfile
import numpy as np
from tensorflow.keras.utils import to_categorical
import cv2
import os
import regex as re
import click

def load_dataset(inputPath:str,outputPath:str):
    """ Load input file (images) and create images numpy files in output_dir directory.
    Args:
         
         output_dir: Path to a directory that will contain images.npy(normalized) and labels.npy files.

    Machine Learning Artifacts:
        Input: ${input_file}/all.tar.gz
        Output: ${output_dir}/images.npy, ${output_dir}/labels.npy
    """
    # grab the paths to all images in our dataset directory, then
    # initialize our lists of images

    graph_env = os.getenv("NEO4J", "True")
    graph = True if graph_env == "True" or graph_env == "TRUE" else False
    normalizer=255
    metawriter = cmf.Cmf(filepath="cmf", pipeline_name="WILDFIRE", graph=graph)
    _ = metawriter.create_context(pipeline_stage="data_load")
    _ = metawriter.create_execution(execution_type="classify_images", custom_properties={"normalizer":normalizer})
    datasetName="all.tar.gz"
    dataset_path = os.path.join(inputPath,datasetName)

    with tarfile.open(dataset_path, "r:gz") as tar:
            tar.extractall(path=inputPath)
    imagePaths = [d for d in os.listdir(inputPath) if os.path.isdir(os.path.join(inputPath, d))]
    print(imagePaths)
    print(inputPath)
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
@click.argument('input_dir', required=True, type=str)
@click.argument('output_dir', required=True, type=str)
def load_dataset_cli(input_dir:str, output_dir: str) -> None:
    load_dataset(input_dir, output_dir)


if __name__ == '__main__':
    load_dataset_cli()