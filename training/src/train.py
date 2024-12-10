from FireDetectionNet import *
import os 
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import click
import numpy as np
import matplotlib.pyplot as plt
import yaml
from cmflib import cmf
from utils import *

def choose_learning_rate(data:np.array)->float:
    lrs, losses = data[:, 0], data[:, 1]
    gradients = np.gradient(losses)

    # Find the index where the gradient is the most negative (steepest descent)
    min_gradient_index = np.argmin(gradients)

    # Choose the learning rate corresponding to this index
    optimal_lr = lrs[min_gradient_index]
    return optimal_lr



def train(config_file:str, dir_config_file:str):
    # input_dir1: training data/test data
    # input_dir2: model data
    # input_dir3: learning rate 
    # output_dir: model data

    params=yaml.safe_load(open(config_file))["train"]
    rotation_range=params["ImageDataGenerator"]["rotation_range"]
    zoom_range=params["ImageDataGenerator"]["zoom_range"]
    width_shift_range=params["ImageDataGenerator"]["width_shift_range"]
    height_shift_range=params["ImageDataGenerator"]["height_shift_range"]
    shear_range=params["ImageDataGenerator"]["shear_range"]
    horizontal_flip=params["ImageDataGenerator"]["horizontal_flip"]
    fill_mode=params["ImageDataGenerator"]["fill_mode"]

    BATCH_SIZE=params["batch_size"]
    NUM_EPOCHS=params["epochs"]
    

    dir_config_file=yaml.safe_load(open(dir_config_file))["dir_config"]["train"]
    input_dir1=dir_config_file["input1"]
    input_dir2=dir_config_file["input2"]
    input_dir3=dir_config_file["input3"]

    output_path=dir_config_file["output"]
    os.makedirs(output_path, exist_ok=True)

    
    trainX_path=os.path.join(input_dir1, "train_images.npy")
    trainY_path=os.path.join(input_dir1, "train_labels.npy")
    testX_path=os.path.join(input_dir1, "val_images.npy")
    testY_path=os.path.join(input_dir1, "val_labels.npy")
    class_weight_file=os.path.join(input_dir1,"classWeight.npy")
    model_path=os.path.join(input_dir2, "init_model.keras")
    model=tf.keras.models.load_model(model_path)
    trainX= np.load(trainX_path)
    trainY= np.load(trainY_path) 
    testX= np.load(testX_path)
    testY= np.load(testY_path)
    classWeight = np.load(class_weight_file)
    class_weight_dict = {i: weight for i, weight in enumerate(classWeight)}
    print(class_weight_dict)

    learning_rate_path=os.path.join(input_dir3, "lr_loss.npy")
    optimal_lr=choose_learning_rate(np.load(learning_rate_path))
    
    tf.keras.backend.set_value(model.optimizer.learning_rate, optimal_lr)
    aug=ImageDataGenerator(
        rotation_range=rotation_range,
        zoom_range=zoom_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        horizontal_flip=horizontal_flip,
        fill_mode=fill_mode)

    modelCheckpointFile=os.path.sep.join([output_path, 'best_model.h5'])
    mc = tf.keras.callbacks.ModelCheckpoint(modelCheckpointFile, monitor='val_loss', mode='min', verbose=1)
    H = model.fit(
        aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
        validation_data=(testX, testY),
        steps_per_epoch=trainX.shape[0] // BATCH_SIZE,
        epochs=NUM_EPOCHS,
        class_weight=class_weight_dict,
        callbacks=[mc],
        verbose=1)
    model_finish_path=os.path.join(output_path, "trained_model.keras")
    tf.keras.models.save_model(model, model_finish_path)
    history_path=os.path.join(output_path, "history.npy")
    np.save(history_path, H.history)

    metawriter= set_cmf_environment("cmf","WILDFIRE")
    _ = metawriter.create_context(pipeline_stage="model training")
    _ = metawriter.create_execution(execution_type="train", custom_properties=params)
    _ = metawriter.log_dataset(trainX_path,"input")
    _ = metawriter.log_dataset(trainY_path,"input")
    _ = metawriter.log_dataset(testX_path,"input")
    _ = metawriter.log_dataset(testY_path,"input")
    _ = metawriter.log_dataset(class_weight_file,"input")
    _ = metawriter.log_dataset(model_path ,"input")
    _ = metawriter.log_dataset(history_path  ,"output")
    _ = metawriter.log_model(
        path=modelCheckpointFile,event="output",model_framework="tensorflow",
    model_type="CNN",custom_properties={"learning_rate":optimal_lr, "batch_size":BATCH_SIZE, "num_epochs":NUM_EPOCHS})
    _ = metawriter.log_model(
        path=model_finish_path,event="output",model_framework="tensorflow",
    model_type="CNN", custom_properties={"learning_rate":optimal_lr, "batch_size":BATCH_SIZE, "num_epochs":NUM_EPOCHS})  






@click.command()
@click.argument('config_file', required=True, type=str)
@click.argument('dir_config_file', required=True, type=str)
def train_cli(config_file:str, dir_config_file: str) -> None:
    train(config_file, dir_config_file)

if __name__=="__main__":
    train_cli()   