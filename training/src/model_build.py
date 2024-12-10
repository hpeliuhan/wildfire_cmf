from FireDetectionNet import *
import yaml
from cmflib import cmf
import os
import click
import tensorflow as tf
from utils import *

def create_model(config_file: str,  dir_config_file: str):
    params = yaml.safe_load(open(config_file))["model_build"]
    Width = params["Width"]
    Height = params["Height"]
    Depth = params["Depth"]
    Classes = params["Classes"]

    INIT_LR = params["init_lr"]
    NUM_EPOCHS = params["num_epochs"]
    MOMENTUM = params["momentum"]
    LOSS= params["loss"]
    METRICS= params["metrics"]

    dir_config_file=yaml.safe_load(open(dir_config_file))["dir_config"]
    output_path=dir_config_file["model_build"]["output"] 
    os.makedirs(output_path, exist_ok=True)

    model=FireDetectionNet.build(width=Width, height=Height, depth=Depth, classes=Classes) 
    model_dict_path=os.path.join(output_path, "init_model_dict.npy")
    

    FireDetectionNet.save_model_dict(model, filename=model_dict_path)

    opt= tf.keras.optimizers.legacy.SGD(learning_rate=INIT_LR, momentum= MOMENTUM,decay=INIT_LR / NUM_EPOCHS)
    model.compile(loss=LOSS, optimizer=opt, metrics = METRICS)
    
    model_path=os.path.join(output_path, "init_model.keras")
    model.save(model_path)

    metawriter = set_cmf_environment("cmf", "WILDFIRE")

    _ = metawriter.create_context(pipeline_stage="model training")
    _ = metawriter.create_execution(execution_type="model_build", custom_properties=params)
    _ = metawriter.log_dataset(model_dict_path,event="output",custom_properties={"model_dict":model_dict_path})
    _ = metawriter.log_model(
        path=model_path,event="output",model_framework="tensorflow",model_type="CNN", custom_properties={"model_dict":model_dict_path})


@click.command()
@click.argument('config_file', required=True, type=str)
@click.argument('dir_config_file', required=True, type=str)
def create_model_cli(config_file: str, dir_config_file: str) -> None:
    create_model(config_file, dir_config_file)


if __name__=="__main__":
    create_model_cli()
