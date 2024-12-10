from tensorflow import lite
import click
from cmflib import cmf
import os
from utils import *
import yaml
import tensorflow as tf

def model_convert(dir_config_file:str):
    dir_config_file=yaml.safe_load(open(dir_config_file))["dir_config"]["model_convert"]
    input_dir=dir_config_file["input"]
    output_dir=dir_config_file["output"]
    os.makedirs(output_dir, exist_ok=True)

    model_path=os.path.join(input_dir, "trained_model.keras")
    modelCheckpointFile=os.path.join(input_dir, "best_model.h5")
    
    
    model = tf.keras.models.load_model(model_path)
    converter = lite.TFLiteConverter.from_keras_model(model)
    model.load_weights(modelCheckpointFile)
    tflite_model = converter.convert()

    tfLiteModelFile = os.path.join(output_dir, "model.tflite")
    open(tfLiteModelFile, "wb").write(tflite_model)

    metawriter = set_cmf_environment("cmf","WILDFIRE")
    _ = metawriter.create_context(pipeline_stage="model_convert") 
    _ = metawriter.create_execution(execution_type="model_convert") 
    _ = metawriter.log_model(
    path=modelCheckpointFile,event="input"
    )
    _ = metawriter.log_model(
    path=tfLiteModelFile, event="output" 
)


@click.command()
@click.argument('dir_config_file', required=True, type=str)  
def model_convert_cli(dir_config_file:str) -> None:
    model_convert(dir_config_file)

if __name__ == '__main__':
    model_convert_cli()
