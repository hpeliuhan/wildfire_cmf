from FireDetectionNet import *
import yaml
import tensorflow as tf
import click
import os
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import cmflib as cmf
import json
from utils import *

def evaluation(config_file:str, dir_config_file:str):
    #input_dir1: test data
    #input_dir2: model data
    #output: test result
    params=yaml.safe_load(open(config_file))["evaluation"]


    dir_config_file=yaml.safe_load(open(dir_config_file))["dir_config"]["evaluation"]
    input_dir1=dir_config_file["input1"]
    input_dir2=dir_config_file["input2"]
    output_dir=dir_config_file["output"]
    os.makedirs(output_dir, exist_ok=True)

    testX_path=os.path.join(input_dir1, "val_images.npy")
    testY_path=os.path.join(input_dir1, "val_labels.npy")
    model_path=os.path.join(input_dir2, "trained_model.keras")
    history_path=os.path.join(input_dir2, "history.npy")
    modelCheckpointFile=os.path.join(input_dir2, "best_model.h5")

    testX= np.load(testX_path)
    testY= np.load(testY_path)
    model=tf.keras.models.load_model(model_path)

    BATCH_SIZE=params["batch_size"]
    CLASSES=params["CLASSES"]
    NUM_EPOCHS=params["epochs"]
    print("[INFO] evaluating network...")
    predictions = model.predict(testX, batch_size=BATCH_SIZE)

    print(classification_report(testY.argmax(axis=1),
        predictions.argmax(axis=1), target_names=CLASSES))
    history_path=os.path.join(input_dir2, "history.npy")
    H=np.load(history_path,allow_pickle='TRUE').item()


    TRAINING_PLOT_PATH=os.path.join(output_dir, "training_plot.png")

    N = np.arange(0, NUM_EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H["loss"], label="train_loss")
    plt.plot(N, H["val_loss"], label="val_loss")
    plt.plot(N, H["accuracy"], label="train_acc")
    plt.plot(N, H["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(TRAINING_PLOT_PATH)

    model.load_weights(modelCheckpointFile)
    loss,acc=model.evaluate(testX,testY)
    print("Test Accuracy: ",str(acc))

    report = classification_report(testY.argmax(axis=1),
                               predictions.argmax(axis=1), 
                               target_names=CLASSES, 
                               output_dict=True) 

    # Save the report to a JSON file
    report_path = os.path.sep.join([output_dir, "classification_report.json"]) 
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4) 

    # Log individual metrics (optional)
    metrics = {
        'accuracy': report['accuracy'],
        'fire_precision': report['Fire']['precision'],
        # ... other metrics
    }



    metawriter=set_cmf_environment("cmf","WILDFIRE")
    _ = metawriter.create_context(pipeline_stage="model training") 
    _ = metawriter.create_execution(execution_type="evaluation") 
    _ = metawriter.log_dataset(testX_path,"input")
    _ = metawriter.log_dataset(testY_path,"input")
    _ = metawriter.log_model(path=model_path,event="input")
    _ = metawriter.log_model(path=modelCheckpointFile,event="input")
    _ = metawriter.log_dataset(history_path,"input")
    _ = metawriter.log_dataset(report_path, "output", custom_properties={"type": "classification_report"})
    _ = metawriter.log_execution_metrics("evaluation_metrics", metrics)


@click.command()
@click.argument('config_file', required=True, type=str)
@click.argument('dir_config_file', required=True, type=str)
def evaluation_cli(config_file:str, dir_config_file: str) -> None:
    evaluation(config_file, dir_config_file)

if __name__=="__main__":
    evaluation_cli()   