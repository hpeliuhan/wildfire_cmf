import click
import yaml
import os
import cv2
import onnx
import numpy as np
import sys
import click
import tflite_runtime.interpreter as tflite
import yaml
sys.path.append(os.path.join(os.path.dirname(__file__), '../training/src'))

from utils import set_cmf_environment

def load_model(model_path):
    try:
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print(f"Model loaded successfully from {model_path}")
        return interpreter
    except Exception as e:
        print(f"Error: Unable to load model from {model_path}")
        print(e)
        return None

def process_frame(frame, interpreter, input_size):
    # Preprocess the frame
    frame_resized = cv2.resize(frame, input_size)
    frame_normalized = frame_resized.astype("float32") / 255.0
    frame_expanded = np.expand_dims(frame_normalized, axis=0)

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the tensor to the input data
    interpreter.set_tensor(input_details[0]['index'], frame_expanded)

    # Perform inference
    interpreter.invoke()

    # Get the output tensor
    predictions = interpreter.get_tensor(output_details[0]['index'])

    return predictions

def inference_video(config_file:str,dir_config_file:str):
    # Load configuration
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    with open(dir_config_file, 'r') as file:
        dir_config = yaml.safe_load(file)
    model_dir = dir_config['dir_config']["inference"]["input"]
    model_name = config["inference"]["model_name"]
    model_path = os.path.join(model_dir, model_name)
    output_dir = dir_config['dir_config']["inference"]["output"]
    os.makedirs(output_dir, exist_ok=True)

    input_size = tuple(config["inference"]["input_size"])
    class_labels = config["inference"]["class_labels"]
    video_dir = config["inference"]["video_path"]

    node_id = config["inference"]["node_id"]
    video_path = os.path.join(video_dir, str(node_id)+".mp4")
    result_path=os.path.join(output_dir,str(node_id))
    os.makedirs(result_path, exist_ok=True)
    # Load the trained model
    interpreter = load_model(model_path)
    if interpreter is None:
        return
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return


    frame_count = 0
    predictions_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference on the frame
        predictions = process_frame(frame, interpreter, input_size)
        predicted_class = np.argmax(predictions, axis=1)[0]
        label = class_labels[predicted_class]

        # Draw the label on the frame
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Save the frame as .jpg
        frame_filename = os.path.join(result_path, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

        # Save the frame data as .npy
        frame_npy_filename = os.path.join(result_path, f"frame_{frame_count:04d}.npy")
        np.save(frame_npy_filename, frame)
        print(frame_npy_filename,"has",predictions)
        # Append predictions to the list
        predictions_list.append(predictions)

        frame_count += 1
    predictions_npy_filename = os.path.join(result_path, "predictions.npy")
    #print(predictions_list)
    np.save(predictions_npy_filename, predictions_list)

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    metawriter = set_cmf_environment("cmf","WILDFIRE")
    _ = metawriter.create_context(pipeline_stage="inference") 
    _ = metawriter.create_execution(execution_type="inferencing") 
   # _ = metawriter.log_model(    path=modelCheckpointFile,event="input")
    _ = metawriter.log_model(
    path=model_path, event="input" 
    )
    for image in os.listdir(result_path):
        if image.endswith(".jpg"):
            image_numpy_path=os.path.join(result_path, image.replace(".jpg", ".npy"))
            np.save(image_numpy_path, cv2.imread(os.path.join(result_path, image)))
            _ = metawriter.log_dataset(image_numpy_path, event="output")
    _ = metawriter.log_dataset(predictions_npy_filename, event="output")





@click.command()
@click.argument('config_file', required=True, type=str)
@click.argument('dir_config_file', required=True, type=str)
def inference_cli(config_file:str, dir_config_file: str) -> None:
    inference_video(config_file, dir_config_file)


if __name__ == '__main__':
    inference_cli()