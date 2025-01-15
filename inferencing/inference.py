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
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime
from minio import Minio
from minio.error import S3Error

def upload_to_minio(file_path, bucket_name, object_name, minio_client):
    try:
        # Make the bucket if it doesn't exist
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
        
        # Upload the file
        minio_client.fput_object(bucket_name, object_name, file_path)
        print(f"File {file_path} uploaded successfully to bucket {bucket_name} with object name {object_name}")
        return bucket_name
    except S3Error as e:
        print(f"Error: Unable to upload file to MinIO")
        print(e)
        return None

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
    url = "http://192.168.30.115:8086"
    token = "R0F_eqrU8jAWam5_DwAdZ2GSngmb1coFNQjjVedPGrdurSKXNnMhSzFkIORupl8QxmkMLlrD9Aq0B-U9ncUXLQ=="
    org = "waggle"
    bucket = "waggle"

    # Initialize the InfluxDB client
    client = InfluxDBClient(url=url, token=token, org=org)
    write_api = client.write_api(write_options=SYNCHRONOUS)

    #minio setting
    minio_client = Minio(
        "192.168.30.115:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=False
    )
        
    bucket_name = "waggle"
  


    # Load configuration
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    print(config)
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
    #predictions_list = []
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
        if label == "Fire":
            frame_name=f"frame_{frame_count:04d}.jpg"
            print(frame_name)
            frame_filename = os.path.join(result_path, frame_name)
            cv2.imwrite(frame_filename, frame)
            point = Point("image_metadata") \
                .tag("node_id", node_id) \
                .field("image_path", frame_name) \
                .time(datetime.utcnow(), WritePrecision.NS)
            write_api.write(bucket=bucket, org=org, record=point)
            frame_count += 1
    #predictions_npy_filename = os.path.join(result_path, "predictions.npy")
    #print(predictions_list)
    #np.save(predictions_npy_filename, predictions_list)

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

 
    for image in os.listdir(result_path):
        if image.endswith(".jpg"):
            image_path=os.path.join(result_path, image)
            object_name = image
            #upload to minio
            upload_to_minio(image_path, bucket_name, object_name, minio_client)
                        
            #np.save(image_numpy_path, cv2.imread(os.path.join(result_path, image)))






@click.command()
@click.argument('config_file', required=True, type=str)
@click.argument('dir_config_file', required=True, type=str)
def inference_cli(config_file:str, dir_config_file: str) -> None:
    inference_video(config_file, dir_config_file)


if __name__ == '__main__':
    inference_cli()