import os
import cv2
import sys
import numpy as np
import tensorflow as tf
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '../training/src'))

def extract_features(frame, feature_extractor, input_size):
    # Preprocess the frame
    frame_resized = cv2.resize(frame, input_size)
    frame_normalized = frame_resized.astype("float32") / 255.0
    frame_expanded = np.expand_dims(frame_normalized, axis=0)

    # Get input and output details
    input_details = feature_extractor.get_input_details()
    output_details = feature_extractor.get_output_details()

    # Set the tensor to the input data
    feature_extractor.set_tensor(input_details[0]['index'], frame_expanded)

    # Perform inference
    feature_extractor.invoke()

    # Get the output tensor (features)
    features = feature_extractor.get_tensor(output_details[0]['index'])
    return features

def run_feature_extraction(config_file: str, dir_config_file: str):
    # Load configuration
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    with open(dir_config_file, 'r') as file:
        dir_config = yaml.safe_load(file)
    
    model_dir = dir_config['dir_config']["inference"]["input"]
    feature_extractor_path = os.path.join(model_dir, "feature_extractor.tflite")
    output_dir = dir_config['dir_config']["inference"]["output"]
    os.makedirs(output_dir, exist_ok=True)

    input_size = tuple(config["inference"]["input_size"])
    video_dir = config["inference"]["video_path"]

    node_id = config["inference"]["node_id"]
    video_path = os.path.join(video_dir, str(node_id) + ".mp4")
    result_path = os.path.join(output_dir, str(node_id))
    os.makedirs(result_path, exist_ok=True)

    # Load the feature extractor model
    feature_extractor = tf.lite.Interpreter(model_path=feature_extractor_path)
    feature_extractor.allocate_tensors()

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_count = 0
    features_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract features using the feature extractor
        features = extract_features(frame, feature_extractor, input_size)
        features_list.append(features)

        # Save the frame data as .npy
        frame_npy_filename = os.path.join(result_path, f"frame_{frame_count:04d}.npy")
        np.save(frame_npy_filename, frame)

        frame_count += 1

    # Save all extracted features
    features_npy_filename = os.path.join(result_path, "features.npy")
    np.save(features_npy_filename, features_list)
    print(f"Features saved to {features_npy_filename}")

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def extract_latent_vectors(image_dir, feature_extractor_path, output_dir, input_size):
    # Load the feature extractor model
    feature_extractor = tf.lite.Interpreter(model_path=feature_extractor_path)
    feature_extractor.allocate_tensors()

    # Get input and output details
    input_details = feature_extractor.get_input_details()
    output_details = feature_extractor.get_output_details()

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    _ = metawriter.create_context(pipeline_stage="inference")
    _ = metawriter.create_execution(execution_type="inferencing")
    _ = metawriter.log_model(path=model_path, event="input")
    # Process each image in the directory
    for image_name in os.listdir(image_dir):
        if image_name.endswith(".jpg"):
            image_path = os.path.join(image_dir, image_name)

            # Load and preprocess the image
            image = cv2.imread(image_path)
            image_resized = cv2.resize(image, input_size)
            image_normalized = image_resized.astype("float32") / 255.0
            image_expanded = np.expand_dims(image_normalized, axis=0)

            # Set the input tensor
            feature_extractor.set_tensor(input_details[0]['index'], image_expanded)

            # Perform inference
            feature_extractor.invoke()

            # Get the latent vector (output tensor)
            latent_vector = feature_extractor.get_tensor(output_details[0]['index'])

            # Save the latent vector
            latent_vector_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_latent.npy")
            np.save(latent_vector_path, latent_vector)

            print(f"Saved latent vector for {image_name} to {latent_vector_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run feature extraction on a video or extract latent vectors from images.")
    parser.add_argument("config_file", type=str, help="Path to the configuration file.")
    parser.add_argument("dir_config_file", type=str, help="Path to the directory configuration file.")
    parser.add_argument("--image_dir", type=str, help="Directory containing .jpg images for latent vector extraction.")
    parser.add_argument("--output_dir", type=str, help="Directory to save latent vectors.")
    args = parser.parse_args()

    run_feature_extraction(args.config_file, args.dir_config_file)

    if args.image_dir and args.output_dir:
        # Example usage for latent vector extraction
        feature_extractor_path = os.path.join(args.dir_config_file, "feature_extractor.tflite")
        input_size = (224, 224)  # Input size expected by the feature extractor

        extract_latent_vectors(args.image_dir, feature_extractor_path, args.output_dir, input_size)