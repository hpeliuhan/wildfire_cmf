import tflite_runtime.interpreter as tflite

model_path = "model.tflite"
try:
    interpreter = tflite.Interpreter(model_path=model_path)
    print("Model loaded successfully!")
except ValueError as e:
    print(f"Error loading model: {e}")