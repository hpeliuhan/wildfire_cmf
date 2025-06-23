# Export the feature extraction model
import tensorflow as tf

# Assume 'feature_extractor' is your trained model
feature_extractor.save('artifacts/model_convert/feature_extractor')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model('artifacts/model_convert/feature_extractor')
tflite_model = converter.convert()

# Save the TFLite model
with open('artifacts/model_convert/feature_extractor.tflite', 'wb') as f:
    f.write(tflite_model)