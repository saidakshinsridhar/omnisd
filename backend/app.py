import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# --- 1. Initialize the Flask App ---
app = Flask(__name__)
CORS(app)

# --- 2. Load the Trained Model ---
# This is done only once when the server starts
print("Loading model, this may take a moment...")
MODEL_PATH = 'models/trash_classifier.h5'
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# Define the class names based on the training folders
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# --- 3. Create the Prediction Function ---
def predict_image(img_path):
    """Loads an image, preprocesses it, and returns the predicted class."""
    # Load the image and resize it to the model's expected input size
    img = image.load_img(img_path, target_size=(224, 224))
    
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    
    # Expand dimensions to create a batch of 1
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Normalize the image data (same as in training)
    img_preprocessed = img_batch / 255.
    
    # Make the prediction
    prediction = model.predict(img_preprocessed)
    
    # Get the class with the highest probability
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    
    return predicted_class, float(confidence)

# --- 4. Define the API Endpoint ---
@app.route('/predict', methods=['POST'])
def handle_prediction():
    """Handles the file upload and returns the prediction as JSON."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file:
        # Save the file temporarily
        filepath = os.path.join('uploads', file.filename)
        # Create uploads directory if it doesn't exist
        os.makedirs('uploads', exist_ok=True)
        file.save(filepath)
        
        # Get the prediction
        predicted_class, confidence = predict_image(filepath)
        
        # Clean up the saved file
        os.remove(filepath)
        
        # Return the result
        return jsonify({
            'prediction': predicted_class,
            'confidence': f'{confidence:.2%}' # Format as percentage
        })

# --- This part runs the server ---
if __name__ == '__main__':
    app.run(debug=True, port=5001)