"""
FastAPI service for Firethorn classification
Receives an image, runs it through the model, returns predictions with confidence percentages
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from typing import Dict
import logging
from tensorflow.keras.applications.efficientnet import preprocess_input

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Firethorn Classifier API",
    description="API for classifying Firethorn vs not-Firethorn images",
    version="1.0.0"
)

# Global variable to store the model
model = None

# Configuration
# Using best_model.keras (saved at best validation accuracy: 93.81%)
MODEL_PATH = os.getenv('MODEL_PATH', 'models/best_model.keras')
IMG_SIZE = (260, 260)
THRESHOLD = 0.5  # Adjust based on your needs

@app.on_event("startup")
async def load_model():
    """Load the model when the API starts"""
    global model
    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
    except FileNotFoundError:
        logger.warning(f"Model file not found at {MODEL_PATH}. Using placeholder model.")
        # Create a dummy model for testing without actual training
        model = create_placeholder_model()
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def create_placeholder_model():
    """Create a placeholder model structure for testing"""
    from tensorflow.keras.applications import EfficientNetB2
    from tensorflow.keras.applications.efficientnet import preprocess_input
    
    base_model = EfficientNetB2(
        weights='imagenet',
        include_top=False,
        input_shape=(260, 260, 3)
    )
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    
    # Compile but don't train
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    logger.info("Placeholder model created")
    return model

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess image for EfficientNetB2 using the same preprocessing as training
    
    Args:
        image: PIL Image object
        
    Returns:
        Preprocessed image array ready for model
    """
    # Resize to model input size
    image = image.resize(IMG_SIZE)
    
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Ensure 3 channels (RGB)
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:  # RGBA, take RGB
        img_array = img_array[:, :, :3]
    
    # Convert to float32 for preprocessing
    img_array = img_array.astype(np.float32)
    
    # Expand dimensions for batch: (1, 260, 260, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Use EfficientNet preprocessing (same as training)
    img_array = preprocess_input(img_array)
    
    return img_array

def predict_image(image_array: np.ndarray) -> Dict:
    """
    Run image through the model and return predictions
    
    Args:
        image_array: Preprocessed image array
        
    Returns:
        Dictionary with predictions and confidence percentages
    """
    # Get predictions
    predictions = model.predict(image_array, verbose=0)
    
    # Extract probabilities
    firethorn_prob = float(predictions[0][0])
    not_firethorn_prob = float(predictions[0][1])
    
    # Determine predicted class based on threshold
    predicted_class = "Firethorn" if firethorn_prob >= THRESHOLD else "not_Firethorn"
    
    result = {
        "predicted_class": predicted_class,
        "confidence": {
            "firethorn_percent": round(firethorn_prob * 100, 2),
            "not_firethorn_percent": round(not_firethorn_prob * 100, 2)
        },
        "threshold": THRESHOLD,
        "firethorn_confidence": firethorn_prob
    }
    
    return result

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "Firethorn Classifier API",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict Firethorn vs not-Firethorn from uploaded image
    
    Request:
    - file: Image file (jpg, jpeg, png, webp)
    
    Response:
    - predicted_class: "Firethorn" or "not_Firethorn"
    - confidence: Dictionary with percentages for each class
    - threshold: The threshold used for classification
    """
    # Validate file type
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {allowed_extensions}"
        )
    
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Open image with PIL
        image = Image.open(io.BytesIO(contents))
        
        # Handle RGBA images (convert to RGB)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        logger.info(f"Processing image: {file.filename}, size: {image.size}, mode: {image.mode}")
        
        # Preprocess image
        img_array = preprocess_image(image)
        
        # Get prediction
        result = predict_image(img_array)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Predict multiple images at once
    
    Response:
    - predictions: List of prediction results
    """
    results = []
    
    for file in files:
        try:
            # Read uploaded file
            contents = await file.read()
            
            # Open image with PIL
            image = Image.open(io.BytesIO(contents))
            
            # Handle RGBA images
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            # Preprocess
            img_array = preprocess_image(image)
            
            # Predict
            result = predict_image(img_array)
            result['filename'] = file.filename
            results.append(result)
            
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e)
            })
    
    return JSONResponse(content={"predictions": results})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

