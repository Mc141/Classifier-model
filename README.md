# Firethorn Classifier API

FastAPI service for classifying Firethorn vs not-Firethorn images using EfficientNetB2.

## Setup

### Local Development

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the API:

```bash
uvicorn main:app --reload
```

3. API will be available at `http://localhost:8000`

### With Docker

1. Build the image:

```bash
docker build -t firethorn-api .
```

2. Run the container:

```bash
docker run -p 8000:8000 firethorn-api
```

### Deploy on Render

1. Push this folder to GitHub
2. On Render dashboard, create new Web Service
3. Connect your GitHub repo
4. Render will automatically detect `render.yaml` and deploy

## Endpoints

### GET `/`

Health check endpoint

### GET `/health`

Detailed health check with model status

### POST `/predict`

Predict a single image

**Request:**

- `file`: Image file (jpg, jpeg, png, webp)

**Response:**

```json
{
  "predicted_class": "Firethorn",
  "confidence": {
    "firethorn_percent": 92.5,
    "not_firethorn_percent": 7.5
  },
  "threshold": 0.5,
  "firethorn_confidence": 0.925
}
```

### POST `/predict/batch`

Predict multiple images at once

**Request:**

- `files`: Array of image files

**Response:**

```json
{
  "predictions": [
    {
      "predicted_class": "Firethorn",
      "confidence": {...},
      "filename": "image1.jpg"
    },
    ...
  ]
}
```

## Testing

### Using curl:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/image.jpg"
```

### Using Python:

```python
import requests

url = "http://localhost:8000/predict"
with open("test_image.jpg", "rb") as f:
    response = requests.post(url, files={"file": f})
    print(response.json())
```

## Model Configuration

- **Model**: EfficientNetB2
- **Input Size**: 260x260 pixels
- **Classes**: Firethorn, not_Firethorn
- **Default Threshold**: 0.5 (configurable)

## Environment Variables

- `MODEL_PATH`: Path to the trained model (default: `models/best_model.keras`)
- `PORT`: Server port (set by Render)

## Model Files

The trained models are included in the `models/` directory:

- `best_model.keras` - Model with best validation accuracy
- `final_model.keras` - Final trained model
