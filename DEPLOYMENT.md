# Deployment Guide for Render

This API service is ready to deploy on Render.

## Quick Deploy Steps

### 1. Push to GitHub

```bash
cd PRJ_project/api_service
git init
git add .
git commit -m "Initial commit: Firethorn API"
git remote add origin <your-github-repo-url>
git push -u origin main
```

### 2. Deploy on Render

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Render will auto-detect `render.yaml`
5. Confirm settings and click "Apply"

## Files Structure

```
api_service/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── Dockerfile          # Container configuration
├── render.yaml         # Render deployment config
├── models/             # Trained models (copied from training)
│   ├── best_model.keras
│   └── final_model.keras
├── README.md           # API documentation
├── test_api.py         # Testing script
└── .dockerignore       # Files to exclude from Docker
```

## Model Loading

The API will:

1. Load `models/best_model.keras` on startup
2. If model file not found, create a placeholder model (for testing)
3. Use EfficientNet preprocessing matching training

## Environment Variables

- `MODEL_PATH`: Path to model (default: `models/best_model.keras`)
- `PORT`: Server port (set by Render)

## Testing Locally

### With Docker:

```bash
docker build -t firethorn-api .
docker run -p 8000:8000 firethorn-api
```

### Without Docker:

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

### Test the API:

```bash
curl http://localhost:8000/health
```

## API Endpoints

- `GET /` - Basic health check
- `GET /health` - Detailed health check with model status
- `POST /predict` - Classify single image (returns percentage)
- `POST /predict/batch` - Classify multiple images

## Response Format

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

## Troubleshooting

### Model Not Found

If you see "Model file not found", ensure:

- Models are copied to `api_service/models/` directory
- Model files are committed to Git
- File paths in code match actual locations

### Memory Issues on Render Free Tier

If you hit memory limits:

- Use `best_model.keras` instead of `final_model.keras`
- Reduce batch processing
- Consider upgrading Render plan

## Notes

- The model uses EfficientNetB2 with input size 260x260
- Default classification threshold is 0.5 (adjustable)
- API uses same preprocessing as training (EfficientNet preprocess_input)
