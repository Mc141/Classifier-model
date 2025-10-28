"""
Test script for the Firethorn API
Run this after the API is running to test the endpoints
"""

import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_single_prediction(image_path):
    """Test single image prediction"""
    print(f"Testing prediction for {image_path}...")
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    with open(image_path, "rb") as f:
        files = {"file": f}
        response = requests.post(f"{API_URL}/predict", files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Prediction successful")
        print(f"Predicted class: {result['predicted_class']}")
        print(f"Firethorn confidence: {result['confidence']['firethorn_percent']:.2f}%")
        print(f"Not-Firethorn confidence: {result['confidence']['not_firethorn_percent']:.2f}%")
        print()
        return result
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"Response: {response.text}")
        print()
        return None

def test_batch_prediction(image_paths):
    """Test batch prediction"""
    print(f"Testing batch prediction for {len(image_paths)} images...")
    
    files = []
    for path in image_paths:
        if os.path.exists(path):
            files.append(("files", open(path, "rb")))
    
    response = requests.post(f"{API_URL}/predict/batch", files=files)
    
    # Close files
    for _, f in files:
        f.close()
    
    if response.status_code == 200:
        results = response.json()["predictions"]
        print(f"✓ Batch prediction successful for {len(results)} images")
        for result in results:
            print(f"  {result.get('filename', 'unknown')}: {result.get('predicted_class', 'N/A')}")
        print()
        return results
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"Response: {response.text}")
        print()
        return None

if __name__ == "__main__":
    print("="*60)
    print("FIREthorn API Test Script")
    print("="*60)
    print()
    
    # Test health check
    try:
        test_health_check()
    except Exception as e:
        print(f"Error connecting to API: {e}")
        print("Make sure the API is running!")
        exit(1)
    
    # Test single prediction (replace with actual test images from your dataset)
    # Uncomment and adjust path when you have a trained model
    # test_image = "../test/Firethorn/image1.jpg"
    # test_single_prediction(test_image)
    
    print("="*60)
    print("Test completed!")
    print("="*60)

