from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# Load the model once when the server starts
model = tf.keras.models.load_model('../mnist_resnet/model.keras') if os.path.exists('../mnist_resnet/model.keras') else None

class PredictView(APIView):
    def post(self, request):
        if not model:
            return Response({'error': 'Model not found. Please train the model first.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        file = request.FILES['image']
        file_name = default_storage.save(file.name, ContentFile(file.read()))
        file_path = default_storage.path(file_name)
        
        # Load and preprocess the image
        image = Image.open(file_path).convert('L')
        image = image.resize((28, 28))
        image = np.array(image) / 255.0
        image = image.reshape(1, 28, 28, 1)

        # Make prediction
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Clean up the saved file
        default_storage.delete(file_name)
        classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        return Response({'prediction': classes[int(predicted_class)], 'probability': float(prediction[0][predicted_class])}, status=status.HTTP_200_OK)

class GuideView(APIView):
    def get(self, request):
        guide = {
            "message": "To use the prediction API, send a POST request to /api/predict/ with an image file.",
            "example_request": {
                "url": "/api/predict/",
                "method": "POST",
                "headers": {
                    "Content-Type": "multipart/form-data"
                },
                "body": {
                    "image": "Upload an image file here"
                }
            },
            "example_response": {
                "prediction": "T-shirt/top",
                "probability": 0.98
            }
        }
        return Response(guide, status=status.HTTP_200_OK)
