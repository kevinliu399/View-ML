import logging
import joblib
import numpy as np
import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

class AAPL_LSTM_MODEL(APIView):
    def __init__(self):
        super().__init__()
        loaded_data = joblib.load('./models/AAPL_lstm_model.pkl')
        self.model = loaded_data['model']
        self.scaler = loaded_data['scaler']
        self.sequence_length = loaded_data['sequence_length']

    def preprocess_data(self, data):
        # Assuming data is a list of stock prices
        data = np.array(data).reshape(-1, 1)
        data = self.scaler.transform(data)
        # Prepare the sequence
        if len(data) < self.sequence_length:
            raise ValueError(f"Input data must have at least {self.sequence_length} points")
        sequence = data[-self.sequence_length:]
        return np.expand_dims(sequence, axis=0)  # Add batch dimension

    def post(self, request):
        try:
            data = request.data['input']
            preprocessed_data = self.preprocess_data(data)
            prediction = self.model.predict(preprocessed_data)
            # Inverse transform the prediction
            prediction = self.scaler.inverse_transform(prediction)
            return Response({'prediction': prediction[0][0]}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        
class Housing_Prediction(APIView):
    def __init__(self):
        super().__init__()
        loaded_data = joblib.load('./models/housing_prediction.pkl')
        self.model = loaded_data['model']
        self.feature_names = loaded_data['feature_names']
        self.ocean_proximity_categories = loaded_data['ocean_proximity_categories']

    def preprocess_data(self, data):
        df = pd.DataFrame(data, index=[0])
        for col in ['total_rooms', 'total_bedrooms', 'population', 'households']:
            df[col] = np.log(df[col]) + 1
        ocean_proximity_encoded = pd.get_dummies(df.ocean_proximity, prefix='ocean_proximity')
        for category in self.ocean_proximity_categories:
            if f'ocean_proximity_{category}' not in ocean_proximity_encoded.columns:
                ocean_proximity_encoded[f'ocean_proximity_{category}'] = 0
        df = df.join(ocean_proximity_encoded).drop(["ocean_proximity"], axis=1)
        df["bedroom_ratio"] = df["total_bedrooms"] / df["total_rooms"]
        df["household_rooms"] = df["total_rooms"] / df["households"]
        df = df.reindex(columns=self.feature_names, fill_value=0)
        return df

    def post(self, request):
        try:
            data = request.data
            preprocessed_data = self.preprocess_data(data)
            prediction = self.model.predict(preprocessed_data)
            return Response({'prediction': prediction[0]}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)


class MNIST(APIView):
    def __init__(self):
        super().__init__()
        self.model = joblib.load('./models/MNIST.pkl')

    def preprocess_data(self, data):
        # Ensure data is a list of 784 pixel values (28x28 image flattened)
        if len(data) != 784:
            raise ValueError("Input data must be a list of 784 elements")
        image = np.array(data, dtype=np.float32).reshape(1, 28, 28, 1)
        image = image / 255.0  # Normalize pixel values
        return image

    def post(self, request):
        try:
            data = request.data.get('input', None)
            if data is None:
                raise ValueError("No input data provided")

            logging.info(f"Received data: {data[:10]}...") 

            preprocessed_data = self.preprocess_data(data)
            logging.info(f"Preprocessed data shape: {preprocessed_data.shape}")

            prediction = self.model.predict(preprocessed_data)
            predicted_class = np.argmax(prediction)

            return Response({'prediction': int(predicted_class)}, status=status.HTTP_200_OK)
        except ValueError as ve:
            logging.error(f"ValueError: {str(ve)}")
            return Response({'error': str(ve)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logging.error(f"General Exception: {str(e)}")
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

class Temperature_Model(APIView):
    def __init__(self):
        super().__init__()
        loaded_data = joblib.load('./models/temperature_model.pkl')
        self.model = loaded_data['model']
        self.scaler = loaded_data['scaler']
        self.feature_names = loaded_data['feature_names']
        self.label_encoders = loaded_data['label_encoders']

    def preprocess_data(self, data):
        df = pd.DataFrame(data, index=[0])
        for col, le in self.label_encoders.items():
            df[f'{col}_encoded'] = le.transform(df[col])
        df = df[self.feature_names]
        df_scaled = self.scaler.transform(df)
        return df_scaled

    def post(self, request):
        try:
            data = request.data
            preprocessed_data = self.preprocess_data(data)
            prediction = self.model.predict(preprocessed_data)
            return Response({'prediction': prediction[0]}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)



class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(16 * 32 * 32, 256)
        self.fc2 = torch.nn.Linear(256, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

class Sentiment_classifier(APIView):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ConvNet().to(self.device)
        self.model.load_state_dict(torch.load('./models/sentiment_classifier.pth', map_location=self.device))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def preprocess_data(self, image_data):
        image = Image.open(io.BytesIO(image_data))
        image = self.transform(image).unsqueeze(0).to(self.device)
        return image

    def post(self, request):
        try:
            image_data = request.FILES['image'].read()
            preprocessed_data = self.preprocess_data(image_data)
            with torch.no_grad():
                prediction = self.model(preprocessed_data)
            sentiment = "Sad" if prediction.item() > 0.5 else "Happy"
            return Response({'prediction': sentiment, 'confidence': prediction.item()}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)