import logging
import pickle
import joblib
import numpy as np
import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from sklearn.preprocessing import MinMaxScaler
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import torch
import torch.nn as nn
import torch.nn.functional as F

from django.http import JsonResponse
from rest_framework.views import APIView
import joblib
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class AAPL_LSTM_MODEL(APIView):
    def __init__(self):
        super().__init__()
        try:
            with open('./models/AAPL_lstm_model.pkl', 'rb') as f:
                loaded_data = pickle.load(f)
            
            self.input_size = loaded_data['input_size']
            self.hidden_size = loaded_data['hidden_size']
            self.num_layers = loaded_data['num_layers']
            self.output_size = loaded_data['output_size']
            self.dropout_rate = loaded_data['dropout_rate']
            
            self.model = LSTMModel(self.input_size, self.hidden_size, self.num_layers, self.output_size, self.dropout_rate)
            self.model.load_state_dict(loaded_data['model_state_dict'])
            self.model.eval()  # Set the model to evaluation mode
            
            # Load scaler and sequence_length
            self.scaler = loaded_data.get('scaler')
            self.sequence_length = loaded_data.get('sequence_length')
            
            if self.scaler is None:
                print("Warning: Scaler is missing. Using default MinMaxScaler.")
                self.scaler = MinMaxScaler(feature_range=(0,1))

            print(f"Scaler initialized: {self.scaler}")
        except FileNotFoundError as e:
            raise FileNotFoundError("The model file was not found. Ensure the file path is correct.") from e
        except Exception as e:
            raise ValueError(f"Error loading the model: {str(e)}") from e

    def preprocess_data(self, data):
        print(f"Preprocessing data: {data}")  # Debugging line
        if not data:
            raise ValueError("Input data is empty.")
        try:
            data = np.array(data, dtype=float).reshape(-1, 1)
        except ValueError:
            raise ValueError("All input values must be numeric.")
        
        print(f"Data shape after reshape: {data.shape}")  # Debugging line
        
        if data.shape[0] == 0:
            raise ValueError("Reshaped data is empty.")
        
        data = self.scaler.fit_transform(data)
        print(f"Data shape after scaling: {data.shape}")  # Debugging line
        
        if len(data) < self.sequence_length:
            raise ValueError(f"Input data must have at least {self.sequence_length} points. Received {len(data)}.")
        sequence = data[-self.sequence_length:]
        return torch.FloatTensor(sequence).unsqueeze(0)

    def post(self, request, *args, **kwargs):
        try:
            input_data = request.data.get('input')
            print(f"Received input data: {input_data}")  # Debugging line
            if input_data is None:
                return JsonResponse({'error': "No 'input' key found in request data."}, status=400)
            if not isinstance(input_data, list):
                return JsonResponse({'error': "Input data must be a list of numbers."}, status=400)
            if len(input_data) < self.sequence_length:
                return JsonResponse({'error': f"Input data must have at least {self.sequence_length} points. Received {len(input_data)}."}, status=400)
            
            processed_data = self.preprocess_data(input_data)
            with torch.no_grad():
                predictions = self.model(processed_data)
            predictions = self.scaler.inverse_transform(predictions.numpy())
            return JsonResponse({'prediction': predictions[0][0].item()})
        except ValueError as e:
            return JsonResponse({'error': f"ValueError: {str(e)}"}, status=400)
        except Exception as e:
            return JsonResponse({'error': f"An error occurred during the prediction: {str(e)}"}, status=500)


def post(self, request):
    try:
        data = request.data.get('input')
        if not data:
            return Response({'error': 'No input data provided'}, status=status.HTTP_400_BAD_REQUEST)
        
        preprocessed_data = self.preprocess_data(data)
        prediction = self.model.predict(preprocessed_data)
        prediction = self.scaler.inverse_transform(prediction)
        return Response({'prediction': prediction[0][0]}, status=status.HTTP_200_OK)
    except ValueError as ve:
        return Response({'error': str(ve)}, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")  # Log the error
        return Response({'error': 'An unexpected error occurred'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
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

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

import logging

# Ensure logging is configured to output to console
logging.basicConfig(level=logging.INFO)

class MNIST(APIView):
    def __init__(self):
        super().__init__()
        try:
            self.model = CNN()
            self.model.load_state_dict(torch.load('./models/mnist_cnn.pt'))
            self.model.eval()  # Set model to evaluation mode
            logging.info("Model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise e

    def preprocess_data(self, data):
        try:
            if len(data) != 784:
                raise ValueError("Input data must be a list of 784 elements")
            image = np.array(data, dtype=np.float32).reshape(1, 1, 28, 28)
            # Remove the line: image = image / 255.0
            return torch.tensor(image)
        except Exception as e:
            logging.error(f"Error in preprocessing data: {str(e)}")
            raise e

    def post(self, request):
        try:
            data = request.data.get('input', None)
            if data is None:
                raise ValueError("No input data provided")

            logging.info(f"Received data: {data[:10]}...")

            preprocessed_data = self.preprocess_data(data)
            logging.info(f"Preprocessed data shape: {preprocessed_data.shape}")
            logging.info(f"Preprocessed data sample: {preprocessed_data[0,0,0,:10]}")  # Log first 10 values of preprocessed data

            with torch.no_grad():
                prediction = self.model(preprocessed_data)
                raw_output = prediction.exp().tolist()[0]  # Convert from log_softmax to probabilities
                predicted_class = torch.argmax(prediction).item()

            logging.info(f"Raw output: {raw_output}")
            logging.info(f"Prediction: {predicted_class}")

            return Response({'prediction': predicted_class, 'raw_output': raw_output}, status=status.HTTP_200_OK)
        except ValueError as ve:
            logging.error(f"ValueError: {str(ve)}")
            return Response({'error': str(ve)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logging.error(f"General Exception: {str(e)}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



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