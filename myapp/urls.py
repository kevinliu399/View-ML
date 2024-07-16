from django.urls import path
from .views import (
    AAPL_LSTM_MODEL,
    Housing_Prediction,
    MNIST,
    Temperature_Model,
    Sentiment_classifier
)

urlpatterns = [
    path('predict/aapl_lstm/', AAPL_LSTM_MODEL.as_view(), name='aapl_lstm'),
    path('predict/housing/', Housing_Prediction.as_view(), name='housing_prediction'),
    path('predict/mnist/', MNIST.as_view(), name='mnist'),
    path('predict/temperature/', Temperature_Model.as_view(), name='temperature_model'),
    path('predict/sentiment/', Sentiment_classifier.as_view(), name='sentiment_classifier'),
]
