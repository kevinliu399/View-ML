import requests

sample_data = [0.0] * 784
response = requests.post('http://localhost:8000/api/predict/mnist/', json={'input': sample_data})
print(response.json())
