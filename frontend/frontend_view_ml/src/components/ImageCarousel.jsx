import React, { useState } from 'react';
import axios from 'axios';

const images = [
    'assets/MNIST/zero.png',
    'assets/MNIST/one.png',
    'assets/MNIST/two.png',
    'assets/MNIST/three.png',
    'assets/MNIST/four.png',
    'assets/MNIST/five.png',
    'assets/MNIST/six.png',
    'assets/MNIST/seven.png',
    'assets/MNIST/eight.png',
    'assets/MNIST/nine.png',

];

const CustomCarousel = () => {
    const [currentIndex, setCurrentIndex] = useState(0);
    const [prediction, setPrediction] = useState(null);

    const prevSlide = () => {
        setCurrentIndex((prevIndex) => (prevIndex - 1 + images.length) % images.length);
    };

    const nextSlide = () => {
        setCurrentIndex((prevIndex) => (prevIndex + 1) % images.length);
    };

    const preprocessImage = async (imageSrc) => {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.src = imageSrc;
            img.crossOrigin = "Anonymous";
            img.onload = () => {
                const canvas = document.createElement('canvas');
                canvas.width = 28;
                canvas.height = 28;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, 28, 28);
                const imageData = ctx.getImageData(0, 0, 28, 28);
                const pixels = [];
                for (let i = 0; i < imageData.data.length; i += 4) {
                    const grayscaleValue = imageData.data[i] * 0.299 + imageData.data[i + 1] * 0.587 + imageData.data[i + 2] * 0.114;
                    pixels.push(grayscaleValue / 255.0);
                }
                resolve(pixels);
            };
            img.onerror = reject;
        });
    };

    const onSubmit = async () => {
        try {
            const pixels = await preprocessImage(images[currentIndex]);
            console.log('Preprocessed Data:', pixels.slice(0, 10)); // Log first 10 preprocessed values
            const response = await axios.post('http://localhost:8000/api/predict/mnist/', { input: pixels });
            console.log('Response:', response.data);
            setPrediction(response.data.prediction);
            console.log('Raw output:', response.data.raw_output);
        } catch (error) {
            console.error('There was an error making the request:', error);
        }
    };

    return (
        <div className="flex flex-col">
            <button onClick={onSubmit}>
                <img src={images[currentIndex]} className="w-full h-full cursor-pointer border-4 hover:border-red-400 border-red-200 rounded-md transition-all shadow-lg" />
            </button>
            <div className="flex justify-between mt-4">
                <button onClick={prevSlide} className="px-4 py-2 bg-red-50 hover:bg-red-200 hover:shadow-md transition-all rounded-lg shadow-sm">Previous</button>
                <button onClick={nextSlide} className="px-4 py-2 bg-red-50 hover:bg-red-200 hover:shadow-md transition-all rounded-lg shadow-sm">Next</button>
            </div>
            <p>
                {prediction !== null ? `The model predicted: ${prediction}` : 'Make a prediction by clicking the image'}
            </p>
        </div>
    );
};

export default CustomCarousel;
