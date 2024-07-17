import React, { useState } from 'react';
import axios from 'axios';

const images = [
    'assets/MNIST/zero.jpg',
    'assets/MNIST/one.jpg',
    'assets/MNIST/two.jpg',
    'assets/MNIST/three.jpg',
    'assets/MNIST/four.jpg',
    'assets/MNIST/five.jpg',
    'assets/MNIST/six.jpg',
    'assets/MNIST/seven.jpg',
    'assets/MNIST/eight.jpg',
    'assets/MNIST/nine.jpg'
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
    
    const onSubmit = async () => {
        try {
            // Load the image
            const img = new Image();
            img.src = images[currentIndex];
            await new Promise((resolve) => (img.onload = resolve));
    
            // Create a canvas to draw the image
            const canvas = document.createElement('canvas');
            canvas.width = 28;
            canvas.height = 28;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0, 28, 28);
    
            // Get image data
            const imageData = ctx.getImageData(0, 0, 28, 28);
            const pixelData = Array.from(imageData.data)
                .filter((_, i) => i % 4 === 0)  // Take only the first channel (assuming grayscale)
                .map(p => p / 255);  // Normalize to [0, 1]
    
            console.log('Sending Pixel Data:', pixelData);
    
            const response = await axios.post('http://localhost:8000/api/predict/mnist/', { input: pixelData });
            console.log('Response:', response.data);
            setPrediction(response.data.prediction);
        } catch (error) {
            console.error('There was an error making the request:', error);
        }
    }
    

    return (
        <div className="flex flex-col">
            <button onClick={onSubmit}>
                <img src={images[currentIndex]} className="w-full h-full cursor-pointer border-2 border-black rounded-md transition-all" />
            </button>
            <div className="flex justify-between mt-4">
                <button onClick={prevSlide} className="px-4 py-2 bg-red-50 hover:bg-red-200 hover:shadow-md transition-all rounded-lg shadow-sm">Previous</button>
                <button onClick={nextSlide} className="px-4 py-2 bg-red-50 hover:bg-red-200 hover:shadow-md transition-all rounded-lg shadow-sm">Next</button>
            </div>
            <p> 
                {prediction !== null ? `The model predicted: ${prediction}` : 'Make a prediction'}
            </p>
        </div>
    );
};

export default CustomCarousel;