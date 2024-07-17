import React from 'react';
import ImageCarousel from '../components/ImageCarousel';

export default function MNIST() {
    return( 
        <div>
            <div className="flex flex-col pb-8">
                <h1 className="text-xl font-medium">MNIST Model</h1>
                <p>This model detects handwritten digits</p>
            </div>
            <ImageCarousel />
        </div>
    )
}