import React from 'react';
import MyDropzone from '../components/ImageDropzone';

export default function Sentiment() {
    return(
        <div>
            <div className="flex flex-col pb-8">
                <h1 className="text-xl font-medium">Sentiment Model</h1>
                <p>This model detects whether the image is happy or sad</p>
            </div>
            <MyDropzone />
        </div> 
    )
}