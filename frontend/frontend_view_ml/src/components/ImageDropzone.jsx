import React, { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import axios from 'axios'
import '../styles/ImageDropzone.css'

function MyDropzone() {
    const [image, setImage] = useState(null)
    const [error, setError] = useState(null)
    const [prediction, setPrediction] = useState(null)

    const onDrop = useCallback(acceptedFiles => {
        const file = acceptedFiles[0]
        if (file.type === "image/jpeg" || file.type === "image/png" || file.type === "image/jpg" || file.type === "image/bmp") {
            setImage(URL.createObjectURL(file))
            setError(null)
            predictSentiment(file)
        } else {
            setError("Only JPEG, JPG, BMP, and PNG files are accepted.")
            setImage(null)
        }
    }, [])

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { 'image/jpeg': [], 'image/png': [], 'image/jpg': [], 'image/bmp': [] }
    })

    const predictSentiment = async (file) => {
        const formData = new FormData()
        formData.append('image', file)

        try {
            const response = await axios.post('http://localhost:8000/api/predict/sentiment/', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            })
            setPrediction(response.data)
        } catch (error) {
            console.error("Error predicting sentiment:", error)
            if (error.response) {
                console.error("Data:", error.response.data);
                console.error("Status:", error.response.status);
                console.error("Headers:", error.response.headers);
            } else if (error.request) {
                console.error("Request:", error.request);
            } else {
                console.error("Error:", error.message);
            }
            setError("Error predicting sentiment. Please try again.")
        }
    }

    return (
        <div className="flex flex-row space-x-4">
            <div>
                <div {...getRootProps()} className="dropzone">
                    <input {...getInputProps()} />
                    {
                        isDragActive ?
                            <p>Drop the image here ...</p> :
                            <p>Drag and drop an image here, or click to select an image</p>
                    }
                </div>
                {error && <p className="error">{error}</p>}
                {image && (
                        <div className="my-4 flex items-center justify-center w-full h-1/2">
                            <img src={image} alt="Uploaded" style={{ maxWidth: '300px', maxHeight: '300px' }} className="border-4 border-red-200 rounded-md shadow-lg" />
                        </div>
                    )}
                <div className="p-4">
                    <h2 className="font-medium text-xl pb-2">Prediction:</h2>
                    <div className="flex flex-col">
                        <p>
                            {prediction ? `Sentiment: ${prediction.prediction}` : 'Sentiment: '}
                        </p>
                        <p>
                            {prediction ? `Confidence: ${prediction.confidence.toFixed(2)}` : 'Confidence: '}
                        </p>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default MyDropzone;