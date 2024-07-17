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
        if (file.type === "image/jpeg" || file.type === "image/png" || file.type === "image/jpg" || file.type === "image/png" || file.type === "image/bmp") {
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
        accept: {'image/jpeg': [], 'image/png': []}
    })

    const predictSentiment = async (file) => {
        const formData = new FormData()
        formData.append('image', file)
    
        try {
            // Replace 'http://localhost:8000' with your Django server's address
            const response = await axios.post('http://localhost:8000/predict/sentiment/', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            })
            setPrediction(response.data)
        } catch (error) {
            console.error("Error predicting sentiment:", error)
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
                {prediction && (
                <div>
                    <h2>Prediction:</h2>
                    <p>{prediction}</p>
                </div>
                )}
            </div>
            
            {image && (
                <div>
                    <img src={image} alt="Uploaded" style={{maxWidth: '300px'}} />
                </div>
            )}
        </div>
    )
}

export default MyDropzone;