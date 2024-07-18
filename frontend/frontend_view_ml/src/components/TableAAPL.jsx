import React, { useState, useRef, useEffect, useCallback } from 'react';
import axios from 'axios';
import { useDropzone } from 'react-dropzone';
import Papa from 'papaparse';
import * as XLSX from 'xlsx';
import { Trash, CirclePlus } from 'lucide-react';

const AppleStockPredictor = () => {
    const [prices, setPrices] = useState(['']);
    const [prediction, setPrediction] = useState(null);
    const inputRefs = useRef([]);

    useEffect(() => {
        inputRefs.current = inputRefs.current.slice(0, prices.length);
    }, [prices]);

    const handleInputChange = (index, value) => {
        const newPrices = [...prices];
        newPrices[index] = value;
        setPrices(newPrices);
    };

    const handleKeyPress = (e, index) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            addRow();
        }
    };

    const addRow = () => {
        setPrices([...prices, '']);
        setTimeout(() => {
            inputRefs.current[prices.length].focus();
        }, 0);
    };

    const removeRow = (index) => {
        if (prices.length > 1) {
            const newPrices = prices.filter((_, i) => i !== index);
            setPrices(newPrices);
        }
    };

    const handleSubmit = async () => {
        try {
            const response = await axios.post('http://localhost:8000/api/predict/aapl_lstm/', {
                input: prices.filter(price => price !== '').map(Number)
            });
            setPrediction(response.data.prediction);
        } catch (error) {
            console.error('Error fetching prediction:', error);
        }
    };

    const handleClear = () => {
        setPrices(['']);
        setPrediction(null);
        setTimeout(() => {
            if (inputRefs.current[0]) {
                inputRefs.current[0].focus();
            }
        }, 0);
    };

    const onDrop = useCallback((acceptedFiles) => {
        acceptedFiles.forEach((file) => {
            const reader = new FileReader();

            reader.onabort = () => console.log('file reading was aborted');
            reader.onerror = () => console.log('file reading has failed');
            reader.onload = () => {
                const binaryStr = reader.result;
                if (file.type === 'text/csv') {
                    Papa.parse(binaryStr, {
                        complete: (result) => {
                            const data = result.data.flat();
                            setPrices(data.filter(item => !isNaN(item) && item !== '').map(Number));
                        },
                        header: false
                    });
                } else if (
                    file.type === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' ||
                    file.type === 'application/vnd.ms-excel'
                ) {
                    const workbook = XLSX.read(binaryStr, { type: 'binary' });
                    const sheetName = workbook.SheetNames[0];
                    const sheet = workbook.Sheets[sheetName];
                    const data = XLSX.utils.sheet_to_json(sheet, { header: 1 }).flat();
                    setPrices(data.filter(item => !isNaN(item) && item !== '').map(Number));
                }
            };

            if (file.type === 'text/csv') {
                reader.readAsText(file);
            } else {
                reader.readAsBinaryString(file);
            }
        });
    }, []);

    const { getRootProps, getInputProps } = useDropzone({ onDrop });

    return (
        <div className="flex flex-row">
            <div className="flex-1">
                <div className="w-2/3">
                    <h1 className="font-bold text-lg mb-4">Prices</h1>
                    <div className="max-h-96 overflow-x-hidden max-w-64 overflow-y-auto relative hide-scrollbar">
                        <table className="w-full">
                            <tbody>
                                {prices.map((price, index) => (
                                    <tr key={index}>
                                        <td>
                                            <input
                                                type="number"
                                                value={price}
                                                onChange={(e) => handleInputChange(index, e.target.value)}
                                                onKeyPress={(e) => handleKeyPress(e, index)}
                                                ref={el => inputRefs.current[index] = el}
                                                className="border border-gray-300 rounded-lg p-2"
                                            />
                                            <button onClick={() => removeRow(index)} className="ml-4">
                                                <Trash size={20} className="hover:text-red-400 transition-all" />
                                            </button>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                        <div className="sticky bottom-0 bg-white w-full flex items-center justify-center p-4">
                            <button onClick={addRow} className="flex px-2 py-1 border-2 border-gray-300 text-gray-300 rounded-xl w-full items-center justify-center mt-2 shadow-md hover:scale-105 transition-all">
                                <CirclePlus size={20} />
                            </button>
                        </div>
                    </div>
                    <div className="flex justify-between mt-4">
                        <button className="px-4 py-2 bg-red-50 hover:bg-red-200 hover:shadow-md transition-all rounded-lg shadow-sm" onClick={handleSubmit}>Submit</button>
                        <button className="px-4 py-2 bg-red-50 hover:bg-red-200 hover:shadow-md transition-all rounded-lg shadow-sm" onClick={handleClear}>Clear</button>
                    </div>
                    {prediction !== null && (
                        <div>
                            <h3>Predicted Price for Tomorrow:</h3>
                            <p>${prediction.toFixed(2)}</p>
                        </div>
                    )}
                </div>

                <style jsx>{`
                .hide-scrollbar::-webkit-scrollbar {
                    display: none;
                }
                .hide-scrollbar {
                    -ms-overflow-style: none; /* IE and Edge */
                    scrollbar-width: none; /* Firefox */
                }
            `}</style>

            </div>
            <div className="w-1/3 ml-4 p-4 border border-dashed border-gray-300 rounded-lg flex items-center justify-center" {...getRootProps()}>
                <input {...getInputProps()} />
                <p>Drag 'n' drop a CSV or Excel file here, or click to select one</p>
            </div>
        </div>
    );
};

export default AppleStockPredictor;
