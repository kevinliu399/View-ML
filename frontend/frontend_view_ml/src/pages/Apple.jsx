import React from 'react';
import AppleStockPredictor from '../components/TableAAPL';

export default function Apple() {
    return( 
        <div>
            <div className="flex flex-col pb-2">
                <h1 className="text-xl font-medium">Apple Stocks Model</h1>
                <p>This model detects apple stocks</p>
            </div>
            <AppleStockPredictor />
        </div>
    )
}