import React from 'react';
import CaliforniaModelInputs from '../components/InputHousing';

export default function Housing() {
    return( 
        <div>
            <div className="flex flex-col pb-8">
                <h1 className="text-xl font-medium">Housing Model</h1>
                <p>This model predicts the median house value in California</p>

                <CaliforniaModelInputs />
            </div>
        </div> 
    )
}