import React, { useState } from 'react';
import axios from 'axios';

export default function CaliforniaModelInputs() {
    const [longitude, setLongitude] = useState('');
    const [latitude, setLatitude] = useState('');
    const [housingMedianAge, setHousingMedianAge] = useState('');
    const [totalRooms, setTotalRooms] = useState('');
    const [totalBedrooms, setTotalBedrooms] = useState('');
    const [population, setPopulation] = useState('');
    const [households, setHouseholds] = useState('');
    const [medianIncome, setMedianIncome] = useState('');
    const [oceanProximity, setOceanProximity] = useState('');
    const [prediction, setPrediction] = useState(null);

    const handleSubmit = async (e) => {
        e.preventDefault();

        const data = {
            longitude: parseFloat(longitude),
            latitude: parseFloat(latitude),
            housing_median_age: parseInt(housingMedianAge),
            total_rooms: parseInt(totalRooms),
            total_bedrooms: parseInt(totalBedrooms),
            population: parseInt(population),
            households: parseInt(households),
            median_income: parseFloat(medianIncome),
            ocean_proximity: oceanProximity
        };

        try {
            const response = await axios.post('http://localhost:8000/api/predict/housing/', data);
            setPrediction(Math.round(response.data.prediction));
        } catch (error) {
            console.error('There was an error making the request:', error);
        }
    };

    return (
        <div className="mt-8">
            <form onSubmit={handleSubmit}>
                <div className="grid grid-cols-2 gap-4">
                    <div className="flex flex-col">
                        <label htmlFor="longitude">Longitude</label>
                        <input
                            type="number"
                            id="longitude"
                            required
                            value={longitude}
                            onChange={(e) => setLongitude(e.target.value)}
                            placeholder="-122.4194"
                        />
                    </div>
                    <div className="flex flex-col">
                        <label htmlFor="latitude">Latitude</label>
                        <input
                            type="number"
                            id="latitude"
                            required
                            value={latitude}
                            onChange={(e) => setLatitude(e.target.value)}
                            placeholder="37.7749"
                        />
                    </div>
                    <div className="flex flex-col">
                        <label htmlFor="housingMedianAge">Housing Median Age</label>
                        <input
                            type="number"
                            id="housingMedianAge"
                            required
                            value={housingMedianAge}
                            onChange={(e) => setHousingMedianAge(e.target.value)}
                            placeholder="35"
                        />
                    </div>
                    <div className="flex flex-col">
                        <label htmlFor="totalRooms">Total Rooms</label>
                        <input
                            type="number"
                            id="totalRooms"
                            required
                            value={totalRooms}
                            onChange={(e) => setTotalRooms(e.target.value)}
                            placeholder="5000"
                        />
                    </div>
                    <div className="flex flex-col">
                        <label htmlFor="totalBedrooms">Total Bedrooms</label>
                        <input
                            type="number"
                            id="totalBedrooms"
                            required
                            value={totalBedrooms}
                            onChange={(e) => setTotalBedrooms(e.target.value)}
                            placeholder="1000"
                        />
                    </div>
                    <div className="flex flex-col">
                        <label htmlFor="population">Population</label>
                        <input
                            type="number"
                            id="population"
                            required
                            value={population}
                            onChange={(e) => setPopulation(e.target.value)}
                            placeholder="5000"
                        />
                    </div>
                    <div className="flex flex-col">
                        <label htmlFor="households">Households</label>
                        <input
                            type="number"
                            id="households"
                            required
                            value={households}
                            onChange={(e) => setHouseholds(e.target.value)}
                            placeholder="1000"
                        />
                    </div>
                    <div className="flex flex-col">
                        <label htmlFor="medianIncome">Median Income</label>
                        <input
                            type="number"
                            id="medianIncome"
                            required
                            value={medianIncome}
                            onChange={(e) => setMedianIncome(e.target.value)}
                            placeholder="50000"
                        />
                    </div>
                    <div className="flex flex-col">
                        <label htmlFor="oceanProximity">Ocean Proximity</label>
                        <select
                            id="oceanProximity"
                            required
                            value={oceanProximity}
                            onChange={(e) => setOceanProximity(e.target.value)}
                        >
                            <option value="NEAR BAY">Near Bay</option>
                            <option value="INLAND">Inland</option>
                            <option value="<1H OCEAN">&lt;1 Hour From Ocean</option>
                            <option value="NEAR OCEAN">Near Ocean</option>
                            <option value="ISLAND">Island</option>
                        </select>
                    </div>
                </div>

                <div className="flex flex-col mt-5">
                    <button type="submit" className="mt-2 px-3 py-2 text-md bg-red-50 rounded-full hover:bg-red-200 transition-all shadow-sm">Submit</button>
                    <div className="mt-1">
                        <h3 className="flex flex-row">Predicted Median House Value:<p className="font-semibold px-1 text-red-400 underline transition-all cursor-default">{prediction}</p>$</h3>
                    </div>
                </div>
            </form>
        </div>
    );
}
