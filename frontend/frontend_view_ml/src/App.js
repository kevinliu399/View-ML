import React from 'react';
import Sidebar, { SidebarItem } from './components/Sidebar';
import './App.css';
import { Thermometer, LineChart, Sigma, Smile, House } from 'lucide-react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Housing from './pages/Housing';
import ThermometerModel from './pages/ThermometerModel';
import Apple from './pages/Apple';
import MNIST from './pages/MNIST';
import Sentiment from './pages/Sentiment';

function App() {
  return (
    <Router>
      <div className="flex">
        <Sidebar>
          <SidebarItem icon={<Thermometer />} text="Temperature" to="/temperature" />
          <SidebarItem icon={<LineChart />} text="Apple Stocks" to="/apple" />
          <SidebarItem icon={<Sigma />} text="MNIST" to="/mnist" />
          <SidebarItem icon={<Smile />} text="Sentiment" to="/sentiment" />
          <SidebarItem icon={<House />} text="California Housing" to="/housing" />
        </Sidebar>

        <div className="flex-1 flex items-center justify-center">
          <Routes>
            <Route path="/" element={<Navigate to="/temperature" />} />
            <Route path="/housing" element={<Housing />} />
            <Route path="/temperature" element={<ThermometerModel />} />
            <Route path="/apple" element={<Apple />} />
            <Route path="/mnist" element={<MNIST />} />
            <Route path="/sentiment" element={<Sentiment />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;