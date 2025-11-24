import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import MLDashboard from './MLDashboard';
import './App.css';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<MLDashboard />} />
      </Routes>
    </Router>
  );
}

export default App;
