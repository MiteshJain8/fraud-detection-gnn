import React, { useEffect, useState } from 'react'; // Import React and hooks
import axios from 'axios'; // Import axios
import Dashboard from '../components/Dashboard'; // Import Dashboard component
// Remove unused imports like logos if not needed for the dashboard
// import reactLogo from './assets/react.svg'
// import viteLogo from '/vite.svg'
// import './App.css' // Remove if App.css is not used

export default function App() {
  const [anomalies, setAnomalies] = useState([]); // State for anomalies

  useEffect(() => {
    // Fetch data from the backend API endpoint
    axios.get('http://localhost:8000/anomalies') // Ensure backend URL is correct
      .then(res => {
        // Check if the response has the anomalies array
        if (res.data && Array.isArray(res.data.anomalies)) {
          setAnomalies(res.data.anomalies);
        } else {
          console.error("Invalid data format received:", res.data);
          setAnomalies([]); // Set to empty array on error or invalid format
        }
      })
      .catch(error => {
        console.error("Error fetching anomalies:", error);
        setAnomalies([]); // Set to empty array on fetch error
      });
  }, []); // Empty dependency array means this effect runs once on mount

  // Render the Dashboard component, passing the anomalies data
  return <Dashboard anomalies={anomalies} />;
}
