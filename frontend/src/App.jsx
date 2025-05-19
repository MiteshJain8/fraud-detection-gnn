import React, { useEffect, useState } from 'react';
import Dashboard from '../components/Dashboard';

export default function App() {
  const [anomalies, setAnomalies] = useState([]);

  useEffect(() => {
    axios.get('http://localhost:8000/anomalies')
      .then(res => {
        if (res.data && Array.isArray(res.data.anomalies)) {
          setAnomalies(res.data.anomalies);
        } else {
          console.error("Invalid data format received:", res.data);
          setAnomalies([]); // Set to empty array on error or invalid format
        }
      })
      .catch(error => {
        console.error("Error fetching anomalies:", error);
        setAnomalies([]);
      });
  }, []);
  return <Dashboard anomalies={anomalies} />;
}
