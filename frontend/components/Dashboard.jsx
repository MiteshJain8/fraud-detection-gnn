import React from 'react';
import ClaimTable from './ClaimTable'; // Import the ClaimTable component

export default function Dashboard({ anomalies }) {
  return (
    // Use Tailwind classes for styling
    <div className="p-4 max-w-4xl mx-auto"> {/* Padding, max-width, center */}
      <h1 className="text-2xl font-bold mb-4 text-center">Fraud Detection Dashboard</h1> {/* Title styling */}
      {/* Render the ClaimTable component, passing the anomaly data */}
      <ClaimTable data={anomalies} />
    </div>
  );
}