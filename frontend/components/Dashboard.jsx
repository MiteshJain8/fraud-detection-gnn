import React from 'react';
import ClaimTable from './ClaimTable';

export default function Dashboard({ anomalies }) {
  return (
    <div className="p-4 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-4 text-center">Fraud Detection Dashboard</h1>
      <ClaimTable data={anomalies} />
    </div>
  );
}