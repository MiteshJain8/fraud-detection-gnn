import React from 'react';

export default function ClaimTable({ data }) {
  // Handle cases where data might be null, undefined, or not an array
  if (!Array.isArray(data)) {
    return <p className="text-center text-gray-500">Loading anomalies or no data available...</p>;
  }

  // Handle case where data array is empty
  if (data.length === 0) {
    return <p className="text-center text-gray-500">No anomalies detected.</p>;
  }

  return (
    // Use Tailwind classes for table styling
    <div className="overflow-x-auto shadow-md sm:rounded-lg"> {/* Add shadow and rounded corners */}
      <table className="min-w-full border-collapse border border-gray-300 text-sm text-left text-gray-500 dark:text-gray-400">
        <thead className="text-xs text-gray-700 uppercase bg-gray-50 dark:bg-gray-700 dark:text-gray-400">
          <tr>
            {/* Define table headers */}
            <th scope="col" className="px-4 py-2 border border-gray-300">ID</th>
            <th scope="col" className="px-4 py-2 border border-gray-300">Node</th>
            <th scope="col" className="px-4 py-2 border border-gray-300">Anomaly Score</th>
          </tr>
        </thead>
        <tbody>
          {/* Map over the data array to create table rows */}
          {data.map((row, i) => (
            // Use a unique key for each row, preferably a stable ID from the data if available
            <tr key={row.id || i} className="bg-white border-b dark:bg-gray-800 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-600">
              {/* Display data in table cells */}
              <td className="px-4 py-2 border border-gray-300 font-medium text-gray-900 whitespace-nowrap dark:text-white">{row.id}</td>
              <td className="px-4 py-2 border border-gray-300">{row.node}</td>
              {/* Format score to a few decimal places */}
              <td className="px-4 py-2 border border-gray-300">{typeof row.score === 'number' ? row.score.toFixed(4) : row.score}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}