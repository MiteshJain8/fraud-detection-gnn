import { useEffect, useState } from 'react'

export default function AnomalyTable() {
  const [rows, setRows] = useState([])

  useEffect(() => {
    fetch('http://localhost:8000/anomalies?top_k=10')
      .then(res => res.json())
      .then(setRows)
      .catch(console.error)
  }, [])

  return (
    <div className="mt-10">
      <h3 className="text-lg font-semibold mb-2">Top 10 Anomalous Nodes</h3>
      <table className="w-full border text-sm">
        <thead className="bg-gray-100">
          <tr>
            <th className="px-3 py-2 border">Node</th>
            <th className="px-3 py-2 border">Anomaly Score</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              <td className="px-3 py-2 border">{r.node_index}</td>
              <td className="px-3 py-2 border">{r.score}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
