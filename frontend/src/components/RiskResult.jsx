export default function RiskResult({ result }) {
  return (
    <div className="mt-6 p-4 bg-white shadow rounded border">
      <h3 className="text-lg font-semibold mb-2">Fraud Prediction Result</h3>

      <div className="mb-2">
        <p className="text-sm font-medium text-gray-700">
          Fraud Score:{' '}
          <span className={`font-bold ${result.fraud_score > 0.85 ? 'text-red-600' : result.fraud_score > 0.6 ? 'text-yellow-600' : 'text-green-600'}`}>
            {result.fraud_score}
          </span>
        </p>
        <p className="text-sm font-medium text-gray-700">
          Anomaly Score: <span className="font-bold">{result.anomaly_score}</span>
        </p>
      </div>

      <div className="mt-2">
        <p className="text-sm font-semibold">Top Similar Nodes:</p>
        <ul className="list-disc pl-5 text-sm text-gray-800">
          {result.top_neighbors.map((n, i) => (
            <li key={i}>Node {n.node_index} — score: {n.score}</li>
          ))}
        </ul>
      </div>
    </div>
  )
}
