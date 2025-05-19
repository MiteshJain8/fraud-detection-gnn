import { useEffect, useState } from 'react';
import axios from '../api/axiosInstance';
import ClaimTable from './ClaimTable';

export default function Dashboard() {
  const [claims, setClaims] = useState([]);

  useEffect(() => {
    axios.get('/claims')
      .then(res => {
        const formatted = res.data.map((item, i) => ({
          id: i + 1,
          node: item.claim_id,
          score: item.fraud_score
        }));
        setClaims(formatted);
      });
  }, []);

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">Anomaly Dashboard</h1>
      <ClaimTable data={claims} />
    </div>
  );
}
