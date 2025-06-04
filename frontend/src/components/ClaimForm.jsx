import { useState } from 'react'
import RiskResult from './RiskResult'

const defaultConditions = {
  SP_ALZHDMTA: 0, SP_CHF: 0, SP_CHRNKIDN: 0, SP_CNCR: 0, SP_COPD: 0,
  SP_DEPRESSN: 0, SP_DIABETES: 0, SP_ISCHMCHT: 0, SP_OSTEOPRS: 0,
  SP_RA_OA: 0, SP_STRKETIA: 0,
}

const defaultPayments = {
  MEDREIMB_IP: 0, BENRES_IP: 0, PPPYMT_IP: 0,
  MEDREIMB_OP: 0, BENRES_OP: 0, PPPYMT_OP: 0,
  MEDREIMB_CAR: 0, BENRES_CAR: 0, PPPYMT_CAR: 0,
}

export default function ClaimForm() {
  const [form, setForm] = useState({
    bene_sex_ident_cd: 1,
    bene_race_cd: 1,
    bene_esrd_ind: 0,
    sp_state_code: 1,
    bene_county_cd: 1,
    sp_conditions: defaultConditions,
    payments: defaultPayments,
    provider_id: '',
  })

  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleChange = (e, group = null) => {
    const { name, value } = e.target
    if (group) {
      setForm(prev => ({
        ...prev,
        [group]: { ...prev[group], [name]: Number(value) },
      }))
    } else {
      setForm(prev => ({ ...prev, [name]: name === 'provider_id' ? value : Number(value) }))
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError(null)
    try {
      const res = await fetch('http://localhost:8000/submit_claim', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      })
      if (!res.ok) throw new Error('Claim submission failed.')
      const data = await res.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    }
  }

  return (
    <div className="max-w-4xl mx-auto text-lg">
      <h2 className="text-2xl font-bold mb-6 text-blue-700">📝 Submit a New Insurance Claim</h2>

      <form onSubmit={handleSubmit} className="space-y-6 bg-white p-6 rounded-lg shadow-lg">

        {/* Basic Info */}
        <div>
          <h3 className="text-xl font-semibold mb-3 text-gray-800">Beneficiary & Provider Info</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block font-medium mb-1">Provider ID</label>
              <input name="provider_id" required placeholder="e.g. 123456" value={form.provider_id}
                onChange={handleChange} className="w-full border p-2 rounded" />
            </div>

            <div>
              <label className="block font-medium mb-1">Sex (1 = Male, 2 = Female)</label>
              <input name="bene_sex_ident_cd" type="number" value={form.bene_sex_ident_cd}
                onChange={handleChange} className="w-full border p-2 rounded" />
            </div>

            <div>
              <label className="block font-medium mb-1">Race Code</label>
              <input name="bene_race_cd" type="number" value={form.bene_race_cd}
                onChange={handleChange} className="w-full border p-2 rounded" />
            </div>

            <div>
              <label className="block font-medium mb-1">ESRD (End-Stage Renal Disease) Indicator</label>
              <input name="bene_esrd_ind" type="number" value={form.bene_esrd_ind}
                onChange={handleChange} className="w-full border p-2 rounded" />
            </div>

            <div>
              <label className="block font-medium mb-1">State Code</label>
              <input name="sp_state_code" type="number" value={form.sp_state_code}
                onChange={handleChange} className="w-full border p-2 rounded" />
            </div>

            <div>
              <label className="block font-medium mb-1">County Code</label>
              <input name="bene_county_cd" type="number" value={form.bene_county_cd}
                onChange={handleChange} className="w-full border p-2 rounded" />
            </div>
          </div>
        </div>

        {/* Chronic Conditions */}
        <div>
          <h3 className="text-xl font-semibold mb-3 text-gray-800">Chronic Conditions</h3>
          <div className="grid grid-cols-3 md:grid-cols-4 gap-3 text-sm">
            {Object.keys(defaultConditions).map((key) => (
              <div key={key}>
                <label className="block text-gray-600 mb-1">{key.replace("SP_", "").replace("_", " ")}</label>
                <input type="number" min="0" max="1" name={key}
                  value={form.sp_conditions[key]}
                  onChange={(e) => handleChange(e, 'sp_conditions')}
                  className="w-full border p-1 rounded" />
              </div>
            ))}
          </div>
        </div>

        {/* Payment Info */}
        <div>
          <h3 className="text-xl font-semibold mb-3 text-gray-800">Payment & Reimbursement</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
            {Object.keys(defaultPayments).map((key) => (
              <div key={key}>
                <label className="block text-gray-600 mb-1">{key.replace("_", " ")}</label>
                <input type="number" name={key} value={form.payments[key]} onChange={(e) => handleChange(e, 'payments')} className="w-full border p-1 rounded" />
              </div>
            ))}
          </div>
        </div>

        <button type="submit"
                className="bg-blue-700 text-white text-lg px-6 py-2 rounded hover:bg-blue-800">
          🚀 Submit Claim for Analysis
        </button>
      </form>

      {error && <p className="text-red-600 mt-3 font-medium">{error}</p>}
      {result && <RiskResult result={result} />}
    </div>
  )
}
