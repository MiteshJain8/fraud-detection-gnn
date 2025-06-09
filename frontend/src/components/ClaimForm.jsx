import { useRef, useState } from 'react';
import RiskResult from './RiskResult';

const chronicLabels = {
    SP_ALZHDMTA: "Alzheimer's Disease/Dementia",
    SP_CHF: "Congestive Heart Failure",
    SP_CHRNKIDN: "Chronic Kidney Disease",
    SP_CNCR: "Cancer",
    SP_COPD: "Chronic Obstructive Pulmonary Disease",
    SP_DEPRESSN: "Depression",
    SP_DIABETES: "Diabetes",
    SP_ISCHMCHT: "Ischemic Heart Disease",
    SP_OSTEOPRS: "Osteoporosis",
    SP_RA_OA: "Rheumatoid/Osteoarthritis",
    SP_STRKETIA: "Stroke or Transient Ischemic Attack",
};

const paymentLabels = {
    MEDREIMB_IP: "Inpatient Medicare Reimbursement",
    BENRES_IP: "Inpatient Beneficiary Responsibility",
    PPPYMT_IP: "Inpatient Primary Payer",
    MEDREIMB_OP: "Outpatient Medicare Reimbursement",
    BENRES_OP: "Outpatient Beneficiary Responsibility",
    PPPYMT_OP: "Outpatient Primary Payer",
    MEDREIMB_CAR: "Carrier Medicare Reimbursement",
    BENRES_CAR: "Carrier Beneficiary Responsibility",
    PPPYMT_CAR: "Carrier Primary Payer",
};

export default function ClaimForm() {
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(false);

    // Create refs
    const refs = useRef({});

    // Utility to render Input
    const InputField = ({ label, name, defaultValue = "0" }) => (
        <div className="mb-4">
            <label className="block font-medium text-gray-800 mb-1">{label}</label>
            <input
                type={name === "provider_id" ? "text" : "number"}
                name={name}
                defaultValue={defaultValue}
                ref={el => (refs.current[name] = el)}
                className="w-full border border-gray-300 rounded px-3 py-2"
            />
        </div>
    );

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        setResult(null);

        try {
            const payload = {
                provider_id: refs.current["provider_id"].value,
                bene_sex_ident_cd: Number(refs.current["bene_sex_ident_cd"].value),
                bene_race_cd: Number(refs.current["bene_race_cd"].value),
                bene_esrd_ind: Number(refs.current["bene_esrd_ind"].value),
                sp_state_code: Number(refs.current["sp_state_code"].value),
                bene_county_cd: Number(refs.current["bene_county_cd"].value),
                sp_conditions: Object.fromEntries(
                    Object.keys(chronicLabels).map(k => [k, Number(refs.current[k].value)])
                ),
                payments: Object.fromEntries(
                    Object.keys(paymentLabels).map(k => [k, Number(refs.current[k].value)])
                )
            };

            const res = await fetch("http://localhost:8000/submit_claim", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            if (!res.ok) throw new Error("Claim submission failed.");
            const data = await res.json();
            setResult(data);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="max-w-2xl mx-auto p-6 bg-white shadow-md rounded-md">
            <h2 className="text-2xl font-bold mb-6 text-blue-700">📝 Submit Insurance Claim</h2>
            <form onSubmit={handleSubmit}>
                <InputField name="provider_id" label="Provider ID" defaultValue="4900NA" />
                <InputField name="bene_sex_ident_cd" label="Sex (1 = Male, 2 = Female)" defaultValue="1" />
                <InputField name="bene_race_cd" label="Race Code" defaultValue="1" />
                <InputField name="bene_esrd_ind" label="ESRD Indicator (0 = No, 1 = Yes)" defaultValue="0" />
                <InputField name="sp_state_code" label="State Code" defaultValue="1" />
                <InputField name="bene_county_cd" label="County Code" defaultValue="1" />

                <hr className="my-6" />
                <h3 className="text-lg font-semibold mb-3">Chronic Conditions</h3>
                {Object.entries(chronicLabels).map(([key, label]) => (
                    <InputField key={key} name={key} label={label} defaultValue="0" />
                ))}

                <hr className="my-6" />
                <h3 className="text-lg font-semibold mb-3">Payments & Reimbursements (USD)</h3>
                {Object.entries(paymentLabels).map(([key, label]) => (
                    <InputField key={key} name={key} label={label} defaultValue="0" />
                ))}

                <button
                    type="submit"
                    disabled={loading}
                    className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 rounded text-lg mt-6"
                >
                    {loading ? "⏳ Analyzing..." : "🚀 Submit Claim for Analysis"}
                </button>
            </form>

            {error && <p className="text-red-600 mt-4 font-medium">{error}</p>}
            {result && <RiskResult result={result} />}
        </div>
    );
}