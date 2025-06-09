from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch
from model import GCNEncoder, FraudGAE
from config import MODEL_SAVE_PATH, EMBEDDING_DIM
from utils import load_graph, load_node_mapping
from inference import get_fraud_score, compute_temp_embedding
from anomaly import top_anomalies, isolate_anomalies, z_score_anomaly_scores

router = APIRouter()

class NewClaim(BaseModel):
    bene_sex_ident_cd: int
    bene_race_cd: int
    bene_esrd_ind: int
    sp_state_code: int
    bene_county_cd: int
    sp_conditions: dict   # e.g., {"SP_CHF": 1, "SP_CNCR": 0, ...}
    payments: dict        # e.g., {"MEDREIMB_IP": 2000, ...}
    provider_id: str      # Existing provider ID


@router.post("/submit_claim")
def submit_claim(claim: NewClaim):
    try:
        data = load_graph()
        node_map = load_node_mapping()
        encoder = GCNEncoder(data.num_node_features, EMBEDDING_DIM)
        model = FraudGAE(encoder)
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        model.eval()

        # Compute new node embedding with context injection
        with torch.no_grad():
            fraud_score, top_neighbors, z_new = compute_temp_embedding(
                data=data,
                model=model,
                claim=claim.model_dump(),
                node_mapping=node_map
            )

            # Compute anomaly score from z
            z_all = torch.cat([model.encode(data.x, data.edge_index), z_new.unsqueeze(0)], dim=0)
            anomaly_scores = z_score_anomaly_scores(z_all)
            anomaly = anomaly_scores[-1].item()

        return {
            "fraud_score": round(fraud_score, 4),
            "anomaly_score": round(anomaly, 4),
            "top_neighbors": top_neighbors
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class PredictionRequest(BaseModel):
    claim_id: str

class AnomalyResponse(BaseModel):
    node_index: int
    score: float

data = load_graph()
node_mapping = load_node_mapping()
encoder = GCNEncoder(data.num_node_features, EMBEDDING_DIM)
model = FraudGAE(encoder)
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

with torch.no_grad():
    z = model.encode(data.x, data.edge_index)

@router.post("/predict")
def predict(request: PredictionRequest):
    claim_id = request.claim_id
    if claim_id not in node_mapping:
        raise HTTPException(status_code=404, detail="Claim ID not found in graph")

    result = get_fraud_score(claim_id, data, z, node_mapping, num_hops=2, topk=5)
    return result

@router.get("/anomalies")
def get_anomalies(top_k: int = 10):
    indices, scores = top_anomalies(z, top_k)
    return [{"node_index": i, "score": round(s, 4)} for i, s in zip(indices, scores)]

@router.get("/isolation")
def isolation_anomaly(top_k: int = 10):
    z = model.encode(data.x, data.edge_index)
    result = isolate_anomalies(z, n_top=top_k)
    return [{"node_index": idx, "score": score} for idx, score in result]
