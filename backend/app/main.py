# # import torch
# # from model import GCNEncoder, FraudGAE
# # from utils import load_graph, load_model, load_node_mapping
# # from inference import get_fraud_score
# # from config import MODEL_SAVE_PATH, EMBEDDING_DIM

# # def run_inference(claim_id: str):
# #     print(f"Running fraud inference for claim: {claim_id}")

# #     # Load graph and trained model
# #     data = load_graph()
# #     node_mapping = load_node_mapping()

# #     encoder = GCNEncoder(data.num_node_features, EMBEDDING_DIM)
# #     model = FraudGAE(encoder)
# #     model.load_state_dict(torch.load(MODEL_SAVE_PATH))
# #     model.eval()

# #     with torch.no_grad():
# #         z = model.encode(data.x, data.edge_index)
# #         result = get_fraud_score(claim_id, data, z, node_mapping)

# #     print(f"\nFraud risk score for claim {result['claim_id']}: {result['fraud_score']:.4f}")
# #     print(f"Top similar nodes:")
# #     for neighbor in result['top_neighbors']:
# #         print(f"Node {neighbor['node_index']} → score: {neighbor['score']}")

# # if __name__ == "__main__":
# #     run_inference(claim_id="002C9FB269ECB7C8")


# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import List, Dict, Optional, Union
# import numpy as np
# import torch
# import joblib
# import os
# from contextlib import asynccontextmanager

# # Import your GNN model and utilities
# from app.model import GAE  # Your Graph Autoencoder model
# from app.graph_builder import build_subgraph_for_claim
# from app.utils import process_claim_data

# # Model state container
# ml_models = {}

# # Define the lifespan context manager to load models on startup
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Load the GNN model on startup
#     try:
#         # Path to your saved model
#         model_path = os.path.join("models", "gnn_model.pt")
        
#         # Load model
#         model = GAE(
#             input_dim=64,  # Your feature dimensions
#             hidden_dim=32,
#             embedding_dim=16
#         )
#         model.load_state_dict(torch.load(model_path))
#         model.eval()  # Set to evaluation mode
        
#         # Store in the models dictionary
#         ml_models["gnn_model"] = model
#         ml_models["graph_data"] = torch.load(os.path.join("data", "processed", "graph_data.pt"))
        
#         print("Model loaded successfully")
#     except Exception as e:
#         print(f"Error loading model: {e}")
    
#     yield  # Yield control back to FastAPI
    
#     # Clean up resources when the app shuts down
#     ml_models.clear()
#     print("Resources cleaned up")

# # Initialize FastAPI app
# app = FastAPI(
#     title="Healthcare Fraud Detection API",
#     description="API for detecting fraud in healthcare claims using Graph Neural Networks",
#     version="1.0.0",
#     lifespan=lifespan
# )

# # Define input data model
# class ClaimData(BaseModel):
#     patient_id: str
#     provider_id: str
#     claim_amount: float
#     claim_date: str
#     patient_info: Dict[str, Union[str, float]]
#     provider_info: Dict[str, str]
#     claim_details: Dict[str, str]

# # Define output data model
# class FraudPrediction(BaseModel):
#     claim_id: str
#     fraud_score: float
#     contributing_factors: List[Dict[str, Union[str, float]]]

# # Health check endpoint
# @app.get("/health")
# def health_check():
#     return {"status": "healthy", "model_loaded": "gnn_model" in ml_models}

# # Prediction endpoint
# @app.post("/predict_fraud", response_model=FraudPrediction)
# async def predict_fraud(claim: ClaimData):
#     if "gnn_model" not in ml_models:
#         raise HTTPException(status_code=503, detail="Model not loaded")
    
#     try:
#         # Process the claim data
#         processed_data = process_claim_data(claim)
        
#         # Build a subgraph for this claim
#         subgraph = build_subgraph_for_claim(
#             claim_data=processed_data,
#             graph_data=ml_models["graph_data"]
#         )
        
#         # Get model prediction
#         model = ml_models["gnn_model"]
#         with torch.no_grad():
#             # Forward pass through the model
#             embedding = model.encode(subgraph.x, subgraph.edge_index)
            
#             # Calculate anomaly score (e.g., reconstruction error or distance from normal patterns)
#             fraud_score = calculate_fraud_score(embedding, processed_data)
            
#             # Generate a unique claim ID
#             claim_id = f"CL{np.random.randint(1000, 9999)}"
            
#             # Identify contributing factors
#             contributing_factors = identify_contributing_factors(
#                 claim_data=processed_data,
#                 embedding=embedding,
#                 fraud_score=fraud_score
#             )
            
#             return FraudPrediction(
#                 claim_id=claim_id,
#                 fraud_score=float(fraud_score),
#                 contributing_factors=contributing_factors
#             )
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# # Helper function to calculate fraud score
# def calculate_fraud_score(embedding, claim_data):
#     # This would be your actual implementation based on your GNN model
#     # For example, you might compute the reconstruction error or
#     # the distance from normal patterns in the embedding space
    
#     # Placeholder implementation
#     # In a real system, this would use the GNN embeddings to compute a score
#     fraud_score = 0.1  # Base score
    
#     if claim_data["claim_amount"] > 2000:
#         fraud_score += 0.3
#     if claim_data["provider_info"]["type"] == "Specialist":
#         fraud_score += 0.2
#     if claim_data["provider_info"]["state"] in ["FL", "CA", "NY"]:
#         fraud_score += 0.1
    
#     # Add some randomness for demonstration
#     fraud_score += np.random.uniform(-0.1, 0.1)
#     fraud_score = max(0, min(1, fraud_score))  # Ensure between 0 and 1
    
#     return fraud_score

# # Helper function to identify contributing factors
# def identify_contributing_factors(claim_data, embedding, fraud_score):
#     # This would analyze the embedding and claim data to determine
#     # which factors contributed most to the fraud score
    
#     # Placeholder implementation
#     factors = [
#         {"factor": "Claim amount", "impact": 0.4 if claim_data["claim_amount"] > 2000 else 0.1},
#         {"factor": "Provider type", "impact": 0.3 if claim_data["provider_info"]["type"] == "Specialist" else 0.1},
#         {"factor": "Geographic region", "impact": 0.2 if claim_data["provider_info"]["state"] in ["FL", "CA", "NY"] else 0.1},
#         {"factor": "Service code", "impact": 0.1}
#     ]
    
#     return factors

# # Endpoint to get claim history
# @app.get("/claim_history")
# async def get_claim_history():
#     # In a real implementation, this would query a database
#     # Placeholder implementation returning mock data
#     return {
#         "claims": [
#             {"id": "CL001", "patient_id": "P12345", "provider_id": "PR678", "amount": 1250.50, "date": "2025-04-15", "fraud_score": 0.82},
#             {"id": "CL002", "patient_id": "P54321", "provider_id": "PR789", "amount": 450.75, "date": "2025-04-18", "fraud_score": 0.23},
#             {"id": "CL003", "patient_id": "P98765", "provider_id": "PR456", "amount": 2100.00, "date": "2025-04-25", "fraud_score": 0.56},
#             {"id": "CL004", "patient_id": "P12345", "provider_id": "PR123", "amount": 875.25, "date": "2025-05-02", "fraud_score": 0.18},
#             {"id": "CL005", "patient_id": "P45678", "provider_id": "PR678", "amount": 3200.50, "date": "2025-05-10", "fraud_score": 0.91}
#         ]
#     }

# # Endpoint to get similar claims
# @app.get("/similar_claims/{claim_id}")
# async def get_similar_claims(claim_id: str):
#     # In a real implementation, this would use embeddings to find similar claims
#     # Placeholder implementation returning mock data
#     return {
#         "similar_claims": [
#             {"id": "CL006", "similarity": 0.92, "patient_id": "P11111", "provider_id": "PR678", "amount": 1280.50, "date": "2025-03-20", "fraud_score": 0.78},
#             {"id": "CL007", "similarity": 0.85, "patient_id": "P22222", "provider_id": "PR678", "amount": 1150.25, "date": "2025-02-15", "fraud_score": 0.81},
#             {"id": "CL008", "similarity": 0.79, "patient_id": "P33333", "provider_id": "PR456", "amount": 1350.00, "date": "2025-01-10", "fraud_score": 0.75}
#         ]
#     }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Union
import numpy as np
import torch
import joblib
import os
from contextlib import asynccontextmanager

# Import your GNN model and utilities
from app.model import GAE  # Your Graph Autoencoder model
from app.graph_builder import build_subgraph_for_claim
from app.utils import process_claim_data

# Model state container
ml_models = {}

# Define the lifespan context manager to load models on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the GNN model on startup
    try:
        # Path to your saved model
        model_path = os.path.join("models", "gnn_model.pt")
        
        # Load model
        model = GAE(
            input_dim=64,  # Your feature dimensions
            hidden_dim=32,
            embedding_dim=16
        )
        
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set to evaluation mode
        
        # Store in the models dictionary
        ml_models["gnn_model"] = model
        ml_models["graph_data"] = torch.load(os.path.join("data", "processed", "graph_data.pt"))
        
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
    
    yield  # Yield control back to FastAPI
    
    # Clean up resources when the app shuts down
    ml_models.clear()
    print("Resources cleaned up")

# Initialize FastAPI app
app = FastAPI(
    title="Healthcare Fraud Detection API",
    description="API for detecting fraud in healthcare claims using Graph Neural Networks",
    version="1.0.0",
    lifespan=lifespan
)

# Define input data model
class ClaimData(BaseModel):
    patient_id: str
    provider_id: str
    claim_amount: float
    claim_date: str
    patient_info: Dict[str, Union[str, float]]
    provider_info: Dict[str, str]
    claim_details: Dict[str, str]

# Define output data model
class FraudPrediction(BaseModel):
    claim_id: str
    fraud_score: float
    contributing_factors: List[Dict[str, Union[str, float]]]

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": "gnn_model" in ml_models}

# Prediction endpoint
@app.post("/predict_fraud", response_model=FraudPrediction)
async def predict_fraud(claim: ClaimData):
    if "gnn_model" not in ml_models:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Process the claim data
        processed_data = process_claim_data(claim)
        
        # Build a subgraph for this claim
        subgraph = build_subgraph_for_claim(
            claim_data=processed_data,
            graph_data=ml_models["graph_data"]
        )
        
        # Get model prediction
        model = ml_models["gnn_model"]
        with torch.no_grad():
            # Forward pass through the model
            embedding = model.encode(subgraph.x, subgraph.edge_index)
            
            # Calculate anomaly score (e.g., reconstruction error or distance from normal patterns)
            fraud_score = calculate_fraud_score(embedding, processed_data)
            
            # Generate a unique claim ID
            claim_id = f"CL{np.random.randint(1000, 9999)}"
            
            # Identify contributing factors
            contributing_factors = identify_contributing_factors(
                claim_data=processed_data,
                embedding=embedding,
                fraud_score=fraud_score
            )
            
            return FraudPrediction(
                claim_id=claim_id,
                fraud_score=float(fraud_score),
                contributing_factors=contributing_factors
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Helper function to calculate fraud score
def calculate_fraud_score(embedding, claim_data):
    # This would be your actual implementation based on your GNN model
    # For example, you might compute the reconstruction error or
    # the distance from normal patterns in the embedding space
    
    # Placeholder implementation
    # In a real system, this would use the GNN embeddings to compute a score
    fraud_score = 0.1  # Base score
    
    if claim_data["claim_amount"] > 2000:
        fraud_score += 0.3
    if claim_data["provider_info"]["type"] == "Specialist":
        fraud_score += 0.2
    if claim_data["provider_info"]["state"] in ["FL", "CA", "NY"]:
        fraud_score += 0.1
    
    # Add some randomness for demonstration
    fraud_score += np.random.uniform(-0.1, 0.1)
    fraud_score = max(0, min(1, fraud_score))  # Ensure between 0 and 1
    
    return fraud_score

# Helper function to identify contributing factors
def identify_contributing_factors(claim_data, embedding, fraud_score):
    # This would analyze the embedding and claim data to determine
    # which factors contributed most to the fraud score
    
    # Placeholder implementation
    factors = [
        {"factor": "Claim amount", "impact": 0.4 if claim_data["claim_amount"] > 2000 else 0.1},
        {"factor": "Provider type", "impact": 0.3 if claim_data["provider_info"]["type"] == "Specialist" else 0.1},
        {"factor": "Geographic region", "impact": 0.2 if claim_data["provider_info"]["state"] in ["FL", "CA", "NY"] else 0.1},
        {"factor": "Service code", "impact": 0.1}
    ]
    
    return factors

# Endpoint to get claim history
@app.get("/claim_history")
async def get_claim_history():
    # In a real implementation, this would query a database
    # Placeholder implementation returning mock data
    return {
        "claims": [
            {"id": "CL001", "patient_id": "P12345", "provider_id": "PR678", "amount": 1250.50, "date": "2025-04-15", "fraud_score": 0.82},
            {"id": "CL002", "patient_id": "P54321", "provider_id": "PR789", "amount": 450.75, "date": "2025-04-18", "fraud_score": 0.23},
            {"id": "CL003", "patient_id": "P98765", "provider_id": "PR456", "amount": 2100.00, "date": "2025-04-25", "fraud_score": 0.56},
            {"id": "CL004", "patient_id": "P12345", "provider_id": "PR123", "amount": 875.25, "date": "2025-05-02", "fraud_score": 0.18},
            {"id": "CL005", "patient_id": "P45678", "provider_id": "PR678", "amount": 3200.50, "date": "2025-05-10", "fraud_score": 0.91}
        ]
    }

# Endpoint to get similar claims
@app.get("/similar_claims/{claim_id}")
async def get_similar_claims(claim_id: str):
    # In a real implementation, this would use embeddings to find similar claims
    # Placeholder implementation returning mock data
    return {
        "similar_claims": [
            {"id": "CL006", "similarity": 0.92, "patient_id": "P11111", "provider_id": "PR678", "amount": 1280.50, "date": "2025-03-20", "fraud_score": 0.78},
            {"id": "CL007", "similarity": 0.85, "patient_id": "P22222", "provider_id": "PR678", "amount": 1150.25, "date": "2025-02-15", "fraud_score": 0.81},
            {"id": "CL008", "similarity": 0.79, "patient_id": "P33333", "provider_id": "PR456", "amount": 1350.00, "date": "2025-01-10", "fraud_score": 0.75}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
