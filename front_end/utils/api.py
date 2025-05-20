# import requests
# import json
# import pandas as pd
# import numpy as np

# # Define API endpoints
# API_BASE_URL = "http://localhost:8000"
# PREDICT_ENDPOINT = f"{API_BASE_URL}/predict_fraud"
# HISTORY_ENDPOINT = f"{API_BASE_URL}/claim_history"
# SIMILAR_CLAIMS_ENDPOINT = f"{API_BASE_URL}/similar_claims"

# def get_claim_history():
#     """Fetch claim history from the backend API"""
#     try:
#         # In production, replace with actual API call
#         # response = requests.get(HISTORY_ENDPOINT)
#         # return response.json()
        
#         # Mock data for development
#         return {
#             "claims": [
#                 {"id": "CL001", "patient_id": "P12345", "provider_id": "PR678", "amount": 1250.50, "date": "2025-04-15", "fraud_score": 0.82},
#                 {"id": "CL002", "patient_id": "P54321", "provider_id": "PR789", "amount": 450.75, "date": "2025-04-18", "fraud_score": 0.23},
#                 {"id": "CL003", "patient_id": "P98765", "provider_id": "PR456", "amount": 2100.00, "date": "2025-04-25", "fraud_score": 0.56},
#                 {"id": "CL004", "patient_id": "P12345", "provider_id": "PR123", "amount": 875.25, "date": "2025-05-02", "fraud_score": 0.18},
#                 {"id": "CL005", "patient_id": "P45678", "provider_id": "PR678", "amount": 3200.50, "date": "2025-05-10", "fraud_score": 0.91}
#             ]
#         }
#     except Exception as e:
#         print(f"Error fetching claim history: {e}")
#         return {"claims": []}

# def get_similar_claims(claim_id):
#     """Fetch similar claims from the backend API"""
#     try:
#         # In production, replace with actual API call
#         # response = requests.get(f"{SIMILAR_CLAIMS_ENDPOINT}/{claim_id}")
#         # return response.json()
        
#         # Mock data for development
#         return {
#             "similar_claims": [
#                 {"id": "CL006", "similarity": 0.92, "patient_id": "P11111", "provider_id": "PR678", "amount": 1280.50, "date": "2025-03-20", "fraud_score": 0.78},
#                 {"id": "CL007", "similarity": 0.85, "patient_id": "P22222", "provider_id": "PR678", "amount": 1150.25, "date": "2025-02-15", "fraud_score": 0.81},
#                 {"id": "CL008", "similarity": 0.79, "patient_id": "P33333", "provider_id": "PR456", "amount": 1350.00, "date": "2025-01-10", "fraud_score": 0.75}
#             ]
#         }
#     except Exception as e:
#         print(f"Error fetching similar claims: {e}")
#         return {"similar_claims": []}

# def predict_fraud(claim_data):
#     """Submit a claim for fraud prediction"""
#     try:
#         # In production, replace with actual API call
#         # response = requests.post(PREDICT_ENDPOINT, json=claim_data)
#         # return response.json()
        
#         # Mock prediction for development
#         import time
#         time.sleep(2)  # Simulate API delay
        
#         # Generate a fraud score based on some simple rules
#         fraud_score = 0.1  # Base score
        
#         if claim_data.get("claim_amount", 0) > 2000:
#             fraud_score += 0.3
            
#         if claim_data.get("provider_info", {}).get("type") == "Specialist":
#             fraud_score += 0.2
            
#         if claim_data.get("provider_info", {}).get("state") in ["FL", "CA", "NY"]:
#             fraud_score += 0.1
        
#         # Add some randomness
#         fraud_score += np.random.uniform(-0.1, 0.1)
#         fraud_score = max(0, min(1, fraud_score))  # Ensure between 0 and 1
        
#         return {
#             "claim_id": f"CL{np.random.randint(1000, 9999)}",
#             "fraud_score": fraud_score,
#             "contributing_factors": [
#                 {"factor": "Claim amount", "impact": 0.4 if claim_data.get("claim_amount", 0) > 2000 else 0.1},
#                 {"factor": "Provider type", "impact": 0.3 if claim_data.get("provider_info", {}).get("type") == "Specialist" else 0.1},
#                 {"factor": "Geographic region", "impact": 0.2 if claim_data.get("provider_info", {}).get("state") in ["FL", "CA", "NY"] else 0.1},
#                 {"factor": "Service code", "impact": 0.1}
#             ]
#         }
#     except Exception as e:
#         print(f"Error predicting fraud: {e}")
#         return {"error": str(e)}


import requests
import json
import pandas as pd
import numpy as np

# Define API endpoints
API_BASE_URL = "http://localhost:8000"
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict_fraud"
HISTORY_ENDPOINT = f"{API_BASE_URL}/claim_history"
SIMILAR_CLAIMS_ENDPOINT = f"{API_BASE_URL}/similar_claims"

def get_claim_history():
    """Fetch claim history from the backend API"""
    try:
        response = requests.get(HISTORY_ENDPOINT)
        return response.json()
    except Exception as e:
        print(f"Error fetching claim history: {e}")
        # Return mock data as fallback
        return {
            "claims": [
                {"id": "CL001", "patient_id": "P12345", "provider_id": "PR678", "amount": 1250.50, "date": "2025-04-15", "fraud_score": 0.82},
                {"id": "CL002", "patient_id": "P54321", "provider_id": "PR789", "amount": 450.75, "date": "2025-04-18", "fraud_score": 0.23},
                {"id": "CL003", "patient_id": "P98765", "provider_id": "PR456", "amount": 2100.00, "date": "2025-04-25", "fraud_score": 0.56},
                {"id": "CL004", "patient_id": "P12345", "provider_id": "PR123", "amount": 875.25, "date": "2025-05-02", "fraud_score": 0.18},
                {"id": "CL005", "patient_id": "P45678", "provider_id": "PR678", "amount": 3200.50, "date": "2025-05-10", "fraud_score": 0.91}
            ]
        }

def get_similar_claims(claim_id):
    """Fetch similar claims from the backend API"""
    try:
        response = requests.get(f"{SIMILAR_CLAIMS_ENDPOINT}/{claim_id}")
        return response.json()
    except Exception as e:
        print(f"Error fetching similar claims: {e}")
        # Return mock data as fallback
        return {
            "similar_claims": [
                {"id": "CL006", "similarity": 0.92, "patient_id": "P11111", "provider_id": "PR678", "amount": 1280.50, "date": "2025-03-20", "fraud_score": 0.78},
                {"id": "CL007", "similarity": 0.85, "patient_id": "P22222", "provider_id": "PR678", "amount": 1150.25, "date": "2025-02-15", "fraud_score": 0.81},
                {"id": "CL008", "similarity": 0.79, "patient_id": "P33333", "provider_id": "PR456", "amount": 1350.00, "date": "2025-01-10", "fraud_score": 0.75}
            ]
        }

def predict_fraud(claim_data):
    """Submit a claim for fraud prediction"""
    try:
        response = requests.post(PREDICT_ENDPOINT, json=claim_data)
        return response.json()
    except Exception as e:
        print(f"Error predicting fraud: {e}")
        # Generate mock prediction as fallback
        import time
        time.sleep(2)  # Simulate API delay
        
        # Generate a fraud score based on some simple rules
        fraud_score = 0.1  # Base score
        
        if claim_data.get("claim_amount", 0) > 2000:
            fraud_score += 0.3
            
        if claim_data.get("provider_info", {}).get("type") == "Specialist":
            fraud_score += 0.2
            
        if claim_data.get("provider_info", {}).get("state") in ["FL", "CA", "NY"]:
            fraud_score += 0.1
        
        # Add some randomness
        fraud_score += np.random.uniform(-0.1, 0.1)
        fraud_score = max(0, min(1, fraud_score))  # Ensure between 0 and 1
        
        return {
            "claim_id": f"CL{np.random.randint(1000, 9999)}",
            "fraud_score": fraud_score,
            "contributing_factors": [
                {"factor": "Claim amount", "impact": 0.4 if claim_data.get("claim_amount", 0) > 2000 else 0.1},
                {"factor": "Provider type", "impact": 0.3 if claim_data.get("provider_info", {}).get("type") == "Specialist" else 0.1},
                {"factor": "Geographic region", "impact": 0.2 if claim_data.get("provider_info", {}).get("state") in ["FL", "CA", "NY"] else 0.1},
                {"factor": "Service code", "impact": 0.1}
            ]
        }
