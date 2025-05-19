import pandas as pd
import numpy as np

def get_risk_class(score):
    """Determine risk class and label based on fraud score"""
    if score >= 0.7:
        return "risk-high", "High Risk"
    elif score >= 0.4:
        return "risk-medium", "Medium Risk"
    else:
        return "risk-low", "Low Risk"

def process_claims_data(claims):
    """Process raw claims data into a pandas DataFrame with additional columns"""
    df = pd.DataFrame(claims)
    
    # Add risk level and class based on fraud score
    df["risk_level"] = df["fraud_score"].apply(lambda x: get_risk_class(x)[1])
    df["risk_class"] = df["fraud_score"].apply(lambda x: get_risk_class(x)[0])
    
    # Convert date to datetime if it's not already
    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    
    return df

def prepare_timeline_data(claims_df):
    """Prepare data for timeline visualization"""
    # Group by date and calculate metrics
    timeline_df = claims_df.groupby(claims_df["date"].dt.date).agg({
        "fraud_score": "mean",
        "amount": "sum",
        "id": "count"
    }).reset_index()
    
    timeline_df.columns = ["date", "avg_fraud_score", "total_amount", "claim_count"]
    return timeline_df

def prepare_provider_data(claims_df):
    """Prepare data for provider risk analysis"""
    # Group by provider
    provider_df = claims_df.groupby("provider_id").agg({
        "fraud_score": ["mean", "max"],
        "amount": ["sum", "mean"],
        "id": "count"
    }).reset_index()
    
    provider_df.columns = ["provider_id", "avg_fraud_score", "max_fraud_score", "total_amount", "avg_amount", "claim_count"]
    provider_df["risk_level"] = provider_df["avg_fraud_score"].apply(lambda x: get_risk_class(x)[1])
    provider_df["risk_class"] = provider_df["avg_fraud_score"].apply(lambda x: get_risk_class(x)[0])
    provider_df = provider_df.sort_values("avg_fraud_score", ascending=False)
    
    return provider_df
