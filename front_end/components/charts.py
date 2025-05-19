import plotly.express as px
import plotly.graph_objects as go

def create_risk_distribution_chart(risk_data):
    """Create a pie chart showing risk distribution"""
    fig = px.pie(
        risk_data, 
        values="Count", 
        names="Risk Level",
        title="Distribution of Risk Levels",
        color="Risk Level",
        color_discrete_map={
            "High Risk": "#d32f2f",
            "Medium Risk": "#ff9800",
            "Low Risk": "#4caf50"
        }
    )
    return fig

def create_risk_vs_amount_chart(claims_df):
    """Create a scatter plot of risk score vs claim amount"""
    fig = px.scatter(
        claims_df,
        x="amount",
        y="fraud_score",
        color="risk_level",
        size="amount",
        hover_data=["id", "patient_id", "provider_id", "date"],
        title="Fraud Risk Score vs. Claim Amount",
        labels={"amount": "Claim Amount ($)", "fraud_score": "Fraud Risk Score"},
        color_discrete_map={
            "High Risk": "#d32f2f",
            "Medium Risk": "#ff9800",
            "Low Risk": "#4caf50"
        }
    )
    return fig

def create_timeline_chart(timeline_df):
    """Create a dual-axis timeline chart"""
    fig = go.Figure()
    
    # Add fraud score line
    fig.add_trace(
        go.Scatter(
            x=timeline_df["date"],
            y=timeline_df["avg_fraud_score"],
            name="Avg. Fraud Score",
            line=dict(color="#1E88E5", width=3)
        )
    )
    
    # Add claim count bars
    fig.add_trace(
        go.Bar(
            x=timeline_df["date"],
            y=timeline_df["claim_count"],
            name="Claim Count",
            marker=dict(color="#90CAF9"),
            opacity=0.7,
            yaxis="y2"
        )
    )
    
    # Update layout for dual y-axes
    fig.update_layout(
        title="Average Fraud Score and Claim Count Over Time",
        xaxis=dict(title="Date"),
        yaxis=dict(
            title=dict(
                text="Average Fraud Score",
                font=dict(color="#1E88E5")
            ),
            tickfont=dict(color="#1E88E5")
        ),
        yaxis2=dict(
            title=dict(
                text="Claim Count",
                font=dict(color="#90CAF9")
            ),
            tickfont=dict(color="#90CAF9"),
            anchor="x",
            overlaying="y",
            side="right"
        ),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig
