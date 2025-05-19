import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import plotly.graph_objects as go
import json

def main():
    # Display user info in sidebar
    if st.session_state.user:
        st.sidebar.markdown(f"### Welcome, {st.session_state.user['name']}")
        st.sidebar.markdown(f"Role: {st.session_state.user['role'].capitalize()}")
        st.sidebar.markdown("---")
    
    # App title
    st.markdown("<h1 class='main-header'>Healthcare Fraud Detection System</h1>", unsafe_allow_html=True)

    # Add CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E88E5;
            text-align: center;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #0D47A1;
        }
        .card {
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            background-color: #f8f9fa;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .risk-high {
            color: #d32f2f;
            font-weight: bold;
        }
        .risk-medium {
            color: #ff9800;
            font-weight: bold;
        }
        .risk-low {
            color: #4caf50;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Submit New Claim", "Dashboard", "Claim History", "System Info"])

    # Function to get risk class
    def get_risk_class(score):
        if score >= 0.7:
            return "risk-high", "High Risk"
        elif score >= 0.4:
            return "risk-medium", "Medium Risk"
        else:
            return "risk-low", "Low Risk"

    # Function to fetch claim history (mock data for now)
    def get_claim_history():
        try:
            # In production, replace with actual API call
            # response = requests.get(HISTORY_ENDPOINT)
            # return response.json()
            
            # Mock data
            return {
                "claims": [
                    {"id": "CL001", "patient_id": "P12345", "provider_id": "PR678", "amount": 1250.50, "date": "2025-04-15", "fraud_score": 0.82},
                    {"id": "CL002", "patient_id": "P54321", "provider_id": "PR789", "amount": 450.75, "date": "2025-04-18", "fraud_score": 0.23},
                    {"id": "CL003", "patient_id": "P98765", "provider_id": "PR456", "amount": 2100.00, "date": "2025-04-25", "fraud_score": 0.56},
                    {"id": "CL004", "patient_id": "P12345", "provider_id": "PR123", "amount": 875.25, "date": "2025-05-02", "fraud_score": 0.18},
                    {"id": "CL005", "patient_id": "P45678", "provider_id": "PR678", "amount": 3200.50, "date": "2025-05-10", "fraud_score": 0.91}
                ]
            }
        except Exception as e:
            st.error(f"Error fetching claim history: {e}")
            return {"claims": []}

    # Function to get similar claims (mock data for now)
    def get_similar_claims(claim_id):
        try:
            # In production, replace with actual API call
            # response = requests.get(f"{SIMILAR_CLAIMS_ENDPOINT}/{claim_id}")
            # return response.json()
            
            # Mock data
            return {
                "similar_claims": [
                    {"id": "CL006", "similarity": 0.92, "patient_id": "P11111", "provider_id": "PR678", "amount": 1280.50, "date": "2025-03-20", "fraud_score": 0.78},
                    {"id": "CL007", "similarity": 0.85, "patient_id": "P22222", "provider_id": "PR678", "amount": 1150.25, "date": "2025-02-15", "fraud_score": 0.81},
                    {"id": "CL008", "similarity": 0.79, "patient_id": "P33333", "provider_id": "PR456", "amount": 1350.00, "date": "2025-01-10", "fraud_score": 0.75}
                ]
            }
        except Exception as e:
            st.error(f"Error fetching similar claims: {e}")
            return {"similar_claims": []}

    # Submit New Claim Page
    if page == "Submit New Claim":
        st.markdown("<h2 class='sub-header'>Submit New Healthcare Claim</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Patient Information")
            patient_id = st.text_input("Patient ID", placeholder="e.g., P12345")
            patient_first_name = st.text_input("First Name", placeholder="John")
            patient_last_name = st.text_input("Last Name", placeholder="Doe")
            patient_dob = st.date_input("Date of Birth", datetime.date(1970, 1, 1))
            patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Provider Information")
            provider_id = st.text_input("Provider ID", placeholder="e.g., PR678")
            provider_name = st.text_input("Provider Name", placeholder="General Hospital")
            provider_type = st.selectbox("Provider Type", ["Hospital", "Clinic", "Physician", "Specialist", "Laboratory"])
            provider_state = st.selectbox("State", ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"])
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Claim Details")
        col3, col4, col5 = st.columns(3)
        
        with col3:
            claim_date = st.date_input("Claim Date", datetime.date.today())
            claim_amount = st.number_input("Claim Amount ($)", min_value=0.0, step=10.0)
        
        with col4:
            service_code = st.text_input("HCPCS Code", placeholder="e.g., 99213")
            place_of_service = st.selectbox("Place of Service", ["Office", "Inpatient Hospital", "Outpatient Hospital", "Emergency Room", "Ambulatory Surgical Center", "Other"])
        
        with col5:
            diagnosis_code = st.text_input("Diagnosis Code (ICD-10)", placeholder="e.g., J20.9")
            procedure_code = st.text_input("Procedure Code", placeholder="e.g., 31255")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("Submit Claim for Fraud Detection", use_container_width=True):
            if not patient_id or not provider_id or claim_amount <= 0:
                st.error("Please fill in all required fields: Patient ID, Provider ID, and Claim Amount")
            else:
                with st.spinner("Analyzing claim for fraud risk..."):
                    # Prepare data payload
                    data = {
                        "patient_id": patient_id,
                        "provider_id": provider_id,
                        "claim_amount": claim_amount,
                        "claim_date": claim_date.strftime("%Y-%m-%d"),
                        "patient_info": {
                            "first_name": patient_first_name,
                            "last_name": patient_last_name,
                            "dob": patient_dob.strftime("%Y-%m-%d"),
                            "gender": patient_gender
                        },
                        "provider_info": {
                            "name": provider_name,
                            "type": provider_type,
                            "state": provider_state
                        },
                        "claim_details": {
                            "service_code": service_code,
                            "place_of_service": place_of_service,
                            "diagnosis_code": diagnosis_code,
                            "procedure_code": procedure_code
                        }
                    }
                    
                    try:
                        # In production, replace with actual API call
                        # response = requests.post(PREDICT_ENDPOINT, json=data)
                        # if response.status_code == 200:
                        #     result = response.json()
                        
                        # Mock response for demonstration
                        import time
                        time.sleep(2)  # Simulate API delay
                        
                        # Generate a fraud score based on some simple rules for demo
                        fraud_score = 0.1  # Base score
                        if claim_amount > 2000:
                            fraud_score += 0.3
                        if provider_type == "Specialist":
                            fraud_score += 0.2
                        if provider_state in ["FL", "CA", "NY"]:
                            fraud_score += 0.1
                        
                        # Add some randomness
                        fraud_score += np.random.uniform(-0.1, 0.1)
                        fraud_score = max(0, min(1, fraud_score))  # Ensure between 0 and 1
                        
                        result = {
                            "claim_id": f"CL{np.random.randint(1000, 9999)}",
                            "fraud_score": fraud_score,
                            "contributing_factors": [
                                {"factor": "Claim amount", "impact": 0.4 if claim_amount > 2000 else 0.1},
                                {"factor": "Provider type", "impact": 0.3 if provider_type == "Specialist" else 0.1},
                                {"factor": "Geographic region", "impact": 0.2 if provider_state in ["FL", "CA", "NY"] else 0.1},
                                {"factor": "Service code", "impact": 0.1}
                            ]
                        }
                        
                        # Display results
                        risk_class, risk_label = get_risk_class(result["fraud_score"])
                        
                        st.markdown(f"<div class='card'>", unsafe_allow_html=True)
                        st.subheader("Fraud Detection Results")
                        
                        col_res1, col_res2 = st.columns([1, 2])
                        
                        with col_res1:
                            st.markdown(f"<h3>Claim ID: {result['claim_id']}</h3>", unsafe_allow_html=True)
                            st.markdown(f"<h2 class='{risk_class}'>Fraud Risk Score: {result['fraud_score']:.2f}</h2>", unsafe_allow_html=True)
                            st.markdown(f"<h3 class='{risk_class}'>Risk Level: {risk_label}</h3>", unsafe_allow_html=True)
                        
                        with col_res2:
                            # Create a horizontal bar chart for contributing factors
                            factors_df = pd.DataFrame(result["contributing_factors"])
                            fig = px.bar(
                                factors_df, 
                                x="impact", 
                                y="factor", 
                                orientation='h',
                                title="Contributing Factors to Fraud Risk",
                                labels={"impact": "Impact on Risk Score", "factor": "Factor"},
                                color="impact",
                                color_continuous_scale=["green", "yellow", "red"]
                            )
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Show similar claims
                        similar_claims = get_similar_claims(result["claim_id"])
                        
                        if similar_claims["similar_claims"]:
                            st.markdown("<div class='card'>", unsafe_allow_html=True)
                            st.subheader("Similar Claims")
                            
                            similar_df = pd.DataFrame(similar_claims["similar_claims"])
                            similar_df["risk_level"] = similar_df["fraud_score"].apply(lambda x: get_risk_class(x)[1])
                            
                            st.dataframe(
                                similar_df[["id", "similarity", "patient_id", "provider_id", "amount", "date", "fraud_score", "risk_level"]],
                                column_config={
                                    "id": "Claim ID",
                                    "similarity": st.column_config.ProgressColumn("Similarity", format="%.2f", min_value=0, max_value=1),
                                    "patient_id": "Patient ID",
                                    "provider_id": "Provider ID",
                                    "amount": st.column_config.NumberColumn("Amount ($)", format="$%.2f"),
                                    "date": "Date",
                                    "fraud_score": st.column_config.ProgressColumn("Fraud Score", format="%.2f", min_value=0, max_value=1),
                                    "risk_level": "Risk Level"
                                },
                                use_container_width=True
                            )
                            st.markdown("</div>", unsafe_allow_html=True)
                    
                    except Exception as e:
                        st.error(f"Error connecting to backend API: {e}")

    # Dashboard Page
    elif page == "Dashboard":
        st.markdown("<h2 class='sub-header'>Fraud Detection Dashboard</h2>", unsafe_allow_html=True)
        
        # Get mock claim history data
        history_data = get_claim_history()
        claims_df = pd.DataFrame(history_data["claims"])
        
        # Add risk level
        claims_df["risk_level"] = claims_df["fraud_score"].apply(lambda x: get_risk_class(x)[1])
        
        # Convert date to datetime
        claims_df["date"] = pd.to_datetime(claims_df["date"])
        
        # Dashboard metrics
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            st.metric("Total Claims", len(claims_df))
        
        with col_m2:
            high_risk = len(claims_df[claims_df["fraud_score"] >= 0.7])
            st.metric("High Risk Claims", high_risk, f"{high_risk/len(claims_df)*100:.1f}%")
        
        with col_m3:
            avg_score = claims_df["fraud_score"].mean()
            st.metric("Average Risk Score", f"{avg_score:.2f}")
        
        with col_m4:
            total_amount = claims_df["amount"].sum()
            st.metric("Total Claim Amount", f"${total_amount:,.2f}")
        
        # Create two columns for charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            # Risk distribution pie chart
            risk_counts = claims_df["risk_level"].value_counts().reset_index()
            risk_counts.columns = ["Risk Level", "Count"]
            
            fig1 = px.pie(
                risk_counts, 
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
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            # Risk score vs claim amount scatter plot
            fig2 = px.scatter(
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
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Timeline chart
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Fraud Risk Timeline")
        
        # Group by date and calculate average fraud score
        timeline_df = claims_df.groupby(claims_df["date"].dt.date).agg({
            "fraud_score": "mean",
            "amount": "sum",
            "id": "count"
        }).reset_index()
        timeline_df.columns = ["date", "avg_fraud_score", "total_amount", "claim_count"]
        
        # Create a dual-axis chart
        fig3 = go.Figure()
        
        # Add fraud score line
        fig3.add_trace(
            go.Scatter(
                x=timeline_df["date"],
                y=timeline_df["avg_fraud_score"],
                name="Avg. Fraud Score",
                line=dict(color="#1E88E5", width=3)
            )
        )
        
        # Add claim count bars
        fig3.add_trace(
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
        fig3.update_layout(
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
        
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Provider risk table
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Provider Risk Analysis")
        
        # Group by provider
        provider_df = claims_df.groupby("provider_id").agg({
            "fraud_score": ["mean", "max"],
            "amount": ["sum", "mean"],
            "id": "count"
        }).reset_index()
        
        provider_df.columns = ["provider_id", "avg_fraud_score", "max_fraud_score", "total_amount", "avg_amount", "claim_count"]
        provider_df["risk_level"] = provider_df["avg_fraud_score"].apply(lambda x: get_risk_class(x)[1])
        provider_df = provider_df.sort_values("avg_fraud_score", ascending=False)
        
        st.dataframe(
            provider_df,
            column_config={
                "provider_id": "Provider ID",
                "avg_fraud_score": st.column_config.ProgressColumn("Avg. Fraud Score", format="%.2f", min_value=0, max_value=1),
                "max_fraud_score": st.column_config.ProgressColumn("Max Fraud Score", format="%.2f", min_value=0, max_value=1),
                "total_amount": st.column_config.NumberColumn("Total Amount ($)", format="$%.2f"),
                "avg_amount": st.column_config.NumberColumn("Avg. Amount ($)", format="$%.2f"),
                "claim_count": "Claim Count",
                "risk_level": "Risk Level"
            },
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Claim History Page
    elif page == "Claim History":
        st.markdown("<h2 class='sub-header'>Claim History</h2>", unsafe_allow_html=True)
        
        # Get claim history
        history_data = get_claim_history()
        claims_df = pd.DataFrame(history_data["claims"])
        
        # Add risk level
        claims_df["risk_level"] = claims_df["fraud_score"].apply(lambda x: get_risk_class(x)[1])
        
        # Convert date to datetime
        claims_df["date"] = pd.to_datetime(claims_df["date"])
        
        # Filters
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            risk_filter = st.multiselect(
                "Risk Level",
                options=["High Risk", "Medium Risk", "Low Risk"],
                default=["High Risk", "Medium Risk", "Low Risk"]
            )
        
        with col_f2:
            min_date, max_date = claims_df["date"].min().date(), claims_df["date"].max().date()
            date_range = st.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
        
        with col_f3:
            provider_filter = st.multiselect(
                "Provider ID",
                options=claims_df["provider_id"].unique().tolist(),
                default=claims_df["provider_id"].unique().tolist()
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Apply filters
        filtered_df = claims_df.copy()
        
        if risk_filter:
            filtered_df = filtered_df[filtered_df["risk_level"].isin(risk_filter)]
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = filtered_df[(filtered_df["date"].dt.date >= start_date) & (filtered_df["date"].dt.date <= end_date)]
        
        if provider_filter:
            filtered_df = filtered_df[filtered_df["provider_id"].isin(provider_filter)]
        
        # Display filtered claims
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader(f"Claims ({len(filtered_df)})")
        
        if not filtered_df.empty:
            st.dataframe(
                filtered_df,
                column_config={
                    "id": "Claim ID",
                    "patient_id": "Patient ID",
                    "provider_id": "Provider ID",
                    "amount": st.column_config.NumberColumn("Amount ($)", format="$%.2f"),
                    "date": "Date",
                    "fraud_score": st.column_config.ProgressColumn("Fraud Score", format="%.2f", min_value=0, max_value=1),
                    "risk_level": "Risk Level"
                },
                use_container_width=True
            )
            
            # Allow user to select a claim for detailed view
            selected_claim_id = st.selectbox("Select a claim for detailed view", filtered_df["id"].tolist())
            
            if selected_claim_id:
                selected_claim = filtered_df[filtered_df["id"] == selected_claim_id].iloc[0]
                
                st.subheader(f"Detailed View: Claim {selected_claim_id}")
                
                col_d1, col_d2 = st.columns(2)
                
                with col_d1:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.write("**Claim Information**")
                    st.write(f"**Claim ID:** {selected_claim['id']}")
                    st.write(f"**Date:** {selected_claim['date'].strftime('%Y-%m-%d')}")
                    st.write(f"**Amount:** ${selected_claim['amount']:.2f}")
                    
                    risk_class, risk_label = get_risk_class(selected_claim["fraud_score"])
                    st.markdown(f"**Fraud Score:** <span class='{risk_class}'>{selected_claim['fraud_score']:.2f}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Risk Level:** <span class='{risk_class}'>{risk_label}</span>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col_d2:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.write("**Entity Information**")
                    st.write(f"**Patient ID:** {selected_claim['patient_id']}")
                    st.write(f"**Provider ID:** {selected_claim['provider_id']}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Get similar claims
                similar_claims = get_similar_claims(selected_claim_id)
                
                if similar_claims["similar_claims"]:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.subheader("Similar Claims")
                    
                    similar_df = pd.DataFrame(similar_claims["similar_claims"])
                    similar_df["risk_level"] = similar_df["fraud_score"].apply(lambda x: get_risk_class(x)[1])
                    
                    st.dataframe(
                        similar_df[["id", "similarity", "patient_id", "provider_id", "amount", "date", "fraud_score", "risk_level"]],
                        column_config={
                            "id": "Claim ID",
                            "similarity": st.column_config.ProgressColumn("Similarity", format="%.2f", min_value=0, max_value=1),
                            "patient_id": "Patient ID",
                            "provider_id": "Provider ID",
                            "amount": st.column_config.NumberColumn("Amount ($)", format="$%.2f"),
                            "date": "Date",
                            "fraud_score": st.column_config.ProgressColumn("Fraud Score", format="%.2f", min_value=0, max_value=1),
                            "risk_level": "Risk Level"
                        },
                        use_container_width=True
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No claims match the selected filters.")
        
        st.markdown("</div>", unsafe_allow_html=True)

    # System Info Page
    else:
        st.markdown("<h2 class='sub-header'>System Information</h2>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("About the Fraud Detection System")
        st.write("""
        This healthcare fraud detection system uses Graph Neural Networks (GNNs) to identify potentially fraudulent healthcare claims. 
        The system analyzes relationships between patients, providers, and claims to detect anomalous patterns that may indicate fraud.
        """)
        
        st.write("""
        **Key Features:**
        - Unsupervised learning approach using Graph Autoencoders
        - Real-time fraud risk scoring for new claims
        - Identification of similar historical claims for context
        - Comprehensive dashboard for monitoring fraud trends
        - Detailed claim history with filtering capabilities
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Model Architecture")
        
        col_a1, col_a2 = st.columns([1, 2])
        
        with col_a1:
            st.write("""
            The system uses a Graph Neural Network (GNN) architecture with the following components:
            
            - **Graph Construction**: Builds a heterogeneous graph with patients and providers as nodes
            - **Feature Encoding**: Processes demographic and claim information
            - **Graph Convolutional Network (GCN)**: Learns node embeddings
            - **Graph Autoencoder (GAE)**: Detects anomalies in an unsupervised manner
            """)
        
        with col_a2:
            # Placeholder for architecture diagram
            st.image("https://miro.medium.com/v2/resize:fit:1400/1*aTzHYpoLF9JBAaKkK_3hwg.png", caption="GNN Architecture (Example)")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("System Status")
        
        col_s1, col_s2, col_s3 = st.columns(3)
        
        with col_s1:
            st.metric("Backend Status", "Online", "Normal")
        
        with col_s2:
            st.metric("Model Version", "v1.2.3")
        
        with col_s3:
            st.metric("Last Updated", "2025-05-15")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Documentation")
        
        st.write("""
        For more information on using this system, please refer to the following resources:
        
        - User Guide: How to submit claims and interpret results
        - Technical Documentation: System architecture and API reference
        - Model Documentation: Details on the GNN model and training process
        """)
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 30px; padding: 10px; font-size: 0.8rem; color: #666;">
        Healthcare Fraud Detection System © 2025 | Powered by Graph Neural Networks
    </div>
    """, unsafe_allow_html=True)
