import streamlit as st

def risk_card(claim_id, fraud_score, risk_class, risk_label):
    """Display a card with risk information"""
    st.markdown(f"<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<h3>Claim ID: {claim_id}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h2 class='{risk_class}'>Fraud Risk Score: {fraud_score:.2f}</h2>", unsafe_allow_html=True)
    st.markdown(f"<h3 class='{risk_class}'>Risk Level: {risk_label}</h3>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def claim_detail_card(claim, title="Claim Information"):
    """Display a card with claim details"""
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write(f"**{title}**")
    
    for key, value in claim.items():
        if key not in ["fraud_score", "risk_level", "risk_class"]:
            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
    
    if "fraud_score" in claim and "risk_class" in claim:
        st.markdown(f"**Fraud Score:** <span class='{claim['risk_class']}'>{claim['fraud_score']:.2f}</span>", unsafe_allow_html=True)
    
    if "risk_level" in claim and "risk_class" in claim:
        st.markdown(f"**Risk Level:** <span class='{claim['risk_class']}'>{claim['risk_level']}</span>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
