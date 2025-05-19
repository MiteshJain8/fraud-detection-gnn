import streamlit as st

def landing_page():
    # Set page configuration
    st.set_page_config(
        page_title="Healthcare Fraud Detection System",
        page_icon="🏥",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 0;
            padding-bottom: 0;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #0D47A1;
            text-align: center;
            margin-top: 0;
            padding-top: 0;
            margin-bottom: 2rem;
        }
        .feature-card {
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            background-color: #f8f9fa;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        }
        .feature-icon {
            font-size: 2.5rem;
            text-align: center;
            margin-bottom: 10px;
            color: #1E88E5;
        }
        .feature-title {
            font-size: 1.2rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 10px;
            color: #0D47A1;
        }
        .feature-text {
            text-align: center;
            color: #333;
        }
        .cta-button {
            background-color: #1E88E5;
            color: white;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 5px;
            text-align: center;
            display: block;
            margin: 0 auto;
            width: 200px;
            text-decoration: none;
            transition: background-color 0.3s ease;
        }
        .cta-button:hover {
            background-color: #0D47A1;
        }
        .footer {
            text-align: center;
            margin-top: 50px;
            padding: 20px;
            font-size: 0.8rem;
            color: #666;
            border-top: 1px solid #eee;
        }
        .hero-section {
            padding: 40px 20px;
            background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
            border-radius: 10px;
            margin-bottom: 40px;
            text-align: center;
        }
        .section-title {
            font-size: 2rem;
            color: #0D47A1;
            text-align: center;
            margin-bottom: 30px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Hero Section
    st.markdown('<div class="hero-section">', unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">Healthcare Fraud Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Powered by Graph Neural Networks</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://img.freepik.com/free-vector/gradient-network-connection-background_23-2148865392.jpg", use_container_width=True)
    
    # Button to navigate to login page
    if st.button("Get Started"):
        st.session_state.navigate_to_login = True
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Features Section
    st.markdown('<h2 class="section-title">Key Features</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<div class="feature-icon">🔍</div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-title">Advanced Detection</div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-text">Utilizes state-of-the-art Graph Neural Networks to detect complex fraud patterns</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<div class="feature-icon">⚡</div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-title">Real-time Analysis</div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-text">Process claims instantly and get immediate fraud risk assessments</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<div class="feature-icon">📊</div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-title">Interactive Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-text">Comprehensive analytics and visualizations to track fraud trends</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<div class="feature-icon">🧠</div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-title">Explainable AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-text">Understand why claims are flagged with transparent risk factors</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # How It Works Section
    st.markdown('<h2 class="section-title">How It Works</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image("https://miro.medium.com/v2/resize:fit:1400/1*aTzHYpoLF9JBAaKkK_3hwg.png", caption="GNN Architecture", use_container_width=True)
    
    with col2:
        st.markdown("""
        ### Graph-Based Fraud Detection
        
        Our system uses a sophisticated Graph Neural Network to analyze relationships between patients, providers, and claims:
        
        1. **Data Integration**: Healthcare claims data is processed and structured
        2. **Graph Construction**: A heterogeneous graph is built connecting related entities
        3. **Feature Learning**: The GNN learns patterns from historical data
        4. **Anomaly Detection**: Unusual patterns are identified as potential fraud
        5. **Risk Scoring**: Each claim receives a fraud risk score with contributing factors
        
        This approach allows us to detect complex fraud schemes that traditional methods might miss.
        """)
    
    # Testimonials Section
    st.markdown('<h2 class="section-title">What Our Users Say</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        > "This system has revolutionized our fraud detection capabilities. We've seen a 40% increase in fraud identification with a significant reduction in false positives."
        
        **- Sarah Johnson, Healthcare Analytics Manager**
        """)
    
    with col2:
        st.markdown("""
        > "The interactive dashboard makes it easy to track fraud trends and focus our investigation resources where they're needed most."
        
        **- Michael Chen, Compliance Officer**
        """)
    
    with col3:
        st.markdown("""
        > "Implementation was smooth and the results were immediate. The explainable AI feature helps us understand why claims are flagged."
        
        **- Dr. Emily Rodriguez, Medical Director**
        """)
    
    # Call to Action
    st.markdown('<div class="hero-section" style="margin-top: 40px;">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title" style="margin-bottom: 20px;">Ready to Get Started?</h2>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; margin-bottom: 30px;">Join healthcare organizations across the country in leveraging advanced AI to combat fraud.</p>', unsafe_allow_html=True)
    
    # Button to navigate to login page
    if st.button("Log In Now"):
        st.session_state.navigate_to_login = True
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    st.markdown('Healthcare Fraud Detection System © 2025 | Powered by Graph Neural Networks', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
