import streamlit as st
import time

def login_page():
    # Set page configuration
    st.set_page_config(
        page_title="Login - Healthcare Fraud Detection System",
        page_icon="🏥",
        layout="centered"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 2rem;
        }
        .login-container {
            max-width: 400px;
            margin: 0 auto;
            padding: 30px;
            border-radius: 10px;
            background-color: #f8f9fa;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .form-header {
            font-size: 1.5rem;
            color: #0D47A1;
            text-align: center;
            margin-bottom: 20px;
        }
        .stButton>button {
            background-color: #1E88E5;
            color: white;
            font-weight: bold;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #0D47A1;
        }
        .forgot-password {
            text-align: right;
            margin-top: 10px;
            font-size: 0.8rem;
        }
        .signup-prompt {
            text-align: center;
            margin-top: 20px;
            font-size: 0.9rem;
        }
        .footer {
            text-align: center;
            margin-top: 50px;
            padding: 20px;
            font-size: 0.8rem;
            color: #666;
        }
        .error-msg {
            color: #d32f2f;
            font-size: 0.9rem;
            text-align: center;
            margin-top: 10px;
        }
        .success-msg {
            color: #4caf50;
            font-size: 0.9rem;
            text-align: center;
            margin-top: 10px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">Healthcare Fraud Detection System</h1>', unsafe_allow_html=True)
    
    # Login container
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="form-header">Log In</h2>', unsafe_allow_html=True)
    
    # Login form
    username = st.text_input("Username", placeholder="Enter your username")
    password = st.text_input("Password", type="password", placeholder="Enter your password")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        remember_me = st.checkbox("Remember me")
    with col2:
        st.markdown('<div class="forgot-password"><a href="#">Forgot password?</a></div>', unsafe_allow_html=True)
    
    login_button = st.button("Log In", use_container_width=True)
    
    # Mock user data (in production, use a secure database)
    users = {
        "admin": {"password": "admin123", "name": "Administrator", "role": "admin"},
        "analyst": {"password": "analyst123", "name": "Fraud Analyst", "role": "analyst"},
        "user": {"password": "user123", "name": "Regular User", "role": "user"}
    }
    
    # Handle login
    if login_button:
        if not username or not password:
            st.markdown('<div class="error-msg">Please enter both username and password.</div>', unsafe_allow_html=True)
        elif username in users and users[username]["password"] == password:
            # Success, save session info
            user_data = {
                "username": username,
                "name": users[username]["name"],
                "role": users[username]["role"],
                "logged_in": True
            }
            
            # In a real app, you'd use proper session management
            # For this example, we'll use st.session_state
            st.session_state.user = user_data
            
            st.markdown('<div class="success-msg">Login successful! Redirecting to dashboard...</div>', unsafe_allow_html=True)
            
            # Simulate redirect
            time.sleep(2)
            st.rerun()
        else:
            st.markdown('<div class="error-msg">Invalid username or password. Please try again.</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="signup-prompt">Don\'t have an account? <a href="#">Sign up</a></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    st.markdown('Healthcare Fraud Detection System © 2025 | Powered by Graph Neural Networks', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
