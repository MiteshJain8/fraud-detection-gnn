import streamlit as st

def show_sidebar(user=None):
    """Display the sidebar with navigation and user info"""
    st.sidebar.title("Navigation")
    
    # Show user info if logged in
    if user:
        st.sidebar.markdown(f"### Welcome, {user['name']}")
        st.sidebar.markdown(f"Role: {user['role'].capitalize()}")
        st.sidebar.markdown("---")
    
    # Navigation options
    page = st.sidebar.radio("Go to", ["Submit New Claim", "Dashboard", "Claim History", "System Info"])
    
    return page
