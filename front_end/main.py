import streamlit as st
import landing
import login
import app

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "landing"

if "user" not in st.session_state:
    st.session_state.user = None

# Determine which page to show
def show_page():
    if st.session_state.page == "landing":
        landing.landing_page()
        
        # Check for navigation from landing page
        if st.session_state.get("navigate_to_login", False):
            st.session_state.page = "login"
            st.session_state.navigate_to_login = False
            st.rerun()
            
    elif st.session_state.page == "login":
        login.login_page()
        
        # Check if user is logged in
        if st.session_state.user and st.session_state.user.get("logged_in", False):
            st.session_state.page = "app"
            st.rerun()
            
    elif st.session_state.page == "app":
        # Check if user is logged in
        if not st.session_state.user or not st.session_state.user.get("logged_in", False):
            st.session_state.page = "login"
            st.rerun()
        else:
            app.main()
            
            # Handle logout
            if st.sidebar.button("Logout"):
                st.session_state.user = None
                st.session_state.page = "landing"
                st.rerun()

# Show appropriate page
show_page()
