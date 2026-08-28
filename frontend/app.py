# frontend/app.py
import streamlit as st
import requests

# Set page config for a wide, clean look
st.set_page_config(page_title="Churn Intelligence", page_icon="🔥", layout="centered")

st.title("🔥 Customer Churn Intelligence")
st.markdown("Enter a Stripe Customer ID to predict their churn risk in real-time.")

# Simple input form
with st.form("prediction_form"):
    customer_id = st.text_input("Customer ID", placeholder="e.g., cus_mock_500")
    submitted = st.form_submit_button("Predict Churn Risk")

if submitted:
    if not customer_id:
        st.warning("Please enter a Customer ID.")
    else:
        with st.spinner("Querying database and running ML model..."):
            try:
                # Call our FastAPI backend
                response = requests.get(f"http://localhost:8000/predictions/churn/{customer_id}")
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Determine color based on risk
                    risk = data["risk_tier"]
                    prob = data["churn_probability"]
                    
                    if risk == "Critical":
                        st.error(f"🚨 {risk} Risk")
                    elif risk == "High":
                        st.warning(f"⚠️ {risk} Risk")
                    elif risk == "Medium":
                        st.info(f"📊 {risk} Risk")
                    else:
                        st.success(f"✅ {risk}")
                        
                    # Display the exact probability
                    st.metric(label="Churn Probability", value=f"{prob * 100:.1f}%")
                    
                elif response.status_code == 404:
                    st.error("❌ Customer not found. Make sure you typed the ID correctly.")
                else:
                    st.error("API Error. Is the FastAPI server running?")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to the API. Did you start Uvicorn?")