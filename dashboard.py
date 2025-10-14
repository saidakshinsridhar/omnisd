import streamlit as st
import requests
import pandas as pd
import json # Import json for sending data

# --- Page Configuration ---
st.set_page_config(page_title="Urban Waste Intelligence", page_icon="♻️", layout="wide")

# --- API Endpoint ---
FLASK_API_URL = "http://127.0.0.1:5001"

# --- Main App ---
st.title("Urban Waste Intelligence Platform 🌿")
view = st.sidebar.radio("Select Your View", ('Citizen Reporting', 'Admin Dashboard'))
st.sidebar.markdown("---")


# ==============================================================================
#                                CITIZEN VIEW
# ==============================================================================
# Find this line in your dashboard.py file:
# if view == 'Citizen Reporting':

# And replace the entire block with this:
if view == 'Citizen Reporting':
    st.header("Report a Waste Pile")
    st.write("Help us keep the city clean! Upload an image and provide the location.")

    # --- NEW: Use session_state to manage points and workflow ---
    if 'prediction' not in st.session_state:
        st.session_state.prediction = None
    if 'eco_points' not in st.session_state:
        st.session_state.eco_points = 125 # Starting points

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader("1. Upload an image of the waste...", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Your Uploaded Image", width=300)

            if st.button("Classify Waste Type"):
                with st.spinner('AI is analyzing the image...'):
                    # ... (your existing API call for prediction) ...
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{FLASK_API_URL}/predict", files=files)
                    if response.status_code == 200:
                        st.session_state.prediction = response.json()['prediction']
                    else:
                        st.error("Error: Could not classify the image.")
                        st.session_state.prediction = None

    with col2:
        st.subheader("Your Impact Score")
        # --- NEW: Display the live points from session state ---
        st.metric(label="Eco Points Earned", value=f"{st.session_state.eco_points} Points")
        st.progress(min(st.session_state.eco_points / 200, 1.0)) # Progress towards a 200 point goal
        st.write("You're making a real difference!")

    # --- Show this part only AFTER classification is done ---
    if st.session_state.prediction:
        st.success(f"**AI has identified the primary waste type as:** {st.session_state.prediction}")

        with st.form("report_form"):
            location = st.text_input("2. Enter Location (e.g., Jayanagar, Koramangala)")
            submitted = st.form_submit_button("Submit Report & Earn 10 Points")

            if submitted:
                if location:
                    payload = {'location': location, 'waste_type': st.session_state.prediction}
                    response = requests.post(f"{FLASK_API_URL}/report_waste", json=payload)
                    if response.status_code == 200:
                        st.session_state.eco_points += 10 # Add points!
                        st.balloons() # Fun celebration!
                        st.success(response.json()['message'])
                        st.session_state.prediction = None # Reset for next report
                        st.experimental_rerun() # Refresh the page to update points
                    else:
                        st.error("Failed to submit report.")
                else:
                    st.warning("Please enter a location before submitting.")


# ==============================================================================
#                                ADMIN VIEW
# ==============================================================================
elif view == 'Admin Dashboard':
    st.header("Live Waste Reports Inbox")
    st.write("Monitor incoming reports and track collection status.")

    try:
        response = requests.get(f"{FLASK_API_URL}/get_reports")
        if response.status_code == 200:
            reports = response.json()
            if reports:
                df = pd.DataFrame(reports)
                
                # --- Display Key Metrics ---
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Reports", len(df))
                col2.metric("Pending Assignment", len(df[df['status'] == 'Pending Assignment']))
                col3.metric("Most Reported Location", df['location'].mode()[0])

                # --- Display the Live Data Table ---
                st.write("### Incoming Reports:")
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No waste reports yet. Great job, team!")
        else:
            st.error("Could not fetch reports from the server.")
    except requests.exceptions.ConnectionError:
        st.error("Connection Error: Could not connect to the backend server.")