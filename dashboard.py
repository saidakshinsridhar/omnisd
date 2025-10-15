# dashboard.py
"""
Urban Waste Intelligence — Streamlit Frontend (Admin + Citizen Views)
Connects to backend at: http://127.0.0.1:5001

Dependencies:
    pip install streamlit requests pandas python-dateutil

Run:
    streamlit run dashboard.py
"""

import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime
from dateutil import parser
from typing import Optional, Any

# -----------------------
# Configuration
# -----------------------
BASE_URL = "http://127.0.0.1:5001"
REQUEST_TIMEOUT = 10  # seconds
STATUS_OPTIONS = ["New", "Assigned", "In Progress", "Collected", "Closed", "Ignored"]
PRIORITY_OPTIONS = ["Low", "Medium", "High", "Critical"]

st.set_page_config(page_title="TRASHLYTICS — Admin & Citizen Dashboard",
                   page_icon="🗑️", layout="wide")

# -----------------------
# Custom CSS (sleek/professional)
# -----------------------
CUSTOM_CSS = """
<style>
/* Page background, fonts */
body { background: linear-gradient(180deg,#071028,#04111a); color:#eaf4ff; font-family:Inter,Roboto,Arial,sans-serif; }

/* KPI card */
.kpi-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
    border-radius: 12px;
    padding: 14px;
    box-shadow: 0 8px 20px rgba(2,6,23,0.6);
    margin-bottom: 8px;
}
.kpi-title { color:#a9c0e6; font-size:13px; }
.kpi-value { color:#ffffff; font-size:26px; font-weight:700; margin-top:6px; }
.kpi-sub { color:#8ea8d1; font-size:12px; margin-top:6px; }

/* badge */
.badge { background: rgba(255,255,255,0.04); padding:6px 10px; border-radius:999px; color:#dbeeff; font-size:12px; }

/* footer */
.footer { color:#9fb5d8; font-size:12px; padding-top:10px; }

/* data editor container */
div[data-baseweb="table"] { background: rgba(255,255,255,0.02); border-radius:8px; padding:6px; }

/* sidebar */
section[aria-label="Sidebar"] { background: linear-gradient(180deg,#061224,#03111b); }

/* small text */
.small-muted { color:#8ea8d1; font-size:12px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------
# Utility: API wrappers (support optional auth header)
# -----------------------
def build_headers():
    token = st.session_state.get("_auth_token") if "_auth_token" in st.session_state else None
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers

def safe_json_response(resp: requests.Response):
    try:
        return resp.json()
    except Exception:
        try:
            return resp.text
        except Exception:
            return None

def api_get(path: str, params: Optional[dict] = None) -> Optional[Any]:
    try:
        url = f"{BASE_URL}{path}"
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT, headers=build_headers())
        resp.raise_for_status()
        return safe_json_response(resp)
    except Exception as e:
        st.error(f"GET {path} failed: {e}")
        return None

def api_post(path: str, json_data: Optional[dict] = None, files: Optional[dict] = None) -> Optional[Any]:
    try:
        url = f"{BASE_URL}{path}"
        headers = build_headers()
        if files:
            # do not set Content-Type header; requests will set multipart boundary
            resp = requests.post(url, files=files, timeout=REQUEST_TIMEOUT, headers=headers)
        else:
            headers.update({"Content-Type": "application/json"})
            resp = requests.post(url, json=json_data, timeout=REQUEST_TIMEOUT, headers=headers)
        resp.raise_for_status()
        return safe_json_response(resp)
    except Exception as e:
        st.error(f"POST {path} failed: {e}")
        return None

def api_patch(path: str, json_data: dict) -> Optional[Any]:
    try:
        url = f"{BASE_URL}{path}"
        headers = build_headers()
        headers.update({"Content-Type": "application/json"})
        resp = requests.patch(url, json=json_data, timeout=REQUEST_TIMEOUT, headers=headers)
        resp.raise_for_status()
        return safe_json_response(resp)
    except Exception as e:
        st.error(f"PATCH {path} failed: {e}")
        return None

# -----------------------
# Helpers: UI components
# -----------------------
def kpi_card(title: str, value: str, subtitle: str = "", badge: Optional[str] = None):
    html = f"""
    <div class="kpi-card">
        <div style="display:flex;justify-content:space-between;align-items:center">
            <div>
                <div class="kpi-title">{title}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-sub">{subtitle}</div>
            </div>
            {"<div class='badge'>"+badge+"</div>" if badge else ""}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, default=str)
    except Exception:
        return str(obj)

# -----------------------
# Cached data fetchers
# -----------------------
@st.cache_data(ttl=30)
def fetch_reports() -> Optional[pd.DataFrame]:
    data = api_get("/get_reports")
    if data is None:
        return None
    try:
        df = pd.json_normalize(data)
    except Exception:
        df = pd.DataFrame(data)
    # ensure columns
    for c in ["id", "location", "waste_type", "status", "user_id", "created_at", "priority", "notes"]:
        if c not in df.columns:
            df[c] = None
    # parse created_at
    def try_parse(x):
        try:
            return parser.parse(x)
        except Exception:
            return pd.NaT
    if "created_at" in df.columns:
        df["created_at_parsed"] = df["created_at"].apply(try_parse)
    else:
        df["created_at_parsed"] = pd.NaT
    return df

@st.cache_data(ttl=30)
def fetch_alerts() -> Optional[pd.DataFrame]:
    data = api_get("/get_alerts")
    if data is None:
        return None
    try:
        df = pd.json_normalize(data)
    except Exception:
        df = pd.DataFrame(data)
    return df

@st.cache_data(ttl=30)
def fetch_forecast(location: str) -> Optional[pd.DataFrame]:
    data = api_get("/forecast", params={"location": location})
    if data is None:
        return None
    try:
        df = pd.DataFrame(data)
        if "ds" in df.columns:
            df["ds"] = pd.to_datetime(df["ds"])
        return df
    except Exception:
        return None

@st.cache_data(ttl=30)
def fetch_resource_plan(location: str) -> Optional[dict]:
    return api_get("/resource_plan", params={"location": location})

# -----------------------
# Handle edits from st.data_editor for status updates
# -----------------------
def handle_status_updates(original_df: pd.DataFrame, edited_df: pd.DataFrame):
    if original_df is None or edited_df is None:
        return []
    orig = original_df.set_index("id", drop=False)
    edited = edited_df.set_index("id", drop=False)
    changes = []
    for rid in edited.index:
        if rid not in orig.index:
            continue
        before = orig.at[rid, "status"] if "status" in orig.columns else None
        after = edited.at[rid, "status"] if "status" in edited.columns else None
        if before != after:
            # call PATCH /reports/<id>/status
            resp = api_patch(f"/reports/{rid}/status", json_data={"status": str(after)})
            changes.append({"id": int(rid), "before": before, "after": after, "response": resp})
    if changes:
        st.success(f"Applied {len(changes)} change(s).")
        # clear caches so updated data shows
        try:
            st.cache_data.clear()
        except Exception:
            # fallback
            st.experimental_memo_clear()
    return changes

# -----------------------
# Sidebar: role selection + optional auth token
# -----------------------
st.sidebar.title("Views")
role = st.sidebar.radio("Choose view", ["Admin", "Citizen"], index=0)
st.sidebar.markdown("---")
st.sidebar.markdown("**Optional:** Add Bearer token (if backend requires auth).")
token_input = st.sidebar.text_input("Bearer token", type="password", key="auth_token_input")
if token_input:
    st.session_state["_auth_token"] = token_input
elif "_auth_token" in st.session_state and token_input == "":
    # allow clearing token
    st.session_state.pop("_auth_token", None)

st.sidebar.markdown("---")
st.sidebar.markdown(f"Backend base: `{BASE_URL}`")
st.sidebar.markdown("App version: 1.0.0")
st.sidebar.markdown("---")
st.sidebar.markdown("Tip: Use 'Citizen' to report quickly; 'Admin' to triage and plan.")

# -----------------------
# Header
# -----------------------
col1, col2 = st.columns([3, 1])
with col1:
    st.title("Urban Waste Intelligence")
    st.markdown("Admin & Citizen Dashboard — connect, triage, forecast, and operate")
with col2:
    st.markdown(f"**{datetime.now().strftime('%b %d, %Y — %H:%M:%S')}**")

# -----------------------
# Load shared data
# -----------------------
reports_df = fetch_reports()
alerts_df = fetch_alerts()

# -----------------------
# ADMIN VIEW
# -----------------------
if role == "Admin":
    tabs = st.tabs(["📊 Overview", "📥 Live Inbox", "🧾 Forecast & Resources", "🚨 Alerts & Anomalies", "🛠 Operations Console"])

    # Overview
    with tabs[0]:
        st.subheader("Admin — Overview")
        total_reports = int(len(reports_df)) if reports_df is not None else 0
        open_reports = 0
        if reports_df is not None and "status" in reports_df.columns:
            open_reports = int(len(reports_df[reports_df["status"].isin(["New", "Assigned", "In Progress"])]))
        critical_alerts = int(len(alerts_df)) if alerts_df is not None else 0
        last_points = st.session_state.get("last_points_added", 0)

        cols = st.columns(4)
        with cols[0]:
            kpi_card("Total Reports", str(total_reports), "All-time reports", badge="Reports")
        with cols[1]:
            kpi_card("Open Reports", str(open_reports), "Awaiting resolution", badge="Action")
        with cols[2]:
            kpi_card("Active Alerts", str(critical_alerts), "High severity", badge="Alerts")
        with cols[3]:
            kpi_card("Last Points Added", str(last_points), "Most recent add_points", badge="Points")

        st.markdown("---")
        st.markdown("Recent reports preview")
        if reports_df is None:
            st.info("No reports available or failed to fetch.")
        else:
            preview = reports_df.sort_values(by="created_at_parsed", ascending=False).head(10)
            st.dataframe(preview[["id", "created_at_parsed", "location", "waste_type", "status", "user_id", "priority"]].fillna("-"), use_container_width=True)

    # Live Inbox - editable
    with tabs[1]:
        st.subheader("Live Inbox — Command Center")
        st.markdown("Edit the **status** directly. Press **Commit changes** to push updates to backend.")
        if reports_df is None:
            st.info("Failed to load reports.")
        elif reports_df.empty:
            st.info("No reports.")
        else:
            display_cols = ["id", "created_at_parsed", "location", "waste_type", "priority", "status", "user_id", "notes"]
            for c in display_cols:
                if c not in reports_df.columns:
                    reports_df[c] = None
            editable = reports_df[display_cols].copy()
            editable["created_at_parsed"] = editable["created_at_parsed"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if not pd.isna(x) else "")
            edited = st.data_editor(
                editable,
                column_config={
                    "status": st.column_config.SelectboxColumn(options=STATUS_OPTIONS, help="Change status"),
                    "priority": st.column_config.SelectboxColumn(options=PRIORITY_OPTIONS, help="Change priority"),
                    "notes": st.column_config.TextColumn(),
                },
                use_container_width=True,
                key="admin_live_inbox"
            )
            if st.button("✅ Commit changes", key="admin_commit"):
                results = handle_status_updates(reports_df, edited)
                if results:
                    st.json(results)

    # Forecast & Resources
    with tabs[2]:
        st.subheader("Forecast & Resource Planning")
        loc = st.text_input("Location", value="Adyar", key="admin_forecast_loc")
        if st.button("🔁 Refresh forecast & resource plan"):
            try:
                st.cache_data.clear()
            except Exception:
                st.experimental_memo_clear()

        if loc:
            forecast_df = fetch_forecast(loc)
            resource_plan = fetch_resource_plan(loc)
            st.markdown(f"#### Forecast for **{loc}**")
            if forecast_df is None or forecast_df.empty:
                st.info("Forecast not available.")
            else:
                # show yhat line chart
                chart_df = forecast_df.set_index("ds").sort_index()
                st.line_chart(chart_df["yhat"], height=320, use_container_width=True)
                st.markdown("Forecast sample:")
                st.dataframe(chart_df.reset_index().head(10), use_container_width=True)
            st.markdown("---")
            st.markdown("#### Resource Plan (simulation)")
            if resource_plan is None:
                st.info("Resource plan not available.")
            else:
                rp_cols = st.columns(3)
                with rp_cols[0]:
                    kpi_card("Truck Trips", str(resource_plan.get("required_truck_trips", "—")), subtitle="Estimated trips")
                with rp_cols[1]:
                    kpi_card("Simulated Cost", str(resource_plan.get("simulated_cost_inr", "—")), subtitle="Estimated cost (INR)")
                with rp_cols[2]:
                    kpi_card("Extra", str(resource_plan.get("notes", "—")), subtitle="Details")
                st.json(resource_plan)

    # Alerts & Anomalies
    with tabs[3]:
        st.subheader("Alerts & Anomaly Detection")
        if alerts_df is None or alerts_df.empty:
            st.info("No alerts found.")
        else:
            st.dataframe(alerts_df, use_container_width=True)
        st.markdown("---")
        st.markdown("Run anomaly detection for a location")
        anomaly_loc = st.text_input("Location for anomalies", value="Adyar", key="admin_anomaly_loc")
        if st.button("Run anomaly check"):
            with st.spinner("Running anomaly detection..."):
                resp = api_post("/anomalies/check", json_data={"location": anomaly_loc})
                if resp:
                    st.success("Anomaly check result:")
                    st.json(resp)
        st.markdown("---")
        st.markdown("Simulate operational failure (test)")
        sim_loc = st.text_input("Location to simulate", value="Adyar", key="admin_sim_loc")
        if st.button("Simulate failure"):
            with st.spinner("Simulating failure..."):
                resp = api_post("/simulate/failure", json_data={"location": sim_loc})
                if resp:
                    st.success("Simulation response:")
                    st.json(resp)

    # Operations Console
    with tabs[4]:
        st.subheader("Operations Console")
        st.markdown("Add points to a user (reward system)")
        with st.form("add_points_form"):
            u_id = st.number_input("User ID", min_value=1, value=1, step=1)
            pts = st.number_input("Points to add", min_value=1, value=10, step=1)
            submit = st.form_submit_button("Add Points")
            if submit:
                resp = api_post(f"/users/{int(u_id)}/add_points", json_data={"points": int(pts)})
                if resp:
                    st.success("Points added:")
                    st.json(resp)
                    st.session_state["last_points_added"] = int(pts)
                    try:
                        st.cache_data.clear()
                    except Exception:
                        st.experimental_memo_clear()

        st.markdown("---")
        st.markdown("Manual report submission (admin)")
        with st.form("admin_manual_report"):
            r_loc = st.text_input("Location", value="Adyar", key="admin_r_loc")
            r_waste = st.text_input("Waste Type", value="Plastic", key="admin_r_waste")
            r_user = st.number_input("User ID", min_value=1, value=1, step=1, key="admin_r_user")
            send = st.form_submit_button("Submit Report")
            if send:
                resp = api_post("/report_waste", json_data={"location": r_loc, "waste_type": r_waste, "user_id": int(r_user)})
                if resp:
                    st.success("Report submitted.")
                    st.json(resp)
                    try:
                        st.cache_data.clear()
                    except Exception:
                        st.experimental_memo_clear()

        st.markdown("---")
        st.markdown("Raw API Explorer (debug)")
        raw_path = st.text_input("Relative path (e.g. /get_reports)", value="/get_reports", key="admin_raw_path")
        raw_method = st.selectbox("Method", ["GET", "POST", "PATCH"], index=0, key="admin_raw_method")
        raw_payload = st.text_area("JSON payload (for POST/PATCH)", height=120, key="admin_raw_payload")
        if st.button("Send request", key="admin_raw_send"):
            try:
                payload = json.loads(raw_payload) if raw_payload.strip() else None
            except Exception as e:
                st.error(f"Invalid JSON: {e}")
                payload = None
            if raw_method == "GET":
                resp = api_get(raw_path, params=payload)
            elif raw_method == "POST":
                resp = api_post(raw_path, json_data=payload)
            else:
                resp = api_patch(raw_path, json_data=payload or {})
            if resp is not None:
                st.json(resp)

# -----------------------
# CITIZEN VIEW
# -----------------------
else:
    st.subheader("Citizen Dashboard — Report & Track")
    user_id = st.number_input("Your User ID", min_value=1, step=1, value=1, key="citizen_user_id")
    st.markdown("Quick interface to submit waste reports and track your reports & points.")
    citizen_tabs = st.tabs(["📮 Report Waste", "📄 My Reports", "🧭 Forecast for My Area", "💰 Points Wallet"])

    # Report Waste
    with citizen_tabs[0]:
        st.markdown("### Submit a waste report")
        uploaded = st.file_uploader("Upload a photo (optional) — helps classification", type=["png", "jpg", "jpeg"], key="citizen_img")
        colA, colB = st.columns(2)
        with colA:
            location = st.text_input("Location", value="Adyar", key="citizen_loc")
        with colB:
            waste_type = st.text_input("Waste Type (leave blank to auto-predict)", value="", key="citizen_waste_type")
        classify = st.button("Classify Image") if uploaded else False
        submit_report = st.button("Submit Report")

        if classify:
            with st.spinner("Classifying image..."):
                files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                resp = api_post("/predict", files=files)
                if resp and isinstance(resp, dict) and "prediction" in resp:
                    pred = resp["prediction"]
                    st.success(f"Prediction: {pred}")
                    st.session_state["citizen_auto_prediction"] = pred
                else:
                    st.warning("No prediction returned.")

        if submit_report:
            chosen = waste_type.strip() or st.session_state.get("citizen_auto_prediction", "")
            if not chosen:
                st.warning("Please enter waste type or classify an image first.")
            else:
                payload = {"location": location, "waste_type": chosen, "user_id": int(user_id)}
                resp = api_post("/report_waste", json_data=payload)
                if resp:
                    st.success("Report submitted — thank you!")
                    st.json(resp)
                    try:
                        st.cache_data.clear()
                    except Exception:
                        st.experimental_memo_clear()

    # My Reports
    with citizen_tabs[1]:
        st.markdown("### My Reports")
        if reports_df is None:
            st.info("No reports available or failed to fetch.")
        else:
            # Filter by user_id (string safe)
            try:
                my_reports = reports_df[reports_df["user_id"].astype(str) == str(int(user_id))]
            except Exception:
                my_reports = reports_df[reports_df["user_id"] == user_id]
            if my_reports.empty:
                st.info("You don't have any reports yet.")
            else:
                display = my_reports.sort_values(by="created_at_parsed", ascending=False)
                st.dataframe(display[["id", "created_at_parsed", "location", "waste_type", "status", "priority", "notes"]].fillna("-"), use_container_width=True)
            st.markdown("If you'd like to change a report status, contact an admin. You can resubmit corrected reports using the Report tab.")

    # Forecast for My Area
    with citizen_tabs[2]:
        st.markdown("### Forecast for your area")
        loc2 = st.text_input("Location", value="Adyar", key="citizen_forecast_loc")
        if st.button("Get forecast"):
            fc = fetch_forecast(loc2)
            if fc is None or fc.empty:
                st.info("Forecast not available.")
            else:
                st.line_chart(fc.set_index("ds")["yhat"], height=320, use_container_width=True)
                st.dataframe(fc.head(10), use_container_width=True)

    # Points Wallet
    with citizen_tabs[3]:
        st.markdown("### Points & Rewards (demo)")
        st.markdown("Add points to your account (this calls `/users/<id>/add_points` endpoint).")
        with st.form("citizen_add_points"):
            p_amt = st.number_input("Points to add", min_value=1, value=10, step=1)
            p_submit = st.form_submit_button("Add points")
            if p_submit:
                resp = api_post(f"/users/{int(user_id)}/add_points", json_data={"points": int(p_amt)})
                if resp:
                    st.success("Points added:")
                    st.json(resp)
                    st.session_state["last_points_added"] = int(p_amt)
                    try:
                        st.cache_data.clear()
                    except Exception:
                        st.experimental_memo_clear()

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.markdown(
    "<div class='footer'>Built for Urban Waste Intelligence • Uses endpoints: /predict, /report_waste, /get_reports, /reports/&lt;id&gt;/status, /users/&lt;id&gt;/add_points, /forecast, /get_alerts, /anomalies/check, /resource_plan, /simulate/failure</div>",
    unsafe_allow_html=True
)
