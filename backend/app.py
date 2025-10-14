import os
import sqlite3
import datetime
import random
import pandas as pd
from prophet import Prophet
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ==============================================================================
#                             1. DATABASE SETUP
# ==============================================================================
def init_db():
    conn = sqlite3.connect('waste_reports.db')
    c = conn.cursor()
    # Main reports table
    c.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY, timestamp TEXT, location TEXT, 
            waste_type TEXT, status TEXT, assigned_worker TEXT, user_id INTEGER
        )''')
    # Users table for rewards
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT UNIQUE, points INTEGER)
    ''')
    c.execute("INSERT OR IGNORE INTO users (id, username, points) VALUES (1, 'citizen_demo', 125)")
    # Workers table for smart dispatch
    # Inside the init_db() function in backend/app.py

    # Workers table with a new 'status' column
    c.execute('''
    CREATE TABLE IF NOT EXISTS workers (
        name TEXT PRIMARY KEY, 
        current_load INTEGER, 
        specialty TEXT,
        status TEXT 
    )''')
    # Add an initial status for each worker
    c.execute("INSERT OR IGNORE INTO workers VALUES ('Ravi Kumar', 0, 'Recyclables', 'Idle'), ('Priya Sharma', 0, 'General', 'Idle'), ('Anil Singh', 0, 'Recyclables', 'Idle'), ('Sunita Devi', 0, 'General', 'Idle')")
    c.execute('''
        CREATE TABLE IF NOT EXISTS anomalies (
            id INTEGER PRIMARY KEY, timestamp TEXT, location TEXT, message TEXT, severity TEXT
        )''')
    # Resources table for cost simulation
    c.execute('''
        CREATE TABLE IF NOT EXISTS resources (item TEXT PRIMARY KEY, value REAL)
    ''')
    c.execute("INSERT OR IGNORE INTO resources VALUES ('cost_per_truck_trip', 1500.0), ('cost_per_worker_hour', 250.0), ('truck_capacity_kg', 1000.0)")
    conn.commit()
    conn.close()
    print("✅ Database initialized with advanced schema.")

# ==============================================================================
#                     2. APP & MODEL INITIALIZATION
# ==============================================================================
app = Flask(__name__)
CORS(app)
init_db()

# --- Load Vision Model ---
print("Loading local Vision Model...")
vision_model = load_model('models/trash_classifier.h5')
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
print("✅ Vision Model loaded successfully!")

# --- Pre-train Forecast Models for Speed ---
FORECAST_MODELS = {}
# NEW CHENNAI LOCATIONS
LOCATIONS = [
    'T. Nagar', 'Adyar', 'Anna Nagar', 'Velachery', 'Mylapore',
    'Nungambakkam', 'Guindy', 'Besant Nagar', 'Thiruvanmiyur',
    'Royapettah', 'Egmore', 'Vadapalani', 'Saidapet', 'Chromepet',
    'Pallavaram'
]

def train_all_forecast_models():
    print("Pre-training forecast models for all locations...")
    try:
        df = pd.read_csv('../data/waste_generation_data.csv')
        df.rename(columns={'timestamp': 'ds', 'waste_kg': 'y'}, inplace=True)
        df['ds'] = pd.to_datetime(df['ds'])
        for location in LOCATIONS:
            print(f"  -> Training model for {location}...")
            df_location = df[df['location_id'] == location]
            if not df_location.empty:
                model = Prophet(daily_seasonality=True, weekly_seasonality=True)
                model.fit(df_location.tail(90 * 24))
                FORECAST_MODELS[location] = model
        print("✅ All forecast models are trained and ready!")
    except FileNotFoundError:
        print("⚠️ WARNING: Mock data file not found. Forecasting will be disabled.")

train_all_forecast_models()

# --- Mappings ---
CATEGORY_MAP = {'cardboard': 'Cardboard', 'paper': 'Paper', 'plastic': 'Plastic', 'metal': 'Metal', 'glass': 'Glass', 'trash': 'General/Bio Waste'}
WORKER_ASSIGNMENTS = {
    'T. Nagar': 'Ravi Kumar', 'Adyar': 'Priya Sharma', 'Anna Nagar': 'Anil Singh',
    'Velachery': 'Sunita Devi', 'Mylapore': 'Ravi Kumar', 'Nungambakkam': 'Priya Sharma',
    'Guindy': 'Anil Singh', 'Besant Nagar': 'Sunita Devi', 'Thiruvanmiyur': 'Ravi Kumar',
    'Royapettah': 'Priya Sharma', 'Egmore': 'Anil Singh', 'Vadapalani': 'Sunita Devi',
    'Saidapet': 'Ravi Kumar', 'Chromepet': 'Priya Sharma', 'Pallavaram': 'Anil Singh'
}
# ==============================================================================
#                               3. API ENDPOINTS
# ==============================================================================

# --- Endpoint 1: AI Waste Classification ---
@app.route('/predict', methods=['POST'])
def handle_prediction():
    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if not file or not file.filename: return jsonify({'error': 'No file selected'}), 400
    filepath = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True); file.save(filepath)
    img = image.load_img(filepath, target_size=(224, 224)); img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0); img_preprocessed = img_batch / 255.
    prediction = vision_model.predict(img_preprocessed)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]; os.remove(filepath)
    return jsonify({'prediction': CATEGORY_MAP.get(predicted_class, 'Uncategorized')})

# --- Endpoint 2: Citizen Waste Reporting (with Smart Dispatch) ---
@app.route('/report_waste', methods=['POST'])
def report_waste():
    data = request.json
    location = data.get('location'); waste_type = data.get('waste_type'); user_id = data.get('user_id', 1)
    if not location or not waste_type: return jsonify({'error': 'Location and waste_type are required'}), 400
    conn = sqlite3.connect('waste_reports.db'); conn.row_factory = sqlite3.Row; c = conn.cursor()
    specialty_needed = 'Recyclables' if waste_type != 'General/Bio Waste' else 'General'
    c.execute("SELECT name FROM workers WHERE specialty = ? ORDER BY current_load ASC LIMIT 1", (specialty_needed,))
    result = c.fetchone()
    assigned_worker = result['name'] if result else "Unassigned"
    if result: c.execute("UPDATE workers SET current_load = current_load + 1 WHERE name = ?", (assigned_worker,))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO reports (timestamp, location, waste_type, status, assigned_worker, user_id) VALUES (?, ?, ?, ?, ?, ?)",
              (timestamp, location, waste_type, "Pending Collection", assigned_worker, user_id))
    conn.commit(); conn.close()
    return jsonify({'success': True, 'message': f'Report submitted! Smart-assigned to worker: {assigned_worker}.'})

# --- Endpoint 3: Admin Data Retrieval ---
@app.route('/get_reports', methods=['GET'])
def get_reports():
    conn = sqlite3.connect('waste_reports.db'); conn.row_factory = sqlite3.Row; c = conn.cursor()
    c.execute("SELECT * FROM reports ORDER BY id DESC"); reports = [dict(row) for row in c.fetchall()]; conn.close()
    return jsonify(reports)

# --- Endpoint 4: Dynamic Forecasting & Proactive Alerts ---
@app.route('/forecast', methods=['GET'])
def handle_forecast():
    location = request.args.get('location')
    if not location: return jsonify({'error': 'Location parameter is required'}), 400
    model = FORECAST_MODELS.get(location)
    if not model: return jsonify({'error': f'No forecast model available for {location}.'}), 404
    future = model.make_future_dataframe(periods=7, freq='D'); forecast = model.predict(future)
    max_predicted_waste = forecast['yhat'].tail(7).max()
    if max_predicted_waste > 200:
        alert_msg = f"Critical waste level ({max_predicted_waste:.0f} kg) predicted for {location}."
        conn = sqlite3.connect('waste_reports.db'); c = conn.cursor()
        c.execute("INSERT INTO anomalies (timestamp, location, message, severity) SELECT ?, ?, ?, ? WHERE NOT EXISTS (SELECT 1 FROM anomalies WHERE location = ? AND severity = 'High')",
                  (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), location, alert_msg, "High", location))
        conn.commit(); conn.close(); print(f"🔥 PROACTIVE ALERT GENERATED for {location}")
    return jsonify(forecast[['ds', 'yhat']].tail(7).to_dict('records'))

# --- Endpoint 5: Digital Tracking (Status Update) ---
@app.route('/reports/<int:report_id>/status', methods=['PATCH'])
def update_report_status(report_id):
    new_status = request.json.get('status')
    if not new_status: return jsonify({'error': 'New status is required'}), 400
    conn = sqlite3.connect('waste_reports.db'); c = conn.cursor()
    c.execute("UPDATE reports SET status = ? WHERE id = ?", (new_status, report_id))
    conn.commit();
    if c.rowcount == 0: conn.close(); return jsonify({'error': 'Report ID not found'}), 404
    conn.close(); return jsonify({'success': True, 'message': f'Report {report_id} status updated to {new_status}'})

# --- Endpoint 6: Reward System (Add Points) ---
@app.route('/users/<int:user_id>/add_points', methods=['POST'])
def add_points(user_id):
    points_to_add = request.json.get('points', 0)
    conn = sqlite3.connect('waste_reports.db'); c = conn.cursor()
    c.execute("UPDATE users SET points = points + ? WHERE id = ?", (points_to_add, user_id))
    conn.commit(); c.execute("SELECT points FROM users WHERE id = ?", (user_id,)); new_total = c.fetchone()[0]
    conn.close(); return jsonify({'success': True, 'new_total_points': new_total})

# --- Endpoint 7: Proactive Alert Retrieval ---
@app.route('/get_alerts', methods=['GET'])
def get_alerts():
    conn = sqlite3.connect('waste_reports.db'); conn.row_factory = sqlite3.Row; c = conn.cursor()
    c.execute("SELECT * FROM anomalies ORDER BY id DESC"); alerts = [dict(row) for row in c.fetchall()]; conn.close()
    return jsonify(alerts)
# Inside backend/app.py

# --- INNOVATIVE FEATURE 1: Waste Composition Anomaly Detection ---
@app.route('/anomalies/check', methods=['POST'])
def check_for_anomalies():
    data = request.json
    location = data.get('location')
    if not location:
        return jsonify({'error': 'Location is required'}), 400

    print(f"Running anomaly detection for {location}...")
    conn = sqlite3.connect('waste_reports.db')
    
    # 1. Fetch all historical report data for this location
    # We use pandas here for its powerful data manipulation capabilities
    df = pd.read_sql_query(f"SELECT timestamp, waste_type FROM reports WHERE location = '{location}'", conn)
    
    if len(df) < 10: # Need at least 10 reports to get a baseline
        conn.close()
        return jsonify({'message': f'Not enough data to run anomaly detection for {location}.'}), 200

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 2. Define our time windows: the historical baseline vs. the recent period
    three_months_ago = pd.Timestamp.now() - pd.DateOffset(months=3)
    last_7_days = pd.Timestamp.now() - pd.DateOffset(days=7)

    baseline_df = df[df['timestamp'] < last_7_days]
    recent_df = df[df['timestamp'] >= last_7_days]

    if baseline_df.empty or recent_df.empty:
        conn.close()
        return jsonify({'message': 'Not enough recent or historical data to compare.'}), 200

    # 3. Calculate the normal (baseline) vs. recent waste composition percentages
    baseline_composition = baseline_df['waste_type'].value_counts(normalize=True)
    recent_composition = recent_df['waste_type'].value_counts(normalize=True)

    # 4. Find statistically significant deviations
    anomalies_found = []
    for waste_type, recent_perc in recent_composition.items():
        baseline_perc = baseline_composition.get(waste_type, 0)
        
        # Anomaly definition: if the recent percentage is more than double the baseline
        if recent_perc > (baseline_perc * 2) and recent_perc > 0.1: # and makes up at least 10% of recent waste
            change = (recent_perc - baseline_perc) * 100
            message = f"Anomaly Detected: Reports of '{waste_type}' in {location} have increased by {change:.0f}% recently, which is statistically unusual."
            anomalies_found.append(message)
            
            # Save the anomaly to our database
            c = conn.cursor()
            alert_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            c.execute("INSERT INTO anomalies (timestamp, location, message, severity) VALUES (?, ?, ?, ?)",
                      (alert_timestamp, location, message, "Medium"))
            conn.commit()

    conn.close()

    if anomalies_found:
        print(f"🔎 Anomaly detected for {location}!")
        return jsonify({'status': 'Anomalies Detected', 'details': anomalies_found})
    else:
        print(f"✅ No anomalies found for {location}.")
        return jsonify({'status': 'No Anomalies Found'})

# --- Endpoint 8: INNOVATIVE - Resource & Cost Simulation ---
@app.route('/resource_plan', methods=['GET'])
def get_resource_plan():
    location = request.args.get('location')
    if not location: return jsonify({'error': 'Location parameter is required'}), 400
    model = FORECAST_MODELS.get(location)
    if not model: return jsonify({'error': f'No forecast model available for {location}.'}), 404
    future = model.make_future_dataframe(periods=7, freq='D'); forecast = model.predict(future)
    total_predicted_waste_kg = forecast['yhat'].tail(7).sum()
    conn = sqlite3.connect('waste_reports.db'); conn.row_factory = sqlite3.Row; c = conn.cursor()
    c.execute("SELECT * FROM resources"); resources = {row['item']: row['value'] for row in c.fetchall()}; conn.close()
    truck_capacity = resources.get('truck_capacity_kg', 1000); cost_per_trip = resources.get('cost_per_truck_trip', 1500)
    cost_per_hour = resources.get('cost_per_worker_hour', 250)
    required_truck_trips = np.ceil(total_predicted_waste_kg / truck_capacity)
    estimated_worker_hours = required_truck_trips * 8
    simulated_cost = (required_truck_trips * cost_per_trip) + (estimated_worker_hours * cost_per_hour)
    return jsonify({
        'location': location, 'forecast_period_days': 7,
        'total_predicted_waste_kg': round(total_predicted_waste_kg, 2),
        'required_truck_trips': int(required_truck_trips),
        'estimated_worker_hours': int(estimated_worker_hours),
        'simulated_cost_inr': f"₹{simulated_cost:,.2f}"
    })
# Inside backend/app.py

# --- INNOVATIVE FEATURE 2: Predictive Failure & Dynamic Re-Routing ---
@app.route('/simulate/failure', methods=['POST'])
def simulate_failure():
    data = request.json
    failed_location = data.get('location')
    if not failed_location:
        return jsonify({'error': 'Location of the failure is required'}), 400

    print(f"🚨 Simulating operational failure at {failed_location}...")
    conn = sqlite3.connect('waste_reports.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # 1. Identify the worker originally assigned to the failed location
    # We use the WORKER_ASSIGNMENTS dict for simplicity
    failed_worker = [worker for worker, loc in WORKER_ASSIGNMENTS.items() if loc == failed_location]

    # 2. Find the best available backup worker
    # The "best" is an IDLE worker from a DIFFERENT zone with the lowest current load
    c.execute("""
        SELECT name FROM workers 
        WHERE status = 'Idle' AND name != ? 
        ORDER BY current_load ASC 
        LIMIT 1
    """, (failed_worker[0] if failed_worker else '',))
    
    backup_worker_row = c.fetchone()

    if not backup_worker_row:
        conn.close()
        return jsonify({'status': 'Crisis', 'message': 'No backup workers available! Manual intervention required.'}), 500

    backup_worker_name = backup_worker_row['name']
    
    # 3. Create a high-priority task and assign the backup worker
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    waste_type = "High-Priority Unscheduled Pickup"
    status = "Dispatched (Re-Routed)"
    
    c.execute("INSERT INTO reports (timestamp, location, waste_type, status, assigned_worker, user_id) VALUES (?, ?, ?, ?, ?, ?)",
              (timestamp, failed_location, waste_type, status, backup_worker_name, 0)) # user_id 0 for system-generated tasks
    
    # 4. Update the backup worker's status to show they are now busy
    c.execute("UPDATE workers SET status = 'On Job' WHERE name = ?", (backup_worker_name,))

    conn.commit()
    conn.close()
    
    response_message = f"Failure at {failed_location} simulated. System automatically re-routed the nearest available worker, {backup_worker_name}, to handle the capacity gap."
    print(f"✅ {response_message}")
    return jsonify({'status': 'Crisis Averted', 'message': response_message})

# ==============================================================================
#                                4. SERVER START
# ==============================================================================
if __name__ == '__main__':
    from waitress import serve
    print("Starting professional-grade server with Waitress on http://127.0.0.1:5001")
    serve(app, host="127.0.0.1", port=5001)