import sqlite3
import datetime

# --- Configuration ---
DB_PATH = 'backend/waste_reports.db'
LOCATION = 'Adyar'

# --- The data we want to insert ---
# This represents a CLEAR historical baseline from 15-90 days ago.
# Normal ratio: ~25% Plastic, ~75% other waste.
historical_data = {
    'Paper': 30,
    'Plastic': 25,
    'General/Bio Waste': 45
}

# This represents a CLEAR recent spike in Plastic within the last 6 days.
# Recent ratio: ~80% Plastic.
recent_spike_data = {
    'Plastic': 20,
    'General/Bio Waste': 5
}

# --- Connect to the database ---
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
print(f"Connected to database at {DB_PATH}")

# --- Insert historical data with OLD timestamps ---
print("Seeding historical baseline data...")
i = 0
for waste_type, count in historical_data.items():
    for _ in range(count):
        # Create timestamps from 15 to 100+ days in the past
        past_date = datetime.datetime.now() - datetime.timedelta(days=15 + i)
        timestamp = past_date.strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO reports (timestamp, location, waste_type, status, assigned_worker, user_id) VALUES (?, ?, ?, ?, ?, ?)",
                  (timestamp, LOCATION, waste_type, "Collected", "N/A", 1))
        i += 1

# --- Insert recent data with NEW timestamps ---
print("Seeding recent spike data...")
j = 0
for waste_type, count in recent_spike_data.items():
    for _ in range(count):
        # Create timestamps strictly within the last 6 days
        recent_date = datetime.datetime.now() - datetime.timedelta(days=j % 6)
        timestamp = recent_date.strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO reports (timestamp, location, waste_type, status, assigned_worker, user_id) VALUES (?, ?, ?, ?, ?, ?)",
                  (timestamp, LOCATION, waste_type, "Pending Collection", "N/A", 1))
        j += 1

# --- Save changes and close ---
conn.commit()
conn.close()
print("✅ Database seeding complete. You can now run the anomaly check.")