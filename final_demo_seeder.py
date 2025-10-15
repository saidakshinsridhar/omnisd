import sqlite3
import datetime

DB_PATH = 'backend/waste_reports.db'
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# WIPE OLD DATA
print("Wiping old data for a clean demo slate...")
c.execute("DELETE FROM reports")
c.execute("DELETE FROM anomalies")
c.execute("UPDATE workers SET current_load = 0, status = 'Idle'")
c.execute("UPDATE users SET points = 125 WHERE id = 1")

# SEED DEMO DATA
print("Seeding with 'Golden' demo data...")

# --- A clear anomaly in T. Nagar ---
# Historical data (normal)
for i in range(15):
    past_date = datetime.datetime.now() - datetime.timedelta(days=20+i)
    c.execute("INSERT INTO reports (timestamp, location, waste_type, status, assigned_worker, user_id) VALUES (?, ?, ?, ?, ?, ?)",
              (past_date, 'T. Nagar', 'Paper', 'Collected', 'N/A', 1))
# Recent data (spike in 'Metal')
for i in range(10):
    recent_date = datetime.datetime.now() - datetime.timedelta(days=i)
    c.execute("INSERT INTO reports (timestamp, location, waste_type, status, assigned_worker, user_id) VALUES (?, ?, ?, ?, ?, ?)",
              (recent_date, 'T. Nagar', 'Metal', 'Pending Collection', 'Ravi Kumar', 1))

# --- A few other reports for the inbox ---
c.execute("INSERT INTO reports (timestamp, location, waste_type, status, assigned_worker, user_id) VALUES (?, ?, ?, ?, ?, ?)",
          (datetime.datetime.now(), 'Adyar', 'Plastic', 'Pending Collection', 'Priya Sharma', 1))
c.execute("INSERT INTO reports (timestamp, location, waste_type, status, assigned_worker, user_id) VALUES (?, ?, ?, ?, ?, ?)",
          (datetime.datetime.now(), 'Velachery', 'General/Bio Waste', 'En Route', 'Sunita Devi', 1))

conn.commit()
conn.close()
print("✅ Demo database is ready!")