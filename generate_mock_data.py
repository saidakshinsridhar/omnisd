import pandas as pd
import numpy as np
import os

# Define our locations for the mock data
locations = [
    'T. Nagar', 'Adyar', 'Anna Nagar', 'Velachery', 'Mylapore',
    'Nungambakkam', 'Guindy', 'Besant Nagar', 'Thiruvanmiyur',
    'Royapettah', 'Egmore', 'Vadapalani', 'Saidapet', 'Chromepet',
    'Pallavaram'
]

# Create a date range for the last year
dates = pd.date_range(end=pd.Timestamp.now(), periods=365*24, freq='H')

# Create a base DataFrame
df = pd.DataFrame({'timestamp': dates})

# Generate realistic data for each location
all_data = []
for location in locations:
    loc_df = df.copy()
    loc_df['location_id'] = location
    
    # Create a baseline waste generation with some random noise
    baseline = np.random.randint(50, 100)
    noise = np.random.normal(0, 15, len(loc_df))
    
    # Add seasonal effects: more waste on weekends and in the evenings
    day_of_week_effect = loc_df['timestamp'].dt.dayofweek.isin([5, 6]) * 30 
    hour_of_day_effect = (loc_df['timestamp'].dt.hour > 18) * 20
    
    loc_df['waste_kg'] = baseline + noise + day_of_week_effect + hour_of_day_effect
    loc_df['waste_kg'] = loc_df['waste_kg'].clip(lower=0) # Make sure waste is not negative
    
    all_data.append(loc_df)

# Combine all location data into one file
final_df = pd.concat(all_data)

# Ensure the 'data' directory exists
os.makedirs('data', exist_ok=True)

# Save the final CSV file
output_path = 'data/waste_generation_data.csv'
final_df.to_csv(output_path, index=False)

print(f"✅ Successfully generated mock waste data at '{output_path}'")