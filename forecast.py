import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# --- 1. Load and Prepare the Data ---
DATA_PATH = 'data/AEP_hourly.csv'
print(f"Loading data from {DATA_PATH}...")

df = pd.read_csv(DATA_PATH)
df['Datetime'] = pd.to_datetime(df['Datetime']) # Convert to datetime objects

# Prophet requires specific column names: 'ds' for date and 'y' for value
df.rename(columns={'Datetime': 'ds', 'AEP_MW': 'y'}, inplace=True)

print("Data loaded and prepared successfully.")

# --- 2. Build and Train the Model ---
# We will use only the last 2 years of data to speed up training
df_subset = df.tail(365 * 24 * 2) 

print("Training the Prophet model... (This might take a minute)")
model = Prophet()
model.fit(df_subset)
print("✅ Model training complete.")

# --- 3. Make a Future Forecast ---
# Create a dataframe for the next 30 days (30 * 24 hours)
future = model.make_future_dataframe(periods=30 * 24, freq='H')
forecast = model.predict(future)

print("✅ Forecast generated successfully.")

# --- 4. Visualize and Save the Forecast ---
print("Plotting the forecast...")
fig = model.plot(forecast)
plt.title('Energy Consumption Forecast (Next 30 Days)')
plt.xlabel('Date')
plt.ylabel('Energy Consumption (MW)')

# Save the plot as an image file
plt.savefig('forecast_plot.png')
print("✅ Forecast plot saved as forecast_plot.png")

# Optional: Display the plot if you are running this interactively
# plt.show()