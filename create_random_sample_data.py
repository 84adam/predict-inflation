# create_random_sample_data.py

import pandas as pd
import numpy as np
np.random.seed(42)

# Generate dates
dates = pd.date_range(start='2019-01-01', end='2023-12-31', freq='M')

# Initialize DataFrame
df = pd.DataFrame(index=dates)

# Generate Core PCE (target variable)
trend = np.linspace(2.0, 3.5, len(dates))
seasonality = 0.2 * np.sin(np.linspace(0, 8*np.pi, len(dates)))
noise = np.random.normal(0, 0.1, len(dates))
df['Core_PCE'] = trend + seasonality + noise

# Generate M2 Money Supply (trillions)
m2_trend = np.linspace(15.5, 19.5, len(dates))
m2_noise = np.random.normal(0, 0.1, len(dates))
df['M2'] = m2_trend + m2_noise

# Generate Yield Curve data
df['2Y_Yield'] = np.random.normal(2.5, 0.3, len(dates))
df['5Y_Yield'] = df['2Y_Yield'] + np.random.normal(0.5, 0.1, len(dates))
df['10Y_Yield'] = df['5Y_Yield'] + np.random.normal(0.3, 0.1, len(dates))

# Generate Credit Spread
df['Credit_Spread'] = np.random.normal(1.0, 0.2, len(dates))

# Generate VIX
base_vix = np.random.normal(20, 5, len(dates))
spikes = np.random.randint(0, len(dates), 5)
base_vix[spikes] *= 2
df['VIX'] = np.maximum(10, base_vix)

# Generate TED Spread
df['Ted_Spread'] = np.random.normal(0.4, 0.1, len(dates))

# Generate Consumer Spending (% change)
spending_trend = np.random.normal(0.3, 0.1, len(dates))
seasonal_effect = 0.2 * np.sin(np.linspace(0, 8*np.pi, len(dates)))
df['Consumer_Spending'] = spending_trend + seasonal_effect

# Generate CPI
cpi_trend = np.linspace(2.2, 3.8, len(dates))
cpi_noise = np.random.normal(0, 0.15, len(dates))
df['CPI'] = cpi_trend + cpi_noise

# Generate Unemployment
unemp_trend = np.random.normal(4.5, 0.3, len(dates))
for i in range(1, len(dates)):
    unemp_trend[i] = 0.9 * unemp_trend[i-1] + 0.1 * unemp_trend[i]
df['Unemployment'] = unemp_trend

# Generate Commodity prices
commodities = ['Commod_Energy', 'Commod_Metals', 'Commod_Agri']
base_trend = np.linspace(0, 0.5, len(dates))
for comm in commodities:
    comm_trend = base_trend + np.random.normal(0, 0.1, len(dates))
    seasonal = 0.1 * np.sin(np.linspace(0, 8*np.pi, len(dates)))
    df[comm] = 100 * np.exp(comm_trend + seasonal)

# Reset index to make Date a column
df.reset_index(inplace=True)
df.rename(columns={'index': 'Date'}, inplace=True)

# Save to CSV
df.round(3).to_csv('macro_data.csv', index=False)

# Print first few rows to verify
print("\nFirst few rows of the generated data:")
print(df.round(3).head())

# Print data info
print("\nDataset information:")
print(df.info())
