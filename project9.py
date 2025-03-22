import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression


plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("coolwarm")
plt.rcParams["figure.figsize"] = (12, 6)


def generate_temperature_data(start_year=1900, end_year=2023):
    """Generate realistic temperature data with warming trend."""
    years = range(start_year, end_year + 1)
    months = range(1, 13)
    
    data = []
    locations = ['Global', 'Northern Hemisphere', 'Southern Hemisphere', 
                'Tropics', 'Arctic', 'Antarctic']
    
    # Base parameters
    baseline_temp = 14.0  # global average in Celsius
    location_offsets = {
        'Global': 0,
        'Northern Hemisphere': 2,
        'Southern Hemisphere': -1,
        'Tropics': 12,
        'Arctic': -25,
        'Antarctic': -35
    }
    
    # Warming trend parameters (degrees per year)
    base_warming_rate = 0.01  # base warming rate
    location_warming_factors = {
        'Global': 1.0,
        'Northern Hemisphere': 1.2,
        'Southern Hemisphere': 0.8,
        'Tropics': 0.7,
        'Arctic': 2.5,
        'Antarctic': 1.8
    }
    
    for year in years:
        for month in months:
            for location in locations:
                # Base temperature for location
                base_temp = baseline_temp + location_offsets[location]
                
                # Seasonal variation (stronger in non-tropical regions)
                seasonal_factor = np.sin((month - 1) * np.pi / 6)
                if location in ['Arctic', 'Antarctic']:
                    seasonal_amplitude = 15.0
                elif location in ['Northern Hemisphere', 'Southern Hemisphere']:
                    seasonal_amplitude = 10.0
                else:
                    seasonal_amplitude = 3.0
                
                # Flip seasons for Southern Hemisphere
                if location in ['Southern Hemisphere', 'Antarctic']:
                    seasonal_factor = -seasonal_factor
                
                # Calculate seasonal effect
                seasonal_effect = seasonal_amplitude * seasonal_factor
                
                # Add warming trend
                years_since_start = year - start_year
                warming_rate = base_warming_rate * location_warming_factors[location]
                
                # Make warming non-linear (accelerating in recent decades)
                if year > 1970:
                    acceleration_factor = min(3.0, 1.0 + (year - 1970) / 50)
                    warming_effect = warming_rate * years_since_start * acceleration_factor
                else:
                    warming_effect = warming_rate * years_since_start
                
                # Add El Niño/La Niña cycles (simplified)
                enso_cycle = 1.0 * np.sin((year - 1900) * 2 * np.pi / 5)
                
                # Add noise (random variations)
                noise_level = 0.3 if location == 'Global' else 0.6
                noise = np.random.normal(0, noise_level)
                
                # Combine all effects
                temperature = base_temp + seasonal_effect + warming_effect + enso_cycle + noise
                
                # Calculate anomaly from 1951-1980 baseline
                is_baseline = (year >= 1951 and year <= 1980)
                anomaly = warming_effect + enso_cycle + noise if not is_baseline else noise
                
                # Add record to data
                data.append({
                    'Date': f"{year}-{month:02d}-01",
                    'Year': year,
                    'Month': month,
                    'Location': location,
                    'Temperature': temperature,
                    'Anomaly': anomaly
                })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Generate temperature dataset
temp_data = generate_temperature_data(1900, 2023)
print("Temperature data generated successfully!")
print(f"Dataset shape: {temp_data.shape}")
print("\nFirst few rows:")
print(temp_data.head())

# 1. Global Temperature Trend Line Chart
plt.figure(figsize=(14, 7))

# Calculate annual averages for global temperature
global_annual = temp_data[temp_data['Location'] == 'Global'].groupby('Year')['Temperature'].mean().reset_index()

# Plot the temperature line
plt.plot(global_annual['Year'], global_annual['Temperature'], color='firebrick', linewidth=2.5, label='Annual Average')

# Add trend line
X = global_annual[['Year']]
y = global_annual['Temperature']
model = LinearRegression().fit(X, y)
trend_temps = model.predict(X)
plt.plot(global_annual['Year'], trend_temps, 'k--', linewidth=1.5, label=f'Trend: {model.coef_[0]:.4f}°C/year')

# Add smoothed line (10-year moving average)
global_annual['10yr_MA'] = global_annual['Temperature'].rolling(window=10, center=True).mean()
plt.plot(global_annual['Year'], global_annual['10yr_MA'], color='darkblue', linewidth=3, alpha=0.7, label='10-year Moving Average')

plt.title('Global Annual Average Temperature (1900-2023)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('global_temperature_trend.png', dpi=300)
plt.close()
print("Created: Global Temperature Trend Line Chart")

# 2. Temperature Anomaly Line Chart
plt.figure(figsize=(14, 7))

# Calculate annual anomalies by location
annual_anomalies = temp_data.groupby(['Year', 'Location'])['Anomaly'].mean().reset_index()

for location in annual_anomalies['Location'].unique():
    location_data = annual_anomalies[annual_anomalies['Location'] == location]
    plt.plot(location_data['Year'], location_data['Anomaly'], linewidth=2, alpha=0.8, label=location)

plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
plt.title('Temperature Anomalies by Region (1951-1980 Baseline)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Temperature Anomaly (°C)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(title='Region')
plt.tight_layout()
plt.savefig('temperature_anomalies_by_region.png', dpi=300)
plt.close()
print("Created: Temperature Anomalies by Region Line Chart")

# 3. Seasonal Temperature Patterns
plt.figure(figsize=(12, 8))

# Filter for the last 30 years
recent_data = temp_data[temp_data['Year'] >= 1993]

# Calculate monthly averages by location
monthly_temps = recent_data.groupby(['Month', 'Location'])['Temperature'].mean().reset_index()

# Plot for each location
for location in monthly_temps['Location'].unique():
    location_data = monthly_temps[monthly_temps['Location'] == location]
    plt.plot(location_data['Month'], location_data['Temperature'], marker='o', linewidth=2, label=location)

plt.title('Average Monthly Temperatures by Region (Last 30 Years)', fontsize=16)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True, alpha=0.3)
plt.legend(title='Region')
plt.tight_layout()
plt.savefig('seasonal_temperature_patterns.png', dpi=300)
plt.close()
print("Created: Seasonal Temperature Patterns Line Chart")

# 4. Temperature Distribution by Decade (Violin Plot)
plt.figure(figsize=(16, 10))

# Create decade column
temp_data['Decade'] = (temp_data['Year'] // 10) * 10
temp_data['Decade'] = temp_data['Decade'].astype(str) + 's'

# Filter for global data
global_data = temp_data[temp_data['Location'] == 'Global']

# Create violin plot
sns.violinplot(x='Decade', y='Temperature', data=global_data, inner='quartile', palette='coolwarm')
plt.title('Global Temperature Distribution by Decade', fontsize=16)
plt.xlabel('Decade', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('temperature_distribution_by_decade.png', dpi=300)
plt.close()
print("Created: Temperature Distribution by Decade Violin Plot")

# 5. Temperature Change Rate Heatmap
plt.figure(figsize=(16, 10))

# Prepare data for heatmap
# Calculate temperature change rate for each 20-year period and location
periods = []
for year in range(1900, 2023, 20):
    if year + 19 <= 2023:
        periods.append((year, year + 19))

heatmap_data = []
for start_year, end_year in periods:
    for location in temp_data['Location'].unique():
        location_data = temp_data[(temp_data['Location'] == location) & 
                                  (temp_data['Year'] >= start_year) & 
                                  (temp_data['Year'] <= end_year)]
        
        # Calculate trend using linear regression
        if len(location_data) > 0:
            X = location_data[['Year']]
            y = location_data['Temperature']
            model = LinearRegression().fit(X, y)
            rate = model.coef_[0]  # degrees per year
            
            heatmap_data.append({
                'Period': f"{start_year}-{end_year}",
                'Location': location,
                'Warming Rate': rate * 10  # convert to degrees per decade
            })

# Create DataFrame and pivot for heatmap
heatmap_df = pd.DataFrame(heatmap_data)
heatmap_pivot = heatmap_df.pivot(index='Location', columns='Period', values='Warming Rate')

# Create heatmap with diverging color palette centered at 0
sns.heatmap(heatmap_pivot, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
            cbar_kws={'label': 'Warming Rate (°C/decade)'})
plt.title('Temperature Change Rate by Region and Time Period', fontsize=16)
plt.xlabel('Time Period', fontsize=12)
plt.ylabel('Region', fontsize=12)
plt.tight_layout()
plt.savefig('warming_rate_heatmap.png', dpi=300)
plt.close()
print("Created: Warming Rate Heatmap")

# 6. Scatter Plot: Arctic vs Global Temperature Anomalies
plt.figure(figsize=(10, 8))

# Filter and prepare data
arctic_data = temp_data[temp_data['Location'] == 'Arctic'].groupby('Year')['Anomaly'].mean().reset_index()
global_data = temp_data[temp_data['Location'] == 'Global'].groupby('Year')['Anomaly'].mean().reset_index()

# Merge the datasets
scatter_data = pd.merge(arctic_data, global_data, on='Year', suffixes=('_Arctic', '_Global'))

# Create colormap based on years
norm = plt.Normalize(scatter_data['Year'].min(), scatter_data['Year'].max())
colors = plt.cm.viridis(norm(scatter_data['Year']))

# Plot scatter with color indicating year
plt.scatter(scatter_data['Anomaly_Global'], scatter_data['Anomaly_Arctic'], c=colors, alpha=0.8, s=50)

# Add colorbar to show year progression
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.set_label('Year', fontsize=12)

# Add diagonal reference line (y=x)
max_val = max(scatter_data['Anomaly_Arctic'].max(), scatter_data['Anomaly_Global'].max())
min_val = min(scatter_data['Anomaly_Arctic'].min(), scatter_data['Anomaly_Global'].min())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Equal Change Line')

# Add trend line
X = scatter_data[['Anomaly_Global']]
y = scatter_data['Anomaly_Arctic']
model = LinearRegression().fit(X, y)
trend_y = model.predict(X)
plt.plot(scatter_data['Anomaly_Global'], trend_y, 'r-', linewidth=2, 
         label=f'Trend (Slope: {model.coef_[0]:.2f})')

plt.title('Arctic vs Global Temperature Anomalies (1900-2023)', fontsize=16)
plt.xlabel('Global Temperature Anomaly (°C)', fontsize=12)
plt.ylabel('Arctic Temperature Anomaly (°C)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('arctic_vs_global_scatter.png', dpi=300)
plt.close()
print("Created: Arctic vs Global Temperature Anomalies Scatter Plot")

print("\nVisualization complete! All temperature charts have been saved as PNG files.")