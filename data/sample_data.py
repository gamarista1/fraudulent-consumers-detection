import pandas as pd
import numpy as np
from datetime import datetime

def generate_sample_data(n_consumers=1000, months=24, start_date='2021-01-01', fraud_rate=0.05):
    """
    Generate comprehensive sample data for power utility fraud detection.
    
    This function creates synthetic customer, consumption, and weather data
    that mimics real-world patterns including socioeconomic stratification,
    seasonal variations, and different types of fraud.
    
    Parameters:
    -----------
    n_consumers : int, optional
        Number of consumers to generate
    months : int, optional
        Number of months of historical data
    start_date : str, optional
        Start date for data generation (YYYY-MM-DD)
    fraud_rate : float, optional
        Proportion of consumers that are fraudulent
        
    Returns:
    --------
    customers_df : DataFrame
        Contains customer information (id, stratum, coordinates, etc.)
    consumption_df : DataFrame
        Contains consumption data over the specified months
    weather_df : DataFrame
        Contains weather data (temperature, humidity, UV index)
    """
    np.random.seed(42)
    
    # Create date range
    date_range = pd.date_range(start=start_date, periods=months, freq='M')
    
    # Create customer IDs
    customer_ids = [f"C{i:04d}" for i in range(1, n_consumers+1)]
    
    # Generate socioeconomic strata (1-6, with 6 being highest)
    strata = np.random.choice([1, 2, 3, 4, 5, 6], size=n_consumers, p=[0.15, 0.25, 0.3, 0.15, 0.1, 0.05])
    
    # Generate coordinates (for Medellín, Colombia)
    latitudes = 6.25 + 0.1 * np.random.randn(n_consumers)
    longitudes = -75.58 + 0.1 * np.random.randn(n_consumers)
    
    # Adjusting coordinates based on strata (higher strata in better areas)
    for i in range(n_consumers):
        if strata[i] >= 5:  # High strata (5-6)
            latitudes[i] = 6.21 + 0.05 * np.random.randn()  # El Poblado area
            longitudes[i] = -75.57 + 0.05 * np.random.randn()
        elif strata[i] >= 3:  # Middle strata (3-4)
            latitudes[i] = 6.25 + 0.05 * np.random.randn()  # Laureles area
            longitudes[i] = -75.60 + 0.05 * np.random.randn()
    
    # Generate zone codes based on location
    zone_codes = []
    for i in range(n_consumers):
        lat, lon = latitudes[i], longitudes[i]
        
        if lat < 6.22 and lon > -75.58:  # Southeast
            zone_codes.append('Z1')
        elif lat < 6.22 and lon <= -75.58:  # Southwest
            zone_codes.append('Z2')
        elif lat >= 6.22 and lon <= -75.58:  # Northwest
            zone_codes.append('Z3')
        else:  # Northeast
            zone_codes.append('Z4')
    
    # Create customers dataframe
    customers_df = pd.DataFrame({
        'customer_id': customer_ids,
        'stratum': strata,
        'latitude': latitudes,
        'longitude': longitudes,
        'zone_code': zone_codes,
        'sanctioned_load': np.random.randint(5, 30, size=n_consumers)
    })
    
    # Normal consumption patterns by strata (higher consumption for higher strata)
    base_consumption = {
        1: 100,  # Base consumption for stratum 1
        2: 150,  # Base consumption for stratum 2
        3: 200,  # Base consumption for stratum 3
        4: 300,  # Base consumption for stratum 4
        5: 400,  # Base consumption for stratum 5
        6: 500   # Base consumption for stratum 6
    }
    
    # Seasonal factors (representing Medellín's climate patterns)
    seasonal_factors = [1.0, 1.0, 0.9, 0.8, 0.7, 0.7, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2] * (months // 12 + 1)
    seasonal_factors = seasonal_factors[:months]
    
    # Initialize consumption data
    consumption_data = []
    
    # Mark some customers as fraudulent
    fraudulent_mask = np.random.choice([0, 1], size=n_consumers, p=[1-fraud_rate, fraud_rate])
    customers_df['is_fraudulent'] = fraudulent_mask
    
    # Assign fraud types for fraudulent customers
    fraud_types = []
    for is_fraud in fraudulent_mask:
        if is_fraud:
            fraud_types.append(np.random.choice([
                'meter_tampering', 'bypass', 'tariff_change', 
                'extension', 'meter_reversal'
            ]))
        else:
            fraud_types.append(None)
    
    customers_df['fraud_type'] = fraud_types
    
    # Generate consumption data
    for i, customer_id in enumerate(customer_ids):
        stratum = strata[i]
        is_fraudulent = fraudulent_mask[i]
        fraud_type = fraud_types[i]
        
        # Start fraud at a random month (if fraudulent)
        fraud_start_month = np.random.randint(1, months // 2) if is_fraudulent else months
        
        for j, month in enumerate(date_range):
            base = base_consumption[stratum]
            seasonal = seasonal_factors[j]
            
            # Add random variation (normal consumers have less variation)
            random_factor = 0.95 + 0.1 * np.random.random()
            
            # Calculate consumption based on normal pattern
            consumption = base * seasonal * random_factor
            
            # Apply fraud patterns after the fraud start month
            if is_fraudulent and j >= fraud_start_month:
                if fraud_type == 'meter_tampering':
                    # Meter tampering - lower consumption than expected
                    consumption *= 0.4 + 0.2 * np.random.random()
                elif fraud_type == 'bypass':
                    # By-passing meter - very low consumption
                    consumption *= 0.2 + 0.1 * np.random.random()
                elif fraud_type == 'tariff_change':
                    # Commercial use on domestic tariff - higher consumption
                    consumption *= 1.5 + 0.5 * np.random.random()
                elif fraud_type == 'extension':
                    # Illegal extension - higher consumption
                    consumption *= 1.3 + 0.3 * np.random.random()
                elif fraud_type == 'meter_reversal':
                    # Meter reversal - inconsistent readings
                    if j % 2 == 0:
                        consumption *= 0.3 + 0.2 * np.random.random()
                    else:
                        consumption *= 1.2 + 0.3 * np.random.random()
            
            consumption_data.append({
                'customer_id': customer_id,
                'date': month,
                'consumption': consumption,
                'month': month.month,
                'year': month.year
            })
    
    consumption_df = pd.DataFrame(consumption_data)
    
    # Generate weather data
    weather_data = []
    
    # Medellín's typical weather patterns
    avg_temps = [22, 22, 22, 22, 22, 21, 21, 22, 22, 21, 21, 22]  # Relatively stable year-round
    avg_humidity = [80, 79, 80, 83, 83, 82, 80, 80, 82, 84, 84, 81]  # Higher in rainy seasons
    avg_uv = [10, 11, 11, 10, 9, 9, 10, 10, 9, 8, 8, 9]  # UV index
    
    for month in date_range:
        month_idx = month.month - 1
        weather_data.append({
            'date': month,
            'temperature': avg_temps[month_idx] + np.random.randn() * 2,
            'humidity': avg_humidity[month_idx] + np.random.randn() * 5,
            'uv_index': avg_uv[month_idx] + np.random.randn()
        })
    
    weather_df = pd.DataFrame(weather_data)
    
    return customers_df, consumption_df, weather_df

def save_sample_data(customers_df, consumption_df, weather_df, path="./"):
    """
    Save generated sample data to CSV files.
    
    Parameters:
    -----------
    customers_df : DataFrame
        Customer information
    consumption_df : DataFrame
        Consumption data
    weather_df : DataFrame
        Weather data
    path : str, optional
        Path to save the files
    """
    customers_df.to_csv(f"{path}customers.csv", index=False)
    consumption_df.to_csv(f"{path}consumption.csv", index=False)
    weather_df.to_csv(f"{path}weather.csv", index=False)
    
    print(f"Data saved to {path}:")
    print(f"  - customers.csv: {len(customers_df)} records")
    print(f"  - consumption.csv: {len(consumption_df)} records")
    print(f"  - weather.csv: {len(weather_df)} records")

def load_saved_data(path="./"):
    """
    Load sample data from CSV files.
    
    Parameters:
    -----------
    path : str, optional
        Path to load the files from
        
    Returns:
    --------
    customers_df : DataFrame
        Customer information
    consumption_df : DataFrame
        Consumption data
    weather_df : DataFrame
        Weather data
    """
    customers_df = pd.read_csv(f"{path}customers.csv")
    consumption_df = pd.read_csv(f"{path}consumption.csv")
    weather_df = pd.read_csv(f"{path}weather.csv")
    
    # Convert date columns to datetime
    consumption_df['date'] = pd.to_datetime(consumption_df['date'])
    weather_df['date'] = pd.to_datetime(weather_df['date'])
    
    return customers_df, consumption_df, weather_df

# Example usage
if __name__ == "__main__":
    # Generate sample data
    customers, consumption, weather = generate_sample_data(n_consumers=1000, months=24)
    
    # Save to CSV files
    save_sample_data(customers, consumption, weather, path="./data/")