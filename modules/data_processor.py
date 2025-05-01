import pandas as pd
import numpy as np
from datetime import datetime

def load_sample_data():
    """
    Generate sample data for demonstration.
    
    This function creates synthetic customer, consumption, and weather data
    for demonstration purposes, replicating the structure of real utility data.
    
    Returns:
    --------
    customers_df : DataFrame
        Contains customer information (id, stratum, coordinates, etc.)
    consumption_df : DataFrame
        Contains consumption data over 24 months
    weather_df : DataFrame
        Contains weather data (temperature, humidity, UV index)
    """
    np.random.seed(42)
    
    # Number of consumers
    n_consumers = 1000
    
    # Create customer IDs
    customer_ids = [f"C{i:04d}" for i in range(1, n_consumers+1)]
    
    # Generate socioeconomic strata (1-6, with 6 being highest)
    strata = np.random.choice([1, 2, 3, 4, 5, 6], size=n_consumers, p=[0.15, 0.25, 0.3, 0.15, 0.1, 0.05])
    
    # Generate coordinates (for Medellín, Colombia)
    latitudes = 6.25 + 0.1 * np.random.randn(n_consumers)
    longitudes = -75.58 + 0.1 * np.random.randn(n_consumers)
    
    # Adjusting coordinates based on strata (higher strata in better areas)
    # This simulates the socioeconomic stratification mentioned in the paper
    for i in range(n_consumers):
        if strata[i] >= 5:  # High strata (5-6)
            latitudes[i] = 6.21 + 0.05 * np.random.randn()  # El Poblado area
            longitudes[i] = -75.57 + 0.05 * np.random.randn()
        elif strata[i] >= 3:  # Middle strata (3-4)
            latitudes[i] = 6.25 + 0.05 * np.random.randn()  # Laureles area
            longitudes[i] = -75.60 + 0.05 * np.random.randn()
    
    # Create customers dataframe
    customers_df = pd.DataFrame({
        'customer_id': customer_ids,
        'stratum': strata,
        'latitude': latitudes,
        'longitude': longitudes,
        'zone_code': np.random.choice(['Z1', 'Z2', 'Z3', 'Z4'], size=n_consumers),
        'sanctioned_load': np.random.randint(5, 30, size=n_consumers)
    })
    
    # Generate features for normal and fraudulent consumers
    # We'll create a time series for 24 months
    # Corregido: usar 'ME' en lugar de 'M' para evitar la advertencia de obsolescencia
    months = pd.date_range(start='2021-01-01', periods=24, freq='ME')
    
    # Normal consumption patterns by strata (higher consumption for higher strata)
    base_consumption = {
        1: 100,  # Base consumption for stratum 1
        2: 150,  # Base consumption for stratum 2
        3: 200,  # Base consumption for stratum 3
        4: 300,  # Base consumption for stratum 4
        5: 400,  # Base consumption for stratum 5
        6: 500   # Base consumption for stratum 6
    }
    
    # Seasonal factors (higher in winter months)
    seasonal_factors = [1.0, 1.0, 0.9, 0.8, 0.7, 0.7, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2] * 2
    
    # Initialize consumption data
    consumption_data = []
    
    # Mark some customers as fraudulent (5%)
    fraudulent_mask = np.random.choice([0, 1], size=n_consumers, p=[0.95, 0.05])
    customers_df['is_fraudulent'] = fraudulent_mask
    
    # Generate consumption data
    for i, customer_id in enumerate(customer_ids):
        stratum = strata[i]
        is_fraudulent = fraudulent_mask[i]
        
        for j, month in enumerate(months):
            base = base_consumption[stratum]
            seasonal = seasonal_factors[j]
            
            # Add random variation
            random_factor = 0.9 + 0.2 * np.random.random()
            
            # Calculate consumption based on normal pattern
            consumption = base * seasonal * random_factor
            
            # Apply fraud patterns
            if is_fraudulent:
                fraud_type = np.random.choice(['meter_tampering', 'bypass', 'tariff_change', 'extension'])
                
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
            
            consumption_data.append({
                'customer_id': customer_id,
                'date': month,
                'consumption': consumption,
                'month': month.month,
                'year': month.year
            })
    
    consumption_df = pd.DataFrame(consumption_data)
    
    # Add weather data (temperature, humidity, UV index)
    # Using Medellin's typical yearly pattern
    weather_data = []
    
    # Typical weather patterns for Medellín
    avg_temps = [22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22]
    avg_humidity = [80, 79, 80, 83, 83, 82, 80, 80, 82, 84, 84, 81]
    avg_uv = [10, 11, 11, 10, 9, 9, 10, 10, 9, 8, 8, 9]
    
    for j, month in enumerate(months):
        month_idx = month.month - 1
        weather_data.append({
            'date': month,
            'temperature': avg_temps[month_idx] + np.random.randn() * 2,
            'humidity': avg_humidity[month_idx] + np.random.randn() * 5,
            'uv_index': avg_uv[month_idx] + np.random.randn()
        })
    
    weather_df = pd.DataFrame(weather_data)
    
    return customers_df, consumption_df, weather_df

def prepare_features(customers_df, consumption_df, weather_df, selected_month=None, selected_year=None, selected_zone=None):
    """
    Prepare features for the MGD model.
    
    This function extracts and processes features from the raw data,
    including current and previous consumption, customer information,
    and weather data.
    
    Parameters:
    -----------
    customers_df : DataFrame
        Customer information
    consumption_df : DataFrame
        Consumption data
    weather_df : DataFrame
        Weather data
    selected_month : int, optional
        Month to analyze
    selected_year : int, optional
        Year to analyze
    selected_zone : str, optional
        Zone code to filter by
        
    Returns:
    --------
    features_for_model : DataFrame
        Processed features ready for the model
    customer_ids : Series
        Customer IDs corresponding to the features
    filtered_customers : DataFrame
        Filtered customer information
    """
    # Verificar si hay datos válidos
    if customers_df.empty or consumption_df.empty:
        return pd.DataFrame(), pd.Series(), pd.DataFrame()
    
    # Filter data if month and year are selected
    if selected_month and selected_year:
        current_month_data = consumption_df[(consumption_df['month'] == selected_month) & 
                                           (consumption_df['year'] == selected_year)]
        
        # Get the same month from the previous year for comparison
        prev_year = selected_year - 1
        prev_month_data = consumption_df[(consumption_df['month'] == selected_month) & 
                                        (consumption_df['year'] == prev_year)]
    else:
        # Use the latest month by default
        latest_date = consumption_df['date'].max()
        current_month_data = consumption_df[consumption_df['date'] == latest_date]
        
        # Get same month from previous year
        prev_month = latest_date.month
        prev_year = latest_date.year - 1
        prev_month_data = consumption_df[(consumption_df['month'] == prev_month) & 
                                        (consumption_df['year'] == prev_year)]
    
    # Filter by zone if selected
    if selected_zone:
        zone_customers = customers_df[customers_df['zone_code'] == selected_zone]['customer_id'].tolist()
        current_month_data = current_month_data[current_month_data['customer_id'].isin(zone_customers)]
        prev_month_data = prev_month_data[prev_month_data['customer_id'].isin(zone_customers)]
        filtered_customers = customers_df[customers_df['zone_code'] == selected_zone]
    else:
        filtered_customers = customers_df
    
    # Verificar si hay datos después del filtrado
    if current_month_data.empty or prev_month_data.empty:
        return pd.DataFrame(), pd.Series(), filtered_customers
    
    # Merge current and previous month data
    merged_data = pd.merge(
        current_month_data[['customer_id', 'consumption']], 
        prev_month_data[['customer_id', 'consumption']], 
        on='customer_id', 
        how='inner',
        suffixes=('_current', '_prev')
    )
    
    # Verificar si hay datos después del merge
    if merged_data.empty:
        return pd.DataFrame(), pd.Series(), filtered_customers
    
    # Add customer information (stratum, sanctioned load)
    features = pd.merge(
        merged_data, 
        filtered_customers[['customer_id', 'stratum', 'sanctioned_load']], 
        on='customer_id', 
        how='inner'
    )
    
    # Calculate additional features with manejo seguro de divisiones por cero
    features['consumption_ratio'] = features.apply(
        lambda row: row['consumption_current'] / row['consumption_prev'] 
        if row['consumption_prev'] > 0 else 1.0, 
        axis=1
    )
    
    features['consumption_diff'] = features['consumption_current'] - features['consumption_prev']
    
    features['per_capita_consumption'] = features.apply(
        lambda row: row['consumption_current'] / row['sanctioned_load'] 
        if row['sanctioned_load'] > 0 else row['consumption_current'], 
        axis=1
    )
    
    # Get current month's weather
    if selected_month and selected_year:
        current_weather = weather_df[(weather_df['date'].dt.month == selected_month) & 
                                    (weather_df['date'].dt.year == selected_year)]
    else:
        current_weather = weather_df[weather_df['date'] == latest_date]
    
    # Add weather information (will be the same for all customers in the same period)
    # Manejo mejorado de errores
    if not current_weather.empty:
        try:
            temp = current_weather['temperature'].iloc[0]
            humidity = current_weather['humidity'].iloc[0]
            uv = current_weather['uv_index'].iloc[0]
            
            features['temperature'] = temp
            features['humidity'] = humidity
            features['uv_index'] = uv
        except (IndexError, KeyError) as e:
            # Usar valores predeterminados si hay algún error
            features['temperature'] = 22  # Temperatura promedio para Medellín
            features['humidity'] = 80     # Humedad promedio
            features['uv_index'] = 10     # Índice UV promedio
    else:
        # Si no hay datos de clima, usar valores predeterminados
        features['temperature'] = 22
        features['humidity'] = 80
        features['uv_index'] = 10
    
    # Convertir explícitamente a tipos numéricos para evitar errores
    numeric_columns = ['consumption_current', 'consumption_prev', 'consumption_ratio', 
                     'consumption_diff', 'temperature', 'humidity', 'uv_index', 
                     'stratum', 'sanctioned_load', 'per_capita_consumption']
    
    for col in numeric_columns:
        if col in features.columns:
            features[col] = pd.to_numeric(features[col], errors='coerce')
    
    # Manejar valores NaN
    features = features.fillna({
        'consumption_ratio': 1.0,
        'consumption_diff': 0,
        'per_capita_consumption': features['consumption_current'].mean(),
        'temperature': 22,
        'humidity': 80,
        'uv_index': 10
    })
    
    # Store customer_id for reference but drop it for modeling
    customer_ids = features['customer_id']
    features_for_model = features.drop('customer_id', axis=1)
    
    return features_for_model, customer_ids, filtered_customers

def clean_data(data_df, columns=None, min_consumption=0, max_consumption=None):
    """
    Clean the data by removing outliers and handling missing values.
    
    Parameters:
    -----------
    data_df : DataFrame
        Data to clean
    columns : list, optional
        Columns to clean
    min_consumption : float, optional
        Minimum valid consumption value
    max_consumption : float, optional
        Maximum valid consumption value
        
    Returns:
    --------
    cleaned_df : DataFrame
        Cleaned data
    """
    if data_df.empty:
        return data_df.copy()
        
    if columns is None:
        columns = data_df.columns
    
    cleaned_df = data_df.copy()
    
    # Remove rows with missing values in specified columns
    cleaned_df = cleaned_df.dropna(subset=columns)
    
    # Remove rows with invalid consumption values
    if 'consumption' in cleaned_df.columns:
        cleaned_df = cleaned_df[cleaned_df['consumption'] >= min_consumption]
        
        if max_consumption is not None:
            cleaned_df = cleaned_df[cleaned_df['consumption'] <= max_consumption]
    
    return cleaned_df

def filter_by_stratum(customers_df, consumption_df, stratum):
    """
    Filter data for a specific socioeconomic stratum.
    
    Parameters:
    -----------
    customers_df : DataFrame
        Customer information
    consumption_df : DataFrame
        Consumption data
    stratum : int or list
        Stratum or list of strata to filter by
        
    Returns:
    --------
    filtered_customers : DataFrame
        Filtered customer information
    filtered_consumption : DataFrame
        Filtered consumption data
    """
    if isinstance(stratum, (int, float)):
        stratum = [stratum]
    
    # Filter customers by stratum
    filtered_customers = customers_df[customers_df['stratum'].isin(stratum)]
    
    # Filter consumption data for these customers
    filtered_consumption = consumption_df[
        consumption_df['customer_id'].isin(filtered_customers['customer_id'])
    ]
    
    return filtered_customers, filtered_consumption

def calculate_monthly_stats(consumption_df, group_by='stratum'):
    """
    Calculate monthly consumption statistics grouped by a given field.
    
    Parameters:
    -----------
    consumption_df : DataFrame
        Consumption data
    group_by : str
        Field to group by (e.g., 'stratum', 'zone_code')
        
    Returns:
    --------
    monthly_stats : DataFrame
        Monthly statistics by group
    """
    # Verificar si hay datos
    if consumption_df.empty:
        return pd.DataFrame()
        
    # Merge with customers data if necessary
    if group_by not in consumption_df.columns:
        raise ValueError(f"Column '{group_by}' not found in consumption data")
    
    # Group by date and the specified field
    grouped = consumption_df.groupby(['date', group_by])
    
    # Calculate statistics
    monthly_stats = grouped['consumption'].agg([
        'mean', 'median', 'std', 'min', 'max', 'count'
    ]).reset_index()
    
    return monthly_stats

def detect_consumption_changes(consumption_df, customer_ids=None, threshold=50, window=3):
    """
    Detect significant changes in consumption patterns.
    
    Parameters:
    -----------
    consumption_df : DataFrame
        Consumption data
    customer_ids : list, optional
        List of customer IDs to analyze (default: all)
    threshold : float, optional
        Percent change threshold to flag
    window : int, optional
        Number of months to consider for moving average
        
    Returns:
    --------
    changes_df : DataFrame
        DataFrame with flagged consumption changes
    """
    # Verificar si hay datos
    if consumption_df.empty:
        return pd.DataFrame(columns=[
            'customer_id', 'date', 'consumption', 'previous_avg', 
            'percent_change', 'direction'
        ])
    
    if customer_ids is None:
        customer_ids = consumption_df['customer_id'].unique()
    
    changes = []
    
    for customer_id in customer_ids:
        # Get customer's consumption history
        history = consumption_df[consumption_df['customer_id'] == customer_id].sort_values('date')
        
        if len(history) < window + 1:
            continue
        
        # Calculate moving average
        history['rolling_avg'] = history['consumption'].rolling(window=window).mean()
        
        # Skip first 'window' months where rolling average is not available
        valid_history = history.dropna(subset=['rolling_avg'])
        
        for i in range(1, len(valid_history)):
            current = valid_history.iloc[i]
            previous = valid_history.iloc[i-1]
            
            # Evitar división por cero
            if previous['rolling_avg'] > 0:
                percent_change = ((current['consumption'] - previous['rolling_avg']) / 
                                 previous['rolling_avg'] * 100)
                
                if abs(percent_change) >= threshold:
                    changes.append({
                        'customer_id': customer_id,
                        'date': current['date'],
                        'consumption': current['consumption'],
                        'previous_avg': previous['rolling_avg'],
                        'percent_change': percent_change,
                        'direction': 'increase' if percent_change > 0 else 'decrease'
                    })
    
    if changes:
        return pd.DataFrame(changes)
    else:
        return pd.DataFrame(columns=[
            'customer_id', 'date', 'consumption', 'previous_avg', 
            'percent_change', 'direction'
        ])

def integrate_weather_data(consumption_df, weather_df):
    """
    Integrate weather data with consumption data.
    
    Parameters:
    -----------
    consumption_df : DataFrame
        Consumption data
    weather_df : DataFrame
        Weather data
        
    Returns:
    --------
    integrated_df : DataFrame
        Consumption data with added weather information
    """
    # Verificar si hay datos
    if consumption_df.empty or weather_df.empty:
        return consumption_df.copy()
    
    # Ensure date columns are datetime
    consumption_df['date'] = pd.to_datetime(consumption_df['date'])
    weather_df['date'] = pd.to_datetime(weather_df['date'])
    
    # Merge consumption data with weather data
    integrated_df = pd.merge(
        consumption_df,
        weather_df,
        on='date',
        how='left'
    )
    
    return integrated_df