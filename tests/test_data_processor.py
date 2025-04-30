import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from modules.data_processor import (
    clean_data, filter_by_stratum, calculate_monthly_stats,
    detect_consumption_changes, integrate_weather_data
)

class TestDataProcessor(unittest.TestCase):
    """Tests for data processing functions."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample customer data
        self.customers_df = pd.DataFrame({
            'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005'],
            'stratum': [1, 2, 3, 4, 5],
            'latitude': [6.25, 6.26, 6.24, 6.23, 6.22],
            'longitude': [-75.58, -75.59, -75.57, -75.56, -75.55],
            'zone_code': ['Z1', 'Z1', 'Z2', 'Z2', 'Z3'],
            'sanctioned_load': [10, 15, 20, 25, 30]
        })
        
        # Create sample consumption data
        dates = pd.date_range(start='2020-01-01', periods=12, freq='M')
        consumption_data = []
        
        for customer_id in self.customers_df['customer_id']:
            stratum = self.customers_df[self.customers_df['customer_id'] == customer_id]['stratum'].values[0]
            base = 100 * stratum  # Base consumption depends on stratum
            
            for i, date in enumerate(dates):
                # Add seasonal variation
                seasonal = 1.0 + 0.2 * np.sin(np.pi * i / 6)
                # Add random noise
                noise = 0.9 + 0.2 * np.random.random()
                
                consumption = base * seasonal * noise
                
                consumption_data.append({
                    'customer_id': customer_id,
                    'date': date,
                    'consumption': consumption,
                    'month': date.month,
                    'year': date.year
                })
        
        self.consumption_df = pd.DataFrame(consumption_data)
        
        # Create sample weather data
        weather_data = []
        
        for date in dates:
            # Simplified seasonal weather patterns
            month = date.month
            temp_base = 22 - 2 * np.cos(np.pi * month / 6)
            humid_base = 80 + 5 * np.sin(np.pi * month / 6)
            uv_base = 10 - 2 * np.cos(np.pi * month / 6)
            
            weather_data.append({
                'date': date,
                'temperature': temp_base + np.random.randn(),
                'humidity': humid_base + 3 * np.random.randn(),
                'uv_index': uv_base + np.random.randn()
            })
        
        self.weather_df = pd.DataFrame(weather_data)
    
    def test_clean_data(self):
        """Test clean_data function."""
        # Add some missing values and outliers
        dirty_df = self.consumption_df.copy()
        dirty_df.loc[0, 'consumption'] = np.nan
        dirty_df.loc[1, 'consumption'] = -10
        dirty_df.loc[2, 'consumption'] = 5000
        
        # Clean the data
        cleaned_df = clean_data(
            dirty_df, 
            columns=['consumption'],
            min_consumption=0,
            max_consumption=3000
        )
        
        # Check if missing values and outliers are removed
        self.assertEqual(len(cleaned_df), len(dirty_df) - 3)
        self.assertTrue(cleaned_df['consumption'].min() >= 0)
        self.assertTrue(cleaned_df['consumption'].max() <= 3000)
    
    def test_filter_by_stratum(self):
        """Test filter_by_stratum function."""
        # Filter for a single stratum
        customers_s1, consumption_s1 = filter_by_stratum(
            self.customers_df, 
            self.consumption_df, 
            stratum=1
        )
        
        # Check results
        self.assertEqual(len(customers_s1), 1)
        self.assertEqual(customers_s1['customer_id'].iloc[0], 'C001')
        self.assertEqual(len(consumption_s1), 12)  # 12 months for 1 customer
        
        # Filter for multiple strata
        customers_s23, consumption_s23 = filter_by_stratum(
            self.customers_df, 
            self.consumption_df, 
            stratum=[2, 3]
        )
        
        # Check results
        self.assertEqual(len(customers_s23), 2)
        self.assertEqual(set(customers_s23['customer_id']), {'C002', 'C003'})
        self.assertEqual(len(consumption_s23), 24)  # 12 months for 2 customers
    
    def test_calculate_monthly_stats(self):
        """Test calculate_monthly_stats function."""
        # Add stratum information to consumption data
        consumption_with_stratum = pd.merge(
            self.consumption_df,
            self.customers_df[['customer_id', 'stratum']],
            on='customer_id'
        )
        
        # Calculate statistics by stratum
        stats = calculate_monthly_stats(consumption_with_stratum, group_by='stratum')
        
        # Check results
        self.assertEqual(len(stats), 60)  # 12 months * 5 strata
        self.assertIn('mean', stats.columns)
        self.assertIn('median', stats.columns)
        self.assertIn('std', stats.columns)
    
    def test_detect_consumption_changes(self):
        """Test detect_consumption_changes function."""
        # Create a modified consumption dataframe with some significant changes
        modified_consumption = self.consumption_df.copy()
        
        # Add a significant increase for customer C001 in month 6
        idx = (modified_consumption['customer_id'] == 'C001') & (modified_consumption['month'] == 6)
        modified_consumption.loc[idx, 'consumption'] *= 2.5
        
        # Add a significant decrease for customer C002 in month 8
        idx = (modified_consumption['customer_id'] == 'C002') & (modified_consumption['month'] == 8)
        modified_consumption.loc[idx, 'consumption'] *= 0.4
        
        # Detect changes
        changes = detect_consumption_changes(modified_consumption, threshold=50, window=3)
        
        # Check results
        self.assertGreaterEqual(len(changes), 2)  # At least the two changes we added
        
        # Check if the specific changes are detected
        c001_changes = changes[changes['customer_id'] == 'C001']
        c002_changes = changes[changes['customer_id'] == 'C002']
        
        self.assertGreaterEqual(len(c001_changes), 1)
        self.assertGreaterEqual(len(c002_changes), 1)
        
        # Check directions
        c001_increase = c001_changes[c001_changes['direction'] == 'increase']
        c002_decrease = c002_changes[c002_changes['direction'] == 'decrease']
        
        self.assertGreaterEqual(len(c001_increase), 1)
        self.assertGreaterEqual(len(c002_decrease), 1)
    
    def test_integrate_weather_data(self):
        """Test integrate_weather_data function."""
        # Integrate weather data
        integrated_df = integrate_weather_data(self.consumption_df, self.weather_df)
        
        # Check results
        self.assertEqual(len(integrated_df), len(self.consumption_df))
        self.assertIn('temperature', integrated_df.columns)
        self.assertIn('humidity', integrated_df.columns)
        self.assertIn('uv_index', integrated_df.columns)
        
        # Check if weather data is correctly aligned with dates
        for date in self.consumption_df['date'].unique():
            weather_data = self.weather_df[self.weather_df['date'] == date]
            integrated_data = integrated_df[integrated_df['date'] == date]
            
            self.assertEqual(
                weather_data['temperature'].iloc[0],
                integrated_data['temperature'].iloc[0]
            )

if __name__ == '__main__':
    unittest.main()