# Fraudulent Consumer Detection for Power Utilities

This application implements the Multivariate Gaussian Distribution (MGD) method for detecting fraudulent consumers in power utilities of developing countries, as described in the paper "A Novel Features-Based Multivariate Gaussian Distribution Method for the Fraudulent Consumers Detection in the Power Utilities of Developing Countries."

## Features

- **Multivariate Gaussian Distribution** model for anomaly detection
- **Social Class Stratification** integration to account for socioeconomic factors
- **Weather Profile** incorporation to handle seasonal variations
- **Interactive Dashboard** with visualizations and geospatial mapping
- **Flexible Threshold** settings to adjust sensitivity
- **Feature Importance** analysis

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To run the application:

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501` in your web browser.

## Data Sources

The application currently uses synthetically generated sample data that simulates:

- Customer information (ID, stratum, location, zone)
- Historical consumption data (24 months)
- Weather data (temperature, humidity, UV index)

In a production environment, this would be replaced with real utility company data.

## Types of Fraud Detected

### In Electricity Distribution:
- Meter tampering
- By-passing meter
- Meter reversal
- Commercial use on domestic tariff
- Illegal extension

### In Natural Gas Distribution:
- Meter tampering
- Tariff violation
- Meter reversal
- Illegal line extension
- Illegal use of compressors
- Electricity generation from natural gas

## Configuration

The application provides several configuration options:

- **Data Filters**: Select specific time periods and zones
- **Model Parameters**: Adjust the anomaly threshold sensitivity
- **Feature Selection**: Choose which data categories to include in the model

## License

This project is licensed under the MIT License - see the LICENSE file for details.