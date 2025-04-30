import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static

# Import modules
from modules.mgd_model import MGDAnomaly
from modules.data_processor import load_sample_data, prepare_features
from modules.visualization import (
    plot_consumption_patterns, create_anomaly_map, plot_anomaly_distribution,
    plot_feature_importance, plot_stratum_distribution, plot_scatter_comparison,
    create_kpi_cards
)

# Set page configuration
st.set_page_config(
    page_title="Fraudulent Consumer Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        font-weight: bold;
        margin-top: 1rem;
    }
    .card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
    }
    .stAlert {
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown("<div class='main-header'>Fraudulent Consumer Detection in Power Utilities</div>", unsafe_allow_html=True)
st.markdown("""
This application implements the Multivariate Gaussian Distribution method for detecting fraudulent consumers in power utilities, 
focusing on socioeconomic stratification and consumption patterns.
""")

# Main function
def main():
    # Load data
    if 'customers_df' not in st.session_state:
        st.session_state.customers_df, st.session_state.consumption_df, st.session_state.weather_df = load_sample_data()
    
    customers_df = st.session_state.customers_df
    consumption_df = st.session_state.consumption_df
    weather_df = st.session_state.weather_df
    
    # Sidebar filters
    st.sidebar.markdown("### Data Filters")
    
    # Filter by date
    available_years = sorted(consumption_df['year'].unique())
    selected_year = st.sidebar.selectbox("Select Year", available_years, index=len(available_years)-1)
    
    available_months = sorted(consumption_df[consumption_df['year'] == selected_year]['month'].unique())
    selected_month = st.sidebar.selectbox("Select Month", available_months, index=len(available_months)-1)
    
    # Filter by zone
    available_zones = ['All'] + sorted(customers_df['zone_code'].unique().tolist())
    selected_zone_option = st.sidebar.selectbox("Select Zone", available_zones)
    selected_zone = None if selected_zone_option == 'All' else selected_zone_option
    
    # MGD parameters
    st.sidebar.markdown("### Model Parameters")
    anomaly_threshold_factor = st.sidebar.slider(
        "Anomaly Threshold Factor", 
        min_value=1.5, 
        max_value=5.0, 
        value=3.0, 
        step=0.1,
        help="Multiplier for standard deviation to set anomaly threshold"
    )
    
    # Feature selection
    st.sidebar.markdown("### Feature Selection")
    use_consumption = st.sidebar.checkbox("Consumption Data", value=True)
    use_stratum = st.sidebar.checkbox("Social Stratification", value=True)
    use_weather = st.sidebar.checkbox("Weather Profile", value=True)
    
    # Prepare features
    try:
        features, customer_ids, filtered_customers = prepare_features(
            customers_df, 
            consumption_df, 
            weather_df,
            selected_month=selected_month,
            selected_year=selected_year,
            selected_zone=selected_zone
        )
        
        # Select features based on user choices
        selected_features = []
        if use_consumption:
            selected_features.extend(['consumption_current', 'consumption_prev', 'consumption_ratio', 'consumption_diff'])
        if use_stratum:
            selected_features.extend(['stratum', 'sanctioned_load', 'per_capita_consumption'])
        if use_weather:
            selected_features.extend(['temperature', 'humidity', 'uv_index'])
        
        # Check if any features selected
        if not selected_features:
            st.error("Please select at least one feature category")
            return
        
        # Check if we have any data
        if features.empty:
            st.warning("No data available for the selected filters. Please try different filter settings.")
            
            # Display empty visualizations
            threshold = 0
            anomaly_scores = np.array([])
            anomaly_mask = np.array([])
            
            # Display KPI cards with zeros
            kpi_metrics = {
                'total_customers': 0,
                'anomalies_detected': 0,
                'detection_rate': 0,
                'precision': None,
                'recall': None,
                'f1': None
            }
        else:
            # Filter features
            common_features = list(set(selected_features).intersection(set(features.columns)))
            if not common_features:
                st.warning(f"No selected features found in data. Available features: {', '.join(features.columns)}")
                return
                
            features_subset = features[common_features]
            
            # Train MGD model
            mgd_model = MGDAnomaly()
            mgd_model.fit(features_subset)
            
            # Predict anomalies
            anomaly_scores = mgd_model.score_samples(features_subset)
            
            # Calculate threshold
            threshold = np.mean(anomaly_scores) + anomaly_threshold_factor * np.std(anomaly_scores)
            anomaly_mask = anomaly_scores > threshold
            
            # Create KPI metrics
            kpi_metrics = create_kpi_cards(anomaly_scores, threshold, filtered_customers)
        
        # Display KPI cards
        st.markdown("<div class='sub-header'>Detection Results</div>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-label'>Total Customers</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{kpi_metrics['total_customers']}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-label'>Detected Anomalies</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{kpi_metrics['anomalies_detected']}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-label'>Detection Rate</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{kpi_metrics['detection_rate']:.1f}%</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col4:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-label'>Hit Rate (Precision)</div>", unsafe_allow_html=True)
            if kpi_metrics['precision'] is not None:
                st.markdown(f"<div class='metric-value'>{kpi_metrics['precision']:.1f}%</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='metric-value'>N/A</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Anomaly map
        st.markdown("<div class='sub-header'>Geospatial Analysis</div>", unsafe_allow_html=True)
        
        try:
            # Usar st_folium en lugar de folium_static para evitar la advertencia de depreciación
            import streamlit.components.v1 as components
            
            anomaly_map = create_anomaly_map(filtered_customers, anomaly_scores, threshold)
            
            # Convertir el mapa a HTML y mostrarlo con components
            map_html = anomaly_map._repr_html_()
            components.html(map_html, height=600, width=1200)
        except Exception as e:
            st.error(f"Error displaying map: {str(e)}")
            st.write("Falling back to basic map:")
            folium_static(create_anomaly_map(filtered_customers[:10], anomaly_scores[:10] if len(anomaly_scores) >= 10 else anomaly_scores, threshold), width=1200, height=600)
        
        # Visualizations
        st.markdown("<div class='sub-header'>Consumption Analysis</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Anomaly distribution
            st.plotly_chart(plot_anomaly_distribution(anomaly_scores, threshold), use_container_width=True)
        
        with col2:
            # Scatter plot
            st.plotly_chart(plot_scatter_comparison(features, anomaly_mask), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Consumption patterns by stratum
            st.plotly_chart(plot_consumption_patterns(consumption_df, customers_df), use_container_width=True)
        
        with col2:
            # Stratum distribution
            st.plotly_chart(plot_stratum_distribution(filtered_customers, anomaly_mask), use_container_width=True)
        
        # Feature importance (only if we have data and model is fitted)
        if not features.empty and 'mgd_model' in locals() and mgd_model.fitted:
            st.markdown("<div class='sub-header'>Feature Analysis</div>", unsafe_allow_html=True)
            st.plotly_chart(plot_feature_importance(mgd_model, common_features), use_container_width=True)
        
        # Display anomalous customers (only if we have anomalies)
        if kpi_metrics['anomalies_detected'] > 0:
            st.markdown("<div class='sub-header'>Detected Fraudulent Consumers</div>", unsafe_allow_html=True)
            
            anomaly_indices = np.where(anomaly_mask)[0]
            if len(anomaly_indices) > 0:
                anomaly_customers = filtered_customers.iloc[anomaly_indices].copy()
                anomaly_customers['anomaly_score'] = anomaly_scores[anomaly_indices]
                
                # Add consumption data
                try:
                    current_month_consumption = consumption_df[
                        (consumption_df['month'] == selected_month) & 
                        (consumption_df['year'] == selected_year) &
                        (consumption_df['customer_id'].isin(anomaly_customers['customer_id']))
                    ][['customer_id', 'consumption']]
                    
                    anomaly_customers = pd.merge(
                        anomaly_customers,
                        current_month_consumption,
                        on='customer_id',
                        how='left'
                    )
                    
                    # Get previous month consumption
                    prev_month_consumption = consumption_df[
                        (consumption_df['month'] == selected_month) & 
                        (consumption_df['year'] == selected_year - 1) &
                        (consumption_df['customer_id'].isin(anomaly_customers['customer_id']))
                    ][['customer_id', 'consumption']]
                    
                    anomaly_customers = pd.merge(
                        anomaly_customers,
                        prev_month_consumption,
                        on='customer_id',
                        how='left',
                        suffixes=('_current', '_prev')
                    )
                    
                    # Calculate percent change
                    anomaly_customers['percent_change'] = ((anomaly_customers['consumption_current'] - 
                                                         anomaly_customers['consumption_prev']) / 
                                                        anomaly_customers['consumption_prev'] * 100)
                    
                    # Display as table
                    display_cols = ['customer_id', 'zone_code', 'stratum', 'consumption_current', 
                                  'consumption_prev', 'percent_change', 'anomaly_score']
                    
                    st.dataframe(
                        anomaly_customers[display_cols].sort_values('anomaly_score', ascending=False),
                        column_config={
                            'customer_id': 'Customer ID',
                            'zone_code': 'Zone',
                            'stratum': 'Stratum',
                            'consumption_current': st.column_config.NumberColumn('Current Consumption (kWh)', format="%.1f"),
                            'consumption_prev': st.column_config.NumberColumn('Previous Year Consumption (kWh)', format="%.1f"),
                            'percent_change': st.column_config.NumberColumn('Change (%)', format="%.1f%%"),
                            'anomaly_score': st.column_config.NumberColumn('Anomaly Score', format="%.2f")
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    # Export options
                    csv_data = anomaly_customers[display_cols].to_csv(index=False)
                    st.download_button(
                        label="Download Detected Anomalies",
                        data=csv_data,
                        file_name=f"fraudulent_consumers_{selected_year}_{selected_month}.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Error displaying anomalous customers: {str(e)}")
                    st.write("Detailed anomaly information is not available.")
            else:
                st.info("No fraudulent consumers detected with the current threshold.")
        else:
            st.info("No fraudulent consumers detected with the current threshold.")
        
        # Add explanation
        st.markdown("<div class='sub-header'>About the Model</div>", unsafe_allow_html=True)
        st.markdown("""
        This application implements the Multivariate Gaussian Distribution method described in the paper 
        "A Novel Features-Based Multivariate Gaussian Distribution Method for the Fraudulent Consumers Detection 
        in the Power Utilities of Developing Countries."
        
        The model incorporates:
        
        1. **Social Class Stratification** - Different consumption patterns based on socioeconomic status
        2. **Weather Profile** - Accounting for seasonal variations in consumption
        3. **Consumption Patterns** - Analyzing changes in consumption over time
        
        Anomalies are detected based on the Mahalanobis distance, which identifies customers whose 
        consumption patterns deviate significantly from the norm within their socioeconomic group.
        """)

        # Add fraud types explanation
        with st.expander("Types of Fraud Detected"):
            st.markdown("""
            ### Fraud Types Detected by the Model
            
            #### In Electricity Distribution:
            - **Meter Tampering**: Manipulation resulting in lower electricity readings
            - **By-passing Meter**: Direct connection to electricity supply bypassing the meter
            - **Meter Reversal**: Inverting meter connection to reduce consumption readings
            - **Commercial Use on Domestic Tariff**: Using cheaper residential rates for business
            - **Illegal Extension**: Unauthorized extension of electricity to other premises
            
            #### In Natural Gas Distribution:
            - **Meter Tampering**: Manipulation to reduce gas consumption readings
            - **Tariff Violation**: Commercial use on domestic tariff
            - **Meter Reversal**: Inverting meter connections
            - **Illegal Line Extension**: Unauthorized extension of gas lines
            - **Illegal Use of Compressors**: Using compressors to draw more gas
            - **Electricity Generation from Natural Gas**: Unauthorized use of natural gas for power generation
            """)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please try different filter settings or refresh the page.")

# Run the app
if __name__ == "__main__":
    main()