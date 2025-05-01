import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
from folium.plugins import MarkerCluster, HeatMap, Fullscreen, MiniMap

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

# Custom CSS with improved styling
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.8rem;
        color: #0066cc;
        font-weight: bold;
        margin-bottom: 1rem;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #f8f9fa, #e9ecef, #f8f9fa);
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* Sub-header styling */
    .sub-header {
        font-size: 1.6rem;
        color: #0066cc;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e9ecef;
    }
    
    /* Card styling - updated for 2x2 grid */
    .card-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 15px;
        margin-bottom: 1.5rem;
    }
    
    .kpi-card {
        background: linear-gradient(145deg, #ffffff, #f5f7fa);
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-left: 4px solid transparent;
        height: 100%;
        display: flex;
        flex-direction: column;
        margin-bottom: 15px;
    }
    
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.12);
    }
    
    .kpi-card.customers {
        border-left-color: #4361ee;
    }
    
    .kpi-card.anomalies {
        border-left-color: #ef476f;
    }
    
    .kpi-card.rate {
        border-left-color: #ffd166;
    }
    
    .kpi-card.precision {
        border-left-color: #06d6a0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        font-weight: 600;
        margin-bottom: 4px;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #212529;
        line-height: 1.2;
    }
    
    .metric-trend {
        display: flex;
        align-items: center;
        margin-top: auto;
        padding-top: 5px;
        font-size: 0.75rem;
        color: #6c757d;
    }

    .trend-up {
        color: #ef476f;
    }
    
    .trend-down {
        color: #06d6a0;
    }
    
    /* Maps and chart containers */
    .map-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        margin-bottom: 2rem;
        background: white;
    }
    
    .chart-container {
        background: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
    }
    
    /* Table styling */
    .dataframe-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        margin: 1.5rem 0;
        padding: 20px;
        background: white;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #0066cc;
        color: white;
        border-radius: 50px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #0052a3;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border-radius: 5px 5px 0px 0px;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #0066cc;
        color: white;
    }
    
    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Global page background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4ecf7 100%);
    }
    
    /* Make the sidebar more attractive */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e9ecef;
    }
    
    /* Improved sidebar header */
    .sidebar-header {
        background-color: #0066cc;
        color: white;
        padding: 15px;
        margin: -1rem -1rem 1rem -1rem;
        border-radius: 0 0 10px 0;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown("<div class='main-header'>⚡ Fraudulent Consumer Detection in Power Utilities</div>", unsafe_allow_html=True)
st.markdown("""
This intelligent system implements the Multivariate Gaussian Distribution method for detecting fraudulent consumers in power utilities, 
analyzing socioeconomic stratification, consumption patterns, and weather impacts to identify anomalies with high precision.
""")

# Main function
def main():
    # Load data
    if 'customers_df' not in st.session_state:
        with st.spinner("Loading data..."):
            st.session_state.customers_df, st.session_state.consumption_df, st.session_state.weather_df = load_sample_data()
    
    customers_df = st.session_state.customers_df
    consumption_df = st.session_state.consumption_df
    weather_df = st.session_state.weather_df
    
    # Sidebar filters and styling
    st.sidebar.markdown("<div class='sidebar-header'>Fraud Detection Controls</div>", unsafe_allow_html=True)
    
    st.sidebar.markdown("### 📅 Data Filters")
    
    # Filter by date with improved UI
    col1, col2 = st.sidebar.columns(2)
    
    available_years = sorted(consumption_df['year'].unique())
    selected_year = col1.selectbox("Year", available_years, index=len(available_years)-1)
    
    available_months = sorted(consumption_df[consumption_df['year'] == selected_year]['month'].unique())
    selected_month = col2.selectbox("Month", available_months, index=len(available_months)-1)
    
    # Filter by zone with search
    available_zones = ['All'] + sorted(customers_df['zone_code'].unique().tolist())
    selected_zone_option = st.sidebar.selectbox("Zone Filter", available_zones, 
                                              help="Filter data by specific geographic zone")
    selected_zone = None if selected_zone_option == 'All' else selected_zone_option
    
    # MGD parameters with improved sliders
    st.sidebar.markdown("### ⚙️ Model Parameters")
    anomaly_threshold_factor = st.sidebar.slider(
        "Anomaly Threshold Factor", 
        min_value=1.5, 
        max_value=5.0, 
        value=3.0, 
        step=0.1,
        help="Lower values detect more anomalies (may increase false positives). Higher values detect fewer, more certain anomalies."
    )
    
    # Feature selection with better organization
    st.sidebar.markdown("### 🔍 Feature Selection")
    
    feature_col1, feature_col2 = st.sidebar.columns(2)
    
    use_consumption = feature_col1.checkbox("Consumption", value=True, help="Include consumption patterns in analysis")
    use_stratum = feature_col1.checkbox("Stratification", value=True, help="Include socioeconomic factors")
    use_weather = feature_col2.checkbox("Weather", value=True, help="Include climate factors")
    use_historical = feature_col2.checkbox("Historical", value=True, help="Include year-over-year comparison")
    
    # Prepare features
    try:
        with st.spinner("Analyzing data..."):
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
                selected_features.extend(['consumption_current', 'consumption_diff'])
                if use_historical:
                    selected_features.extend(['consumption_prev', 'consumption_ratio'])
            elif use_historical:  # Solo si no se seleccionó consumption
                selected_features.extend(['consumption_prev', 'consumption_ratio'])
            if use_stratum:
                selected_features.extend(['stratum', 'sanctioned_load', 'per_capita_consumption'])
            if use_weather:
                selected_features.extend(['temperature', 'humidity', 'uv_index'])
            
            # Check if any features selected
            if not selected_features:
                st.error("Please select at least one feature category for analysis")
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
        
        # Display KPI cards with improved design - now 2x2 layout
        st.markdown("<div class='sub-header'>📊 Detection Results</div>", unsafe_allow_html=True)
        
        # Create a container for the cards
        col1, col2 = st.columns(2)
        
        with col1:
            # Card 1: Total Customers
            st.markdown(f"""
            <div class='kpi-card customers'>
                <div class='metric-label'>Total Customers</div>
                <div class='metric-value'>{kpi_metrics['total_customers']:,}</div>
                <div class='metric-trend'>
                    <span>Active consumers</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Card 3: Detection Rate
            st.markdown(f"""
            <div class='kpi-card rate'>
                <div class='metric-label'>Detection Rate</div>
                <div class='metric-value'>{kpi_metrics['detection_rate']:.1f}%</div>
                <div class='metric-trend'>
                    <span>Anomalous %</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Card 2: Detected Anomalies
            st.markdown(f"""
            <div class='kpi-card anomalies'>
                <div class='metric-label'>Detected Anomalies</div>
                <div class='metric-value'>{kpi_metrics['anomalies_detected']:,}</div>
                <div class='metric-trend {"trend-up" if kpi_metrics["anomalies_detected"] > 0 else ""}'>
                    <span>Potential fraud</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Card 4: Hit Rate (Precision)
            precision_value = f"{kpi_metrics['precision']:.1f}%" if kpi_metrics['precision'] is not None else "N/A"
            st.markdown(f"""
            <div class='kpi-card precision'>
                <div class='metric-label'>Hit Rate (Precision)</div>
                <div class='metric-value'>{precision_value}</div>
                <div class='metric-trend'>
                    <span>Validation</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # Close the container
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Create tabs for better organization
        tab1, tab2, tab3 = st.tabs(["📍 Geospatial Analysis", "📈 Consumption Analysis", "🔎 Fraud Detection"])

        with tab1:
            st.markdown("<div class='sub-header'>Distribución Geográfica de Anomalías</div>", unsafe_allow_html=True)
            
            try:
                # Aseguramos que los tamaños coincidan
                if len(filtered_customers) != len(anomaly_scores):
                    n_samples = min(len(filtered_customers), len(anomaly_scores))
                    map_customers = filtered_customers.iloc[:n_samples].reset_index(drop=True)
                    map_scores = anomaly_scores[:n_samples]
                else:
                    map_customers = filtered_customers
                    map_scores = anomaly_scores
                
                # Creamos y mostramos el mapa
                anomaly_map = create_anomaly_map(map_customers, map_scores, threshold)
                folium_static(anomaly_map, width=1200, height=600)
                
                # Información útil
                st.info("**Nota:** Azul = consumidores normales, Rojo = anomalías potencialmente fraudulentas.")
                
            except Exception as e:
                st.error(f"Error al mostrar el mapa: {str(e)}")
                # Crear un mapa básico como fallback
                basic_map = folium.Map(location=[6.25, -75.58], zoom_start=12)
                folium.Marker(
                    location=[6.25, -75.58],
                    popup="Error mostrando el mapa de datos",
                    icon=folium.Icon(color="red")
                ).add_to(basic_map)
                folium_static(basic_map, width=1200, height=600)
            
        with tab2:
            # Consumption Analysis Tab
            st.markdown("<div class='sub-header'>Consumption Patterns Analysis</div>", unsafe_allow_html=True)
            
            # Improved layout with 2x2 grid
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                # Enhanced anomaly distribution
                anomaly_dist_fig = plot_anomaly_distribution(anomaly_scores, threshold)
                
                # Add shaded area for threshold if we have data
                if len(anomaly_scores) > 0:
                    anomaly_dist_fig.add_shape(
                        type="rect",
                        x0=threshold, y0=0,
                        x1=max(anomaly_scores) if len(anomaly_scores) > 0 else threshold + 5, 
                        y1=1000,
                        fillcolor="rgba(255, 0, 0, 0.1)",
                        line=dict(width=0),
                        layer="below"
                    )
                
                anomaly_dist_fig.update_layout(
                    title="Distribution of Anomaly Scores",
                    height=400,
                    margin=dict(t=50, b=50, l=50, r=25)
                )
                st.plotly_chart(anomaly_dist_fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                # Enhanced scatter comparison
                scatter_fig = plot_scatter_comparison(features, anomaly_mask)
                scatter_fig.update_layout(
                    title="Current vs. Previous Consumption by Anomaly Status",
                    height=400,
                    margin=dict(t=50, b=50, l=50, r=25)
                )
                st.plotly_chart(scatter_fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                # Enhanced consumption patterns
                patterns_fig = plot_consumption_patterns(consumption_df, customers_df)
                patterns_fig.update_layout(
                    title="Consumption Patterns by Social Stratum",
                    height=400,
                    margin=dict(t=50, b=50, l=50, r=25)
                )
                st.plotly_chart(patterns_fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                # Enhanced stratum distribution
                stratum_fig = plot_stratum_distribution(filtered_customers, anomaly_mask)
                stratum_fig.update_layout(
                    title="Anomaly Distribution by Social Stratum",
                    height=400,
                    margin=dict(t=50, b=50, l=50, r=25)
                )
                st.plotly_chart(stratum_fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Feature importance (only if we have data and model is fitted)
            if not features.empty and 'mgd_model' in locals() and mgd_model.fitted:
                st.markdown("<div class='sub-header'>Feature Significance Analysis</div>", unsafe_allow_html=True)
                st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                
                # Enhanced feature importance visualization
                feature_fig = plot_feature_importance(mgd_model, common_features)
                feature_fig.update_layout(
                    title="Relative Importance of Features in Anomaly Detection",
                    height=500,
                    margin=dict(t=50, b=70, l=70, r=25)
                )
                st.plotly_chart(feature_fig, use_container_width=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Add feature explanation
                with st.expander("📚 Understanding Feature Importance"):
                    st.markdown("""
                    ### How to Interpret Feature Importance
                    
                    **Feature importance** in this model indicates how much each variable contributes to identifying fraudulent activity:
                    
                    - **Higher values** mean the feature is more critical for detecting anomalies
                    - **Related features** often show similar importance values
                    - **Consumption-related features** typically have significant importance as they directly reflect usage patterns
                    - **Socioeconomic factors** help contextualize consumption relative to expected patterns for specific demographic groups
                    
                    The model uses these features to establish a multidimensional profile of normal consumer behavior, then identifies outliers that deviate from these established patterns.
                    """)
            
        with tab3:
            # Fraud Detection Tab
            st.markdown("<div class='sub-header'>Detected Fraudulent Consumers</div>", unsafe_allow_html=True)
            
            # Display anomalous customers (only if we have anomalies)
            if kpi_metrics['anomalies_detected'] > 0:
                anomaly_indices = np.where(anomaly_mask)[0]
                if len(anomaly_indices) > 0:
                    try:
                        anomaly_customers = filtered_customers.iloc[anomaly_indices].copy()
                        anomaly_customers['anomaly_score'] = anomaly_scores[anomaly_indices]
                        
                        # Add consumption data
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
                        
                        # Avoid division by zero errors
                        anomaly_customers['consumption_prev'] = anomaly_customers['consumption_prev'].replace(0, np.nan)
                        
                        # Calculate percent change only where we have previous consumption data
                        anomaly_customers['percent_change'] = None
                        valid_mask = ~anomaly_customers['consumption_prev'].isna()
                        if valid_mask.any():
                            anomaly_customers.loc[valid_mask, 'percent_change'] = (
                                (anomaly_customers.loc[valid_mask, 'consumption_current'] - 
                                 anomaly_customers.loc[valid_mask, 'consumption_prev']) / 
                                anomaly_customers.loc[valid_mask, 'consumption_prev'] * 100
                            )
                        
                        # Enhanced risk score (composite of anomaly score and percent change)
                        # Normalize both values to 0-100 scale
                        if not anomaly_customers.empty:
                            max_anomaly = anomaly_customers['anomaly_score'].max()
                            min_anomaly = anomaly_customers['anomaly_score'].min()
                            
                            # Normalize anomaly score to 0-100
                            if max_anomaly > min_anomaly:
                                anomaly_customers['normalized_score'] = 100 * (anomaly_customers['anomaly_score'] - min_anomaly) / (max_anomaly - min_anomaly)
                            else:
                                anomaly_customers['normalized_score'] = 50  # Default if all scores are the same
                            
                            # Calculate a risk level based on normalized score
                            def get_risk_level(score):
                                if score >= 80:
                                    return "Critical"
                                elif score >= 60:
                                    return "High"
                                elif score >= 40:
                                    return "Medium"
                                else:
                                    return "Low"
                            
                            anomaly_customers['risk_level'] = anomaly_customers['normalized_score'].apply(get_risk_level)
                        
                        # Display as enhanced table
                        st.markdown("<div class='dataframe-container'>", unsafe_allow_html=True)
                        
                        # Configure columns for better display
                        display_cols = ['customer_id', 'zone_code', 'stratum', 'consumption_current', 
                                      'consumption_prev', 'percent_change', 'anomaly_score', 'risk_level']
                        
                        st.dataframe(
                            anomaly_customers[display_cols].sort_values('anomaly_score', ascending=False),
                            column_config={
                                'customer_id': st.column_config.TextColumn('Customer ID'),
                                'zone_code': st.column_config.TextColumn('Zone'),
                                'stratum': st.column_config.NumberColumn('Stratum', format="%d"),
                                'consumption_current': st.column_config.NumberColumn('Current Consumption (kWh)', format="%.1f"),
                                'consumption_prev': st.column_config.NumberColumn('Previous Year (kWh)', format="%.1f"),
                                'percent_change': st.column_config.NumberColumn('Change (%)', format="%.1f%%"),
                                'anomaly_score': st.column_config.NumberColumn('Anomaly Score', format="%.2f"),
                                'risk_level': st.column_config.TextColumn('Risk Level')
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Chart showing anomaly customers by risk level
                        risk_counts = anomaly_customers['risk_level'].value_counts().reset_index()
                        risk_counts.columns = ['Risk Level', 'Count']
                        
                        if not risk_counts.empty:
                            # Define color map for risk levels
                            risk_colors = {
                                'Critical': '#d32f2f',
                                'High': '#f57c00',
                                'Medium': '#ffd166',
                                'Low': '#06d6a0'
                            }
                            
                            # Create risk level distribution chart
                            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                            
                            # Sort by risk severity
                            risk_order = ['Critical', 'High', 'Medium', 'Low']
                            risk_counts['Risk Level'] = pd.Categorical(
                                risk_counts['Risk Level'], 
                                categories=risk_order, 
                                ordered=True
                            )
                            risk_counts = risk_counts.sort_values('Risk Level')
                            
                            risk_fig = px.bar(
                                risk_counts, 
                                x='Risk Level', 
                                y='Count',
                                color='Risk Level',
                                color_discrete_map=risk_colors,
                                title="Distribution of Anomalies by Risk Level"
                            )
                            
                            risk_fig.update_layout(
                                xaxis_title="Risk Level",
                                yaxis_title="Number of Customers",
                                height=400
                            )
                            
                            st.plotly_chart(risk_fig, use_container_width=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Export options with improved button
                        csv_data = anomaly_customers[display_cols].to_csv(index=False)
                        col1, col2 = st.columns([3, 1])
                        
                        with col2:
                            st.download_button(
                                label="📥 Download Report",
                                data=csv_data,
                                file_name=f"fraudulent_consumers_{selected_year}_{selected_month}.csv",
                                mime="text/csv",
                                help="Download the list of detected fraudulent consumers"
                            )
                        
                        with col1:
                            st.info("📋 Report ready for download. The CSV file contains all detected anomalies with their risk levels and consumption patterns for further investigation.")
                        
                    except Exception as e:
                        st.error(f"Error processing anomaly data: {str(e)}")
                        st.write("Detailed anomaly information is not available. Try adjusting filter settings.")
                else:
                    st.info("🔍 No fraudulent consumers detected with the current threshold. Try lowering the threshold to detect more potential anomalies.")
            else:
                st.info("🔍 No fraudulent consumers detected with the current threshold. Try adjusting parameters or selecting different time periods.")
            
            # Add fraud detection explanation
            st.markdown("<div class='sub-header'>Understanding Fraud Detection</div>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                st.markdown("""
                ### How Fraud is Detected
                
                The model analyzes multiple dimensions of consumer behavior to identify patterns that deviate significantly from the norm:
                
                1. **Consumption Patterns** - Unusual changes in electricity usage compared to historical data
                2. **Socioeconomic Context** - Consumption that's inconsistent with the socioeconomic stratum
                3. **Weather Correlation** - Consumption that doesn't follow expected seasonal patterns
                4. **Spatial Analysis** - Geographic clustering of anomalies that may indicate organized fraud
                
                The anomaly score represents the statistical distance of a consumer's behavior from the expected pattern, with higher scores indicating greater likelihood of fraudulent activity.
                """)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                st.markdown("""
                ### Risk Level Classification
                
                Risk levels are determined based on anomaly severity:
                
                - **Critical** (80-100): Extremely abnormal behavior with high confidence of fraud
                - **High** (60-79): Significantly unusual patterns warranting immediate investigation
                - **Medium** (40-59): Moderately suspicious activity requiring follow-up
                - **Low** (0-39): Slightly unusual but may be explained by legitimate factors
                
                Consumers with higher risk levels should be prioritized for field inspections or technical reviews.
                """)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Add fraud types explanation in an expander
            with st.expander("📋 Types of Fraud Detected by the System"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    ### Electricity Distribution Fraud
                    
                    - **Meter Tampering** - Physical manipulation of meters to reduce readings
                    - **Meter Bypassing** - Direct connection to power lines bypassing the meter
                    - **Meter Reversal** - Inverting meter connections to reverse counting
                    - **Commercial Use on Domestic Tariff** - Business operations under residential rates
                    - **Illegal Line Extension** - Unauthorized extension to unmetered premises
                    - **Current Transformer (CT) Tampering** - Manipulation of CT ratios
                    - **Neutral Disturbance** - Disrupting neutral wire to affect meter readings
                    """)
                
                with col2:
                    st.markdown("""
                    ### Natural Gas Distribution Fraud
                    
                    - **Meter Tampering** - Physically altering gas meters to show lower consumption
                    - **Tariff Misuse** - Using domestic rates for commercial operations
                    - **Illegal Connections** - Unauthorized connections to gas pipelines
                    - **Meter Reversal** - Inverting meter connections to affect readings
                    - **Illegal Line Extension** - Unauthorized extension to unmetered premises
                    - **Compressor Usage** - Unauthorized use of compressors to draw more gas
                    - **Electricity Generation** - Using natural gas for power generation without proper tariffs
                    """)
    
    except Exception as e:
        st.error(f"An error occurred in the application: {str(e)}")
        st.warning("Please try different filter settings or refresh the page.")
        
        # Provide debugging information in an expander
        with st.expander("Technical Details"):
            st.code(str(e))
            st.write("If this error persists, please contact the system administrator.")

# Add an about section in the sidebar
with st.sidebar.expander("ℹ️ About This System"):
    st.write("""
    This fraud detection system implements the Multivariate Gaussian Distribution method described in research on fraudulent consumer detection in power utilities of developing countries.
    
    **Version:** 2.0
    **Last Updated:** April 2025
    
    Developed for power distribution companies to identify potential energy theft and fraudulent consumption patterns using advanced anomaly detection algorithms.
    """)

# Run the app
if __name__ == "__main__":
    main()