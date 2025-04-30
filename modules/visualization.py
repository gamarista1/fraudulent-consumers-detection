import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
import matplotlib.pyplot as plt
import seaborn as sns

def plot_consumption_patterns(consumption_df, customers_df, selected_customers=None):
    """
    Plot consumption patterns over time by socioeconomic stratum.
    
    Parameters:
    -----------
    consumption_df : DataFrame
        Consumption data
    customers_df : DataFrame
        Customer information
    selected_customers : list, optional
        List of customer IDs to include
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Plotly figure object
    """
    if consumption_df.empty or customers_df.empty:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="No data available for consumption patterns",
            height=500
        )
        return fig
    
    if selected_customers is not None:
        consumption_df = consumption_df[consumption_df['customer_id'].isin(selected_customers)]
    
    # Merge with customer data to get stratum
    plot_data = consumption_df.merge(
        customers_df[['customer_id', 'stratum']], 
        on='customer_id'
    )
    
    if plot_data.empty:
        # Return empty figure if no data after merging
        fig = go.Figure()
        fig.update_layout(
            title="No consumption data available for selected criteria",
            height=500
        )
        return fig
    
    # Convert stratum to string for better legend
    plot_data['stratum'] = 'Stratum ' + plot_data['stratum'].astype(str)
    
    fig = px.line(
        plot_data,
        x='date',
        y='consumption',
        color='stratum',
        title='Consumption Patterns by Socioeconomic Stratum',
        labels={'consumption': 'Monthly Consumption (kWh)', 'date': 'Date'}
    )
    
    fig.update_layout(
        height=500,
        legend_title="Socioeconomic Stratum",
        hovermode="x unified"
    )
    
    return fig

def create_anomaly_map(customers_df, anomaly_scores, threshold):
    """
    Create an interactive map with anomaly scores.
    
    Parameters:
    -----------
    customers_df : DataFrame
        Customer information with coordinates
    anomaly_scores : array-like
        Anomaly scores for each customer
    threshold : float
        Threshold for anomaly detection
        
    Returns:
    --------
    m : folium.Map
        Folium map object
    """
    # Create base map centered on Medellín
    m = folium.Map(location=[6.25, -75.58], zoom_start=12)
    
    # Check if data is available
    if customers_df.empty or len(anomaly_scores) == 0:
        # Add a message to the map
        folium.Marker(
            location=[6.25, -75.58],
            popup="No data available for the selected filters",
            icon=folium.Icon(color="gray")
        ).add_to(m)
        return m
    
    # Ensure we have the same number of customers and scores
    if len(customers_df) != len(anomaly_scores):
        # Use only the common subset
        n_samples = min(len(customers_df), len(anomaly_scores))
        customers_df = customers_df.iloc[:n_samples].reset_index(drop=True)
        anomaly_scores = anomaly_scores[:n_samples]
    
    # Add markers for each customer
    for idx, row in customers_df.iterrows():
        # Ensure we don't go out of bounds
        if idx >= len(anomaly_scores):
            break
            
        score = anomaly_scores[idx]
        is_anomaly = score > threshold
        
        # Define color based on anomaly score
        color = 'red' if is_anomaly else 'blue'
        radius = 8 if is_anomaly else 5
        
        # Create popup content
        popup_content = f"""
        <b>Customer ID:</b> {row['customer_id']}<br>
        <b>Stratum:</b> {int(row['stratum'])}<br>
        <b>Zone:</b> {row['zone_code']}<br>
        <b>Anomaly Score:</b> {score:.2f}<br>
        <b>Threshold:</b> {threshold:.2f}<br>
        <b>Is Anomaly:</b> {'Yes' if is_anomaly else 'No'}
        """
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(popup_content, max_width=300)
        ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
        bottom: 50px; left: 50px; width: 150px; height: 90px; 
        border:2px solid grey; z-index:9999; font-size:14px;
        background-color:white;
        padding: 10px;
        border-radius: 5px;
        ">
    <p><span style="color:blue;font-size:24px;">●</span> Normal</p>
    <p><span style="color:red;font-size:24px;">●</span> Anomaly</p>
    </div>
    '''
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

def plot_anomaly_distribution(anomaly_scores, threshold):
    """
    Plot distribution of anomaly scores.
    
    Parameters:
    -----------
    anomaly_scores : array-like
        Anomaly scores
    threshold : float
        Threshold for anomaly detection
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Plotly figure object
    """
    fig = go.Figure()
    
    if len(anomaly_scores) == 0:
        fig.update_layout(
            title='No data available for anomaly score distribution',
            xaxis_title='Anomaly Score',
            yaxis_title='Count',
            height=400
        )
        return fig
    
    fig.add_trace(go.Histogram(
        x=anomaly_scores,
        name='Score Distribution',
        nbinsx=50,
        marker_color='lightblue'
    ))
    
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text="Anomaly Threshold",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title='Distribution of Anomaly Scores',
        xaxis_title='Anomaly Score',
        yaxis_title='Count',
        height=400,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    
    return fig

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance based on MGD components.
    
    Parameters:
    -----------
    model : MGDAnomaly
        Fitted MGD model
    feature_names : list
        Names of the features
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Plotly figure object
    """
    fig = go.Figure()
    
    if not model.fitted or len(feature_names) == 0:
        fig.update_layout(
            title='Model not fitted or no features available',
            height=400
        )
        return fig
    
    # Calculate feature importance using covariance matrix
    importance = np.abs(np.diag(model.cov_estimator.covariance_))
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance in Anomaly Detection',
        color='Importance',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(height=400)
    
    return fig

def plot_stratum_distribution(customers_df, anomaly_mask):
    """
    Plot distribution of anomalies by socioeconomic stratum.
    
    Parameters:
    -----------
    customers_df : DataFrame
        Customer information
    anomaly_mask : array-like
        Boolean mask indicating anomalies
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Plotly figure object
    """
    fig = go.Figure()
    
    if customers_df.empty or len(anomaly_mask) == 0:
        fig.update_layout(
            title='No data available for stratum distribution',
            height=400
        )
        return fig
    
    # Make sure the mask is the right length
    if len(anomaly_mask) != len(customers_df):
        # Use only the available data
        n_samples = min(len(customers_df), len(anomaly_mask))
        customers_df = customers_df.iloc[:n_samples].reset_index(drop=True)
        anomaly_mask = anomaly_mask[:n_samples]
    
    # Count anomalies by stratum
    strata_counts = pd.DataFrame({
        'Stratum': customers_df['stratum'].astype(int),
        'Is_Anomaly': anomaly_mask
    })
    
    strata_agg = strata_counts.groupby(['Stratum', 'Is_Anomaly']).size().reset_index(name='Count')
    strata_agg['Status'] = strata_agg['Is_Anomaly'].map({True: 'Anomaly', False: 'Normal'})
    
    fig = px.bar(
        strata_agg,
        x='Stratum',
        y='Count',
        color='Status',
        barmode='group',
        title='Distribution of Customers by Stratum',
        color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'}
    )
    
    fig.update_layout(
        xaxis_title='Socioeconomic Stratum',
        yaxis_title='Number of Customers',
        height=400
    )
    
    return fig

def plot_scatter_comparison(features, anomaly_mask):
    """
    Plot scatter comparison of current vs previous consumption with anomaly highlighting.
    
    Parameters:
    -----------
    features : DataFrame
        Feature DataFrame containing consumption data
    anomaly_mask : array-like
        Boolean mask indicating anomalies
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Plotly figure object
    """
    fig = go.Figure()
    
    if features.empty or len(anomaly_mask) == 0:
        fig.update_layout(
            title='No data available for consumption comparison',
            height=500
        )
        return fig
    
    # Make sure the mask is the right length
    if len(anomaly_mask) != len(features):
        # Use only the available data
        n_samples = min(len(features), len(anomaly_mask))
        features = features.iloc[:n_samples].reset_index(drop=True)
        anomaly_mask = anomaly_mask[:n_samples]
    
    scatter_data = pd.DataFrame({
        'Previous Month': features['consumption_prev'],
        'Current Month': features['consumption_current'],
        'Is_Anomaly': anomaly_mask,
        'Status': ['Anomaly' if a else 'Normal' for a in anomaly_mask]
    })
    
    fig = px.scatter(
        scatter_data,
        x='Previous Month',
        y='Current Month',
        color='Status',
        color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'},
        title='Consumption Comparison: Current vs Previous Month',
        hover_data=['Status']
    )
    
    # Add diagonal line (y=x)
    max_val = scatter_data[['Previous Month', 'Current Month']].max().max()
    if not np.isnan(max_val):
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                line=dict(color='grey', dash='dash'),
                name='Same Consumption Line'
            )
        )
    
    fig.update_layout(
        height=500,
        xaxis_title='Previous Month Consumption (kWh)',
        yaxis_title='Current Month Consumption (kWh)'
    )
    
    return fig

def create_kpi_cards(anomaly_scores, threshold, customers_df):
    """
    Create KPI metrics for the dashboard.
    
    Parameters:
    -----------
    anomaly_scores : array-like
        Anomaly scores
    threshold : float
        Threshold for anomaly detection
    customers_df : DataFrame
        Customer information
        
    Returns:
    --------
    kpi_metrics : dict
        Dictionary of KPI metrics
    """
    # Handle empty data case
    if len(anomaly_scores) == 0 or customers_df.empty:
        return {
            'total_customers': 0,
            'anomalies_detected': 0,
            'detection_rate': 0,
            'precision': None,
            'recall': None,
            'f1': None
        }
    
    # Make sure we have matching length
    if len(anomaly_scores) != len(customers_df):
        n_samples = min(len(customers_df), len(anomaly_scores))
        customers_df = customers_df.iloc[:n_samples].reset_index(drop=True)
        anomaly_scores = anomaly_scores[:n_samples]
    
    anomaly_mask = anomaly_scores > threshold
    total_customers = len(customers_df)
    anomalies_detected = sum(anomaly_mask)
    detection_rate = (anomalies_detected / total_customers) * 100 if total_customers > 0 else 0
    
    # If we have ground truth (is_fraudulent)
    if 'is_fraudulent' in customers_df.columns:
        true_positives = sum(anomaly_mask & customers_df['is_fraudulent'])
        true_frauds = sum(customers_df['is_fraudulent'])
        
        if anomalies_detected > 0:
            precision = (true_positives / anomalies_detected) * 100
        else:
            precision = 0
            
        recall = (true_positives / true_frauds) * 100 if true_frauds > 0 else 0
        f1 = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0
    else:
        precision = None
        recall = None
        f1 = None
    
    return {
        'total_customers': total_customers,
        'anomalies_detected': anomalies_detected,
        'detection_rate': detection_rate,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def plot_heatmap(data, x_field, y_field, value_field, title=None):
    """
    Create a heatmap visualization.
    
    Parameters:
    -----------
    data : DataFrame
        Data for heatmap
    x_field : str
        Field for x-axis
    y_field : str
        Field for y-axis
    value_field : str
        Field for values
    title : str, optional
        Title for the plot
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Plotly figure object
    """
    fig = go.Figure()
    
    if data.empty:
        fig.update_layout(
            title='No data available for heatmap',
            height=400
        )
        return fig
    
    # Pivot data for heatmap
    try:
        heatmap_data = data.pivot_table(
            index=y_field, 
            columns=x_field, 
            values=value_field,
            aggfunc='mean'
        )
        
        fig = px.imshow(
            heatmap_data,
            labels=dict(x=x_field, y=y_field, color=value_field),
            x=heatmap_data.columns,
            y=heatmap_data.index,
            color_continuous_scale="Viridis"
        )
        
        if title:
            fig.update_layout(title=title)
            
    except Exception as e:
        fig.update_layout(
            title=f'Error creating heatmap: {str(e)}',
            height=400
        )
    
    return fig