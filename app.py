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

# Translations
TRANSLATIONS = {
    "ES": {
        "page_title": "Plataforma de Detección de Fraude | DISICO Ingeniería S.A.",
        "main_header": "⚡ Plataforma Corporativa de Detección de Fraude Energético | DISICO Ingeniería S.A.",
        "intro_paragraph": (
            "Este sistema inteligente implementa el método de Distribución Gaussiana Multivariada (MGD) "
            "para detectar consumidores fraudulentos en servicios de energía, analizando la estratificación "
            "socioeconómica, los patrones de consumo y el impacto climático para identificar anomalías con alta precisión."
        ),
        "language_selector": "Idioma de la interfaz",
        "language_es": "Español",
        "language_en": "Inglés",
        "loading_data": "Cargando datos...",
        "analyzing_data": "Analizando datos...",
        "sidebar_header": "Controles de Detección de Fraude",
        "sidebar_data_filters": "### 📅 Filtros de datos",
        "filter_year": "Año",
        "filter_month": "Mes",
        "filter_zone": "Filtro de zona",
        "filter_zone_all": "Todas",
        "filter_zone_help": "Filtrar datos por zona geográfica específica",
        "sidebar_model_params": "### ⚙️ Parámetros del modelo",
        "anomaly_threshold_label": "Factor de umbral de anomalía",
        "anomaly_threshold_help": "Valores bajos detectan más anomalías (pueden aumentar falsos positivos). Valores altos detectan menos, pero con mayor certeza.",
        "sidebar_feature_selection": "### 🔍 Selección de características",
        "feature_consumption_label": "Consumo",
        "feature_consumption_help": "Incluir patrones de consumo en el análisis",
        "feature_stratification_label": "Estratificación",
        "feature_stratification_help": "Incluir factores socioeconómicos",
        "feature_weather_label": "Clima",
        "feature_weather_help": "Incluir factores climáticos",
        "feature_historical_label": "Histórico",
        "feature_historical_help": "Incluir comparación interanual",
        "error_select_feature": "Seleccione al menos una categoría de características para el análisis",
        "warning_no_data": "No hay datos disponibles para los filtros seleccionados. Intente con otros parámetros.",
        "warning_no_features_found": "No se encontraron características seleccionadas en los datos. Características disponibles: {available}",
        "detection_results_header": "📊 Resultados de detección",
        "kpi_total_customers": "Total de clientes",
        "kpi_total_customers_trend": "Consumidores activos",
        "kpi_detection_rate": "Tasa de detección",
        "kpi_detection_rate_trend": "Porcentaje anómalo",
        "kpi_detected_anomalies": "Anomalías detectadas",
        "kpi_detected_anomalies_trend": "Fraude potencial",
        "kpi_precision": "Tasa de acierto (Precisión)",
        "kpi_precision_trend": "Validación",
        "kpi_not_available": "N/D",
        "tab_geospatial": "📍 Análisis geoespacial",
        "tab_consumption": "📈 Análisis de consumo",
        "tab_fraud": "🔎 Detección de fraude",
        "geo_subheader": "Distribución geográfica de anomalías",
        "geo_info_note": "**Nota:** Azul = consumidores normales, rojo = anomalías potencialmente fraudulentas.",
        "geo_error": "Error al mostrar el mapa: {error}",
        "geo_map_error_popup": "Error mostrando el mapa de datos",
        "consumption_subheader": "Análisis de patrones de consumo",
        "anomaly_distribution_title": "Distribución de puntajes de anomalía",
        "scatter_title": "Consumo actual vs. consumo previo por estado de anomalía",
        "scatter_x_label": "Consumo mes anterior (kWh)",
        "scatter_y_label": "Consumo mes actual (kWh)",
        "consumption_patterns_title": "Patrones de consumo por estrato socioeconómico",
        "stratum_distribution_title": "Distribución de anomalías por estrato socioeconómico",
        "feature_significance_header": "Análisis de relevancia de características",
        "feature_importance_title": "Importancia relativa de variables en la detección de anomalías",
        "feature_importance_expander": "📚 Comprender la importancia de variables",
        "feature_importance_markdown": (
            "### Cómo interpretar la importancia de variables\n\n"
            "**La importancia de variables** en este modelo indica cuánto contribuye cada "
            "variable a identificar actividad fraudulenta:\n\n"
            "- **Valores más altos** significan que la variable es más crítica para detectar anomalías\n"
            "- **Variables relacionadas** suelen mostrar niveles de importancia similares\n"
            "- **Variables de consumo** suelen tener alta importancia porque reflejan patrones de uso\n"
            "- **Factores socioeconómicos** contextualizan el consumo según el perfil esperado\n\n"
            "El modelo utiliza estas variables para establecer un perfil multidimensional de comportamiento "
            "normal del consumidor e identificar atípicos que se desvían de los patrones establecidos."
        ),
        "fraud_detected_header": "Consumidores fraudulentos detectados",
        "anomaly_process_error": "Error procesando los datos de anomalías: {error}",
        "anomaly_process_detail_unavailable": "La información detallada de anomalías no está disponible. Ajuste los filtros.",
        "no_fraud_detected": "🔍 No se detectaron consumidores fraudulentos con el umbral actual. Pruebe bajar el umbral para identificar más anomalías.",
        "no_fraud_detected_alt": "🔍 No se detectaron consumidores fraudulentos con el umbral actual. Ajuste parámetros o seleccione otros periodos.",
        "understanding_fraud_header": "Comprender la detección de fraude",
        "fraud_detected_markdown": (
            "### Cómo se detecta el fraude\n\n"
            "El modelo analiza múltiples dimensiones del comportamiento del consumidor para identificar patrones "
            "que se desvían significativamente de la norma:\n\n"
            "1. **Patrones de consumo** - Cambios inusuales en el uso de energía frente al histórico\n"
            "2. **Contexto socioeconómico** - Consumo inconsistente con el estrato socioeconómico\n"
            "3. **Correlación climática** - Consumo que no sigue patrones estacionales esperados\n"
            "4. **Análisis espacial** - Agrupamientos geográficos de anomalías que podrían indicar fraude organizado\n\n"
            "El puntaje de anomalía representa la distancia estadística del comportamiento del consumidor respecto "
            "al patrón esperado; puntajes más altos implican mayor probabilidad de fraude."
        ),
        "risk_level_markdown": (
            "### Clasificación por nivel de riesgo\n\n"
            "Los niveles de riesgo se determinan según la severidad de la anomalía:\n\n"
            "- **Crítico** (80-100): Comportamiento extremadamente anómalo con alta confianza de fraude\n"
            "- **Alto** (60-79): Patrones significativamente inusuales que requieren investigación inmediata\n"
            "- **Medio** (40-59): Actividad moderadamente sospechosa que requiere seguimiento\n"
            "- **Bajo** (0-39): Ligeramente inusual, puede explicarse por factores legítimos\n\n"
            "Los consumidores con mayor nivel de riesgo deben priorizarse para inspecciones de campo o revisiones técnicas."
        ),
        "fraud_types_expander": "📋 Tipos de fraude detectados por el sistema",
        "fraud_types_electricity_title": "Fraude en distribución eléctrica",
        "fraud_types_electricity_list": (
            "- **Manipulación del medidor** - Alteración física para reducir lecturas\n"
            "- **Bypass del medidor** - Conexión directa a la red sin medición\n"
            "- **Inversión del medidor** - Invertir conexiones para revertir el conteo\n"
            "- **Uso comercial en tarifa residencial** - Operación comercial con tarifa doméstica\n"
            "- **Extensión ilegal de línea** - Derivaciones no autorizadas\n"
            "- **Manipulación de TC** - Alteración de relaciones de transformadores de corriente\n"
            "- **Perturbación del neutro** - Interferencia del neutro para afectar lecturas\n"
        ),
        "fraud_types_gas_title": "Fraude en distribución de gas natural",
        "fraud_types_gas_list": (
            "- **Manipulación del medidor** - Alteración física para reducir consumo reportado\n"
            "- **Uso indebido de tarifa** - Uso comercial con tarifa doméstica\n"
            "- **Conexiones ilegales** - Conexiones no autorizadas a la red\n"
            "- **Inversión del medidor** - Invertir conexiones para alterar lecturas\n"
            "- **Extensión ilegal de línea** - Derivaciones no autorizadas\n"
            "- **Uso de compresores** - Uso no autorizado para extraer más gas\n"
            "- **Generación eléctrica** - Uso de gas para generación sin tarifa adecuada\n"
        ),
        "app_error": "Se produjo un error en la aplicación: {error}",
        "app_error_warning": "Intente con otros filtros o actualice la página.",
        "technical_details_expander": "Detalles técnicos",
        "technical_details_persist": "Si este error persiste, contacte al administrador del sistema.",
        "about_expander": "ℹ️ Acerca del sistema",
        "about_text": (
            "Esta plataforma de detección de fraude implementa el método de Distribución Gaussiana Multivariada "
            "descrito en investigaciones sobre detección de fraude en servicios públicos.\n\n"
            "**Versión:** 2.0\n"
            "**Última actualización:** abril de 2025\n\n"
            "Desarrollado para compañías distribuidoras con el fin de identificar posibles robos de energía y "
            "patrones de consumo fraudulentos mediante algoritmos avanzados de detección de anomalías."
        ),
        "download_report_label": "📥 Descargar reporte",
        "download_report_help": "Descargar el listado de consumidores fraudulentos detectados",
        "report_ready_info": "📋 Reporte listo para descargar. El CSV contiene anomalías detectadas con su nivel de riesgo y patrones de consumo para investigación.",
        "risk_level_column": "Nivel de riesgo",
        "risk_level_critical": "Crítico",
        "risk_level_high": "Alto",
        "risk_level_medium": "Medio",
        "risk_level_low": "Bajo",
        "risk_distribution_title": "Distribución de anomalías por nivel de riesgo",
        "risk_xaxis_title": "Nivel de riesgo",
        "risk_yaxis_title": "Número de clientes",
        "csv_filename_template": "consumidores_fraudulentos_{year}_{month}.csv",
        "column_customer_id": "ID de cliente",
        "column_zone": "Zona",
        "column_stratum": "Estrato",
        "column_current_consumption": "Consumo actual (kWh)",
        "column_previous_consumption": "Año anterior (kWh)",
        "column_change_pct": "Variación (%)",
        "column_anomaly_score": "Puntaje de anomalía",
        "feature_importance_title_no_data": "Modelo no ajustado o sin variables disponibles",
        "anomaly_distribution_no_data": "No hay datos para distribución de anomalías",
        "scatter_no_data": "No hay datos para comparación de consumo",
        "consumption_no_data": "No hay datos disponibles para patrones de consumo",
        "consumption_no_data_filtered": "No hay datos de consumo para los criterios seleccionados",
        "stratum_no_data": "No hay datos para distribución por estrato",
        "stratum_axis_title": "Estrato socioeconómico",
        "stratum_legend_normal": "Normal",
        "stratum_legend_anomaly": "Anomalía",
        "scatter_status_normal": "Normal",
        "scatter_status_anomaly": "Anomalía",
        "scatter_same_line": "Línea de consumo igual",
        "plot_x_date": "Fecha",
        "plot_y_consumption": "Consumo mensual (kWh)",
        "plot_legend_stratum": "Estrato socioeconómico",
        "plot_stratum_prefix": "Estrato ",
        "plot_anomaly_score_label": "Puntaje de anomalía",
        "plot_count_label": "Conteo",
        "plot_threshold_label": "Umbral de anomalía",
        "map_no_data": "No hay datos disponibles para los filtros seleccionados",
        "map_cluster_normal": "Consumidores normales",
        "map_cluster_anomaly": "Anomalías",
        "map_popup_anomaly_title": "Anomalía detectada",
        "map_popup_normal_title": "Consumidor normal",
        "map_label_customer_id": "ID cliente",
        "map_label_stratum": "Estrato",
        "map_label_zone": "Zona",
        "map_label_anomaly_score": "Puntaje de anomalía",
        "map_label_risk_level": "Nivel de riesgo",
        "map_tooltip_anomaly": "Anomalía",
        "map_tooltip_normal": "Normal",
        "map_legend_normal": "Normal",
        "map_legend_anomaly": "Anomalía",
        "chart_container_end": "</div>"
    },
    "EN": {
        "page_title": "Fraud Detection Platform | DISICO Ingeniería S.A.",
        "main_header": "⚡ Corporate Energy Fraud Detection Platform | DISICO Ingeniería S.A.",
        "intro_paragraph": (
            "This intelligent system implements the Multivariate Gaussian Distribution (MGD) method "
            "to detect fraudulent consumers in energy utilities, analyzing socioeconomic stratification, "
            "consumption patterns, and climate impacts to identify anomalies with high precision."
        ),
        "language_selector": "Interface language",
        "language_es": "Spanish",
        "language_en": "English",
        "loading_data": "Loading data...",
        "analyzing_data": "Analyzing data...",
        "sidebar_header": "Fraud Detection Controls",
        "sidebar_data_filters": "### 📅 Data Filters",
        "filter_year": "Year",
        "filter_month": "Month",
        "filter_zone": "Zone Filter",
        "filter_zone_all": "All",
        "filter_zone_help": "Filter data by specific geographic zone",
        "sidebar_model_params": "### ⚙️ Model Parameters",
        "anomaly_threshold_label": "Anomaly Threshold Factor",
        "anomaly_threshold_help": "Lower values detect more anomalies (may increase false positives). Higher values detect fewer, more certain anomalies.",
        "sidebar_feature_selection": "### 🔍 Feature Selection",
        "feature_consumption_label": "Consumption",
        "feature_consumption_help": "Include consumption patterns in analysis",
        "feature_stratification_label": "Stratification",
        "feature_stratification_help": "Include socioeconomic factors",
        "feature_weather_label": "Weather",
        "feature_weather_help": "Include climate factors",
        "feature_historical_label": "Historical",
        "feature_historical_help": "Include year-over-year comparison",
        "error_select_feature": "Please select at least one feature category for analysis",
        "warning_no_data": "No data available for the selected filters. Please try different filter settings.",
        "warning_no_features_found": "No selected features found in data. Available features: {available}",
        "detection_results_header": "📊 Detection Results",
        "kpi_total_customers": "Total Customers",
        "kpi_total_customers_trend": "Active consumers",
        "kpi_detection_rate": "Detection Rate",
        "kpi_detection_rate_trend": "Anomalous %",
        "kpi_detected_anomalies": "Detected Anomalies",
        "kpi_detected_anomalies_trend": "Potential fraud",
        "kpi_precision": "Hit Rate (Precision)",
        "kpi_precision_trend": "Validation",
        "kpi_not_available": "N/A",
        "tab_geospatial": "📍 Geospatial Analysis",
        "tab_consumption": "📈 Consumption Analysis",
        "tab_fraud": "🔎 Fraud Detection",
        "geo_subheader": "Geospatial distribution of anomalies",
        "geo_info_note": "**Note:** Blue = normal consumers, red = potentially fraudulent anomalies.",
        "geo_error": "Error displaying the map: {error}",
        "geo_map_error_popup": "Error displaying the data map",
        "consumption_subheader": "Consumption patterns analysis",
        "anomaly_distribution_title": "Distribution of Anomaly Scores",
        "scatter_title": "Current vs. Previous Consumption by Anomaly Status",
        "scatter_x_label": "Previous Month Consumption (kWh)",
        "scatter_y_label": "Current Month Consumption (kWh)",
        "consumption_patterns_title": "Consumption Patterns by Socioeconomic Stratum",
        "stratum_distribution_title": "Anomaly Distribution by Socioeconomic Stratum",
        "feature_significance_header": "Feature Significance Analysis",
        "feature_importance_title": "Relative Importance of Features in Anomaly Detection",
        "feature_importance_expander": "📚 Understanding Feature Importance",
        "feature_importance_markdown": (
            "### How to Interpret Feature Importance\n\n"
            "**Feature importance** in this model indicates how much each variable contributes to identifying fraudulent activity:\n\n"
            "- **Higher values** mean the feature is more critical for detecting anomalies\n"
            "- **Related features** often show similar importance values\n"
            "- **Consumption-related features** typically have significant importance as they directly reflect usage patterns\n"
            "- **Socioeconomic factors** help contextualize consumption relative to expected patterns for specific demographic groups\n\n"
            "The model uses these features to establish a multidimensional profile of normal consumer behavior, then identifies outliers that deviate from these established patterns."
        ),
        "fraud_detected_header": "Detected Fraudulent Consumers",
        "anomaly_process_error": "Error processing anomaly data: {error}",
        "anomaly_process_detail_unavailable": "Detailed anomaly information is not available. Try adjusting filter settings.",
        "no_fraud_detected": "🔍 No fraudulent consumers detected with the current threshold. Try lowering the threshold to detect more potential anomalies.",
        "no_fraud_detected_alt": "🔍 No fraudulent consumers detected with the current threshold. Try adjusting parameters or selecting different time periods.",
        "understanding_fraud_header": "Understanding Fraud Detection",
        "fraud_detected_markdown": (
            "### How Fraud is Detected\n\n"
            "The model analyzes multiple dimensions of consumer behavior to identify patterns that deviate significantly from the norm:\n\n"
            "1. **Consumption Patterns** - Unusual changes in electricity usage compared to historical data\n"
            "2. **Socioeconomic Context** - Consumption that's inconsistent with the socioeconomic stratum\n"
            "3. **Weather Correlation** - Consumption that doesn't follow expected seasonal patterns\n"
            "4. **Spatial Analysis** - Geographic clustering of anomalies that may indicate organized fraud\n\n"
            "The anomaly score represents the statistical distance of a consumer's behavior from the expected pattern, with higher scores indicating greater likelihood of fraudulent activity."
        ),
        "risk_level_markdown": (
            "### Risk Level Classification\n\n"
            "Risk levels are determined based on anomaly severity:\n\n"
            "- **Critical** (80-100): Extremely abnormal behavior with high confidence of fraud\n"
            "- **High** (60-79): Significantly unusual patterns warranting immediate investigation\n"
            "- **Medium** (40-59): Moderately suspicious activity requiring follow-up\n"
            "- **Low** (0-39): Slightly unusual but may be explained by legitimate factors\n\n"
            "Consumers with higher risk levels should be prioritized for field inspections or technical reviews."
        ),
        "fraud_types_expander": "📋 Types of Fraud Detected by the System",
        "fraud_types_electricity_title": "Electricity Distribution Fraud",
        "fraud_types_electricity_list": (
            "- **Meter Tampering** - Physical manipulation of meters to reduce readings\n"
            "- **Meter Bypassing** - Direct connection to power lines bypassing the meter\n"
            "- **Meter Reversal** - Inverting meter connections to reverse counting\n"
            "- **Commercial Use on Domestic Tariff** - Business operations under residential rates\n"
            "- **Illegal Line Extension** - Unauthorized extension to unmetered premises\n"
            "- **Current Transformer (CT) Tampering** - Manipulation of CT ratios\n"
            "- **Neutral Disturbance** - Disrupting neutral wire to affect meter readings\n"
        ),
        "fraud_types_gas_title": "Natural Gas Distribution Fraud",
        "fraud_types_gas_list": (
            "- **Meter Tampering** - Physically altering gas meters to show lower consumption\n"
            "- **Tariff Misuse** - Using domestic rates for commercial operations\n"
            "- **Illegal Connections** - Unauthorized connections to gas pipelines\n"
            "- **Meter Reversal** - Inverting meter connections to affect readings\n"
            "- **Illegal Line Extension** - Unauthorized extension to unmetered premises\n"
            "- **Compressor Usage** - Unauthorized use of compressors to draw more gas\n"
            "- **Electricity Generation** - Using natural gas for power generation without proper tariffs\n"
        ),
        "app_error": "An error occurred in the application: {error}",
        "app_error_warning": "Please try different filter settings or refresh the page.",
        "technical_details_expander": "Technical Details",
        "technical_details_persist": "If this error persists, please contact the system administrator.",
        "about_expander": "ℹ️ About This System",
        "about_text": (
            "This fraud detection platform implements the Multivariate Gaussian Distribution method described in "
            "research on fraudulent consumer detection in utilities.\n\n"
            "**Version:** 2.0\n"
            "**Last Updated:** April 2025\n\n"
            "Developed for distribution companies to identify potential energy theft and fraudulent consumption patterns "
            "using advanced anomaly detection algorithms."
        ),
        "download_report_label": "📥 Download Report",
        "download_report_help": "Download the list of detected fraudulent consumers",
        "report_ready_info": "📋 Report ready for download. The CSV file contains detected anomalies with their risk levels and consumption patterns for further investigation.",
        "risk_level_column": "Risk Level",
        "risk_level_critical": "Critical",
        "risk_level_high": "High",
        "risk_level_medium": "Medium",
        "risk_level_low": "Low",
        "risk_distribution_title": "Distribution of Anomalies by Risk Level",
        "risk_xaxis_title": "Risk Level",
        "risk_yaxis_title": "Number of Customers",
        "csv_filename_template": "fraudulent_consumers_{year}_{month}.csv",
        "column_customer_id": "Customer ID",
        "column_zone": "Zone",
        "column_stratum": "Stratum",
        "column_current_consumption": "Current Consumption (kWh)",
        "column_previous_consumption": "Previous Year (kWh)",
        "column_change_pct": "Change (%)",
        "column_anomaly_score": "Anomaly Score",
        "feature_importance_title_no_data": "Model not fitted or no features available",
        "anomaly_distribution_no_data": "No data available for anomaly score distribution",
        "scatter_no_data": "No data available for consumption comparison",
        "consumption_no_data": "No data available for consumption patterns",
        "consumption_no_data_filtered": "No consumption data available for selected criteria",
        "stratum_no_data": "No data available for stratum distribution",
        "stratum_axis_title": "Socioeconomic Stratum",
        "stratum_legend_normal": "Normal",
        "stratum_legend_anomaly": "Anomaly",
        "scatter_status_normal": "Normal",
        "scatter_status_anomaly": "Anomaly",
        "scatter_same_line": "Same Consumption Line",
        "plot_x_date": "Date",
        "plot_y_consumption": "Monthly Consumption (kWh)",
        "plot_legend_stratum": "Socioeconomic Stratum",
        "plot_stratum_prefix": "Stratum ",
        "plot_anomaly_score_label": "Anomaly Score",
        "plot_count_label": "Count",
        "plot_threshold_label": "Anomaly Threshold",
        "map_no_data": "No data available for the selected filters",
        "map_cluster_normal": "Normal Consumers",
        "map_cluster_anomaly": "Anomalies",
        "map_popup_anomaly_title": "Anomaly Detected",
        "map_popup_normal_title": "Normal Consumer",
        "map_label_customer_id": "Customer ID",
        "map_label_stratum": "Stratum",
        "map_label_zone": "Zone",
        "map_label_anomaly_score": "Anomaly Score",
        "map_label_risk_level": "Risk Level",
        "map_tooltip_anomaly": "Anomaly",
        "map_tooltip_normal": "Normal",
        "map_legend_normal": "Normal",
        "map_legend_anomaly": "Anomaly",
        "chart_container_end": "</div>"
    }
}

if "language" not in st.session_state:
    st.session_state["language"] = "ES"

def get_text(key):
    language = st.session_state.get("language", "ES")
    translations = TRANSLATIONS.get(language, TRANSLATIONS["ES"])
    if key in translations:
        return translations[key]
    if "missing_translation_keys" not in st.session_state:
        st.session_state["missing_translation_keys"] = set()
    if key not in st.session_state["missing_translation_keys"]:
        st.session_state["missing_translation_keys"].add(key)
        warning_message = f"Missing translation key: {key} (language: {language})"
        try:
            st.warning(warning_message)
        except Exception:
            print(warning_message)
    return TRANSLATIONS.get("EN", {}).get(key, key)

# Set page configuration
st.set_page_config(
    page_title=get_text("page_title"),
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Language selector (sidebar)
current_language = st.session_state.get("language", "ES")
language_labels = {
    "ES": {"ES": TRANSLATIONS["ES"]["language_es"], "EN": TRANSLATIONS["ES"]["language_en"]},
    "EN": {"ES": TRANSLATIONS["EN"]["language_es"], "EN": TRANSLATIONS["EN"]["language_en"]}
}
selected_language = st.sidebar.selectbox(
    get_text("language_selector"),
    ["ES", "EN"],
    index=0 if current_language == "ES" else 1,
    format_func=lambda code: language_labels[current_language][code]
)
st.session_state["language"] = selected_language

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
st.markdown(f"<div class='main-header'>{get_text('main_header')}</div>", unsafe_allow_html=True)
st.markdown(get_text("intro_paragraph"))

# Main function
def main():
    # Load data
    if 'customers_df' not in st.session_state:
        with st.spinner(get_text("loading_data")):
            st.session_state.customers_df, st.session_state.consumption_df, st.session_state.weather_df = load_sample_data()
    
    customers_df = st.session_state.customers_df
    consumption_df = st.session_state.consumption_df
    weather_df = st.session_state.weather_df
    
    # Sidebar filters and styling
    st.sidebar.markdown(f"<div class='sidebar-header'>{get_text('sidebar_header')}</div>", unsafe_allow_html=True)
    
    st.sidebar.markdown(get_text("sidebar_data_filters"))
    
    # Filter by date with improved UI
    col1, col2 = st.sidebar.columns(2)
    
    available_years = sorted(consumption_df['year'].unique())
    selected_year = col1.selectbox(get_text("filter_year"), available_years, index=len(available_years)-1)
    
    available_months = sorted(consumption_df[consumption_df['year'] == selected_year]['month'].unique())
    selected_month = col2.selectbox(get_text("filter_month"), available_months, index=len(available_months)-1)
    
    # Filter by zone with search
    available_zones = [get_text("filter_zone_all")] + sorted(customers_df['zone_code'].unique().tolist())
    selected_zone_option = st.sidebar.selectbox(
        get_text("filter_zone"),
        available_zones,
        help=get_text("filter_zone_help")
    )
    selected_zone = None if selected_zone_option == get_text("filter_zone_all") else selected_zone_option
    
    # MGD parameters with improved sliders
    st.sidebar.markdown(get_text("sidebar_model_params"))
    anomaly_threshold_factor = st.sidebar.slider(
        get_text("anomaly_threshold_label"),
        min_value=1.5, 
        max_value=5.0, 
        value=3.0, 
        step=0.1,
        help=get_text("anomaly_threshold_help")
    )
    
    # Feature selection with better organization
    st.sidebar.markdown(get_text("sidebar_feature_selection"))
    
    feature_col1, feature_col2 = st.sidebar.columns(2)
    
    use_consumption = feature_col1.checkbox(
        get_text("feature_consumption_label"),
        value=True,
        help=get_text("feature_consumption_help")
    )
    use_stratum = feature_col1.checkbox(
        get_text("feature_stratification_label"),
        value=True,
        help=get_text("feature_stratification_help")
    )
    use_weather = feature_col2.checkbox(
        get_text("feature_weather_label"),
        value=True,
        help=get_text("feature_weather_help")
    )
    use_historical = feature_col2.checkbox(
        get_text("feature_historical_label"),
        value=True,
        help=get_text("feature_historical_help")
    )
    
    # Prepare features
    try:
        with st.spinner(get_text("analyzing_data")):
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
                st.error(get_text("error_select_feature"))
                return
            
            # Check if we have any data
            if features.empty:
                st.warning(get_text("warning_no_data"))
                
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
                    st.warning(
                        get_text("warning_no_features_found").format(
                            available=", ".join(features.columns)
                        )
                    )
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
        st.markdown(f"<div class='sub-header'>{get_text('detection_results_header')}</div>", unsafe_allow_html=True)
        
        # Create a container for the cards
        col1, col2 = st.columns(2)
        
        with col1:
            # Card 1: Total Customers
            st.markdown(f"""
            <div class='kpi-card customers'>
                <div class='metric-label'>{get_text('kpi_total_customers')}</div>
                <div class='metric-value'>{kpi_metrics['total_customers']:,}</div>
                <div class='metric-trend'>
                    <span>{get_text('kpi_total_customers_trend')}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Card 3: Detection Rate
            st.markdown(f"""
            <div class='kpi-card rate'>
                <div class='metric-label'>{get_text('kpi_detection_rate')}</div>
                <div class='metric-value'>{kpi_metrics['detection_rate']:.1f}%</div>
                <div class='metric-trend'>
                    <span>{get_text('kpi_detection_rate_trend')}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Card 2: Detected Anomalies
            st.markdown(f"""
            <div class='kpi-card anomalies'>
                <div class='metric-label'>{get_text('kpi_detected_anomalies')}</div>
                <div class='metric-value'>{kpi_metrics['anomalies_detected']:,}</div>
                <div class='metric-trend {"trend-up" if kpi_metrics["anomalies_detected"] > 0 else ""}'>
                    <span>{get_text('kpi_detected_anomalies_trend')}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Card 4: Hit Rate (Precision)
            precision_value = (
                f"{kpi_metrics['precision']:.1f}%" if kpi_metrics['precision'] is not None else get_text("kpi_not_available")
            )
            st.markdown(f"""
            <div class='kpi-card precision'>
                <div class='metric-label'>{get_text('kpi_precision')}</div>
                <div class='metric-value'>{precision_value}</div>
                <div class='metric-trend'>
                    <span>{get_text('kpi_precision_trend')}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # Close the container
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Create tabs for better organization
        tab1, tab2, tab3 = st.tabs([get_text("tab_geospatial"), get_text("tab_consumption"), get_text("tab_fraud")])

        with tab1:
            st.markdown(f"<div class='sub-header'>{get_text('geo_subheader')}</div>", unsafe_allow_html=True)
            
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
                anomaly_map = create_anomaly_map(
                    map_customers,
                    map_scores,
                    threshold,
                    text={
                        "no_data": get_text("map_no_data"),
                        "cluster_normal": get_text("map_cluster_normal"),
                        "cluster_anomaly": get_text("map_cluster_anomaly"),
                        "popup_anomaly_title": get_text("map_popup_anomaly_title"),
                        "popup_normal_title": get_text("map_popup_normal_title"),
                        "label_customer_id": get_text("map_label_customer_id"),
                        "label_stratum": get_text("map_label_stratum"),
                        "label_zone": get_text("map_label_zone"),
                        "label_anomaly_score": get_text("map_label_anomaly_score"),
                        "label_risk_level": get_text("map_label_risk_level"),
                        "tooltip_anomaly": get_text("map_tooltip_anomaly"),
                        "tooltip_normal": get_text("map_tooltip_normal"),
                        "legend_normal": get_text("map_legend_normal"),
                        "legend_anomaly": get_text("map_legend_anomaly"),
                        "risk_levels": {
                            "critical": get_text("risk_level_critical"),
                            "high": get_text("risk_level_high"),
                            "medium": get_text("risk_level_medium"),
                            "low": get_text("risk_level_low")
                        }
                    }
                )
                folium_static(anomaly_map, width=1200, height=600)
                
                # Información útil
                st.info(get_text("geo_info_note"))
                
            except Exception as e:
                st.error(get_text("geo_error").format(error=str(e)))
                # Crear un mapa básico como fallback
                basic_map = folium.Map(location=[6.25, -75.58], zoom_start=12)
                folium.Marker(
                    location=[6.25, -75.58],
                    popup=get_text("geo_map_error_popup"),
                    icon=folium.Icon(color="red")
                ).add_to(basic_map)
                folium_static(basic_map, width=1200, height=600)
            
        with tab2:
            # Consumption Analysis Tab
            st.markdown(f"<div class='sub-header'>{get_text('consumption_subheader')}</div>", unsafe_allow_html=True)
            
            # Improved layout with 2x2 grid
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                # Enhanced anomaly distribution
                anomaly_dist_fig = plot_anomaly_distribution(
                    anomaly_scores,
                    threshold,
                    title_text=get_text("anomaly_distribution_title"),
                    x_label=get_text("plot_anomaly_score_label"),
                    y_label=get_text("plot_count_label"),
                    threshold_label=get_text("plot_threshold_label"),
                    no_data_title=get_text("anomaly_distribution_no_data")
                )
                
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
                    title=get_text("anomaly_distribution_title"),
                    height=400,
                    margin=dict(t=50, b=50, l=50, r=25)
                )
                st.plotly_chart(anomaly_dist_fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                # Enhanced scatter comparison
                scatter_fig = plot_scatter_comparison(
                    features,
                    anomaly_mask,
                    title_text=get_text("scatter_title"),
                    x_label=get_text("scatter_x_label"),
                    y_label=get_text("scatter_y_label"),
                    status_anomaly=get_text("scatter_status_anomaly"),
                    status_normal=get_text("scatter_status_normal"),
                    same_line_label=get_text("scatter_same_line"),
                    no_data_title=get_text("scatter_no_data")
                )
                scatter_fig.update_layout(
                    title=get_text("scatter_title"),
                    height=400,
                    margin=dict(t=50, b=50, l=50, r=25)
                )
                st.plotly_chart(scatter_fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                # Enhanced consumption patterns
                patterns_fig = plot_consumption_patterns(
                    consumption_df,
                    customers_df,
                    title_text=get_text("consumption_patterns_title"),
                    x_label=get_text("plot_x_date"),
                    y_label=get_text("plot_y_consumption"),
                    legend_title=get_text("plot_legend_stratum"),
                    stratum_prefix=get_text("plot_stratum_prefix"),
                    no_data_title=get_text("consumption_no_data"),
                    no_data_filtered_title=get_text("consumption_no_data_filtered")
                )
                patterns_fig.update_layout(
                    title=get_text("consumption_patterns_title"),
                    height=400,
                    margin=dict(t=50, b=50, l=50, r=25)
                )
                st.plotly_chart(patterns_fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                # Enhanced stratum distribution
                stratum_fig = plot_stratum_distribution(
                    filtered_customers,
                    anomaly_mask,
                    title_text=get_text("stratum_distribution_title"),
                    x_label=get_text("stratum_axis_title"),
                    y_label=get_text("risk_yaxis_title"),
                    status_anomaly=get_text("stratum_legend_anomaly"),
                    status_normal=get_text("stratum_legend_normal"),
                    no_data_title=get_text("stratum_no_data")
                )
                stratum_fig.update_layout(
                    title=get_text("stratum_distribution_title"),
                    height=400,
                    margin=dict(t=50, b=50, l=50, r=25)
                )
                st.plotly_chart(stratum_fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Feature importance (only if we have data and model is fitted)
            if not features.empty and 'mgd_model' in locals() and mgd_model.fitted:
                st.markdown(f"<div class='sub-header'>{get_text('feature_significance_header')}</div>", unsafe_allow_html=True)
                st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                
                # Enhanced feature importance visualization
                feature_fig = plot_feature_importance(
                    mgd_model,
                    common_features,
                    title_text=get_text("feature_importance_title"),
                    no_data_title=get_text("feature_importance_title_no_data")
                )
                feature_fig.update_layout(
                    title=get_text("feature_importance_title"),
                    height=500,
                    margin=dict(t=50, b=70, l=70, r=25)
                )
                st.plotly_chart(feature_fig, use_container_width=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Add feature explanation
                with st.expander(get_text("feature_importance_expander")):
                    st.markdown(get_text("feature_importance_markdown"))
            
        with tab3:
            # Fraud Detection Tab
            st.markdown(f"<div class='sub-header'>{get_text('fraud_detected_header')}</div>", unsafe_allow_html=True)
            
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
                                    return get_text("risk_level_critical")
                                elif score >= 60:
                                    return get_text("risk_level_high")
                                elif score >= 40:
                                    return get_text("risk_level_medium")
                                else:
                                    return get_text("risk_level_low")
                            
                            anomaly_customers['risk_level'] = anomaly_customers['normalized_score'].apply(get_risk_level)
                        
                        # Display as enhanced table
                        st.markdown("<div class='dataframe-container'>", unsafe_allow_html=True)
                        
                        # Configure columns for better display
                        display_cols = ['customer_id', 'zone_code', 'stratum', 'consumption_current', 
                                      'consumption_prev', 'percent_change', 'anomaly_score', 'risk_level']
                        
                        st.dataframe(
                            anomaly_customers[display_cols].sort_values('anomaly_score', ascending=False),
                            column_config={
                                'customer_id': st.column_config.TextColumn(get_text("column_customer_id")),
                                'zone_code': st.column_config.TextColumn(get_text("column_zone")),
                                'stratum': st.column_config.NumberColumn(get_text("column_stratum"), format="%d"),
                                'consumption_current': st.column_config.NumberColumn(get_text("column_current_consumption"), format="%.1f"),
                                'consumption_prev': st.column_config.NumberColumn(get_text("column_previous_consumption"), format="%.1f"),
                                'percent_change': st.column_config.NumberColumn(get_text("column_change_pct"), format="%.1f%%"),
                                'anomaly_score': st.column_config.NumberColumn(get_text("column_anomaly_score"), format="%.2f"),
                                'risk_level': st.column_config.TextColumn(get_text("risk_level_column"))
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Chart showing anomaly customers by risk level
                        risk_counts = anomaly_customers['risk_level'].value_counts().reset_index()
                        risk_counts.columns = [get_text("risk_level_column"), get_text("plot_count_label")]
                        
                        if not risk_counts.empty:
                            # Define color map for risk levels
                            risk_colors = {
                                get_text("risk_level_critical"): '#d32f2f',
                                get_text("risk_level_high"): '#f57c00',
                                get_text("risk_level_medium"): '#ffd166',
                                get_text("risk_level_low"): '#06d6a0'
                            }
                            
                            # Create risk level distribution chart
                            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                            
                            # Sort by risk severity
                            risk_order = [
                                get_text("risk_level_critical"),
                                get_text("risk_level_high"),
                                get_text("risk_level_medium"),
                                get_text("risk_level_low")
                            ]
                            risk_counts[get_text("risk_level_column")] = pd.Categorical(
                                risk_counts[get_text("risk_level_column")],
                                categories=risk_order, 
                                ordered=True
                            )
                            risk_counts = risk_counts.sort_values(get_text("risk_level_column"))
                            
                            risk_fig = px.bar(
                                risk_counts, 
                                x=get_text("risk_level_column"),
                                y=get_text("plot_count_label"),
                                color=get_text("risk_level_column"),
                                color_discrete_map=risk_colors,
                                title=get_text("risk_distribution_title")
                            )
                            
                            risk_fig.update_layout(
                                xaxis_title=get_text("risk_xaxis_title"),
                                yaxis_title=get_text("risk_yaxis_title"),
                                height=400
                            )
                            
                            st.plotly_chart(risk_fig, use_container_width=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Export options with improved button
                        csv_data = anomaly_customers[display_cols].to_csv(index=False)
                        col1, col2 = st.columns([3, 1])
                        
                        with col2:
                            st.download_button(
                                label=get_text("download_report_label"),
                                data=csv_data,
                                file_name=get_text("csv_filename_template").format(
                                    year=selected_year,
                                    month=selected_month
                                ),
                                mime="text/csv",
                                help=get_text("download_report_help")
                            )
                        
                        with col1:
                            st.info(get_text("report_ready_info"))
                        
                    except Exception as e:
                        st.error(get_text("anomaly_process_error").format(error=str(e)))
                        st.write(get_text("anomaly_process_detail_unavailable"))
                else:
                    st.info(get_text("no_fraud_detected"))
            else:
                st.info(get_text("no_fraud_detected_alt"))
            
            # Add fraud detection explanation
            st.markdown(f"<div class='sub-header'>{get_text('understanding_fraud_header')}</div>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                st.markdown(get_text("fraud_detected_markdown"))
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                st.markdown(get_text("risk_level_markdown"))
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Add fraud types explanation in an expander
            with st.expander(get_text("fraud_types_expander")):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(
                        f"### {get_text('fraud_types_electricity_title')}\n\n{get_text('fraud_types_electricity_list')}"
                    )
                
                with col2:
                    st.markdown(
                        f"### {get_text('fraud_types_gas_title')}\n\n{get_text('fraud_types_gas_list')}"
                    )
    
    except Exception as e:
        st.error(get_text("app_error").format(error=str(e)))
        st.warning(get_text("app_error_warning"))
        
        # Provide debugging information in an expander
        with st.expander(get_text("technical_details_expander")):
            st.code(str(e))
            st.write(get_text("technical_details_persist"))

# Add an about section in the sidebar
with st.sidebar.expander(get_text("about_expander")):
    st.write(get_text("about_text"))

# Run the app
if __name__ == "__main__":
    main()
