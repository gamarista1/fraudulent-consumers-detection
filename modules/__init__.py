# This file makes the modules directory a Python package
# Import main classes and functions for easier access

from .mgd_model import MGDAnomaly
from .data_processor import load_sample_data, prepare_features, clean_data
from .visualization import (
    plot_consumption_patterns, 
    create_anomaly_map, 
    plot_anomaly_distribution,
    plot_feature_importance,
    plot_stratum_distribution, 
    plot_scatter_comparison,
    create_kpi_cards
)