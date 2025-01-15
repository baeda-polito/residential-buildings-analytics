from .building import Building
from .aggregate import Aggregate
from .kpi import run_kpi
from .benchmarking import run_benchmarking
from settings import PROJECT_ROOT

import warnings
import os

warnings.filterwarnings("ignore")

# Create the repository for the results and figures if they do not exists
if not os.path.exists(os.path.join(PROJECT_ROOT, "results")):
    os.makedirs(os.path.join(PROJECT_ROOT, "results"))

if not os.path.exists(os.path.join(PROJECT_ROOT, "results", "kpi")):
    os.makedirs(os.path.join(PROJECT_ROOT, "results", "kpi"))

if not os.path.exists(os.path.join(PROJECT_ROOT, "results", "benchmarking")):
    os.makedirs(os.path.join(PROJECT_ROOT, "results", "benchmarking"))

if not os.path.exists(os.path.join(PROJECT_ROOT, "results", "anomaly_detection")):
    os.makedirs(os.path.join(PROJECT_ROOT, "results", "anomaly_detection"))

if not os.path.exists(os.path.join(PROJECT_ROOT, "results", "anomaly_detection", "pv_models")):
    os.makedirs(os.path.join(PROJECT_ROOT, "results", "anomaly_detection", "pv_models"))

if not os.path.exists(os.path.join(PROJECT_ROOT, "results", "anomaly_detection", "pv_models", "scalers")):
    os.makedirs(os.path.join(PROJECT_ROOT, "results", "anomaly_detection", "pv_models", "scalers"))

    if not os.path.exists(os.path.join(PROJECT_ROOT, "results", "anomaly_detection", "pv_models", "thresholds")):
        os.makedirs(os.path.join(PROJECT_ROOT, "results", "anomaly_detection", "pv_models", "thresholds"))

if not os.path.exists(os.path.join(PROJECT_ROOT, "results", "anomaly_detection", "pv_models", "loss")):
    os.makedirs(os.path.join(PROJECT_ROOT, "results", "anomaly_detection", "pv_models", "loss"))

if not os.path.exists(os.path.join(PROJECT_ROOT, "results", "anomaly_detection", "pv_models", "metrics")):
    os.makedirs(os.path.join(PROJECT_ROOT, "results", "anomaly_detection", "pv_models", "metrics"))

if not os.path.exists(os.path.join(PROJECT_ROOT, "figures")):
    os.makedirs(os.path.join(PROJECT_ROOT, "figures"))

if not os.path.exists(os.path.join(PROJECT_ROOT, "figures", "pre_processing")):
    os.makedirs(os.path.join(PROJECT_ROOT, "figures", "pre_processing"))

if not os.path.exists(os.path.join(PROJECT_ROOT, "figures", "kpi")):
    os.makedirs(os.path.join(PROJECT_ROOT, "figures", "kpi"))

if not os.path.exists(os.path.join(PROJECT_ROOT, "figures", "benchmarking")):
    os.makedirs(os.path.join(PROJECT_ROOT, "figures", "benchmarking"))

if not os.path.exists(os.path.join(PROJECT_ROOT, "figures", "benchmarking", "feature_distribution")):
    os.makedirs(os.path.join(PROJECT_ROOT, "figures", "benchmarking", "feature_distribution"))

if not os.path.exists(os.path.join(PROJECT_ROOT, "figures", "anomaly_detection")):
    os.makedirs(os.path.join(PROJECT_ROOT, "figures", "anomaly_detection"))

if not os.path.exists(os.path.join(PROJECT_ROOT, "figures", "anomaly_detection", "pv_models")):
    os.makedirs(os.path.join(PROJECT_ROOT, "figures", "anomaly_detection", "pv_models"))
