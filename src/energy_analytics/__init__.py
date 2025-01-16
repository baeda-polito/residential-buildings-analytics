import warnings

from .building import Building
from .aggregate import Aggregate
from .kpi import run_kpi
from .benchmarking import run_benchmarking
from .anomaly_detection import run_train, run_evaluation, detect_anomaly_pv
from settings import PROJECT_ROOT
from .utils import create_directories

warnings.filterwarnings("ignore")


# Define all required directories
required_directories = [
    "results",
    "results/kpi",
    "results/benchmarking",
    "results/benchmarking/models",
    "results/anomaly_detection",
    "results/anomaly_detection/pv_models",
    "results/anomaly_detection/pv_models/scalers",
    "results/anomaly_detection/pv_models/thresholds",
    "results/anomaly_detection/pv_models/loss",
    "results/anomaly_detection/pv_models/metrics",
    "figures",
    "figures/pre_processing",
    "figures/kpi",
    "figures/benchmarking",
    "figures/benchmarking/feature_distribution",
    "figures/anomaly_detection",
    "figures/anomaly_detection/pv_models",
]

# Create all directories
create_directories(PROJECT_ROOT, required_directories)
