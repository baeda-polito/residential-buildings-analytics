from .building import Building
from .aggregate import Aggregate
from .kpi import run_kpi
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

if not os.path.exists(os.path.join(PROJECT_ROOT, "figures")):
    os.makedirs(os.path.join(PROJECT_ROOT, "figures"))

if not os.path.exists(os.path.join(PROJECT_ROOT, "figures", "kpi")):
    os.makedirs(os.path.join(PROJECT_ROOT, "figures", "kpi"))

if not os.path.exists(os.path.join(PROJECT_ROOT, "figures", "benchmarking")):
    os.makedirs(os.path.join(PROJECT_ROOT, "figures", "benchmarking"))

if not os.path.exists(os.path.join(PROJECT_ROOT, "figures", "anomaly_detection")):
    os.makedirs(os.path.join(PROJECT_ROOT, "figures", "anomaly_detection"))