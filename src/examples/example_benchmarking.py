import pandas as pd
import json
import os

from settings import PROJECT_ROOT
from src.energy_analytics import Building, Aggregate, run_benchmarking


data_DU1 = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "energy_meter", "7436df46-294b-4c97-bd1b-8aaa3aed97c5.csv"))
metadata_DU1 = json.load(open(os.path.join(PROJECT_ROOT, "data", "metadata", "7436df46-294b-4c97-bd1b-8aaa3aed97c5.json")))

data_DU2 = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "energy_meter", "80c3bedd-8c41-450c-ae52-1864b9ace7aa.csv"))
metadata_DU2 = json.load(open(os.path.join(PROJECT_ROOT, "data", "metadata", "80c3bedd-8c41-450c-ae52-1864b9ace7aa.json")))

data_DU4 = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "energy_meter", "d93552c8-e7f6-45bb-b382-bd4a2b969502.csv"))
metadata_DU4 = json.load(open(os.path.join(PROJECT_ROOT, "data", "metadata", "d93552c8-e7f6-45bb-b382-bd4a2b969502.json")))

data_DU5 = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "energy_meter", "b87be67b-8133-4b7f-a045-c06da08b5416.csv"))
metadata_DU5 = json.load(open(os.path.join(PROJECT_ROOT, "data", "metadata", "b87be67b-8133-4b7f-a045-c06da08b5416.json")))

data_DU6 = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "energy_meter", "9a3386b3-017c-4848-ac6d-a24bf7f36077.csv"))
metadata_DU6 = json.load(open(os.path.join(PROJECT_ROOT, "data", "metadata", "9a3386b3-017c-4848-ac6d-a24bf7f36077.json")))

data_DU7 = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "energy_meter", "8490da00-eb75-45df-888e-851ea3103ec4.csv"))
metadata_DU7 = json.load(open(os.path.join(PROJECT_ROOT, "data", "metadata", "8490da00-eb75-45df-888e-851ea3103ec4.json")))

data_DU8 = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "energy_meter", "08f2fc03-ce0b-4cd6-ab25-8b3906feb858.csv"))
metadata_DU8 = json.load(open(os.path.join(PROJECT_ROOT, "data", "metadata", "08f2fc03-ce0b-4cd6-ab25-8b3906feb858.json")))

data_DU9 = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "energy_meter", "3d956901-f5ea-4094-9c85-333cc68183d4.csv"))
metadata_DU9 = json.load(open(os.path.join(PROJECT_ROOT, "data", "metadata", "3d956901-f5ea-4094-9c85-333cc68183d4.json")))

data_DU10 = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "energy_meter", "4ef8599c-2c4b-433e-94c8-ca48e23a5a07.csv"))
metadata_DU10 = json.load(open(os.path.join(PROJECT_ROOT, "data", "metadata", "4ef8599c-2c4b-433e-94c8-ca48e23a5a07.json")))

weather = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "weather", "anguillara.csv"))

DU1 = Building(data=data_DU1, weather_data=weather, metadata=metadata_DU1)
DU2 = Building(data=data_DU2, weather_data=weather, metadata=metadata_DU2)
DU4 = Building(data=data_DU4, weather_data=weather, metadata=metadata_DU4)
DU5 = Building(data=data_DU5, weather_data=weather, metadata=metadata_DU5)
DU6 = Building(data=data_DU6, weather_data=weather, metadata=metadata_DU6)
DU7 = Building(data=data_DU7, weather_data=weather, metadata=metadata_DU7)
DU8 = Building(data=data_DU8, weather_data=weather, metadata=metadata_DU8)
DU9 = Building(data=data_DU9, weather_data=weather, metadata=metadata_DU9)
DU10 = Building(data=data_DU10, weather_data=weather, metadata=metadata_DU10)

aggregate = Aggregate(name="anguillara", buildings=[DU1, DU2, DU4, DU5, DU6, DU7, DU8, DU9, DU10])

for building in aggregate.buildings:
    building.pre_process()

cluster = pd.read_csv(os.path.join(PROJECT_ROOT, "results", "benchmarking", "cluster_anguillara.csv"))

run_benchmarking(aggregate)
