import os
import pandas as pd
from src.building import load_anguillara, load_garda
from settings import PROJECT_ROOT


def save_energy_data(aggregate="anguillara"):

    if aggregate == "anguillara":
        building_list = load_anguillara(mode="online")
    elif aggregate == "garda":
        building_list = load_garda(mode="online")

    for building in building_list:
        print(building)
        energy_meter = building.energy_meter.energy_meter_data
        energy_meter.to_csv(os.path.join(PROJECT_ROOT, "data", "energy_meter", f"{building.building_info['id']}.csv"),
                            index=False)


def save_weather_data(prefix="anguillara"):

    files = os.listdir(os.path.join(PROJECT_ROOT, "data", "weather"))
    files = [f for f in files if f.startswith(prefix)]
    files = [f for f in files if f != f"{prefix}.csv"]

    weather_data = pd.concat([pd.read_csv(os.path.join(PROJECT_ROOT, "data", "weather", f)) for f in files])
    weather_data.drop(columns=["period"], inplace=True)
    weather_data.rename(columns={"period_end": "timestamp"}, inplace=True)
    weather_data["timestamp"] = pd.to_datetime(weather_data["timestamp"])
    weather_data.drop_duplicates(subset=["timestamp"], inplace=True)

    cols = weather_data.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    weather_data = weather_data[cols]
    weather_data.to_csv(os.path.join(PROJECT_ROOT, "data", "weather", f"{prefix}.csv"), index=False)


if __name__ == "__main__":
    save_weather_data(prefix="garda")
    save_energy_data(aggregate="garda")
