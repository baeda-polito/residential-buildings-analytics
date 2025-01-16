from src.building import load_anguillara
from pvlib.location import Location
import pandas as pd
import json

building_list = load_anguillara()
data_list = []

for building in building_list:
    energy_data = building.energy_meter.data
    weather_data = pd.read_csv("../data/weather/anguillara.csv")
    weather_data["timestamp"] = pd.to_datetime(weather_data["timestamp"])
    with open(f"../data/metadata/{building.building_info['id']}.json", "r") as f:
        metadata = json.load(f)

    location = Location(latitude=building.building_info["coordinates"][1], longitude=building.building_info["coordinates"][0])

    data = pd.merge(energy_data[["timestamp", "Net"]], weather_data, on="timestamp", how="left")
    data.dropna(inplace=True)
    data.rename(columns={"Net": "load"}, inplace=True)

    data["timeseriesID"] = building.building_info["name"]
    data["time_observation"] = data.reset_index().index

    occupancy = pd.DataFrame.from_dict(metadata["occupancy"], orient="index", columns=["occupancy"])
    occupancy["occupancy"] = occupancy["occupancy"] / occupancy["occupancy"].max()
    occupancy.index = ["occupancy_morning", "occupancy_afternoon", "occupancy_evening", "occupancy_night"]

    # Join the occupancy data
    data["occupancy_morning"] = occupancy.loc["occupancy_morning", "occupancy"]
    data["occupancy_afternoon"] = occupancy.loc["occupancy_afternoon", "occupancy"]
    data["occupancy_evening"] = occupancy.loc["occupancy_evening", "occupancy"]
    data["occupancy_night"] = occupancy.loc["occupancy_night", "occupancy"]

    data["user_type"] = metadata["user_type"]
    data["surface"] = metadata["surface"]
    data["persons"] = metadata["persons"]
    data["rated_power"] = metadata["rated_power"]

    data["n_ac"] = metadata["ac"]["n_ac"]
    data["ac_type"] = metadata["ac"]["type"]

    data["pv_azimuth"] = metadata["pv"]["azimuth"]
    data["pv_tilt"] = metadata["pv"]["tilt"]
    data["storage_capacity"] = metadata["pv"]["storage"]
    data["pv_power"] = metadata["pv"]["rated_power"]
    data["user_name"] = metadata["name"]

    data_list.append(data.reset_index(drop=True))

data_total = pd.concat(data_list)
data_total.to_csv("../data/ts.csv", index=False)
