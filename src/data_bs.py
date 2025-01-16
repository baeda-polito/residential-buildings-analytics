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

    data['date'] = data['timestamp'].dt.date
    data['quarter_hour'] = data['timestamp'].dt.strftime('%H:%M')

    # Step 2: Pivot the DataFrame
    df_pivot = data.pivot_table(index='date', columns='quarter_hour', values=['load', 'ghi', 'dni', 'air_temp'])

    # Step 3: Flatten the multi-level columns
    df_pivot.columns = ['_'.join(col).strip() for col in df_pivot.columns.values]
    df_pivot.dropna(inplace=True)

    occupancy = pd.DataFrame.from_dict(metadata["occupancy"], orient="index", columns=["occupancy"])
    occupancy["occupancy"] = occupancy["occupancy"] / occupancy["occupancy"].max()
    occupancy.index = ["occupancy_morning", "occupancy_afternoon", "occupancy_evening", "occupancy_night"]

    # Join the occupancy data
    df_pivot["occupancy_morning"] = occupancy.loc["occupancy_morning", "occupancy"]
    df_pivot["occupancy_afternoon"] = occupancy.loc["occupancy_afternoon", "occupancy"]
    df_pivot["occupancy_evening"] = occupancy.loc["occupancy_evening", "occupancy"]
    df_pivot["occupancy_night"] = occupancy.loc["occupancy_night", "occupancy"]

    df_pivot["user_type"] = metadata["user_type"]
    df_pivot["surface"] = metadata["surface"]
    df_pivot["persons"] = metadata["persons"]
    df_pivot["rated_power"] = metadata["rated_power"]

    df_pivot["n_ac"] = metadata["ac"]["n_ac"]
    df_pivot["ac_type"] = metadata["ac"]["type"]

    df_pivot["pv_azimuth"] = metadata["pv"]["azimuth"]
    df_pivot["pv_tilt"] = metadata["pv"]["tilt"]
    df_pivot["storage_capacity"] = metadata["pv"]["storage"]
    df_pivot["pv_power"] = metadata["pv"]["rated_power"]
    df_pivot["user_name"] = metadata["name"]

    data_list.append(df_pivot.reset_index(drop=True))

data_total = pd.concat(data_list)
data_total.to_csv("../data/load_profiles.csv", index=False)