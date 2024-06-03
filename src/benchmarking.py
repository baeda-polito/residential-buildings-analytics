from building import load_anguillara
import shape_factor
import pandas as pd


building_list = load_anguillara()
building_sf_dict = {}

for building in building_list:
    data = building.energy_meter.data.copy()
    data.set_index("timestamp", inplace=True)
    data = data["Load"]
    data = data.resample("1H").mean()
    mean_max = shape_factor.sf_mean_max(data)
    min_max = shape_factor.sf_min_max(data)
    min_mean = shape_factor.sf_min_mean(data)
    daytime_mean = shape_factor.sf_daytime_mean(data)
    daytime_max = shape_factor.sf_daytime_max(data)
    daytime_min_mean = shape_factor.sf_daytime_min_mean(data)
    nightime_mean = shape_factor.sf_nightime_mean(data)
    nightime_max = shape_factor.sf_nightime_max(data)
    nightime_min_mean = shape_factor.sf_nightime_min_mean(data)
    afternoon_mean = shape_factor.sf_afternoon_mean(data)
    afternoon_max = shape_factor.sf_afternoon_max(data)
    afternoon_min_mean = shape_factor.sf_afternoon_min_mean(data)

    df_sf = pd.concat([mean_max, min_max, min_mean, daytime_mean, daytime_max, daytime_min_mean, nightime_mean,
                       nightime_max, nightime_min_mean, afternoon_mean, afternoon_max, afternoon_min_mean], axis=1)
    df_sf.columns = ["mean_max", "min_max", "min_mean", "daytime_mean", "daytime_max", "daytime_min_mean",
                     "nightime_mean", "nightime_max", "nightime_min_mean", "afternoon_mean", "afternoon_max",
                     "afternoon_min_mean"]
    building_sf_dict[building.building_info["id"]] = df_sf
