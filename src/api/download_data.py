import os
import pandas as pd
import numpy as np
from loguru import logger

from settings import PROJECT_ROOT
from src.energy_analytics import Aggregate
from src.api.smarthome import get_data_device, get_devices


def save_energy_data(building_id: str, time_from: str, time_to: str):
    """
    Funzione che salva i dati relativi al dispositivo utente di un determinato edificio.

    Args:
        building_id (str): id edificio
        time_from (str): stringa con la data di inizio nel formato "YYYY-MM-DDTHH:MM:SS"
        time_to (str): stringa con la data di fine nel formato "YYYY-MM-DDTHH:MM:SS"

    Returns:
        None
    """

    logger.info(f"Starting data save process for building {building_id}, from {time_from} to {time_to}")

    time_from_dt = pd.to_datetime(time_from)
    time_to_dt = pd.to_datetime(time_to)

    time_to_utc = time_to_dt.tz_localize("Europe/Rome").tz_convert("UTC")
    time_from_utc = time_from_dt.tz_localize("Europe/Rome").tz_convert("UTC")

    building_devices = get_devices(building_id)
    device_id = None
    for device in building_devices:
        if device["name"] == "Dispositivo Utente":
            device_id = device["uuid"]
            break

    if device_id is not None:
        properties = ["power_arithmeticMean_quarter", "impEnergy_delta_quarter", "expEnergy_delta_quarter",
                      "productionEnergy_delta_quarter", "productionPower_arithmeticMean_quarter"]

        df_full_range_utc = pd.DataFrame(pd.date_range(start=time_from_utc, end=time_to_utc, freq='15T'), columns=["timestamp"])
        groups_month = df_full_range_utc.groupby(df_full_range_utc["timestamp"].dt.to_period("M"))

        df_months_list = []
        for group in groups_month.groups:
            logger.debug(f"Downloading data for month {group}")
            time_from_month = groups_month.get_group(group).iloc[0]["timestamp"]
            time_to_month = groups_month.get_group(group).iloc[-1]["timestamp"]
            data = get_data_device(device_id, properties, time_to_month.strftime('%Y-%m-%dT%H:%M:%SZ'), time_from_month.strftime('%Y-%m-%dT%H:%M:%SZ'))
            data_formatted = {col: dict(values) for col, values in data.items()}
            df = pd.DataFrame.from_dict(data_formatted)
            df.rename(columns={"power_arithmeticMean_quarter": "power", "impEnergy_delta_quarter": "impEnergy",
                               "expEnergy_delta_quarter": "expEnergy", "productionEnergy_delta_quarter": "productionEnergy",
                               "productionPower_arithmeticMean_quarter": "productionPower"}, inplace=True)

            df_months_list.append(df)

        data = pd.concat(df_months_list)
        data.index = pd.to_datetime(data.index)
        data = data.sort_index()
        data = data.reindex(pd.date_range(start=time_from_utc, end=time_to_utc, freq='15T'), fill_value=np.nan)
        data = data.replace({None: np.nan})
        index = data.index
        index = pd.to_datetime(index, utc=True).tz_convert("Europe/Rome").tz_localize(None)
        data.index = index
        data = data.reset_index(names=["timestamp"])
        data.sort_values(by="timestamp", inplace=True)

        # Save CSV
        csv_path = os.path.join(PROJECT_ROOT, "data", "energy_meter", f"{building_id}.csv")
        data.to_csv(csv_path, index=False)
        logger.debug(f"CSV file saved for building {building_id} at {csv_path}")
    else:
        warning_msg = f"No user device found for building {building_id}"
        logger.warning(f"{warning_msg}")
        raise Warning(warning_msg)


def save_energy_data_aggregate(aggregate: Aggregate, time_from: str, time_to: str):
    """
    Funzione che salva i dati relativi al dispositivo utente di un determinato edificio

    Args:
        aggregate (Aggregate): oggetto Aggregate con le informazioni sull'aggregato
        time_from (str): stringa con la data di inizio nel formato "YYYY-MM-DDTHH:MM:SS"
        time_to (str): stringa con la data di fine nel formato "YYYY-MM-DDTHH:MM:SS"

    Returns:
        None
    """

    for building in aggregate.buildings:
        save_energy_data(building.building_info["id"], time_from, time_to)


def save_weather_data(aggregate: Aggregate):
    """
    Funzione che raccoglie i dati meteo relativi ad un aggregato e li salva in un unico file CSV.

    Args:
        aggregate (Aggregate): oggetto Aggregate con le informazioni sull'aggregato

    Returns:
        None
    """

    files = os.listdir(os.path.join(PROJECT_ROOT, "data", "weather"))
    files = [f for f in files if f.startswith(aggregate.name)]
    files = [f for f in files if f != f"{aggregate.name}.csv"]

    weather_data = pd.concat([pd.read_csv(os.path.join(PROJECT_ROOT, "data", "weather", f)) for f in files])
    weather_data.drop(columns=["period"], inplace=True)
    weather_data.rename(columns={"period_end": "timestamp"}, inplace=True)
    weather_data["timestamp"] = pd.to_datetime(weather_data["timestamp"])
    weather_data.drop_duplicates(subset=["timestamp"], inplace=True)

    cols = weather_data.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    weather_data = weather_data[cols]
    weather_data.to_csv(os.path.join(PROJECT_ROOT, "data", "weather", f"{aggregate.name}.csv"), index=False)
