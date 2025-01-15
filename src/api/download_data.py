import os
import pandas as pd
import numpy as np
from loguru import logger

from settings import PROJECT_ROOT
from src.energy_analytics import Aggregate
from .smarthome import get_data_device, get_devices


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

    # TODO: Refactor this function

    logger.info(f"Starting data save process for building {building_id}, from {time_from} to {time_to}")

    time_from_dt = pd.to_datetime(time_from)
    time_to_dt = pd.to_datetime(time_to)

    time_from_tz = time_from_dt.tz_localize("Europe/Rome")
    time_to_tz = time_to_dt.tz_localize("Europe/Rome")

    # Obtaining the timedelta between Europe/Rome and UTC
    time_delta_from = time_from_tz.utcoffset().total_seconds() / 3600
    time_delta_to = time_to_tz.utcoffset().total_seconds() / 3600

    # Adjusting the time to UTC
    time_from_dt = time_from_dt - pd.Timedelta(hours=-time_delta_from)
    time_to_dt = time_to_dt - pd.Timedelta(hours=-time_delta_to)

    full_range = pd.date_range(start=time_from, end=time_to, freq='15T', tz="UTC")

    building_devices = get_devices(building_id)
    device_id = None
    for device in building_devices:
        if device["name"] == "Dispositivo Utente":
            device_id = device["uuid"]
            break

    if device_id is not None:
        properties = ["power_arithmeticMean_quarter", "impEnergy_delta_quarter", "expEnergy_delta_quarter",
                      "productionEnergy_delta_quarter", "productionPower_arithmeticMean_quarter"]
        data = get_data_device(device_id, properties, time_to_dt, time_from_dt)
        data_formatted = {col: dict(values) for col, values in data.items()}
        df = pd.DataFrame.from_dict(data_formatted)
        df.rename(columns={"power_arithmeticMean_quarter": "power", "impEnergy_delta_quarter": "impEnergy",
                           "expEnergy_delta_quarter": "expEnergy", "productionEnergy_delta_quarter": "productionEnergy",
                           "productionPower_arithmeticMean_quarter": "productionPower"}, inplace=True)
        df.index = pd.to_datetime(df.index, utc=True)
        # Replace None with NaN
        df.replace({None: np.nan}, inplace=True)
        df = df.reindex(full_range, fill_value=np.nan)
        df.reset_index(inplace=True, names=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.sort_values(by="timestamp", inplace=True)

        # Save CSV
        csv_path = os.path.join(PROJECT_ROOT, "data", "energy_meter", f"{building_id}.csv")
        df.to_csv(csv_path, index=False)
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
