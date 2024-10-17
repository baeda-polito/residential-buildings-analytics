import os
import pandas as pd
import numpy as np
from src.building import load_anguillara, load_garda
from src.api.smarthome import get_data_device, get_devices
from settings import PROJECT_ROOT
import logging

# Configure logging (you can customize the log format, level, etc.)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def save_energy_data(building_id: str, time_from: str, time_to: str):
    """
    Funzione che salva i dati relativi al dispositivo utente di un determinato edificio
    :param building_id: id edificio
    :param time_from: stringa con la data di inizio nel formato "YYYY-MM-DDTHH:MM:SS"
    :param time_to: stringa con la data di fine nel formato "YYYY-MM-DDTHH:MM:SS"
    :return:
    """
    logging.info(f"\033[97mStarting data save process for building {building_id}, from {time_from} to {time_to}")

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
        logging.info(f"\033[92m\u2714 CSV file saved for building {building_id} at {csv_path}")

    else:
        warning_msg = f"No user device found for building {building_id}"
        logging.warning(f"\u2716 {warning_msg}")
        raise Warning(warning_msg)


def save_energy_data_aggregate(aggregate: str, time_from: str, time_to: str):
    """
    Funzione che salva i dati relativi al dispositivo utente di un determinato edificio
    :param aggregate: stringa con il nome dell'aggregato
    :param time_from: stringa con la data di inizio nel formato "YYYY-MM-DDTHH:MM:SS"
    :param time_to: stringa con la data di fine nel formato "YYYY-MM-DDTHH:MM:SS"
    :return:
    """

    building_list = []
    if aggregate == "anguillara":
        building_list = load_anguillara()
    elif aggregate == "garda":
        building_list = load_garda()

    for building in building_list:
        save_energy_data(building.building_info["id"], time_from, time_to)


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
    #save_energy_data_aggregate("anguillara", "2024-01-01T00:00:00", "2024-10-15T23:59:59")
    save_energy_data_aggregate("garda", "2024-01-01T00:00:00", "2024-10-15T23:59:59")