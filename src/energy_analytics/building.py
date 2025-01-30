import pandas as pd
import numpy as np
from loguru import logger

from .pre_processing import pre_process_power, pre_process_production_power


class Building:
    """
    Questa classe modella i dati di un edificio, al quale vengono forniti i dati relativi al contatore di energia e
    i metadati riguardo all'edificio stesso.
    """

    def __init__(self, data: pd.DataFrame, weather_data: pd.DataFrame, metadata: dict):
        """
        Inizializza la classe Building.
        Args:
            data (pd.DataFrame): dati relativi al contatore di energia. Deve contenere una colonna "timestamp" e almeno la colonna "power". Se l'utente è un prosumer/prostormer, deve contenere anche la colonna "productionPower"
            weather_data (pd.DataFrame): dati meteorologici. Necessari solo se l'utente è un prosumer/prostormer
            metadata (dict): metadati dell'edificio. I valori che devono essere presenti sono:
                - "id": identificativo dell'edificio
                - "name": nome dell'edificio
                - "user_type": tipologia di utente
                - "rated_power": potenza nominale
                Valori opzionali:
                - "pv": dizionario con i seguenti valori: "tilt", "azimuth", "rated_power", "storage"
                - "coordinates": lista delle coordinate geografiche dell'edificio
                - "persons": numero di persone nell'edificio
                - "occupancy": dizionario con i seguenti valori: "24-8", "8-13", "13-19", "19-24"
                - "tariff": stringa con il tipo di tariffa
                - "surface": superficie dell'edificio
                - "ac": dizionario con i seguenti valori: "n_ac", "type"
                - "retrofit": 0/1 se l'edificio è stato ristrutturato
        """

        self.building_info = metadata
        self.energy_data = EnergyData(data=data, weather_data=weather_data)
        self._validate_metadata()

    def _validate_metadata(self):
        """
        Controlla che ci siano almeno le chiavi "uuid", "user_type", "rated_power"
        """

        if "id" not in self.building_info:
            raise ValueError("uuid non presente nei metadati")
        if "user_type" not in self.building_info:
            raise ValueError("user_type non presente nei metadati")
        if "rated_power" not in self.building_info:
            raise ValueError("rated_power non presente nei metadati")

    def pre_process(self):
        """
        Pre-processa i dati del contatore di energia.
        """
        logger.info(f"Pre-processing dei dati per l'edificio {self.building_info['name']}")
        self.energy_data.pre_process(metadata=self.building_info)
        self.energy_data.define_load_components(user_type=self.building_info["user_type"])


class EnergyData:
    """
    Questa classe modella i dati di un contatore di energia.
    """

    def __init__(self, data: pd.DataFrame, weather_data: pd.DataFrame = None):
        """
        Inizializza la classe EnergyData.
        Args:
            data (pd.DataFrame): dati relativi al contatore di energia
        """

        self.data_raw = data
        self.weather_data = weather_data
        self.data_cleaned = None
        self.data = None

    def pre_process(self, metadata: dict):
        """
        Pre-processa i dati del contatore di energia.
        Args:
            metadata (dict): metadati dell'edificio
        """

        data = self.data_raw.copy()

        data_cleaned = data.copy()

        data["timestamp"] = pd.to_datetime(data["timestamp"])

        data_power = pre_process_power(
            data=data,
            user_type=metadata["user_type"],
            rated_power=metadata["rated_power"],
            rated_pv_power=metadata["pv"]["rated_power"] if "rated_power" in metadata["pv"] else None
        )
        data_power["timestamp"] = pd.to_datetime(data_power["timestamp"])
        data_power = data_power.set_index("timestamp")
        # data_power.index = pd.to_datetime(data_power.index)
        data_cleaned = data_cleaned.set_index("timestamp")
        data_cleaned.index = pd.to_datetime(data_cleaned.index)
        data_cleaned.loc[data_power.index, "power"] = data_power["power"]
        data_cleaned = data_cleaned.reset_index()

        if metadata["user_type"] != "consumer":
            if metadata["pv"]["tilt"] is not None and metadata["pv"]["azimuth"] is not None and metadata["pv"]["rated_power"] is not None:
                physic_model = True
            else:
                physic_model = False
            weather = self.weather_data.copy()
            weather["timestamp"] = pd.to_datetime(weather["timestamp"])
            production_power = pre_process_production_power(
                data=data_cleaned,
                weather_data=weather,
                physic_model=physic_model,
                pv_params=metadata["pv"],
                coordinates=metadata["coordinates"]
            )
            production_power = production_power.set_index("timestamp")
            data_cleaned = data_cleaned.set_index("timestamp")
            data_cleaned.loc[production_power.index, "productionPower"] = production_power["productionPower"]
            data_cleaned = data_cleaned.reset_index()

        self.data_cleaned = data_cleaned

    def define_load_components(self, user_type: str):
        """
        Definisce le componenti del carico dell'edificio, ovvero "Load", "Net" e "Production".
        Args:
            user_type (str): tipologia di utente
        """

        logger.debug(f"Definizione delle componenti del carico.")
        data = self.data_cleaned.copy()

        if user_type != "consumer":
            data.dropna(inplace=True, subset=["productionPower", "power"])
        else:
            data.dropna(inplace=True, subset=["power"])

        if len(data) > 0:
            day_groups = data.groupby(data["timestamp"].dt.date)

            for date, day in day_groups:
                if len(day) < 96:
                    data.drop(day.index, inplace=True)

            if user_type != "consumer":
                load = np.where(data["power"] < 0, data["productionPower"] - abs(data["power"]),
                                data["productionPower"] + data["power"])
                # Physical constraint
                load[load < 0] = np.nan
                load = pd.Series(load).interpolate(method="linear", limit_direction="both")

                net = data["power"].reset_index(drop=True)
                production = data["productionPower"].reset_index(drop=True)
                timestamp = data["timestamp"].reset_index(drop=True)
                self.data = pd.DataFrame({"timestamp": timestamp, "Load": load, "Net": net, "Production": production})

            else:
                timestamp = data["timestamp"].reset_index(drop=True)
                load = data["power"].reset_index(drop=True)
                net = load
                production = data["productionPower"].reset_index(drop=True)

                self.data = pd.DataFrame({"timestamp": timestamp, "Load": load, "Net": net, "Production": production})

        else:
            self.data = pd.DataFrame(columns=["timestamp", "Load", "Net", "Production"])
