from src.api.smarthome import get_devices
from src.pre_processing import pre_process_power, pre_process_production_power
from settings import PROJECT_ROOT
import os
import json
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class Building:
    """
    Classe che contiene le informazioni utili per un edificio e il suo contatore di energia.
    """

    def __init__(self, uuid: str, aggregate: str):
        """
        Costruttore della classe Building
        :param uuid: identificativo dell'edificio
        :param aggregate: nome dell'aggregato
        """
        self.building_info = {}
        self.get_building_info(uuid)
        self.building_info['aggregate'] = aggregate
        self.energy_meter = EnergyMeter(uuid)
        self.energy_meter.set_data(uuid)
        self.energy_meter.pre_process(self.building_info["user_type"],
                                      self.building_info["rated_power"],
                                      aggregate=aggregate,
                                      pv_params=self.building_info["pv"],
                                      coordinates=self.building_info["coordinates"])
        self.energy_meter.define_load_components(user_type=self.building_info["user_type"])

    def get_building_info(self, uuid):
        """
        Estrae le informazioni dell'edificio dal portale di SmartHome e le carica nell'attributo 'building_info'
        """
        with open(os.path.join(PROJECT_ROOT, "data", "metadata", f"{uuid}.json"), "r") as f:
            self.building_info = json.load(f)


class EnergyMeter:
    """
    Classe che contiene le informazioni utili per un contatore di energia e i suoi dati.
    """

    def __init__(self, building_id):
        self.energy_meter_info = {
            "id": None,
            "properties": None,
            "aggregation_functions": None,
            "name": None
        }
        self.energy_meter_data = None
        self.energy_meter_data_cleaned = None
        self.data = None
        # self.get_energy_meter_info(building_id)
        self.set_data(building_id)

    def get_energy_meter_info(self, building_id):
        """
        Estrae le informazioni del contatore di energia dal portale di SmartHome e le carica nell'attributo
        'energy_meter_info'. In particolare estrae id, proprietà, funzioni di aggregazione e nome del contatore.
        :param building_id: identificativo dell'edificio
        """
        device_list = get_devices(building_id)
        energy_meter = [device for device in device_list if device["name"] == "Dispositivo Utente"][0]
        self.energy_meter_info = {
            "id": energy_meter["uuid"],
            "properties": list(energy_meter["deviceType"]["properties"].keys()),
            "aggregation_functions": ["sumOfEnergy", "", "delta", "delta", "delta", "arithmeticMean", ""],
            "name": energy_meter["name"]
        }

    def set_data(self, building_id):
        """
        Salva i dati del contatore di energia in base alla modalità di recupero dei dati.
        :param building_id: Identificatore dell'edificio
        :param mode: online od offline
        """
        self.energy_meter_data = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "energy_meter", f"{building_id}.csv"))

    def pre_process(self, user_type, rated_power, aggregate, pv_params: dict, coordinates: list):
        """
        Pre-processa i dati del contatore di energia in base al tipo di utente
        """
        data = self.energy_meter_data.copy()
        data_cleaned = data.copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        # Pre-processing 'power'
        data_power = pre_process_power(data,
                                       user_type,
                                       rated_power=rated_power,
                                       rated_pv_power=pv_params["rated_power"])
        data_power.set_index("timestamp", inplace=True)
        data_power.index = pd.to_datetime(data_power.index, utc=True)
        data_cleaned.set_index("timestamp", inplace=True)
        data_cleaned.index = pd.to_datetime(data_cleaned.index)
        data_cleaned.loc[data_power.index, "power"] = data_power["power"]
        data_cleaned.reset_index(inplace=True)

        if user_type != "consumer":
            # Pre-processing productionPower
            if pv_params["tilt"] is not None and pv_params["azimuth"] is not None and pv_params["rated_power"] is not None:
                physic_model = True
            else:
                physic_model = False
            weather = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "weather", f"{aggregate}.csv"))
            weather["timestamp"] = pd.to_datetime(weather["timestamp"])
            production_power = pre_process_production_power(data_cleaned, weather,
                                                            physic_model=physic_model,
                                                            pv_params=pv_params,
                                                            coordinates=coordinates)
            production_power.set_index("timestamp", inplace=True)
            data_cleaned.set_index("timestamp", inplace=True)
            data_cleaned.loc[production_power.index, "productionPower"] = production_power["productionPower"]
            data_cleaned.reset_index(inplace=True)
        self.energy_meter_data_cleaned = data_cleaned

    def define_load_components(self, user_type: str):
        data = self.energy_meter_data_cleaned.copy()

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
                # data.set_index("timestamp", inplace=True)
                timestamp = data["timestamp"].reset_index(drop=True)
                load = data["power"].reset_index(drop=True)
                net = load
                production = data["productionPower"].reset_index(drop=True)

                self.data = pd.DataFrame({"timestamp": timestamp, "Load": load, "Net": net, "Production": production})

        else:
            self.data = pd.DataFrame(columns=["timestamp", "Load", "Net", "Production"])


def load_anguillara():
    DU_1 = Building("7436df46-294b-4c97-bd1b-8aaa3aed97c5", aggregate="anguillara")
    DU_2 = Building("80c3bedd-8c41-450c-ae52-1864b9ace7aa", aggregate="anguillara")
    DU_4 = Building("d93552c8-e7f6-45bb-b382-bd4a2b969502", aggregate="anguillara")
    DU_5 = Building("b87be67b-8133-4b7f-a045-c06da08b5416", aggregate="anguillara")
    DU_6 = Building("9a3386b3-017c-4848-ac6d-a24bf7f36077", aggregate="anguillara")
    DU_7 = Building("8490da00-eb75-45df-888e-851ea3103ec4", aggregate="anguillara")
    DU_8 = Building("08f2fc03-ce0b-4cd6-ab25-8b3906feb858", aggregate="anguillara")
    DU_9 = Building("3d956901-f5ea-4094-9c85-333cc68183d4", aggregate="anguillara")
    DU_10 = Building("4ef8599c-2c4b-433e-94c8-ca48e23a5a07", aggregate="anguillara")

    return [DU_1, DU_2, DU_4, DU_5, DU_6, DU_7, DU_8, DU_9, DU_10]


def load_garda():
    DU_11 = Building("903a9b98-8c2c-49f9-a31d-9a398c4fafb3", aggregate="garda")
    DU_12 = Building("a99641a4-5da6-4922-a6ea-d578847a094d", aggregate="garda")
    DU_13 = Building("d91a0269-386f-4486-854d-a1e11405d97d", aggregate="garda")
    DU_14 = Building("cd2198a5-069a-4ea1-a118-fa0ef8d43005", aggregate="garda")
    DU_15 = Building("2785c76e-1f34-45f1-ab14-39da69957482", aggregate="garda")
    DU_16 = Building("8ae1b59c-a5af-4036-9469-db8a12ba1427", aggregate="garda")
    DU_17 = Building("12eefaba-1024-4883-97a1-719f3b8e2c96", aggregate="garda")
    DU_18 = Building("eaaa2a7f-9631-4869-b33a-fca820464b41", aggregate="garda")
    DU_19 = Building("0af9a404-635f-43d7-82bf-064981cb0145", aggregate="garda")
    DU_20 = Building("5b434307-b3ce-4580-86de-e0b74c8da2b8", aggregate="garda")

    return [DU_11, DU_12, DU_13, DU_14, DU_15, DU_16, DU_17, DU_18, DU_19, DU_20]


if __name__ == "__main__":
    anguillara = load_anguillara()