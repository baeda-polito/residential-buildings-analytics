from src.api.smarthome import get_plant_info, get_devices, get_data_device
from settings import PROJECT_ROOT
import os
import pandas as pd
from datetime import datetime, timedelta


class Building:
    """
    Classe che contiene le informazioni utili per un edificio e il suo contatore di energia.
    :param uuid: identificativo dell'edificio
    :param get_data_mode: modalità di recupero dei dati, online o offline. Se online, i dati verranno scaricati dal
    portale di SmartHome, altrimenti verranno presi da un file csv nella cartella "data".
    """
    def __init__(self, uuid, get_data_mode="offline"):
        self.building_info = {
            "id": uuid,
            "name": None,
            "user_type": None,
            "persons": None,
            "occupancy": None,
            "rated_power": None,
            "tariff": None,
            "surface": None,
            "coordinates": None,
        }
        self.get_building_info()
        self.energy_meter = EnergyMeter(uuid, get_data_mode)

    def get_building_info(self):
        """
        Estrae le informazioni dell'edificio dal portale di SmartHome e le carica nell'attributo 'building_info'
        """
        flag = 0
        try:
            plant_info = get_plant_info(self.building_info["id"])["plant"]
            flag = 1
        except Exception as e:
            print(f"Is not possible to retrieve information for building {self.building_info['id']}")

        if flag == 1:
            self.building_info["name"] = plant_info["name"]
            self.building_info["user_type"] = plant_info["metadata"]["userType"]
            self.building_info["persons"] = plant_info["metadata"]["tipologia_utenza"]["person_number"]
            self.building_info["occupancy"] = plant_info["metadata"]["tipologia_utenza"]["time_person_number"]
            self.building_info["rated_power"] = plant_info["metadata"]["tipologia_utenza"]["contract"]["power"]
            self.building_info["tariff"] = plant_info["metadata"]["tipologia_utenza"]["contract"]["price"]
            self.building_info["surface"] = plant_info["metadata"]["surface"]
            self.building_info["coordinates"] = plant_info["metadata"]["address"]["coordinates"]


class EnergyMeter:
    """
    Classe che contiene le informazioni utili per un contatore di energia e i suoi dati.
    """
    def __init__(self, building_id, mode="offline"):
        self.energy_meter_info = {
            "id": None,
            "properties": None,
            "aggregation_functions": None,
            "name": None
        }
        self.energy_meter_data = None
        self.get_energy_meter_info(building_id)
        self.set_data(building_id, mode)

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
            "aggregation_functions": ["sumOfEnergy", "delta", "delta", "delta", "arithmeticMean"],
            "name": energy_meter["name"]
        }

    def get_data(self, time_from, aggregation_period="quarter"):
        """
        Estrae i dati del contatore di energia dal portale di SmartHome e li restituisce in un DataFrame.
        :param time_from: timestamp in formato stringa dell'istante di inizio richiesta dati
        :param aggregation_period: aggregazione temporale del dato ("aurter, hourly", etc..)
        :return: DataFrame con i dati del contatore di energia
        """

        flag = 0
        yesterday = datetime.utcnow() - timedelta(days=1)
        time_to = yesterday.strftime("%Y-%m-%dT23:59:00Z")
        try:
            data = get_data_device(
                device_id=self.energy_meter_info["id"],
                properties=self.energy_meter_info["properties"],
                time_to=time_to,
                time_from=time_from,
                aggregation_function=self.energy_meter_info["aggregation_functions"],
                aggregation_period=aggregation_period
            )
            flag = 1
        except Exception as e:
            print(f"Is not possible to retrieve data for device {self.energy_meter_info['id']}")

        if flag == 1:
            df_list = {}

            for key, value_list in data.items():
                new_key = key.split("_")[0]
                df = pd.DataFrame(value_list, columns=['timestamp', new_key])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df[new_key] = df[new_key].astype(float)
                df_list[new_key] = df

            df_energy_meter = pd.concat([df.set_index('timestamp') for df in df_list.values()], axis=1).sort_values('timestamp')

            timestamp_range = pd.date_range(start=time_from, end=time_to, freq='15min')
            df_energy_meter = df_energy_meter.reindex(timestamp_range)
            df_energy_meter.reset_index(inplace=True, names=['timestamp'])

            return df_energy_meter
        else:
            return None

    def set_data(self, building_id, mode="offline"):
        """
        Salva i dati del contatore di energia in base alla modalità di recupero dei dati.
        :param building_id: Identificatore dell'edificio
        :param mode: online od offline
        """
        if mode == "online":
            self.energy_meter_data = self.get_data(time_from="2024-03-01T00:00:00Z")
        else:
            self.energy_meter_data = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "energy_meter", f"{building_id}.csv"))


def load_anguillara(mode="offline"):

    DU_1 = Building("7436df46-294b-4c97-bd1b-8aaa3aed97c5", get_data_mode=mode)
    DU_2 = Building("80c3bedd-8c41-450c-ae52-1864b9ace7aa", get_data_mode=mode)
    DU_3 = Building("b8296a26-2a08-417b-92d3-41e37f6a956e", get_data_mode=mode)
    DU_4 = Building("d93552c8-e7f6-45bb-b382-bd4a2b969502", get_data_mode=mode)
    DU_5 = Building("b87be67b-8133-4b7f-a045-c06da08b5416", get_data_mode=mode)
    DU_6 = Building("9a3386b3-017c-4848-ac6d-a24bf7f36077", get_data_mode=mode)
    DU_7 = Building("8490da00-eb75-45df-888e-851ea3103ec4", get_data_mode=mode)
    DU_8 = Building("08f2fc03-ce0b-4cd6-ab25-8b3906feb858", get_data_mode=mode)
    DU_9 = Building("3d956901-f5ea-4094-9c85-333cc68183d4", get_data_mode=mode)
    DU_10 = Building("4ef8599c-2c4b-433e-94c8-ca48e23a5a07", get_data_mode=mode)

    return [DU_1, DU_2, DU_3, DU_4, DU_5, DU_6, DU_7, DU_8, DU_9, DU_10]
