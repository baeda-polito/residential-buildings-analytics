import requests
import json
import urllib
import urllib3
import os
from dotenv import load_dotenv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv("../../.env")
BEARER_TOKEN = os.getenv("BEARER_TOKEN")


def get_plant_info(plant_id: str) -> dict:
    """
    Funzione che restituisce le informazioni di un determinato edificio.
    Args:
        plant_id (str): uuid dell'edificio
    Returns:
        dict: dizionario con le informazioni dell'edificio
    """

    url = f"https://www.smarthome.enea.it/api/projects/43c4bb12-5a38-4ed8-b704-5be64310012a/dashboard/components/plants/{plant_id}"

    payload = {}
    headers = {
        'Authorization': f"Bearer {BEARER_TOKEN}"
    }

    response = requests.request("GET", url, headers=headers, data=payload, verify=False)
    plant_info = json.loads(response.text)["data"]

    return plant_info


def get_devices(plant_id: str) -> dict:
    """
    Funzione che restituisce tutti i dispositivi per un determinato edificio
    Args:
        plant_id (str): uuid dell'edificio
    Returns:
        dict: dizionario con i dispositivi dell'edificio
    """

    url = "https://www.smarthome.enea.it/api/projects/43c4bb12-5a38-4ed8-b704-5be64310012a/devices"

    querystring = {"plantId": plant_id}

    payload = ""
    headers = {
        "authority": "www.smarthome.enea.it",
        "accept": "application/json, text/plain, */*",
        "accept-language": "it,it-IT;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
        "authorization": f"Bearer {BEARER_TOKEN}",
        "referer": "https://www.smarthome.enea.it/app/management/plants",
        "sec-ch-ua": "^\^Chromium^^;v=^\^112^^, ^\^Microsoft",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "^\^Windows^^",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36 Edg/112.0.1722.58"
    }

    response = requests.request("GET", url, data=payload, headers=headers, params=querystring, verify=False)

    devices = json.loads(response.text)["data"]

    return devices


def get_data_device(device_id: str, properties: list, time_to: str, time_from: str) -> dict:
    """
    Funzione che restituisce un dizionario con i dati delle grandezze definite in properties, per il sensore in input
    Args:
        device_id (str): uuid del sensore
        properties (str): lista delle grandezze da estrarre
        time_to (str): data di fine
        time_from (str): data di inizio
    Returns:
        dict: dizionario con i dati del sensore
    """

    url = "https://www.smarthome.enea.it/api/projects/43c4bb12-5a38-4ed8-b704-5be64310012a/telemetry"

    # Initialize the list to store parameters
    params = [("deviceId", device_id), ("timeTo", time_to), ("timeFrom", time_from)]

    for prop in properties:
        params.append(("name[]", prop))

    querystring = urllib.parse.urlencode(params)

    headers = {
        'authority': 'www.smarthome.enea.it',
        'accept': 'application/json, text/plain, */*',
        'accept-language': 'it,it-IT;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
        'authorization': f'Bearer {BEARER_TOKEN}',
        'cookie': '_ga_9WXGGLFQEH=GS1.1.1684230821.1.1.1684230872.0.0.0; _ga=GA1.1.861981974.1684230821',
        'referer': 'https://www.smarthome.enea.it/app/management/plants',
        'sec-ch-ua': '"Microsoft Edge";v="113", "Chromium";v="113", "Not-A.Brand";v="24"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36 Edg/113.0.1774.42'
    }

    # response = requests.request("GET", url, data=payload, headers=headers, params=querystring, verify=False)
    response = requests.get(f"{url}?{querystring}", headers=headers, verify=False)

    data = json.loads(response.text)["data"][device_id]

    return data
