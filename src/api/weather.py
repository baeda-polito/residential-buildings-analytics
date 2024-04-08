import requests
import json


def get_historical_data(location_code: str, start_date: str, end_date: str):
    """
    Estrae i dati storici meteo dal sito www.wunderground.com per massimo un mese, con una granularità di mezz'ora.
    :param location_code: codice della stazione meteo più vicina (Rome: LIRA:9:IT, Pinerolo LIMF:9:IT, Viterbo LIRF:9:IT)
    :param start_date: stringa della data di inizio nel formato %Y%m%d
    :param end_date: stringa della data di fine nel formato %Y%m%d
    :return: lista di dizionari che contengono timestamp, temperatura, punto di rugiada, umidità, pressione e velocità
     del vento
    """

    url = f"https://api.weather.com/v1/location/{location_code}/observations/historical.json"

    querystring = f"apiKey=e1f10a1e78da46f5b10a1e78da96f525&units=h&startDate={start_date}&endDate={end_date}"

    payload = {}

    headers = {
        'sec-ch-ua': '"Microsoft Edge";v="113", "Chromium";v="113", "Not-A.Brand";v="24"',
        'Accept': 'application/json, text/plain, */*',
        'Referer': 'https://www.wunderground.com/',
        'sec-ch-ua-mobile': '?0',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36 Edg/113.0.1774.42',
        'sec-ch-ua-platform': '"Windows"'
    }

    response = requests.request("GET", url, headers=headers, params=querystring, data=payload)

    historical_data = json.loads(response.text)["observations"]

    valid_key = ["valid_time_gmt", "temp", "dewPt", "rh", "pressure", "wspd"]

    result = [{key: entry[key] for key in valid_key} for entry in historical_data]

    return result
