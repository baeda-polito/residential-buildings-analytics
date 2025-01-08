import json
import pandas as pd
from loguru import logger

from ..building import Building
from ..aggregate import Aggregate
from .utils import get_operating_hours
from .kpis import (eui, normalized_eui, percentage_anomalies, off_impact, on_impact, weekend_impact,
                   self_consumption, self_sufficiency, self_sufficiency_potential,
                   additional_self_sufficiency, loss_of_load_probability, energy_autonomy,
                   on_site_generation_ratio, load_volatility, load_factor, flexibility_factor)


def calculate_kpi_load(building: Building, cluster: pd.DataFrame) -> dict:
    """
    Esegue il calcolo dei KPI sul carico per l'edificio.
    I KPI calcolati sono:
    - EUI
    - EUI normalizzato
    - Percentuale di anomalie
    - OFF impact
    - ON impact
    - Weekend impact

    Args:
        building (Building): Oggetto Building contenente i dati relativi all'edificio.
        cluster (pd.DataFrame): DataFrame contenente le colonne "date", "building_name" e "cluster", risultante dall'analisi di clustering (vedi modulo benchmarking).

    Returns:
        dict: Dizionario contenente i KPI calcolati per ogni edificio dell'aggregato. Il dizionario è strutturato in:
            - aggregated_kpi: kpi che ritornano un unico valore per ogni edificio
            - daily_kpi: kpi che ritornano un valore per ogni giorno dell'anno
    """

    if "surface" in building.building_info:
        kpi_eui = eui(building.energy_data.data, building.building_info["surface"])
    else:
        kpi_eui = None
        logger.warning(
            "Superficie dell'edificio non specificata nei metadati. Impossibile calcolare EUI e EUI normalizzato")

    if "persons" in building.building_info and "surface" in building.building_info:
        kpi_eui_normalized = normalized_eui(building.energy_data.data, building.building_info["surface"],
                                            building.building_info["persons"])
    else:
        kpi_eui_normalized = None
        logger.warning(
            "Numero di persone non specificato nei metadati. Impossibile calcolare EUI normalizzato")

    building_cluster = cluster[cluster["building_name"] == building.building_info["name"]]
    kpi_percentage_anomalies = percentage_anomalies(building_cluster)

    data_operation = get_operating_hours(building.energy_data.data, building_cluster)

    data_grouped = data_operation.groupby(data_operation["timestamp"].dt.date)
    daily_off_impact = []
    daily_on_impact = []
    daily_weekend_impact = []
    for date, data_daily in data_grouped:
        daily_off_impact.append((date, off_impact(data_daily)))
        daily_on_impact.append((date, on_impact(data_daily)))
        daily_weekend_impact.append((date, weekend_impact(data_daily)))

    # Create a dataframe with the daily KPIs
    daily_kpi = pd.DataFrame(daily_off_impact, columns=["date", "off_impact"])
    daily_kpi["on_impact"] = pd.DataFrame(daily_on_impact)[1]
    daily_kpi["weekend_impact"] = pd.DataFrame(daily_weekend_impact)[1]

    return {
        "aggregated_kpi": {
            "eui": kpi_eui,
            "eui_normalized": kpi_eui_normalized,
            "percentage_anomalies": kpi_percentage_anomalies,
            "on_impact": daily_kpi["on_impact"].mean(),
            "off_impact": daily_kpi["off_impact"].mean(),
            "weekend_impact": daily_kpi["weekend_impact"].mean()
        },
        "daily_kpi": daily_kpi
    }


def calculate_kpi_renewable(building: Building) -> dict:
    """
    Esegue il calcolo dei KPI relativi alla produzione rinnovabile per l'edificio
    I KPI calcolati sono:
    - Self consumption
    - Self sufficiency
    - Self sufficiency potential
    - Additional self sufficiency
    - Loss of load probability
    - Energy autonomy
    - On site generation ratio

    Args:
        building (Building): Oggetto Building contenente i dati relativi all'edificio.

    Returns:
        dict: Dizionario contenente i KPI calcolati per ogni edificio dell'aggregato. Il dizionario è strutturato in:
            - aggregated_kpi: kpi che ritornano un unico valore per ogni edificio
            - daily_kpi: kpi che ritornano un valore per ogni giorno dell'anno
    """

    data_grouped = building.energy_data.data.groupby(building.energy_data.data["timestamp"].dt.date)

    daily_self_consumption = []
    daily_self_sufficiency = []
    daily_self_sufficiency_potential = []
    daily_additional_self_sufficiency = []
    daily_loss_of_load_probability = []
    daily_energy_autonomy = []
    daily_on_site_generation_ratio = []

    for date, data_daily in data_grouped:
        daily_self_consumption.append((date, self_consumption(data_daily)))
        daily_self_sufficiency.append((date, self_sufficiency(data_daily)))
        daily_self_sufficiency_potential.append((date, self_sufficiency_potential(data_daily)))
        daily_additional_self_sufficiency.append((date, additional_self_sufficiency(data_daily)))
        daily_loss_of_load_probability.append((date, loss_of_load_probability(data_daily)))
        daily_energy_autonomy.append((date, energy_autonomy(data_daily)))
        daily_on_site_generation_ratio.append((date, on_site_generation_ratio(data_daily)))

    # Create a dataframe with the daily KPIs
    daily_kpi = pd.DataFrame(daily_self_consumption, columns=["date", "self_consumption"])
    daily_kpi["self_sufficiency"] = pd.DataFrame(daily_self_sufficiency)[1]
    daily_kpi["self_sufficiency_potential"] = pd.DataFrame(daily_self_sufficiency_potential)[1]
    daily_kpi["additional_self_sufficiency"] = pd.DataFrame(daily_additional_self_sufficiency)[1]
    daily_kpi["loss_of_load_probability"] = pd.DataFrame(daily_loss_of_load_probability)[1]
    daily_kpi["energy_autonomy"] = pd.DataFrame(daily_energy_autonomy)[1]
    daily_kpi["on_site_generation_ratio"] = pd.DataFrame(daily_on_site_generation_ratio)[1]

    return {
        "aggregated_kpi": {
            "self_consumption": daily_kpi["self_consumption"].mean(),
            "self_sufficiency": daily_kpi["self_sufficiency"].mean(),
            "self_sufficiency_potential": daily_kpi["self_sufficiency_potential"].mean(),
            "additional_self_sufficiency": daily_kpi["additional_self_sufficiency"].mean(),
            "loss_of_load_probability": daily_kpi["loss_of_load_probability"].mean(),
            "energy_autonomy": daily_kpi["energy_autonomy"].mean(),
            "on_site_generation_ratio": daily_kpi["on_site_generation_ratio"].mean()
        },
        "daily_kpi": daily_kpi
    }


def calculate_kpi_flexibility(building: Building) -> dict:
    """
    Esegue il calcolo dei KPI relativi alla flessibilità del carico per l'edificio
    I KPI calcolati sono:
    - Load volatility
    - Load factor
    - Flexibility factor

    Args:
        building (Building): Oggetto Building contenente i dati relativi all'edificio.

    Returns:
        dict: Dizionario contenente i KPI calcolati per ogni edificio dell'aggregato. Il dizionario è strutturato in:
            - aggregated_kpi: kpi che ritornano un unico valore per ogni edificio
            - daily_kpi: kpi che ritornano un valore per ogni giorno dell'anno
    """

    kpi_load_volatility = load_volatility(building.energy_data.data)

    data_grouped = building.energy_data.data.groupby(building.energy_data.data["timestamp"].dt.date)

    daily_load_factor = []
    daily_flexibility_factor = []
    daily_load_volatility = []

    for date, data_daily in data_grouped:
        daily_load_factor.append((date, load_factor(data_daily)))
        daily_flexibility_factor.append((date, flexibility_factor(data_daily)))
        daily_load_volatility.append((date, load_volatility(data_daily)))

    daily_kpi = pd.DataFrame(daily_load_factor, columns=["date", "load_factor"])
    daily_kpi["flexibility_factor"] = pd.DataFrame(daily_flexibility_factor)[1]
    daily_kpi["load_volatility"] = pd.DataFrame(daily_load_volatility)[1]

    return {
        "aggregated_kpi": {
            "load_volatility": kpi_load_volatility,
            "load_factor": daily_kpi["load_factor"].mean(),
            "flexibility_factor": daily_kpi["flexibility_factor"].mean()
        },
        "daily_kpi": daily_kpi
    }


def calculate_kpi_aggregate(aggregate: Aggregate, cluster: pd.DataFrame) -> tuple:
    """
    Esegue il calcolo dei KPI per l'aggregato di edifici.

    Args:
        aggregate (Aggregate): Oggetto Aggregate contenente i dati relativi all'aggregato di edifici.
        cluster (pd.DataFrame): DataFrame contenente le colonne "date", "building_name" e "cluster", risultante dall'analisi di clustering (vedi modulo benchmarking).

    Returns:
        tuple: Tuple contenente i DataFrame con i KPI aggregati per l'aggregato di edifici. I DataFrame sono strutturati in:
            - df_kpi_load_aggregate: KPI relativi al carico
            - df_kpi_flexibility_aggregate: KPI relativi alla flessibilità
            - df_kpi_renewable_aggregate: KPI relativi alla produzione rinnovabile
    """


    kpi_load_list = []
    kpi_flexibility_list = []
    kpi_renewable_list = []
    for building in aggregate.buildings:
        kpi_load = calculate_kpi_load(building, cluster)["daily_kpi"]
        kpi_load["building_name"] = building.building_info["name"]
        kpi_load_list.append(kpi_load)
        kpi_flexibility = calculate_kpi_flexibility(building)["daily_kpi"]
        kpi_flexibility["building_name"] = building.building_info["name"]
        kpi_flexibility_list.append(kpi_flexibility)
        if building.building_info["user_type"] != "consumer":
            kpi_renewable = calculate_kpi_renewable(building)["daily_kpi"]
            kpi_renewable["building_name"] = building.building_info["name"]
            kpi_renewable_list.append(kpi_renewable)

    df_kpi_load_aggregate = pd.concat(kpi_load_list, axis=0).reset_index(drop=True)
    df_kpi_flexibility_aggregate = pd.concat(kpi_flexibility_list, axis=0).reset_index(drop=True)
    df_kpi_renewable_aggregate = pd.concat(kpi_renewable_list, axis=0).reset_index(drop=True)

    return df_kpi_load_aggregate, df_kpi_flexibility_aggregate, df_kpi_renewable_aggregate


def calculate_performance_score_user(building: Building, cluster: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola il punteggio di performance per l'utente.
    Il punteggio di performance è una trasformazione 0-100 del valore del KPI (0 rappresenta il peggior valore possibile, 100 il migliore).

    Args:
        building (Building): Oggetto Building contenente i dati relativi all'edificio.
        cluster (pd.DataFrame): DataFrame contenente le colonne "date", "building_name" e "cluster", risultante dall'analisi di clustering (vedi modulo benchmarking).

    Returns:
        pd.DataFrame: DataFrame contenente i punteggi di performance per l'utente.
    """

    kpi_load = calculate_kpi_load(building, cluster)["aggregated_kpi"]
    kpi_flexibility = calculate_kpi_flexibility(building)["aggregated_kpi"]

    if building.building_info["user_type"] != "consumer":
        kpi_renewable = calculate_kpi_renewable(building)["aggregated_kpi"]
    else:
        kpi_renewable = {
            "self_consumption": 0,
            "self_sufficiency": 0,
            "self_sufficiency_potential": 0,
            "additional_self_sufficiency": 100,
            "loss_of_load_probability": 100,
            "energy_autonomy": 0,
            "on_site_generation_ratio": 0
        }

    with open("kpi/config.json", "r") as f:
        config = json.load(f)

    # Create a score_dict where each key, that has the name of the KPI, has a dict with "name" and "value"
    score_dict = {}
    score_dict["percentage_anomalies"] = {
        "name": config["percentage_anomalies"]["name"],
        "value": 100 - kpi_load["percentage_anomalies"]
    }
    score_dict["off_impact"] = {
        "name": config["off_impact"]["name"],
        "value": kpi_load["off_impact"]
    }
    score_dict["on_impact"] = {
        "name": config["on_impact"]["name"],
        "value": kpi_load["on_impact"]
    }
    score_dict["weekend_impact"] = {
        "name": config["weekend_impact"]["name"],
        "value": kpi_load["weekend_impact"]
    }
    score_dict["self_consumption"] = {
        "name": config["self_consumption"]["name"],
        "value": kpi_renewable["self_consumption"]
    }
    score_dict["self_sufficiency"] = {
        "name": config["self_sufficiency"]["name"],
        "value": kpi_renewable["self_sufficiency"]
    }
    score_dict["additional_self_sufficiency"] = {
        "name": config["additional_self_sufficiency"]["name"],
        "value": 100 - kpi_renewable["additional_self_sufficiency"]
    }
    score_dict["loss_of_load_probability"] = {
        "name": config["loss_of_load_probability"]["name"],
        "value": 100 - kpi_renewable["loss_of_load_probability"]
    }
    score_dict["energy_autonomy"] = {
        "name": config["energy_autonomy"]["name"],
        "value": kpi_renewable["energy_autonomy"]
    }
    score_dict["load_volatility"] = {
        "name": config["load_volatility"]["name"],
        "value": kpi_flexibility["load_volatility"]
    }
    score_dict["load_factor"] = {
        "name": config["load_factor"]["name"],
        "value": (1 - kpi_flexibility["load_factor"]) * 100
    }
    score_dict["flexibility_factor"] = {
        "name": config["flexibility_factor"]["name"],
        "value": kpi_flexibility["flexibility_factor"] * 100
    }

    score_df = pd.DataFrame(score_dict).transpose().reset_index(names="kpi")

    return score_df


def calculate_performance_score_aggregate(aggregate: Aggregate, cluster: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola il punteggio di performance per l'aggregato di edifici.

    Args:
        aggregate (Aggregate): Oggetto Aggregate contenente i dati relativi all'aggregato di edifici.
        cluster (pd.DataFrame): DataFrame contenente le colonne "date", "building_name" e "cluster", risultante dall'analisi di clustering (vedi modulo benchmarking).

    Returns:
        pd.DataFrame: DataFrame contenente i punteggi di performance per l'aggregato di edifici.
    """

    score_list = []
    for building in aggregate.buildings:
        df_score_user = calculate_performance_score_user(building, cluster)
        df_score_user["building_name"] = building.building_info["name"]
        score_list.append(df_score_user)

    df_score_aggregate = pd.concat(score_list, axis=0).reset_index(drop=True)
    df_score_aggregate = df_score_aggregate.pivot_table(index="building_name", columns="kpi", values="value")

    return df_score_aggregate
