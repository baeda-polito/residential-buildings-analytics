import pandas as pd
import json
from src.building import Building, load_anguillara, load_garda
from src.utils.operating_hours import get_operating_hours
from src.kpi.kpi_function import (eui, normalized_eui, percentage_anomalies, off_impact, on_impact, weekend_impact,
                                  self_consumption, self_sufficiency, self_sufficiency_potential,
                                  additional_self_sufficiency, loss_of_load_probability, energy_autonomy,
                                  on_site_generation_ratio, load_volatility, load_factor, flexibility_factor)


def calculate_kpi_load(user_id: str, aggregate: str):
    """
    Esegue il calcolo dei KPI sul carico per tutti gli edifici dell'aggregato.
    :param user_id: id dell'utente
    :param aggregate: il nome dell'aggregato ("anguillara" o "garda")
    :return:
    """

    building = Building(user_id, aggregate)

    cluster = pd.read_csv(f"../../results/cluster_{aggregate}_assigned.csv")

    kpi_eui = eui(building.energy_meter.data, building.building_info["surface"])
    kpi_eui_normalized = normalized_eui(building.energy_meter.data, building.building_info["surface"], building.building_info["persons"])

    building_cluster = cluster[cluster["building_name"] == building.building_info["name"]]
    kpi_percentage_anomalies = percentage_anomalies(building_cluster)

    data_operation = get_operating_hours(building.energy_meter.data, building_cluster)

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


def calculate_kpi_renewable(user_id: str, aggregate: str):
    """
    Esegue il calcolo dei KPI sull'utilizzo delle fonti rinnovabili per tutti gli edifici dell'aggregato.
    :param user_id: id dell'utente
    :param aggregate: il nome dell'aggregato ("anguillara" o "garda")
    :return:
    """

    building = Building(user_id, aggregate)

    data_grouped = building.energy_meter.data.groupby(building.energy_meter.data["timestamp"].dt.date)

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


def calculate_kpi_flexibility(user_id: str, aggregate: str):
    """
    Esegue il calcolo dei KPI sulla flessibilità per tutti gli edifici dell'aggregato.
    :param user_id: id dell'utente
    :param aggregate: il nome dell'aggregato ("anguillara" o "garda")
    :return:
    """

    building = Building(user_id, aggregate)

    kpi_load_volatility = load_volatility(building.energy_meter.data)

    data_grouped = building.energy_meter.data.groupby(building.energy_meter.data["timestamp"].dt.date)

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


def calculate_performance_score_user(user_id: str, aggregate: str):
    """
    Calcola il punteggio di performance di un utente per tutti i KPI.
    :param user_id: id dell'utente
    :param aggregate: il nome dell'aggregato ("anguillara" o "garda")
    :return: il dataframe con i KPI e i relativi punteggi
    """

    building = Building(user_id, aggregate, pre_process=False)

    kpi_load = calculate_kpi_load(user_id, aggregate)["aggregated_kpi"]
    kpi_flexibility = calculate_kpi_flexibility(user_id, aggregate)["aggregated_kpi"]

    if building.building_info["user_type"] != "consumer":
        kpi_renewable = calculate_kpi_renewable(user_id, aggregate)["aggregated_kpi"]
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

    with open("config.json", "r") as f:
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


def calculate_performance_score_aggregate(aggregate: str):
    """
    Calcola il punteggio di performance per tutti gli utenti dell'aggregato.
    :param aggregate: il nome dell'aggregato ("anguillara" o "garda")
    :return: il dataframe con i KPI e i relativi punteggi
    """

    building_list = []
    if aggregate == "anguillara":
        building_list = load_anguillara()
    elif aggregate == "garda":
        building_list = load_garda()

    score_list = []
    for building in building_list:
        df_score_user = calculate_performance_score_user(building.building_info["id"], aggregate)
        df_score_user["building_name"] = building.building_info["name"]
        score_list.append(df_score_user)

    df_score_aggregate = pd.concat(score_list, axis=0).reset_index(drop=True)
    df_score_aggregate = df_score_aggregate.pivot_table(index="building_name", columns="kpi", values="value")

    return df_score_aggregate