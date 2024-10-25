import pandas as pd
import numpy as np


def eui(data: pd.DataFrame, surface: float):
    """
    Calcola l'Energy Use Intensity (EUI) come Energia utilizzata diviso la superficie dell'edificio e il numero di
    giorni disponibili nel dataset
    :param data: il dataset con la colonna "Load" in W e datetime timestamp
    :param surface: la superficie dell'edificio in m^2
    :return: ritorna il valore di EUI
    """

    energy = (data["Load"] * 0.25).sum()
    n_days = data["timestamp"].dt.date.nunique()
    return energy / (surface * n_days)


def normalized_eui(data: pd.DataFrame, surface: float, people: int):
    """
    Calcola l'Energy Use Intensity (EUI) normalizzato come Energia utilizzata diviso la superficie dell'edificio,
    il numero di persone e numero di giorni disponibili nel dataset
    :param data: il dataset con la colonna "Load" in W e datetime timestamp
    :param surface: la superficie dell'edificio in m^2
    :param people: numero di persone nell'edificio
    :return: ritorna il valore di EUI normalizzato
    """

    energy = (data["Load"] * 0.25).sum()
    n_days = data["timestamp"].dt.date.nunique()
    return energy / (surface * n_days * people)


def percentage_anomalies(building_cluster: pd.DataFrame):
    """
    Calcola la percentuale di profili anomali presenti nel dataset in base all'analisi di clustering. Conta il numero di
    istanze "Anomalous" presenti nella colonna "cluster".
    :param building_cluster: il dataset con la colonna cluster
    """

    return building_cluster[building_cluster["cluster"] == "Anomalous"].shape[0] / building_cluster.shape[0] * 100


def off_impact(data_operation: pd.DataFrame):
    """
    Calcola il KPI OFF-Impact come la percentuale di energia consumata durante le ore OFF rispetto all'energia totale.
    :param data_operation: dataframe con "Load" e "operating_type"
    :return: il valore di OFF-Impact
    """

    energy_off = (data_operation[data_operation["operating_type"] == "OFF"]["Load"] * 0.25).sum()
    energy_total = (data_operation["Load"] * 0.25).sum()
    off_impact = energy_off / energy_total * 100

    return off_impact


def on_impact(data_operation: pd.DataFrame):
    """
    Calcola il KPI ON-Impact come la percentuale di energia consumata durante le ore ON rispetto all'energia totale.
    :param data_operation: dataframe con "Load" e "operating_type"
    :return: il valore di ON-Impact
    """

    energy_on = (data_operation[data_operation["operating_type"] == "ON"]["Load"] * 0.25).sum()
    energy_total = (data_operation["Load"] * 0.25).sum()
    on_impact = energy_on / energy_total * 100

    return on_impact


def weekend_impact(data_operation: pd.DataFrame):
    """
    Calcola il KPI Weekend-Impact come la percentuale di energia consumata durante il weekend nelle ore ON
    rispetto all'energia totale.
    :param data_operation: dataframe con "Load", "day_type" e "operating_type"
    :return: il valore di Weekend-Impact
    """

    # Check if at least one weekend day is present in the dataset
    if (data_operation["day_type"] == "WEEKEND").any():
        energy_weekend = (data_operation[(data_operation["day_type"] == "WEEKEND") & (data_operation["operating_type"] == "ON")]["Load"] * 0.25).sum()
        energy_total = (data_operation["Load"] * 0.25).sum()
        weekend_impact = energy_weekend / energy_total * 100
    else:
        weekend_impact = np.nan

    return weekend_impact


def self_consumption(data: pd.DataFrame):
    """
    Calcola il KPI Self-Consumption come la percentuale di energia prodotta e consumata internamente rispetto
    alla produzione totale.
    :param data: dataframe con "Load" e "Production" e datetime timestamp
    :return il valore di Self-Consumption
    """

    self_consumed_energy = (data[["Load", "Production"]].min(axis=1) * 0.25).sum()
    production_energy = (data["Production"] * 0.25).sum()
    return self_consumed_energy / production_energy * 100


def self_sufficiency(data: pd.DataFrame):
    """
    Calcola il KPI Self-Sufficiency come la percentuale di energia soddisfatta dal PV rispetto al consumo totale.
    :param data: dataframe con "Load" e "Production" e datetime timestamp
    :return: il valore di Self-Sufficiency
    """
    self_consumed_energy = (data[["Load", "Production"]].min(axis=1) * 0.25).sum()
    total_energy = (data["Load"] * 0.25).sum()
    return self_consumed_energy / total_energy * 100


def self_sufficiency_potential(data: pd.DataFrame):
    """
    Calcola il KPI Self-Sufficiency Potential, ovvero la quota di energia che poteva essere ancora soddisfatta dal PV
    ma che non è stata consumata.
    :param data: dataframe con "Load" e "Production" e datetime timestamp
    :return: il valore di Self-Sufficiency Potential
    """
    return min(data["Load"].sum(), data["Production"].sum()) / data["Load"].sum() * 100


def additional_self_sufficiency(data: pd.DataFrame):
    """
    Calcola il KPI Additional Self-Sufficiency come la differenza tra il potenziale massimo di Self-Sufficiency e il
    valore attuale di Self-Sufficiency.
    :param data: dataframe con "Load" e "Production" e datetime timestamp
    :return: il valore di Additional Self-Sufficiency
    """

    return self_sufficiency_potential(data) - self_sufficiency(data)


def loss_of_load_probability(data: pd.DataFrame):
    """
    Calcola il KPI Loss of Load Probability come la percentuale di tempo in cui la produzione non copre il carico.
    :param data: dataframe con "Net" e "Production" e datetime timestamp
    :return: il valore di Loss of Load Probability
    """

    data = data[data["Production"] > 0]
    data["f"] = data["Net"].apply(lambda x: 1 if x > 0 else 0)
    return 100 - data["f"].sum() / data.shape[0] * 100


def energy_autonomy(data: pd.DataFrame):
    """
    Calcola il KPI Energy Autonomy come la percentuale di energia soddisfatta internamente rispetto al consumo totale.
    :param data: dataframe con "Net" e "Production" e datetime timestamp
    :return: il valore di Energy Autonomy
    """

    return 100 - loss_of_load_probability(data)


def on_site_generation_ratio(data: pd.DataFrame):
    """
    Calcola il KPI On-Site Generation Ratio come la percentuale di energia prodotta internamente rispetto al consumo totale.
    :param data: dataframe con "Load" e "Production" e datetime timestamp
    :return: il valore di On-Site Generation Ratio
    """

    return (data["Production"] * 0.25).sum() / (data["Load"] * 0.25).sum() * 100


def load_volatility(data: pd.DataFrame):
    """
    Calcola il KPI Load Volatility come la deviazione standard dei valori di carico diviso la media (è un coefficiente
    di variazione).
    :param data: dataframe con "Load" e datetime timestamp
    :return: il valore di Load Volatility
    """

    return data["Load"].std() / data["Load"].mean() * 100


def load_factor(data: pd.DataFrame):
    """
    Calcola il KPI Load Factor come il rapporto tra il carico medio e il carico massimo.
    :param data: dataframe con "Load" e datetime timestamp
    :return: il valore di Load Factor
    """

    return data["Load"].mean() / data["Load"].max()


def flexibility_factor(data: pd.DataFrame):
    """
    Calcola il KPI Flexibility Factor come il rapporto tra l'energia consumata tra le 6 e le 9 e tra le 17 e le 20 e
    l'energia totale consumata.
    :param data: dataframe con "Load" e datetime timestamp
    :return: il valore di Flexibility Factor
    """

    on_peak = data[(data["timestamp"].dt.hour >= 6) & (data["timestamp"].dt.hour <= 9) | (data["timestamp"].dt.hour >= 17) & (data["timestamp"].dt.hour <= 20)]["Load"].sum() * 0.25
    return on_peak / (data["Load"].sum() * 0.25)
