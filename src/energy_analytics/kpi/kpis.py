import pandas as pd
import numpy as np


def eui(data: pd.DataFrame, surface: float) -> float:
    """
    Calcola l'Energy Use Intensity (EUI).

    L'EUI è calcolato come l'energia utilizzata divisa per la superficie dell'edificio e il numero
    di giorni disponibili nel dataset.

    Args:
        data (pd.DataFrame): Dataset con una colonna "Load" in watt e una colonna "timestamp" in formato datetime.
        surface (float): Superficie dell'edificio in metri quadrati.

    Returns:
        float: Il valore dell'EUI.
    """
    energy = (data["Load"] * 0.25).sum()
    n_days = data["timestamp"].dt.date.nunique()
    return energy / (surface * n_days)


def normalized_eui(data: pd.DataFrame, surface: float, people: int) -> float:
    """
    Calcola l'Energy Use Intensity (EUI) normalizzato.

    L'EUI normalizzato è calcolato come l'energia utilizzata divisa per la superficie dell'edificio,
    il numero di persone e il numero di giorni disponibili nel dataset.

    Args:
        data (pd.DataFrame): Dataset con una colonna "Load" in watt e una colonna "timestamp" in formato datetime.
        surface (float): Superficie dell'edificio in metri quadrati.
        people (int): Numero di persone presenti nell'edificio.

    Returns:
        float: Il valore dell'EUI normalizzato.
    """
    energy = (data["Load"] * 0.25).sum()
    n_days = data["timestamp"].dt.date.nunique()
    return energy / (surface * n_days * people)


def percentage_anomalies(building_cluster: pd.DataFrame) -> float:
    """
    Calcola la percentuale di profili anomali presenti nel dataset.

    La percentuale è calcolata come il numero di istanze "Anomalous" presenti nella colonna "cluster"
    diviso il numero totale di record nel dataset.

    Args:
        building_cluster (pd.DataFrame): Dataset con una colonna "cluster" che include valori "Anomalous".

    Returns:
        float: Percentuale di profili anomali.
    """
    return (building_cluster[building_cluster["cluster"] == "Anomalous"].shape[0] /
            building_cluster.shape[0]) * 100


def off_impact(data_operation: pd.DataFrame) -> float:
    """
    Calcola il KPI OFF-Impact.

    OFF-Impact è la percentuale di energia consumata durante le ore OFF rispetto all'energia totale.

    Args:
        data_operation (pd.DataFrame): Dataset con colonne "Load" e "operating_type".

    Returns:
        float: Il valore di OFF-Impact.
    """
    energy_off = (data_operation[data_operation["operating_type"] == "OFF"]["Load"] * 0.25).sum()
    energy_total = (data_operation["Load"] * 0.25).sum()
    return energy_off / energy_total * 100


def on_impact(data_operation: pd.DataFrame) -> float:
    """
    Calcola il KPI ON-Impact.

    ON-Impact è la percentuale di energia consumata durante le ore ON rispetto all'energia totale.

    Args:
        data_operation (pd.DataFrame): Dataset con colonne "Load" e "operating_type".

    Returns:
        float: Il valore di ON-Impact.
    """
    energy_on = (data_operation[data_operation["operating_type"] == "ON"]["Load"] * 0.25).sum()
    energy_total = (data_operation["Load"] * 0.25).sum()
    return energy_on / energy_total * 100


def weekend_impact(data_operation: pd.DataFrame) -> float:
    """
    Calcola il KPI Weekend-Impact.

    Weekend-Impact è la percentuale di energia consumata durante il weekend nelle ore ON rispetto all'energia totale.

    Args:
        data_operation (pd.DataFrame): Dataset con colonne "Load", "day_type" e "operating_type".

    Returns:
        float: Il valore di Weekend-Impact, oppure NaN se non sono presenti giorni di weekend.
    """
    if (data_operation["day_type"] == "WEEKEND").any():
        energy_weekend = (data_operation[(data_operation["day_type"] == "WEEKEND") &
                                         (data_operation["operating_type"] == "ON")]["Load"] * 0.25).sum()
        energy_total = (data_operation["Load"] * 0.25).sum()
        return energy_weekend / energy_total * 100
    else:
        return np.nan


def self_consumption(data: pd.DataFrame) -> float:
    """
    Calcola il KPI Self-Consumption.

    Self-Consumption è la percentuale di energia prodotta e consumata internamente rispetto alla produzione totale.

    Args:
        data (pd.DataFrame): Dataset con colonne "Load" e "Production".

    Returns:
        float: Il valore di Self-Consumption.
    """
    self_consumed_energy = (data[["Load", "Production"]].min(axis=1) * 0.25).sum()
    production_energy = (data["Production"] * 0.25).sum()
    return self_consumed_energy / production_energy * 100


def self_sufficiency(data: pd.DataFrame) -> float:
    """
    Calcola il KPI Self-Sufficiency.

    Self-Sufficiency è la percentuale di energia soddisfatta dal PV rispetto al consumo totale.

    Args:
        data (pd.DataFrame): Dataset con colonne "Load" e "Production".

    Returns:
        float: Il valore di Self-Sufficiency.
    """
    self_consumed_energy = (data[["Load", "Production"]].min(axis=1) * 0.25).sum()
    total_energy = (data["Load"] * 0.25).sum()
    return self_consumed_energy / total_energy * 100


def self_sufficiency_potential(data: pd.DataFrame) -> float:
    """
    Calcola il KPI Self-Sufficiency Potential.

    Self-Sufficiency Potential rappresenta la quota di energia che poteva essere ancora soddisfatta dal PV ma che
    non è stata consumata.

    Args:
        data (pd.DataFrame): Dataset con colonne "Load" e "Production".

    Returns:
        float: Il valore di Self-Sufficiency Potential.
    """
    return min(data["Load"].sum(), data["Production"].sum()) / data["Load"].sum() * 100


def additional_self_sufficiency(data: pd.DataFrame) -> float:
    """
    Calcola il KPI Additional Self-Sufficiency.

    Additional Self-Sufficiency è la differenza tra il potenziale massimo di Self-Sufficiency e il valore attuale di
    Self-Sufficiency.

    Args:
        data (pd.DataFrame): Dataset con colonne "Load" e "Production".

    Returns:
        float: Il valore di Additional Self-Sufficiency.
    """
    return self_sufficiency_potential(data) - self_sufficiency(data)


def loss_of_load_probability(data: pd.DataFrame) -> float:
    """
    Calcola il KPI Loss of Load Probability.

    Loss of Load Probability è la percentuale di tempo in cui la produzione non copre il carico.

    Args:
        data (pd.DataFrame): Dataset con colonne "Net" e "Production".

    Returns:
        float: Il valore di Loss of Load Probability.
    """
    data = data[data["Production"] > 0]
    data["f"] = data["Net"].apply(lambda x: 1 if x > 0 else 0)
    return 100 - data["f"].sum() / data.shape[0] * 100


def energy_autonomy(data: pd.DataFrame) -> float:
    """
    Calcola il KPI Energy Autonomy.

    Energy Autonomy è la percentuale di energia soddisfatta internamente rispetto al consumo totale.

    Args:
        data (pd.DataFrame): Dataset con colonne "Net" e "Production".

    Returns:
        float: Il valore di Energy Autonomy.
    """
    return 100 - loss_of_load_probability(data)


def on_site_generation_ratio(data: pd.DataFrame) -> float:
    """
    Calcola il KPI On-Site Generation Ratio.

    On-Site Generation Ratio è la percentuale di energia prodotta internamente rispetto al consumo totale.

    Args:
        data (pd.DataFrame): Dataset con colonne "Load" e "Production".

    Returns:
        float: Il valore di On-Site Generation Ratio.
    """
    return (data["Production"] * 0.25).sum() / (data["Load"] * 0.25).sum() * 100


def load_volatility(data: pd.DataFrame) -> float:
    """
    Calcola il KPI Load Volatility.

    Load Volatility è la deviazione standard dei valori di carico divisa per la media (coefficiente di variazione).

    Args:
        data (pd.DataFrame): Dataset con colonna "Load".

    Returns:
        float: Il valore di Load Volatility.
    """
    return data["Load"].std() / data["Load"].mean() * 100


def load_factor(data: pd.DataFrame) -> float:
    """
    Calcola il KPI Load Factor.

    Load Factor è il rapporto tra il carico medio e il carico massimo.

    Args:
        data (pd.DataFrame): Dataset con colonna "Load".

    Returns:
        float: Il valore di Load Factor.
    """
    return data["Load"].mean() / data["Load"].max()


def flexibility_factor(data: pd.DataFrame) -> float:
    """
    Calcola il KPI Flexibility Factor.

    Flexibility Factor è il rapporto tra l'energia consumata tra le 6 e le 9 e tra le 17 e le 20 rispetto all'energia
    totale consumata.

    Args:
        data (pd.DataFrame): Dataset con colonna "Load" e colonna "timestamp" in formato datetime.

    Returns:
        float: Il valore di Flexibility Factor.
    """
    on_peak = data[(data["timestamp"].dt.hour >= 6) & (data["timestamp"].dt.hour <= 9) |
                   (data["timestamp"].dt.hour >= 17) & (data["timestamp"].dt.hour <= 20)]["Load"].sum() * 0.25
    return on_peak / (data["Load"].sum() * 0.25)
