import pandas as pd


def get_operating_hours(data: pd.DataFrame, building_cluster: pd.DataFrame):
    """
    Tagga i record del dataset come ON, OFF oppure WEEKEND, in base al tipo di funzionamento.

    I tipi di funzionamento sono definiti come:
    - **ON**: La potenza supera la soglia del baseload, calcolata come il 25° percentile della distribuzione di potenza per un determinato cluster.
    - **OFF**: La potenza è inferiore alla soglia del baseload.
    - **WEEKEND**: Identificati in base al giorno della settimana (sabato e domenica) e taggati interamente come WEEKEND.
    - **WEEKDAY**: Giorni lavorativi della settimana.

    Args:
        data (pd.DataFrame): DataFrame contenente la colonna "Load" e un index di tipo timestamp.
        building_cluster (pd.DataFrame): DataFrame contenente le colonne "cluster" e "building_name".

    Returns:
        pd.DataFrame: Dataset taggato con:
            - La colonna `operating_type` contenente i valori ON o OFF.
            - La colonna `day_type` contenente i valori WEEKDAY o WEEKEND.
    """

    building_cluster["date"] = pd.to_datetime(building_cluster["date"]).dt.date
    # Merge building_cluster with data. The data have a timestamp column, while building_cluster has a date column
    data["date"] = data["timestamp"].dt.date
    data = data.merge(building_cluster, on="date", how="left")
    data = data[data["cluster"] != "Anomalous"]

    baseload = data.groupby("cluster")["Load"].quantile(0.25)
    peakload = data.groupby("cluster")["Load"].quantile(0.95)

    dl = 0.25 * (peakload - baseload)

    on_hour_threshold = baseload + dl

    # On "data", create a column called operating_type and set to ON if the "Load" is higher than on_hour_threshold in the same cluster, and OFF is not
    data["on_hour_threshold"] = data["cluster"].map(on_hour_threshold)
    data["operating_type"] = "OFF"
    data.loc[data["Load"] > data["on_hour_threshold"], "operating_type"] = "ON"

    # Handle weekends
    data["weekday"] = data["timestamp"].dt.weekday
    data["day_type"] = "WEEKDAY"
    data.loc[data["weekday"] >= 5, "day_type"] = "WEEKEND"

    return data[["timestamp", "Load", "operating_type", "day_type"]]
