import pandas as pd


def get_operating_hours(data: pd.DataFrame, building_cluster: pd.DataFrame):
    """
    Tagga i record del dataset come ON/OFF oppure WEEKEND, in base al tipo di funzionamento corrispondente.
    I tipi di funzionamento sono definiti come ON se la potenza supera la soglia del baseload, definito come il 25esimo
    percentile della distribuzione di potenza di un determinato cluster, e OFF se la potenza è inferiore alla soglia del
    baseload. I giorni di weekend sono identificati in base al giorno della settimana e sono taggati completamente come
    WEEKEND, mentre i giorni della settimana come WEEKDAY.
    :param data: dataframe con colonna "Load" e timestamp index
    :building_cluster: dataframe con colonna "cluster" e "building_name"
    :return: dataset taggato con i valori ON/OFF nella colonna "operating_type e WEEKDAY/WEEKEND nella colonna "day_type"
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


if __name__ == "__main__":
    from src.building import load_anguillara
    cluster = pd.read_csv("../../results/cluster_anguillara_assigned.csv")

    anguillara = load_anguillara()

    for building in anguillara:
        building_cluster = cluster[cluster["building_name"] == building.building_info["name"]]
        print(get_operating_hours(building.energy_meter.data, building_cluster))
