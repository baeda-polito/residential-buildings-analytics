import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import os
from settings import PROJECT_ROOT


def pre_process(data: pd.DataFrame, user_type: str):
    """
    Pre-processa i dati di "power" e "productionPower". La potenza è pre-processata in due fasi: prima vengono interpolati
    i valore mancanti con un massimo di 4 valori consecutivi, poi viene utilizzato un algoritmo KNN per interpolare i
    valori mancanti per i giorni in cui sono presenti più di 4 valori mancanti consecutivi ma meno di 16 (4 ore). Se vi sono
    più di 16 valori mancanti consecutivi, si lascia NaN.
    Per la productionPower, viene utilizzato un modello di regressione lineare tra Irradianza globale orizzontale e
    potenza di produzione per interpolare i valori mancanti.
    :param data: dataframe del dispositivo utente
    :param user_type: tipo di utente (consumer o prosumer)
    :return:
    """

    metrics = {}
    data.set_index("timestamp", inplace=True)
    data.index = pd.to_datetime(data.index)
    data_pre_processed = pd.DataFrame(index=data.index, columns=["power"])
    data_pre_processed.index = pd.to_datetime(data_pre_processed.index)

    data_pre_processed["power"] = data["power"]

    data_pre_processed['missing_encoded'] = data_pre_processed['power'].isnull().astype(int)
    data_pre_processed['consecutive_missing'] = data_pre_processed['missing_encoded'].groupby(
        (data_pre_processed['missing_encoded'] != data_pre_processed['missing_encoded'].shift()).cumsum()).cumcount() + 1

    start_index = None
    consecutive_missing = 0
    for index, row in data_pre_processed.iterrows():
        if row['missing_encoded'] == 1:
            consecutive_missing = row['consecutive_missing']
            if start_index is None:
                start_index = index
        elif start_index is not None:
            if consecutive_missing <= 4:
                adjusted_index_loc = max(data_pre_processed.index.get_loc(start_index) - 1, 0)
                adjusted_index = data_pre_processed.index[adjusted_index_loc]
                data_pre_processed.loc[adjusted_index:index, "power"] = data_pre_processed.loc[adjusted_index:index, "power"].interpolate(limit=4)
            start_index = None

    data_pre_processed['missing_encoded'] = data_pre_processed['power'].isnull().astype(int)
    data_pre_processed['consecutive_missing'] = data_pre_processed['missing_encoded'].groupby(
        (data_pre_processed['missing_encoded'] != data_pre_processed[
            'missing_encoded'].shift()).cumsum()).cumcount() + 1
    data_pre_processed.loc[data_pre_processed['missing_encoded'] == 0, "consecutive_missing"] = 0

    daily_data = data_pre_processed["consecutive_missing"].resample("D").max()
    daily_data = daily_data[daily_data <= 16]
    knn_data = data_pre_processed[np.isin(data_pre_processed.index.date, daily_data.index.date)]
    knn_data = knn_data[["power"]]
    knn_imputer = KNNImputer(n_neighbors=5)
    knn_data = knn_data.pivot_table(index=knn_data.index.date, columns=knn_data.index.strftime("%H:%M"), values="power")
    knn_data_processed = knn_imputer.fit_transform(knn_data)
    knn_data_processed = pd.DataFrame(knn_data_processed, index=knn_data.index, columns=knn_data.columns)
    knn_data_processed.reset_index(inplace=True)
    knn_data_processed_long = pd.melt(knn_data_processed, id_vars=['index'], value_vars=knn_data_processed.columns,
                                      var_name='hour', value_name='power')
    knn_data_processed_long['index'] = knn_data_processed_long['index'].astype(str)
    knn_data_processed_long['timestamp'] = pd.to_datetime(
        knn_data_processed_long['index'] + ' ' + knn_data_processed_long['hour'])
    knn_data_processed_long.drop(columns=['index', 'hour'], inplace=True)
    knn_data_processed_long.set_index('timestamp', inplace=True)
    knn_data_processed_long.index = pd.to_datetime(knn_data_processed_long.index, utc=True)
    knn_data_processed_long.sort_index(inplace=True)

    power_data = data_pre_processed[["power"]]
    power_data.loc[knn_data_processed_long.index, "power"] = knn_data_processed_long["power"]

    if user_type == "consumer":
        data_final = power_data
        data_final["impEnergy"] = data["impEnergy"]
        data_final["expEnergy"] = data["expEnergy"]
        data_final["productionPower"] = data["productionPower"]
        data_final["productionEnergy"] = data["productionEnergy"]
    else:
        weather_data = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "weather", "anguillara.csv"))
        weather_data["timestamp"] = pd.to_datetime(weather_data["timestamp"])
        data_model = pd.merge(data["productionPower"], weather_data, left_index=True, right_on="timestamp", how="right")
        data_model.set_index("timestamp", inplace=True)

        missing_values = data_model[data_model["productionPower"].isnull()].index
        data_to_reconstruct = data_model.loc[missing_values]

        data_model.dropna(subset=["productionPower"], inplace=True)

        model = LinearRegression()
        model.fit(data_model[["ghi"]], data_model["productionPower"])

        y_true = data_model["productionPower"]
        y_pred = model.predict(data_model[["ghi"]])
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        metrics = {"R2": r2, "MAE": mae}

        data_to_reconstruct["productionPower"] = model.predict(data_to_reconstruct[["ghi"]])

        pv_data = pd.concat([data_model, data_to_reconstruct])[['productionPower']]
        pv_data.sort_index(inplace=True)
        data_final = pd.merge(power_data, pv_data, left_index=True, right_index=True, how="left")
        data_final.iloc[0, 1] = 0

        data_final["impEnergy"] = data["impEnergy"]
        data_final["expEnergy"] = data["expEnergy"]
        data_final["productionEnergy"] = data["productionEnergy"]

    data_final.reset_index(inplace=True, names=['timestamp'])

    return data_final, metrics
