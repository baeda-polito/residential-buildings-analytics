import pandas as pd
from sklearn.impute import KNNImputer
import numpy as np


def pre_process(data: pd.DataFrame, user_type: str):
    """

    :param data:
    :param user_type:
    :return:
    """

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
            # Check if consecutive missing values are less than or equal to the threshold
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
    knn_data_processed_long.index = pd.to_datetime(knn_data_processed_long.index)
    knn_data_processed_long.sort_index(inplace=True)

    power_data = data_pre_processed[["power"]]
    power_data.index = data_pre_processed.index.tz_localize(None)
    power_data.loc[knn_data_processed_long.index, "power"] = knn_data_processed_long["power"]

    if user_type == "consumer":
        data_final = power_data
    else:
        # TODO Implementare modello lineare con irradianza per ricostruire la productionPower
        data_final = data

    data_final.reset_index(inplace=True, names=['timestamp'])
    return data_final


if __name__ == "__main__":
    from src.building import Building
    import plotly.graph_objs as go
    DU_8 = Building("08f2fc03-ce0b-4cd6-ab25-8b3906feb858", get_data_mode="offline")

    data_cleaned = pre_process(DU_8.energy_meter.energy_meter_data.copy(), user_type=DU_8.building_info["user_type"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=data_cleaned["timestamp"], y=data_cleaned["power"], mode='lines', name='Cleaned'))
    fig.add_trace(
        go.Scatter(x=DU_8.energy_meter.energy_meter_data["timestamp"], y=DU_8.energy_meter.energy_meter_data["power"],
                   mode='lines', name='Raw'))
    fig.update_layout(
        yaxis_title="Power [Wh]",
        xaxis_title="Time",
        title="DU 8"
    )
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    fig.show()
