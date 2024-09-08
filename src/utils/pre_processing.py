import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


def replace_constant_values(series, n):
    """
    Individua i valori costanti in una serie temporale e li sostituisce con NaN.
    :param series: la serie temporale
    :param n: numero massimo di valori costanti consecutivi ammesso
    :return: la serie storia ripulita dai valori costanti
    """
    # Identify where the value changes by using shift()
    change_points = series != series.shift()

    # Calculate the cumulative sum of change points to identify groups of consecutive values
    group_ids = change_points.cumsum()

    # Count the size of each group
    group_sizes = group_ids.map(group_ids.value_counts())

    # Replace values in groups where the size is greater than or equal to n
    series[group_sizes >= n] = np.nan

    return series


def reconstruct_missing_values_interp(data: pd.DataFrame, max_missing=4):

    df = data.copy()
    # Ottenimento dei valori mancanti consecutivi
    df["missing_encoded"] = df["power"].isnull().astype(int)
    df["consecutive_missing"] = df["missing_encoded"].groupby(
        (df["missing_encoded"] != df["missing_encoded"].shift()).cumsum()).cumcount() + 1

    # Interpolazione dei valori mancanti consecutivi fino a 4
    start_index = None
    consecutive_missing = 0
    for index, row in df.iterrows():
        if row['missing_encoded'] == 1:
            consecutive_missing = row['consecutive_missing']
            if start_index is None:
                start_index = index
        else:
            if start_index is not None:
                if consecutive_missing <= max_missing:
                    adjusted_index_loc = max(df.index.get_loc(start_index) - 1, 0)
                    adjusted_index = df.index[adjusted_index_loc]
                    df.loc[adjusted_index:index, "power"] = df.loc[adjusted_index:index, "power"].interpolate(limit=4)
                start_index = None

    return df


def reconstruct_missing_values_knn(df, k=5, min_missing=5, max_missing=16):
    """
    Ricostituisce i valori mancanti in un profilo di carico giornaliero utilizzando l'algoritmo KNN. In particolare,
    identifica i giorni con un numero di valori mancanti compreso tra min_missing e max_missing e li imputa con KNN.
    :param df: dataframe con i valori mancanti
    :param k: numero di vicini da considerare in KNN
    :param min_missing: numero minimo di valori mancanti consecutivi
    :param max_missing: numero massimo di valori mancanti consecutivi
    :return: dataframe ricostruito con i valori mancanti imputati
    """
    # Convert timestamp to datetime if it's not already
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Extract the date from the timestamp for grouping by day
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.strftime('%H:%M')

    # Pivot the dataframe to create a daily load profile
    daily_profiles = df.pivot_table(index='date', columns='hour', values='power', dropna=False)

    # Identify rows (days) that have between min_missing and max_missing NaNs
    missing_counts = daily_profiles.isnull().sum(axis=1)
    valid_missing_days = missing_counts[(missing_counts >= min_missing) & (missing_counts <= max_missing)].index

    # Select only the days that need imputation
    incomplete_profiles = daily_profiles.loc[valid_missing_days, :]

    # Use KNNImputer to fill in missing values
    imputer = KNNImputer(n_neighbors=k)
    imputed_profiles = pd.DataFrame(imputer.fit_transform(incomplete_profiles),
                                    index=incomplete_profiles.index,
                                    columns=incomplete_profiles.columns)

    # Replace the original missing values with the imputed ones
    daily_profiles.update(imputed_profiles)

    # Reshape the dataframe back to the original format forming the column timestamp as index + column name
    df_reconstructed = daily_profiles.stack(dropna=False).reset_index()
    df_reconstructed.columns = ['date', 'hour', 'power']
    df_reconstructed['date'] = df_reconstructed['date'].astype(str)
    df_reconstructed['timestamp'] = pd.to_datetime(df_reconstructed['date'] + ' ' + df_reconstructed['hour'] + ':00')
    df_reconstructed = df_reconstructed[['timestamp', 'power']]

    return df_reconstructed
