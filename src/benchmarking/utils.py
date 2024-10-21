import pandas as pd
from src.benchmarking import shape_factor


def calculate_shape_factor(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola i fattori di forma giornalieri per una serie temporale
    :param data: dataframe con la colonna "Load", contenente i dati di potenza elettrica consumata, e timestamp index
    :return: dataframe con i fattori di forma calcolati per ogni giorno disponibile
    """

    mean_max = shape_factor.sf_mean_max(data)
    min_max = shape_factor.sf_min_max(data)
    min_mean = shape_factor.sf_min_mean(data)
    daytime_mean = shape_factor.sf_daytime_mean(data)
    daytime_max = shape_factor.sf_daytime_max(data)
    daytime_min_mean = shape_factor.sf_daytime_min_mean(data)
    nightime_mean = shape_factor.sf_nightime_mean(data)
    nightime_max = shape_factor.sf_nightime_max(data)
    nightime_min_mean = shape_factor.sf_nightime_min_mean(data)
    afternoon_mean = shape_factor.sf_afternoon_mean(data)
    afternoon_max = shape_factor.sf_afternoon_max(data)
    afternoon_min_mean = shape_factor.sf_afternoon_min_mean(data)
    peak_load = shape_factor.peakload(data)
    peak_period = shape_factor.peak_period(data, peak_load)

    df_sf = pd.concat([mean_max, min_max, min_mean, daytime_mean, daytime_max, daytime_min_mean, nightime_mean,
                       nightime_max, nightime_min_mean, afternoon_mean, afternoon_max, afternoon_min_mean,
                       peak_period], axis=1)
    df_sf.columns = ["mean_max", "min_max", "min_mean", "daytime_mean", "daytime_max", "daytime_min_mean",
                     "nightime_mean", "nightime_max", "nightime_min_mean", "afternoon_mean", "afternoon_max",
                     "afternoon_min_mean", "peak_night", "pick_morning", "peak_mid-day", "peak_evening"]
    df_sf.dropna(inplace=True)

    return df_sf
