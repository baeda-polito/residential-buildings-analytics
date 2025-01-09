import pandas as pd
import numpy as np


def sf_mean_max(df: pd.DataFrame) -> pd.Series:
    """
    Calcola il rapporto tra valore medio giornaliero e valore massimo giornaliero.

    Args:
        df (pd.DataFrame): La series del singolo utente con timestamp sull'indice.

    Returns:
        pd.Series: Una serie con un valore del rapporto per ogni giorno.
    """
    daily_stats = df.groupby(df.index.date).agg(['mean', 'max'])
    daily_stats['ratio'] = daily_stats['mean'] / daily_stats['max']
    return daily_stats['ratio']


def sf_min_max(df: pd.DataFrame) -> pd.Series:
    """
    Calcola il rapporto tra valore minimo giornaliero e valore massimo giornaliero.

    Args:
        df (pd.DataFrame): La series del singolo utente con timestamp sull'indice.

    Returns:
        pd.Series: Una serie con un valore del rapporto per ogni giorno.
    """
    daily_stats = df.groupby(df.index.date).agg(['min', 'max'])
    daily_stats['ratio'] = daily_stats['min'] / daily_stats['max']
    return daily_stats['ratio']


def sf_min_mean(df: pd.DataFrame) -> pd.Series:
    """
    Calcola il rapporto tra valore minimo giornaliero e valore medio giornaliero.

    Args:
        df (pd.DataFrame): La series del singolo utente con timestamp sull'indice.

    Returns:
        pd.Series: Una serie con un valore del rapporto per ogni giorno.
    """
    daily_stats = df.groupby(df.index.date).agg(['min', 'mean'])
    daily_stats['ratio'] = daily_stats['min'] / daily_stats['mean']
    return daily_stats['ratio']


def sf_daytime_mean(df: pd.DataFrame) -> pd.Series:
    """
    Calcola il rapporto tra il valore medio di potenza durante le ore giornaliere rispetto al medio totale.

    Args:
        df (pd.DataFrame): La series del singolo utente con timestamp sull'indice.

    Returns:
        pd.Series: Una serie con un valore del rapporto per ogni giorno.
    """
    filtered_df = df.between_time('9:00', '22:00')
    daily_stats = df.groupby(df.index.date).mean()
    hourly_mean = filtered_df.groupby(filtered_df.index.date).mean()
    ratio = 7 / 12 * hourly_mean / daily_stats
    return ratio


def sf_daytime_max(df: pd.DataFrame) -> pd.Series:
    """
    Calcola il rapporto tra il valore massimo di potenza durante le ore giornaliere rispetto al massimo totale.

    Args:
        df (pd.DataFrame): La series del singolo utente con timestamp sull'indice.

    Returns:
        pd.Series: Una serie con un valore del rapporto per ogni giorno.
    """
    filtered_df = df.between_time('9:00', '22:00')
    daily_stats = df.groupby(df.index.date).max()
    hourly_max = filtered_df.groupby(filtered_df.index.date).max()
    ratio = 7 / 12 * hourly_max / daily_stats
    return ratio


def sf_daytime_min_mean(df: pd.DataFrame) -> pd.Series:
    """
    Calcola il rapporto tra il valore minimo di potenza durante le ore giornaliere rispetto al medio totale.

    Args:
        df (pd.DataFrame): La series del singolo utente con timestamp sull'indice.

    Returns:
        pd.Series: Una serie con un valore del rapporto per ogni giorno.
    """
    filtered_df = df.between_time('9:00', '22:00')
    daily_stats = df.groupby(df.index.date).mean()
    hourly_min = filtered_df.groupby(filtered_df.index.date).min()
    ratio = 7 / 12 * hourly_min / daily_stats
    return ratio


def sf_nightime_mean(df: pd.DataFrame) -> pd.Series:
    """
    Calcola il rapporto tra il valore medio di potenza durante le ore notturne rispetto al medio totale.

    Args:
        df (pd.DataFrame): La series del singolo utente con timestamp sull'indice.

    Returns:
        pd.Series: Una serie con un valore del rapporto per ogni giorno.
    """
    filtered_df = df[~df.index.hour.isin(range(9, 23))]
    daily_stats = df.groupby(df.index.date).mean()
    hourly_mean = filtered_df.groupby(filtered_df.index.date).mean()
    ratio = 5 / 12 * hourly_mean / daily_stats
    return ratio


def sf_nightime_max(df: pd.DataFrame) -> pd.Series:
    """
    Calcola il rapporto tra il valore massimo di potenza durante le ore notturne rispetto al massimo totale.

    Args:
        df (pd.DataFrame): La series del singolo utente con timestamp sull'indice.

    Returns:
        pd.Series: Una serie con un valore del rapporto per ogni giorno.
    """
    filtered_df = df[~df.index.hour.isin(range(9, 23))]
    daily_stats = df.groupby(df.index.date).max()
    hourly_max = filtered_df.groupby(filtered_df.index.date).max()
    ratio = 5 / 12 * hourly_max / daily_stats
    return ratio


def sf_nightime_min_mean(df: pd.DataFrame) -> pd.Series:
    """
    Calcola il rapporto tra il valore minimo di potenza durante le ore notturne rispetto al medio totale.

    Args:
        df (pd.DataFrame): La series del singolo utente con timestamp sull'indice.

    Returns:
        pd.Series: Una serie con un valore del rapporto per ogni giorno.
    """
    filtered_df = df[~df.index.hour.isin(range(9, 23))]
    daily_stats = df.groupby(df.index.date).mean()
    hourly_mean = filtered_df.groupby(filtered_df.index.date).min()
    ratio = 5 / 12 * hourly_mean / daily_stats
    return ratio


def sf_afternoon_mean(df: pd.DataFrame) -> pd.Series:
    """
    Calcola il rapporto tra il valore medio di potenza durante le ore pomeridiane rispetto al medio totale.

    Args:
        df (pd.DataFrame): La series del singolo utente con timestamp sull'indice.

    Returns:
        pd.Series: Una serie con un valore del rapporto per ogni giorno.
    """
    filtered_df = df.between_time('13:00', '17:00')
    daily_stats = df.groupby(df.index.date).mean()
    hourly_mean = filtered_df.groupby(filtered_df.index.date).mean()
    ratio = 5 / 24 * hourly_mean / daily_stats
    return ratio


def sf_afternoon_max(df: pd.DataFrame) -> pd.Series:
    """
    Calcola il rapporto tra il valore massimo di potenza durante le ore pomeridiane rispetto al massimo totale.

    Args:
        df (pd.DataFrame): La series del singolo utente con timestamp sull'indice.

    Returns:
        pd.Series: Una serie con un valore del rapporto per ogni giorno.
    """
    filtered_df = df.between_time('13:00', '17:00')
    daily_stats = df.groupby(df.index.date).max()
    hourly_max = filtered_df.groupby(filtered_df.index.date).max()
    ratio = 5 / 24 * hourly_max / daily_stats
    return ratio


def sf_afternoon_min_mean(df: pd.DataFrame) -> pd.Series:
    """
    Calcola il rapporto tra il valore minimo di potenza durante le ore pomeridiane rispetto al medio totale.

    Args:
        df (pd.DataFrame): La series del singolo utente con timestamp sull'indice.

    Returns:
        pd.Series: Una serie con un valore del rapporto per ogni giorno.
    """
    filtered_df = df.between_time('13:00', '17:00')
    daily_stats = df.groupby(df.index.date).mean()
    hourly_min = filtered_df.groupby(filtered_df.index.date).min()
    ratio = 5 / 24 * hourly_min / daily_stats
    return ratio


def sf_evening_mean(df: pd.DataFrame) -> pd.Series:
    """
    Calcola il rapporto tra il valore medio di potenza durante le ore serali rispetto al medio totale.

    Args:
        df (pd.DataFrame): La series del singolo utente con timestamp sull'indice.

    Returns:
        pd.Series: Una serie con un valore del rapporto per ogni giorno.
    """
    filtered_df = df.between_time('21:00', '23:00')
    daily_stats = df.groupby(df.index.date).mean()
    hourly_mean = filtered_df.groupby(filtered_df.index.date).mean()
    ratio = 1 / 8 * hourly_mean / daily_stats
    return ratio


def sf_evening_max(df: pd.DataFrame) -> pd.Series:
    """
    Calcola il rapporto tra il valore massimo di potenza durante le ore serali rispetto al massimo totale.

    Args:
        df (pd.DataFrame): La series del singolo utente con timestamp sull'indice.

    Returns:
        pd.Series: Una serie con un valore del rapporto per ogni giorno.
    """
    filtered_df = df.between_time('21:00', '23:00')
    daily_stats = df.groupby(df.index.date).max()
    hourly_max = filtered_df.groupby(filtered_df.index.date).max()
    ratio = 1 / 8 * hourly_max / daily_stats
    return ratio


def sf_evening_min_mean(df: pd.DataFrame) -> pd.Series:
    """
    Calcola il rapporto tra il valore minimo di potenza durante le ore serali rispetto al medio totale.

    Args:
        df (pd.DataFrame): La series del singolo utente con timestamp sull'indice.

    Returns:
        pd.Series: Una serie con un valore del rapporto per ogni giorno.
    """
    filtered_df = df.between_time('21:00', '23:00')
    daily_stats = df.groupby(df.index.date).mean()
    hourly_min = filtered_df.groupby(filtered_df.index.date).min()
    ratio = 1 / 8 * hourly_min / daily_stats
    return ratio


def baseload(df: pd.DataFrame) -> float:
    """
    Calcola il baseload dell'utente come il 15esimo percentile della distribuzione di potenza.

    Args:
        df (pd.DataFrame): La series del singolo utente con timestamp sull'indice.

    Returns:
        float: Il valore di baseload del periodo.
    """
    base_load = df[df > 0].quantile(0.15)
    return base_load


def peakload(df: pd.DataFrame) -> float:
    """
    Calcola il peakload dell'utente come il 90esimo percentile della distribuzione di potenza.

    Args:
        df (pd.DataFrame): La series del singolo utente con timestamp sull'indice.

    Returns:
        float: Il valore di peakload del periodo.
    """
    peak_load = df[df > 0].quantile(0.9)
    return peak_load


def daily_peaks(df: pd.DataFrame, peak_load: float) -> pd.Series:
    """
    Calcola il numero di picchi giornalieri di potenza.

    Args:
        df (pd.DataFrame): La series del singolo utente con timestamp sull'indice.
        peak_load (float): Il valore di peakload del periodo.

    Returns:
        pd.Series: Il numero di picchi giornalieri.
    """
    df_positive = df[df > 0]
    n_peaks = df_positive[df_positive > peak_load]
    return n_peaks.groupby(n_peaks.index.date).count().reindex(np.unique(df.index.date), fill_value=0)


def peak_period(df: pd.DataFrame, peak_load: float) -> pd.DataFrame:
    """
    Calcolo il periodo del picco di potenza tra notte, mattino, mid-day e sera.

    Args:
        df (pd.DataFrame): La series del singolo utente con timestamp sull'indice.
        peak_load (float): Il valore di peakload del periodo.

    Returns:
        pd.DataFrame: Un dataframe con 4 colonne binarie (night, morning, mid-day, evening).
    """
    df_positive = df[df > 0]
    peak_hours = df_positive[df_positive > peak_load].index
    night, morning, mid_day, evening = range(0, 6), range(6, 12), range(12, 18), range(18, 24)
    peak_period = pd.DataFrame(index=np.unique(df.index.date), columns=["night", "morning", "mid-day", "evening"], data=0)
    for date, hour in zip(peak_hours.date, peak_hours.hour):
        if hour in night:
            peak_period.loc[date, "night"] = 1
        elif hour in morning:
            peak_period.loc[date, "morning"] = 1
        elif hour in mid_day:
            peak_period.loc[date, "mid-day"] = 1
        elif hour in evening:
            peak_period.loc[date, "evening"] = 1
    return peak_period
