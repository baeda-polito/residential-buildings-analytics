import pandas as pd


def sf_mean_max(df: pd.DataFrame):
    """
    Calcola il rapporto tra valore medio giornaliero e valore massimo giornaliero
    :param df: la series del singolo utente con timestamp sull'indice
    :return: una serie con un valore del rapporto per ogni giorno
    """

    daily_stats = df.groupby(df.index.date).agg(['mean', 'max'])
    # daily_stats.columns = daily_stats.columns.droplevel(level=0)
    daily_stats['ratio'] = daily_stats['mean'] / daily_stats['max']

    return daily_stats['ratio']


def sf_min_max(df: pd.DataFrame):
    """
    Calcola il rapporto tra valore minimo giornaliero e valore massimo giornaliero
    :param df: la series del singolo utente con timestamp sull'indice
    :return: una serie con un valore del rapporto per ogni giorno
    """

    daily_stats = df.groupby(df.index.date).agg(['min', 'max'])
    # daily_stats.columns = daily_stats.columns.droplevel(level=0)
    daily_stats['ratio'] = daily_stats['min'] / daily_stats['max']

    return daily_stats['ratio']


def sf_min_mean(df: pd.DataFrame):
    """
    Calcola il rapporto tra valore minimo giornaliero e valore medio giornaliero
    :param df: la series del singolo utente con timestamp sull'indice
    :return: una serie con un valore del rapporto per ogni giorno
    """

    daily_stats = df.groupby(df.index.date).agg(['min', 'mean'])
    # daily_stats.columns = daily_stats.columns.droplevel(level=0)
    daily_stats['ratio'] = daily_stats['min'] / daily_stats['mean']

    return daily_stats['ratio']


def sf_daytime_mean(df: pd.DataFrame):
    """
    Calcola il rapporto tra il valore medio di potenza durante le ore giornaliere rispetto al medio totale
    :param df: la series del singolo utente con timestamp sull'indice
    :return: una serie con un valore del rapporto per ogni giorno
    """
    filtered_df = df.between_time('9:00', '22:00')
    daily_stats = df.groupby(df.index.date).mean()
    hourly_mean = filtered_df.groupby(filtered_df.index.date).mean()
    ratio = 7 / 12 * hourly_mean / daily_stats

    return ratio


def sf_daytime_max(df: pd.DataFrame):
    """
    Calcola il rapporto tra il valore massimo di potenza durante le ore giornaliere rispetto al massimo totale
    :param df: la series del singolo utente con timestamp sull'indice
    :return: una serie con un valore del rapporto per ogni giorno
    """
    filtered_df = df.between_time('9:00', '22:00')
    daily_stats = df.groupby(df.index.date).max()
    hourly_max = filtered_df.groupby(filtered_df.index.date).max()
    ratio = 7 / 12 * hourly_max / daily_stats

    return ratio


def sf_daytime_min_mean(df: pd.DataFrame):
    """
    Calcola il rapporto tra il valore minimo di potenza durante le ore giornaliere rispetto al medio totale
    :param df: la series del singolo utente con timestamp sull'indice
    :return: una serie con un valore del rapporto per ogni giorno
    """
    filtered_df = df.between_time('9:00', '22:00')
    daily_stats = df.groupby(df.index.date).mean()
    hourly_min = filtered_df.groupby(filtered_df.index.date).min()
    ratio = 7 / 12 * hourly_min / daily_stats

    return ratio


def sf_nightime_mean(df: pd.DataFrame):
    """
    Calcola il rapporto tra il valore medio di potenza durante le ore notturne rispetto al medio totale
    :param df: la series del singolo utente con timestamp sull'indice
    :return: una serie con un valore del rapporto per ogni giorno
    """
    filtered_df = df[~df.index.hour.isin(range(9, 23))]
    daily_stats = df.groupby(df.index.date).mean()
    hourly_mean = filtered_df.groupby(filtered_df.index.date).mean()
    ratio = 5 / 12 * hourly_mean / daily_stats

    return ratio


def sf_nightime_max(df: pd.DataFrame):
    """
    Calcola il rapporto tra il valore massimo di potenza durante le ore notturne rispetto al massimo totale
    :param df: la series del singolo utente con timestamp sull'indice
    :return: una serie con un valore del rapporto per ogni giorno
    """
    filtered_df = df[~df.index.hour.isin(range(9, 23))]
    daily_stats = df.groupby(df.index.date).max()
    hourly_max = filtered_df.groupby(filtered_df.index.date).max()
    ratio = 5 / 12 * hourly_max / daily_stats

    return ratio


def sf_nightime_min_mean(df: pd.DataFrame):
    """
    Calcola il rapporto tra il valore minimo di potenza durante le ore notturne rispetto al medio totale
    :param df: la series del singolo utente con timestamp sull'indice
    :return: una serie con un valore del rapporto per ogni giorno
    """
    filtered_df = df[~df.index.hour.isin(range(9, 23))]
    daily_stats = df.groupby(df.index.date).mean()
    hourly_mean = filtered_df.groupby(filtered_df.index.date).min()
    ratio = 5 / 12 * hourly_mean / daily_stats

    return ratio


def sf_afternoon_mean(df: pd.DataFrame):
    """
    Calcola il rapporto tra il valore medio di potenza durante le ore pomeridiane rispetto al medio totale
    :param df: la series del singolo utente con timestamp sull'indice
    :return: una serie con un valore del rapporto per ogni giorno
    """
    filtered_df = df.between_time('13:00', '17:00')
    daily_stats = df.groupby(df.index.date).mean()
    hourly_mean = filtered_df.groupby(filtered_df.index.date).mean()
    ratio = 5 / 24 * hourly_mean / daily_stats

    return ratio


def sf_afternoon_max(df: pd.DataFrame):
    """
    Calcola il rapporto tra il valore massimo di potenza durante le ore pomeridiane rispetto al massimo totale
    :param df: la series del singolo utente con timestamp sull'indice
    :return: una serie con un valore del rapporto per ogni giorno
    """
    filtered_df = df.between_time('13:00', '17:00')
    daily_stats = df.groupby(df.index.date).max()
    hourly_max = filtered_df.groupby(filtered_df.index.date).max()
    ratio = 5 / 24 * hourly_max / daily_stats

    return ratio


def sf_afternoon_min_mean(df: pd.DataFrame):
    """
    Calcola il rapporto tra il valore minimo di potenza durante le ore pomeridiane rispetto al medio totale
    :param df: la series del singolo utente con timestamp sull'indice
    :return: una serie con un valore del rapporto per ogni giorno
    """
    filtered_df = df.between_time('13:00', '17:00')
    daily_stats = df.groupby(df.index.date).mean()
    hourly_min = filtered_df.groupby(filtered_df.index.date).min()
    ratio = 5 / 24 * hourly_min / daily_stats

    return ratio


def sf_evening_mean(df: pd.DataFrame):
    """
    Calcola il rapporto tra il valore medio di potenza durante le ore serali rispetto al medio totale
    :param df: la series del singolo utente con timestamp sull'indice
    :return: una serie con un valore del rapporto per ogni giorno
    """
    filtered_df = df.between_time('21:00', '23:00')
    daily_stats = df.groupby(df.index.date).mean()
    hourly_mean = filtered_df.groupby(filtered_df.index.date).mean()
    ratio = 1 / 8 * hourly_mean / daily_stats

    return ratio


def sf_evening_max(df: pd.DataFrame):
    """
    Calcola il rapporto tra il valore massimo di potenza durante le ore serali rispetto al massimo totale
    :param df: la series del singolo utente con timestamp sull'indice
    :return: una serie con un valore del rapporto per ogni giorno
    """
    filtered_df = df.between_time('21:00', '23:00')
    daily_stats = df.groupby(df.index.date).max()
    hourly_max = filtered_df.groupby(filtered_df.index.date).max()
    ratio = 1 / 8 * hourly_max / daily_stats

    return ratio


def sf_evening_min_mean(df: pd.DataFrame):
    """
    Calcola il rapporto tra il valore minimo di potenza durante le ore serali rispetto al medio totale
    :param df: la series del singolo utente con timestamp sull'indice
    :return: una serie con un valore del rapporto per ogni giorno
    """
    filtered_df = df.between_time('21:00', '23:00')
    daily_stats = df.groupby(df.index.date).mean()
    hourly_min = filtered_df.groupby(filtered_df.index.date).min()
    ratio = 1 / 8 * hourly_min / daily_stats

    return ratio