import os
import pandas as pd
import plotly.io as pio
from loguru import logger
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from plotly.subplots import make_subplots

from settings import PROJECT_ROOT
from ..building import Building
from .utils import get_anomaly_residuals

pio.renderers.default = "browser"


def plot_predictions(data: pd.DataFrame, building_name: str):
    """
    Crea il grafico interattivo con le previsioni del modello rispetto ai valori reali su una serie temporale

    Args:
        data (pd.DataFrame): dataframe con le colonne "true" e "pred" e l'indice timestamp.
        building_name (str): nome dell'edificio.

    Returns:
        go.Figure: oggetto con il grafico interattivo.
    """
    data.index = pd.to_datetime(data.index)
    true = go.Scatter(x=data.index, y=data["true"], mode="lines", name="Reale",
                      line=dict(width=1.5, color='#239b56'),
                      hovertemplate='Data: %{x}<br>Potenza: %{y:.2f} W<extra></extra>')
    pred = go.Scatter(x=data.index, y=data["pred"], mode="lines", name="Predetto",
                      line=dict(width=1.5, color='#154360'),
                      hovertemplate='Date: %{x}<br>Potenza: %{y:.2f} W<extra></extra>')

    layout = go.Layout(
        title=dict(
            text=f"{building_name}",
            x=0.5,
            font=dict(size=20, color='black')
        ),
        font=dict(family='Poppins'),
        xaxis=dict(
            title='Data e ora',
            showgrid=True,
            gridcolor='#dfdfdf',
            gridwidth=0.2,
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=1,
            tickfont=dict(size=14),
            titlefont=dict(size=18)
        ),
        yaxis=dict(
            title='Potenza [W]',
            showgrid=True,
            gridcolor='#dfdfdf',
            gridwidth=0.2,
            tickfont=dict(size=14),
            titlefont=dict(size=18)
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=0.97,
            xanchor='center',
            x=0.5,
            font=dict(size=16, color='black')
        )
    )

    return go.Figure(data=[true, pred], layout=layout)


def plot_distribution(data: pd.DataFrame, building_name: str):
    """
    Crea il grafico con la distribuzione degli errori del modello.

    Args:
        data (pd.DataFrame): dataframe con la colonna "residuals".
        building_name (str): nome dell'edificio.

    Returns:
        go.Figure: oggetto con il grafico.
    """

    # Plot the distribution of the residuals column
    hist = go.Histogram(x=data["residuals"], marker=dict(color='#239b56'), name="Residuo",
                        hovertemplate='Residuo: %{x}<extra></extra>')
    layout = go.Layout(
        title=dict(
            text=f"{building_name}",
            x=0.5,
            font=dict(size=20, color='black')
        ),
        font=dict(family='Poppins'),
        xaxis=dict(
            title='Residuo',
            showgrid=True,
            gridcolor='#dfdfdf',
            gridwidth=0.2,
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=1,
            tickfont=dict(size=14),
            titlefont=dict(size=18)
        ),
        yaxis=dict(
            title='Count',
            showgrid=True,
            gridcolor='#dfdfdf',
            gridwidth=0.2,
            tickfont=dict(size=14),
            titlefont=dict(size=18)
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=False
    )

    return go.Figure(data=[hist], layout=layout)


def plot_pred_vs_true(data: pd.DataFrame, building_name: str):
    """
    Crea il grafico con i valori predetti vs i valori reali.

    Args:
        data (pd.DataFrame): dataframe con le colonne "true" e "pred".
        building_name (str): nome dell'edificio.

    Returns:
        go.Figure: oggetto con il grafico.
    """

    data.index = pd.to_datetime(data.index)
    max_true = data["true"].max()
    max_pred = data["pred"].max()
    max_total = max(max_true, max_pred)
    scatter = go.Scatter(x=data["true"], y=data["pred"], mode="markers", name="Reale vs Predetto",
                         marker=dict(color='#239b56', size=5),
                         customdata=data.index,
                         hovertemplate='Data: %{customdata}<br>Reale: %{x:.2f} W<br>Predetto: %{y:.2f} W<extra></extra>')
    line = go.Scatter(x=[0, max_total], y=[0, max_total], mode="lines",
                      line=dict(width=1.5, color='#d60e0e'),
                      hoverinfo='skip')
    layout = go.Layout(
        title=dict(
            text=f"{building_name}",
            x=0.5,
            font=dict(size=20, color='black')
        ),
        font=dict(family='Poppins'),
        xaxis=dict(
            title='Reale',
            showgrid=True,
            gridcolor='#dfdfdf',
            gridwidth=0.2,
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=1,
            tickfont=dict(size=14),
            titlefont=dict(size=18)
        ),
        yaxis=dict(
            title='Predetto',
            showgrid=True,
            gridcolor='#dfdfdf',
            gridwidth=0.2,
            tickfont=dict(size=14),
            titlefont=dict(size=18)
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=False
    )

    return go.Figure(data=[scatter, line], layout=layout)


def heatmap_anomalies(building: Building):
    """
    Crea un heatmap dove ogni cella rappresenta la severità dell'anomalia per una determinato giorno in una determinata ora.
    I dati su cui vengono valutate le anomalie sono tutti gli energy_data dell'edificio. La figura viene generata e salvata
    in figures/anomaly_detection/pv_models

    Args:
        building (Building): oggetto Building con le informazioni sull'edificio.

    Returns:
        None
    """

    df_add = get_anomaly_residuals(building)
    df_add["date"] = df_add.index.date
    df_add['hour'] = df_add.index.strftime('%H:%M')

    pivot_table = df_add.pivot(index='date', columns='hour', values='severity')
    pivot_table = pivot_table.astype(float).round(1)
    pivot_table.fillna(2, inplace=True)
    pivot_table.replace(0, float('nan'), inplace=True)
    pivot_table.replace(2, 0, inplace=True)
    pivot_table.index = pd.to_datetime(pivot_table.index)
    pivot_table.index = pivot_table.index.strftime('%b-%d')

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(f"Identificazione delle anomalie per l'edificio {building.building_info['name']}", fontsize=16,
                 fontweight='bold')
    ax.set_ylabel("Ora del giorno", fontsize=14)
    ax.set_xlabel("Data", fontsize=14)
    cmap = clr.LinearSegmentedColormap.from_list('custom', ['#cfcfcf', '#ffaa00', '#ff0000'], N=256)
    cax = ax.matshow(pivot_table.T, cmap=cmap, aspect='auto')

    ax.set_yticks(range(0, len(pivot_table.columns), 8))
    ax.set_yticklabels(pivot_table.columns[::8])
    ax.set_xticks(range(0, len(pivot_table.index), 14))
    ax.set_xticklabels(pivot_table.index[::14])
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticks([x - 0.5 for x in range(1, len(pivot_table.columns))], minor=True)
    ax.set_xticks([y - 0.5 for y in range(1, len(pivot_table.index))], minor=True)
    ax.grid(which='minor', color='#e1e1e1', linestyle='-', linewidth=1)
    cbar = fig.colorbar(cax, orientation='horizontal', pad=0.1, aspect=50)
    cbar.set_label('Severità', fontsize=12)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_ticks([0, 0.5, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, "figures", "anomaly_detection", "pv_models",
                             f"{building.building_info['id']}_heatmap.png"))

    count_low = df_add[df_add["severity"] == 0.5].shape[0]
    count_high = df_add[df_add["severity"] == 1].shape[0]

    sum_low = (df_add[df_add["severity"] == 0.5]["residuals"] - df_add[df_add["severity"] == 0.5][
        "LT"]).sum() * 0.25 / 1000
    sum_high = (df_add[df_add["severity"] == 1]["residuals"] - df_add[df_add["severity"] == 1][
        "HT"]).sum() * 0.25 / 1000

    logger.debug(f"Numeri di anomalie di bassa severità: {count_low} -- Energia associata: {sum_low:.1f} kWh")
    logger.debug(f"Numeri di anomalie di alta severità: {count_high} -- Energia associata: {sum_high:.1f} kWh")


def plot_residuals(building: Building):
    """
    Crea i grafici con i residui, i threshold e la severità delle anomalie. Salva i grafici in formato html nella cartella
    figures/anomaly_detection/pv_models.

    Args:
        building (Building): oggetto Building con le informazioni sull'edificio.

    Returns:
        None
    """

    df_add = get_anomaly_residuals(building)

    layout = go.Layout(
        title=dict(
            text=f"{building.building_info['name']}",
            x=0.5,
            font=dict(size=20, color='black')
        ),
        font=dict(family='Poppins'),
        xaxis=dict(
            title='Data e ora',
            showgrid=True,
            gridcolor='#dfdfdf',
            gridwidth=0.2,
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=1,
            tickfont=dict(size=14),
            titlefont=dict(size=18)
        ),
        yaxis=dict(
            title='Residuo [W]',
            showgrid=True,
            gridcolor='#dfdfdf',
            gridwidth=0.2,
            tickfont=dict(size=14),
            titlefont=dict(size=18)
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=0.97,
            xanchor='center',
            x=0.5,
            font=dict(size=16, color='black')
        )
    )

    true = go.Scatter(
        x=df_add.index,
        y=df_add['y_true'],
        mode='lines',
        name='Reale',
        line=dict(width=1.5, color='#2bb06a'),
        hovertemplate='Data: %{x}<br>True: %{y:.2f} W<extra></extra>'
    )

    pred = go.Scatter(
        x=df_add.index,
        y=df_add['y_pred'],
        mode='lines',
        name='Predetto',
        line=dict(width=1.2, color='#b6353f'),
        hovertemplate='Data: %{x}<br>Predicted: %{y:.2f} W<extra></extra>'
    )

    residuo = go.Scatter(
        x=df_add.index,
        y=df_add['residuals'],
        mode='lines',
        name='Residuo',
        line=dict(width=1.2, color='#4c00a6'),
        hovertemplate='Data: %{x}<br>Residuo: %{y:.2f} W<extra></extra>'
    )

    lt = go.Scatter(
        x=df_add.index,
        y=df_add['LT'],
        mode='lines',
        name='Threshold bassa severità',
        line=dict(width=1.2, color='#ee6b04', dash='dash'),
        hovertemplate='Data: %{x}<br>LT: %{y:.2f} W<extra></extra>'
    )

    ht = go.Scatter(
        x=df_add.index,
        y=df_add['HT'],
        mode='lines',
        name='Threshold alta severità',
        line=dict(width=1.2, color='#ee2404', dash='dash'),
        hovertemplate='Data: %{x}<br>HT: %{y:.2f} W<extra></extra>'
    )

    severity = go.Scatter(
        x=df_add.index,
        y=df_add['severity'],
        mode='lines',
        name='Severità',
        line=dict(width=1.2, color='#b6353f'),
        hovertemplate='Data: %{x}<br>Severity: {y}<extra></extra>'
    )

    low_severity_points = df_add[df_add["severity"] == 0.5]
    high_severity_points = df_add[df_add["severity"] == 1]

    low_severity = go.Scatter(
        x=low_severity_points.index,
        y=low_severity_points["residuals"],
        mode='markers',
        name='Anomalia bassa severità',
        marker=dict(color='#ee6b04', size=5)
    )

    high_severity = go.Scatter(
        x=high_severity_points.index,
        y=high_severity_points["residuals"],
        mode='markers',
        name='Anomalia alta severità',
        marker=dict(color='#b6353f', size=5)
    )

    fig_res = go.Figure(data=[residuo, lt, ht, low_severity, high_severity], layout=layout)
    fig_res.write_html(os.path.join(PROJECT_ROOT, "figures", "anomaly_detection", "pv_models",
                                    f"{building.building_info['id']}_residuals.html"))

    fig_sev = go.Figure(data=[severity], layout=layout)
    fig_sev.write_html(os.path.join(PROJECT_ROOT, "figures", "anomaly_detection", "pv_models",
                                    f"{building.building_info['id']}_severity.html"))

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.5, 0.4, 0.1])
    fig.add_trace(true, row=1, col=1)
    fig.add_trace(pred, row=1, col=1)
    fig.add_trace(residuo, row=2, col=1)
    fig.add_trace(lt, row=2, col=1)
    fig.add_trace(ht, row=2, col=1)
    fig.add_trace(low_severity, row=2, col=1)
    fig.add_trace(high_severity, row=2, col=1)
    fig.add_trace(severity, row=3, col=1)

    fig.update_layout(
        annotations=[
            dict(x=0.5, y=1, xref="paper", yref="paper",
                 text="Predetto vs Reale", showarrow=False, font=dict(size=18, color="black")),
            dict(x=0.5, y=0.5, xref="paper", yref="paper",
                 text="Analisi dei residui", showarrow=False, font=dict(size=18, color="black")),
            dict(x=0.5, y=0.1, xref="paper", yref="paper",
                 text="Analisi della severità", showarrow=False, font=dict(size=18, color="black"))
        ],
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # Add y-axis titles for each row
    fig.update_yaxes(title_text="Potenza [W]", row=1, col=1)
    fig.update_yaxes(title_text="Residuo [W]", row=2, col=1)
    fig.update_yaxes(title_text="Severità", row=3, col=1)
    fig.update_yaxes(
        showgrid=True, gridcolor='#f4f4f4',
        zeroline=True, zerolinecolor='black'
    )
    fig.update_xaxes(showgrid=True, gridcolor='#f4f4f4', zeroline=True, zerolinecolor='black')
    fig.write_html(os.path.join(PROJECT_ROOT, "figures", "anomaly_detection", "pv_models",
                                f"{building.building_info['id']}_all.html"))
