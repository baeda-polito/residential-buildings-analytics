import pandas as pd
import plotly.io as pio
import plotly.graph_objs as go

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
