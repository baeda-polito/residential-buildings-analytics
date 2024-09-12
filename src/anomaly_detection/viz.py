import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio

pio.renderers.default = "browser"


def plot_predictions(data: pd.DataFrame, building_name: str):
    data.index = pd.to_datetime(data.index)
    true = go.Scatter(x=data.index, y=data["true"], mode="lines", name="True",
                      line=dict(width=1.5, color='#239b56'),
                      hovertemplate='Date: %{x}<br>Potenza: %{y:.2f} W<extra></extra>')
    pred = go.Scatter(x=data.index, y=data["pred"], mode="lines", name="Predicted",
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
    # Plot the distribution of the residuals column
    hist = go.Histogram(x=data["residuals"], marker=dict(color='#239b56'), name="Residuals",
                        hovertemplate='Residuals: %{x}<extra></extra>')
    layout = go.Layout(
        title=dict(
            text=f"{building_name}",
            x=0.5,
            font=dict(size=20, color='black')
        ),
        font=dict(family='Poppins'),
        xaxis=dict(
            title='Residuals',
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
    data.index = pd.to_datetime(data.index)
    max_true = data["true"].max()
    max_pred = data["pred"].max()
    max_total = max(max_true, max_pred)
    scatter = go.Scatter(x=data["true"], y=data["pred"], mode="markers", name="True vs Pred",
                         marker=dict(color='#239b56', size=5),
                         customdata=data.index,
                         hovertemplate='Date: %{customdata}<br>True: %{x:.2f} W<br>Pred: %{y:.2f} W<extra></extra>')
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
            title='True',
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
            title='Pred',
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
