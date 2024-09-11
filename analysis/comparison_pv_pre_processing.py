import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = "browser"
from src.pre_processing import pre_process_production_power

data_raw = pd.read_csv('../data/energy_meter/3d956901-f5ea-4094-9c85-333cc68183d4.csv')
weather = pd.read_csv('../data/weather/anguillara.csv')

data_pre_processed_datadriven = pre_process_production_power(data_raw, weather, physic_model=False,
                                                             pv_params={"rated_power": 5})
data_pre_processed_physical = pre_process_production_power(data_raw, weather, physic_model=True,
                                                           pv_params={"rated_power": 3,
                                                                      "tilt": 30,
                                                                      "azimuth": 40,
                                                                      "storage": None},
                                                           coordinates=[12.2830132, 42.0837101])

trace1 = go.Scatter(
    x=data_raw["timestamp"],
    y=data_raw['productionPower'],
    mode='lines',
    name='Raw',
    line=dict(width=2, color='#b6353f'),
    hovertemplate='Date: %{x}<br>Power: %{y:.2f} W<extra></extra>'  # Customize hover template
)

trace2 = go.Scatter(
    x=data_pre_processed_datadriven["timestamp"],
    y=data_pre_processed_datadriven['productionPower'],
    mode='lines',
    name='Modello data-driven',
    line=dict(width=1.2, color='#2bb06a'),
    hovertemplate='Date: %{x}<br>Power: %{y:.2f} W<extra></extra>'  # Customize hover template
)

trace3 = go.Scatter(
    x=data_pre_processed_physical["timestamp"],
    y=data_pre_processed_physical['productionPower'],
    mode='lines',
    name='Modello fisico',
    line=dict(width=1.2, color='#1d3d72'),
    hovertemplate='Date: %{x}<br>Power: %{y:.2f} W<extra></extra>'  # Customize hover template
)

# Create the layout with legend positioned at the top center
layout = go.Layout(
    title=dict(
        text=f"Comparazione tra le diverse tecniche di pre-processing",
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

fig_power = go.Figure(data=[trace2, trace3, trace1], layout=layout)
pio.write_html(fig_power, f"../figures/pv_pre_processing/comparison.html")
