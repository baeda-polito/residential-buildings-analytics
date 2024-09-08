import json
import os
import pandas as pd
import numpy as np
from src.pre_processing import pre_process_power
import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = "browser"

files = os.listdir("../data/energy_meter")
for file in files:
    with open(f"../data/metadata/{file.split('.')[0]}.json") as f:
        metadata = json.load(f)

    building_name = metadata["name"]
    df = pd.read_csv(f"../data/energy_meter/{file}")
    df = df[["timestamp", "power"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    try:
        df_pre_processed = pre_process_power(df, "prosumer", 3, 6)
    except Exception as e:
        print(e)
        df_pre_processed = df
        df_pre_processed['power'] = np.nan
    # Create traces for each column in the DataFrame with hover templates
    trace1 = go.Scatter(
        x=df["timestamp"],
        y=df['power'],
        mode='lines',
        name='Power',
        line=dict(width=2, color='#b6353f'),
        hovertemplate='Date: %{x}<br>Power: %{y:.2f} W<extra></extra>'  # Customize hover template
    )

    trace2 = go.Scatter(
        x=df_pre_processed["timestamp"],
        y=df_pre_processed['power'],
        mode='lines',
        name='Power cleaned',
        line=dict(width=1.5, dash='dash', color='#2bb06a'),
        hovertemplate='Date: %{x}<br>Power: %{y:.2f} W<extra></extra>'  # Customize hover template
    )

    # Create the layout with legend positioned at the top center
    layout = go.Layout(
        title=dict(
            text=f"Pre-processing di 'power' per {building_name}",
            x=0.5,
            font=dict(size=20, color='black')
        ),
        font=dict(family='Poppins'),
        xaxis=dict(title='Timestamp', showgrid=False,
                   tickfont=dict(size=14),
                   titlefont=dict(size=18)),
        yaxis=dict(title='Power [W]', showgrid=False,
                   tickfont=dict(size=14),
                   titlefont=dict(size=18)),
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

    # Create the figure and add traces
    fig = go.Figure(data=[trace1, trace2], layout=layout)

    # Save the figure as an HTML file
    pio.write_html(fig, f"../figures/power_pre_processing/{building_name}_power_pre_processing.html")
