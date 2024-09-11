from src.building import load_anguillara
import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = "browser"

"""
Pre-processing
"""

building_list = load_anguillara()

for building in building_list:
    print(building.building_info['name'])
    data_raw = building.energy_meter.energy_meter_data
    data_cleaned = building.energy_meter.energy_meter_data_cleaned

    trace1 = go.Scatter(
        x=data_raw["timestamp"],
        y=data_raw['power'],
        mode='lines',
        name='power raw',
        line=dict(width=2, color='#b6353f'),
        hovertemplate='Date: %{x}<br>Power: %{y:.2f} W<extra></extra>'  # Customize hover template
    )

    trace2 = go.Scatter(
        x=data_cleaned["timestamp"],
        y=data_cleaned['power'],
        mode='lines',
        name='power cleaned',
        line=dict(width=1.2, color='#2bb06a'),
        hovertemplate='Date: %{x}<br>Power: %{y:.2f} W<extra></extra>'  # Customize hover template
    )

    trace3 = go.Scatter(
        x=data_raw["timestamp"],
        y=data_raw['productionPower'],
        mode='lines',
        name='productionPower raw',
        line=dict(width=2, color='#b6353f'),
        hovertemplate='Date: %{x}<br>Power: %{y:.2f} W<extra></extra>'  # Customize hover template
    )

    trace4 = go.Scatter(
        x=data_cleaned["timestamp"],
        y=data_cleaned['productionPower'],
        mode='lines',
        name='productionPower cleaned',
        line=dict(width=1.2, color='#2bb06a', dash='dash'),
        hovertemplate='Date: %{x}<br>Power: %{y:.2f} W<extra></extra>'  # Customize hover template
    )

    # Create the layout with legend positioned at the top center
    layout = go.Layout(
        title=dict(
            text=f"Pre-processing per {building.building_info['name']}",
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

    fig_power = go.Figure(data=[trace2, trace1], layout=layout)
    pio.write_html(fig_power, f"../figures/power_pre_processing/{building.building_info['name']}_power_pre_processing.html")

    if building.building_info['user_type'] != "consumer":
        fig_pv = go.Figure(data=[trace4, trace3], layout=layout)
        pio.write_html(fig_pv, f"../figures/pv_pre_processing/{building.building_info['name']}_productionPower_pre_processing.html")
