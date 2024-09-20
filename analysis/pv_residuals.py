from src.anomaly_detection.mlp import MultiLayerPerceptron
from src.anomaly_detection.data_handler import DataHandler
from src.anomaly_detection.anomaly_detection_functions import get_anomaly_severity, get_anomaly_threhsold
from src.building import load_anguillara
from pvlib.location import Location
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import plotly.io as pio
import pandas as pd
import torch
pio.renderers.default = "browser"


building_list = load_anguillara()

for building in building_list:

    if building.building_info["user_type"] != "consumer":

        uuid = building.building_info["id"]
        aggregate = building.building_info["aggregate"]

        mlp = MultiLayerPerceptron(input_size=5, hidden_layers=[64, 64], output_size=1)
        mlp.load_state_dict(torch.load(f"../data/pv_add/models/{uuid}.pth"))

        location = Location(latitude=building.building_info["coordinates"][1], longitude=building.building_info["coordinates"][0])

        energy_data = building.energy_meter.data
        weather_data = pd.read_csv("../data/weather/anguillara.csv")
        data_handler = DataHandler(energy_data=energy_data, weather_data=weather_data)
        data = data_handler.create_data(location=location)
        X, y, x_scaler, y_scaler = data_handler.preprocess(data)

        X_tensor = torch.tensor(X, dtype=torch.float32)

        y_pred = mlp(X_tensor).detach().numpy()

        y_pred_rescaled = y_scaler.inverse_transform(y_pred)
        y_true_rescaled = y_scaler.inverse_transform(y)

        residuals = y_pred_rescaled - y_true_rescaled

        df_add = pd.DataFrame({"y_true": y_true_rescaled.flatten(), "y_pred": y_pred_rescaled.flatten(), "residuals": residuals.flatten()}, index=data.index)
        df_add["hour"] = df_add.index.hour
        threshold = pd.read_csv(f"../data/pv_add/threshold/{uuid}.csv")
        df_add = pd.merge(df_add, threshold, on="hour", how="left")
        df_add['residuals'] = df_add['residuals'].clip(lower=0)
        df_add.index = data.index

        df_add["severity"] = df_add.apply(lambda x: get_anomaly_severity(x["y_true"], x["y_pred"], x["LT"], x["HT"]), axis=1)

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

        fig_prod = go.Figure(data=[true, pred], layout=layout)

        fig_res = go.Figure(data=[residuo, lt, ht, low_severity, high_severity], layout=layout)
        fig_res.write_html(f"../figures/pv_evaluation/{uuid}_residuals.html")

        fig_sev = go.Figure(data=[severity], layout=layout)
        fig_sev.write_html(f"../figures/pv_evaluation/{uuid}_severity.html")

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
            # Set the layout background to white
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        # Add y-axis titles for each row
        fig.update_yaxes(title_text="Potenza [W]", row=1, col=1)  # Row 1 Y-axis title
        fig.update_yaxes(title_text="Residuo [W]", row=2, col=1)  # Row 2 Y-axis title
        fig.update_yaxes(title_text="Severità", row=3, col=1)  # Row 3 Y-axis title
        fig.update_yaxes(
            showgrid=True, gridcolor='#f4f4f4',  # Light grey gridlines
            zeroline=True, zerolinecolor='black'  # Black zero line for y-axis
        )
        # Similarly, you can update x-axes if needed (for shared x-axis)
        fig.update_xaxes(showgrid=True, gridcolor='#f4f4f4', zeroline=True, zerolinecolor='black')
        fig.write_html(f"../figures/pv_evaluation/{uuid}_all.html")
