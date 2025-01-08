from pvlib.pvsystem import PVSystem, FixedMount, Array
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
import pandas as pd
import numpy as np


def get_pv_production(lat, lon, tilt, azimuth, rated_power, weather):
    """
    Calcola la produzione di energia fotovoltaica tramite modello fisico in pvlib.
    Args:
        lat: latitudine
        lon: longitudine
        tilt: inclinazione del pannello
        azimuth: azimut del pannello
        rated_power: potenza nominale del pannello
        weather: dati meteorologici
    Returns:
        pd.DataFrame con la produzione di energia fotovoltaica
    """

    weather['timestamp'] = pd.to_datetime(weather['timestamp'])
    weather.set_index('timestamp', inplace=True)

    # Create a location object to store the latitude and longitude
    location = Location(latitude=lat, longitude=lon)
    solar_angles = location.get_solarposition(weather.index)
    solar_angles['zenith_radians'] = np.radians(solar_angles['zenith'])
    weather['dhi'] = weather['ghi'] - weather['dni'] * np.cos(solar_angles['zenith_radians'])

    pv_system = PVSystem(
        arrays=Array(mount=FixedMount(surface_azimuth=azimuth, surface_tilt=tilt),
                     module_parameters={"pdc0": rated_power, 'gamma_pdc': -0.004},
                     temperature_model_parameters=TEMPERATURE_MODEL_PARAMETERS["pvsyst"]["freestanding"]),
        inverter_parameters={"pdc0": rated_power},
    )

    model_chain = ModelChain(system=pv_system,
                             location=location,
                             aoi_model='physical',
                             spectral_model='no_loss')
    model_chain.run_model(weather)
    pv_production = pd.DataFrame(data=model_chain.results.ac.astype(float))
    pv_production.columns = ["productionPower"]
    pv_production.index = pd.to_datetime(pv_production.index)

    return pv_production
