# Pre-processing dei dati

In questo modulo vengono implementate le funzionalità per il pre-processing dei dati. I principali componenti di questo modulo sono:

- `pre_processing.py`: contiene le due funzioni principali per il pre-processing dei dati, `pre_processing_power` e `pre_process_production_power`. La prima funzione si occupa di effettuare il pre-processing dei dati riguardanti la potenza consumata dall'edificio, mentre la seconda funzione si occupa di effettuare il pre-processing dei dati riguardanti la potenza prodotta dall'impianto fotovoltaico.
- `utils.py`: contiene le funzioni di supporto per il pre-processing dei dati, come ad esempio la funzione `replace_constant_values` che si occupa di sostituire i valori costanti con NaN, la funzione `reconstruct_missing_values_interp` che si occupa di ricostruire i valori mancanti utilizzando l'interpolazione, la funzione `reconstruct_missing_values_knn` che si occupa di ricostruire i valori mancanti utilizzando il K-Nearest Neighbors.
- `pv_model.py`: contiene la funzione `get_pv_production` che si occupa di calcolare la potenza prodotta dall'impianto fotovoltaico utilizzando un modello fisico basato su metadati dell'impianto fotovoltaico, utilizzando la libreria _pvlib_.

Si assume che i dati vengono forniti con un campionamento di 15 minuti.


Nelle sezioni seguenti viene descritto il funzionamento generale del pre-processing.

### Pre-processing della potenza utilizzata dall'edificio ("power")

L'algoritmo di pre-processing della variabile _power_ identifica valori mancanti, outlier e valori costanti. Gli outlier sono definiti confrontando i dati con i limiti di potenza contrattuale e nominale del fotovoltaico, maggiorati del 10%. I valori costanti sono rilevati quando superano 4 istanti consecutivi, indicando possibili blocchi del sistema. Tutti i valori anomali sono sostituiti con NaN e ricostruiti in base alla durata delle sequenze mancanti:
- Fino a 4 istanti: interpolazione lineare.
- Da 5 a 16 istanti: k-Nearest Neighbor (k=5), che calcola la media dei profili giornalieri più simili tramite distanza euclidea.
- Oltre 16 istanti: i dati non vengono ricostruiti né analizzati. 

### Pre-processing della potenza prodotta dall'impianto fotovoltaico ("productionPower")

Per la variabile productionPower, l’algoritmo verifica che i valori siano nulli in assenza di irradianza e, in sua presenza, che non superino la potenza nominale maggiorata del 10%. I dati anomali vengono sostituiti con NaN e ricostruiti tramite:
- Modello fisico (libreria pvlib) basato su metadati dell'impianto fotovoltaico, se disponibili.
- Modello di regressione lineare che lega la potenza prodotta a irradianza globale (GHI) e diretta normale (DNI). 

Il modello di regressione lineare sviluppato segue l’Equazione 

`productionPower = a * GHI + b * DNI + c`

- GHI è l’irradianza globale sul piano orizzontale in W
- DNI è l’irradianza diretta normale in W
- a, b e c sono i coefficienti di regressione.


## Utilizzo

Per utilizzare le funzionalità di pre-processing dei dati, è sufficiente importare il modulo `pre_processing` e chiamare le funzioni `pre_processing_power` e `pre_process_production_power` passando come argomento i dati da pre-processare. Di seguito un esempio di utilizzo:

```python
import pandas as pd
from src.energy_analytics.pre_processing import pre_process_power, pre_process_production_power

# Caricamento dei dati
power_data = pd.DataFrame()  # Dati relativi alla potenza consumata dall'edificio
weather_data = pd.DataFrame()  # Dati relativi alle condizioni meteorologiche
production_power_data = pd.DataFrame()  # Dati relativi alla potenza prodotta dall'impianto fotovoltaico

rated_power = 10  # Potenza nominale dell'edificio
rated_power_pv = 5  # Potenza nominale dell'impianto fotovoltaico
user_type = 'producer'  # Tipologia di utente

# Pre-processing della potenza consumata dall'edificio
power_data = pre_process_power(power_data, user_type, rated_power, rated_power_pv, max_missing_interp=4, max_missing_knn=24)

physic_model = True
pv_params = {
    "azimuth": 0,
    "tilt": 30,
    "rated_power": rated_power_pv,
    "storage": 5
}

coordinates = [12.28, 41.8]

# Pre-processing della potenza prodotta dall'impianto fotovoltaico
production_power_data = pre_process_production_power(production_power_data, weather_data, physic_model, pv_params, coordinates)
```
