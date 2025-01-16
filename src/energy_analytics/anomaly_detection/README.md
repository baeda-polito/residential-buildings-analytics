# Identificazione e diagnostica di anomalie energetiche in edifici residenziali

Il modulo `anomaly_detection` contiene le funzioni per implementare il processo di identificazione e diagnostica di anomalie energetiche sulla produzione fotovoltaica e sul consumo energetico di un edificio residenziale.

## Identificazione di anomalie sul consumo energetico

Il processo di identificazione di anomalie sul consumo energetico si basa sull'analisi dei dati di consumo energetico e sulla rilevazione di anomalie rispetto ai profili di carico atteso.
Il processo si basa sull'utilizzo dei medioidi calcoli nel modulo di `benchmarking`. I medioidi rappresentano i profili di carico tipici per ciascun cluster di consumo energetico.
Se il profilo di carico attuale mostra una distanza significativa rispetto ai medioidi, allora viene rilevata un'anomalia.

### Utilizzo

Questo script identifica anomalie sul consumo energetico in tempo reale. Utilizza i risultati del modulo di benchmarking per calcolare la distanza tra il profilo di carico attuale e i medioidi dei cluster di consumo energetico.

```python
import pandas as pd
from src.energy_analytics.anomaly_detection import detect_anomaly_power

load_profile = pd.Series()  # Profilo di carico attuale
aggregate_name = ""  # Nome dell'aggregato. Prende i medioidi calcolati per esso
user_type = ""  # Tipo di utente

anomaly = detect_anomaly_power(load_profile, aggregate_name, user_type)

if anomaly:
    print("Anomalia rilevata")
else:
    print("Nessuna anomalia rilevata")
```


## Identificazione di anomalie sulla produzione fotovoltaica

Il processo di identificazione di anomalie sulla produzione fotovoltaica si basa sull'analisi dei dati di produzione fotovoltaica e sulla rilevazione di anomalie rispetto al valore di produzione atteso.
Il processo si basa sull'allenamento di un modello di regressione (MultiLayer Perceptron) per la previsione della produzione fotovoltaica attesa.
Il modello viene allenato sui dati di produzione fotovoltaica e di irraggiamento solare e viene utilizzato per prevedere la produzione attesa.
Le anomalie vengono identificate confrontando la produzione attesa con la produzione reale, tenendo conto di threshold statistici calcolati sui residui del modello (utilizzando lo Zscore).
Le anomalie vengono poi classificate in base alla loro severità: 
- **0**: Nessuna anomalia
- **0.5**: Anomalia moderata
- **1**: Anomalia grave

Il modulo è organizzato nel seguente modo:
- `mlp.py` contiene la classe `MultiLayerPerceptron` che ospita il modello di regressione.
- `data_handler.py` contiene la classe `DataHandler` che si occupa della gestione dei dati di produzione fotovoltaica e irraggiamento solare e li prepara per l'allenamento del modello.
- `trainer.py` contiene la classe `Trainer` che si occupa dell'allenamento e della valutazione del modello.
- `train.py` contiene la funzione `train` che si occupa dell'allenamento del modello e del salvataggio dei risultati.
- `evaluate.py` contiene la funzione `evaluate` che si occupa della valutazione del modello.
- `viz.py` contiene le funzioni per visualizzare i risultati dell'allenamento e della valutazione del modello, come il plot predetto vs reale, l'istogramma dei residui, etc.
- `anomaly_detection_functions.py` contiene le funzioni per il calcolo dei threshold statistici e per l'identificazione della severità dell'anomalia.
- `main_train.py` contiene la pipeline per l'allenamento del modello.
- `main_anomaly_detection.py` contiene la funzione per runnare online il processo di identificazione di anomalie sulla produzione fotovoltaica.

### Utilizzo

#### Modalità offline

##### Trainare un singolo modello di produzione fotovoltaica

Questo script allena un modello di produzione fotovoltaica per un edificio residenziale. Il modello viene allenato utilizzando i dati di produzione fotovoltaica e irraggiamento solare forniti e salvato nella cartella `results/anomaly_dection/pv_models`.

```python
import pandas as pd

from src.energy_analytics.anomaly_detection.train import train
from src.energy_analytics import Building

data_DU1 = pd.DataFrame()  # Dataframe con "timestamp", "power" e "productionPower"
metadata_DU1 = {}  # Dizionario con i metadati dell'edificio
weather_data = pd.DataFrame()  # Dataframe con i dati meteo

DU1 = Building(data=data_DU1, metadata=metadata_DU1, weather_data=weather_data)

train(DU1)
```

##### Valutare un singolo modello di produzione fotovoltaica

Questo script valuta il modello di produzione fotovoltaica allenato in precedenza. Calcola metriche di errore, calcola l'entità delle anomalie nel periodo fornito e visualizza i risultati salvando i plot nella cartella `figures/pv_models`.

```python
import pandas as pd

from src.energy_analytics.anomaly_detection.evaluate import evaluate_pv_model
from src.energy_analytics import Building

data_DU1 = pd.DataFrame()  # Dataframe con "timestamp", "power" e "productionPower"
metadata_DU1 = {}  # Dizionario con i metadati dell'edificio
weather_data = pd.DataFrame()  # Dataframe con i dati meteo

DU1 = Building(data=data_DU1, metadata=metadata_DU1, weather_data=weather_data)

evaluate_pv_model(DU1)
```

#### Modalità online

Questo script identifica anomalie sulla produzione fotovoltaica in tempo reale. Utilizza il modello allenato in precedenza per prevedere la produzione attesa e identificare anomalie rispetto alla produzione reale. Le anomalie vengono classificate in base alla loro severità.

```python
from src.energy_analytics.anomaly_detection import detect_anomaly_pv

building_id = ""
ghi = 100   # Global Horizontal Irradiance
dni = 100   # Direct Normal Irradiance
air_temp = 20   # temperatura esterna
azimuth = 180   # angolo di azimut
zenith = 30   # angolo zenitale
y_true = 100   # produzione reale
hour = 12   # ora del giorno

anomaly, severity, y_pred =  detect_anomaly_pv(building_id, ghi, dni, air_temp, azimuth, zenith, y_true, hour)

if anomaly:
    print(f"Anomalia rilevata con severità {severity}. La produzione attesa era {y_pred}")
else:
    print("Nessuna anomalia rilevata")
```