# Individuazione automatica di anomalie energetiche e analisi di benchmarking in edifici residenziali

Questo pacchetto contiene i moduli e le funzioni necessarie a:
1. individuare automaticamente anomalie energetiche nei dati relativi al consumo e alla produzione di energia in edifici residenziali;
2. analisi di benchmarking esterno all'interno dell'aggregato di edifici residenziali per caratterizzare il consumo energetico degli edifici.
3. calcolo di Key Performance Indicators (KPI) per valutare le prestazioni energetiche degli edifici.

---

### Struttura del pacchetto

Il pacchetto è organizzato nel seguente modo:

```plaintext
.
├── building.py         # Modulo per gestire i dati relativi ai singoli edifici
├── aggregate.py        # Modulo per gestire i dati aggregati di più edifici
├── pre_processing      # Modulo per il pre-processing dei dati
│   ├── ....
├── anomaly_detection   # Modulo per l'individuazione automatica di anomalie energetiche sulla produzione fotovoltaica
│   ├── ....
├── benchmarking        # Modulo per l'analisi di benchmarking esterno e l'identificazione di anomalie sul consumo energetico
│   ├── ....
├── kpi                 # Modulo per il calcolo dei Key Performance Indicators (KPI)
│   ├── ....
└── README.md
```

---

## Utilizzo

### Pre-processing

Il processo di pre-processing prevede la pulizia dei dati relativi al consumo energetico e alla produzione fotovoltaica. Il processo consiste nell'identificazione di inconsistenze (outlier, valori costanti e valori non fisicamente ammissibili) e nella ricostruzione di tali valori.

```python
import pandas as pd
from src.energy_analytics import Building


data = pd.DataFrame()  # Dati relativi alla potenza consumata dall'edificio
weather = pd.DataFrame()  # Dati relativi alle condizioni meteorologiche
metadata = {}  # Metadati relativi all'edificio

DU2 = Building(data=data, weather_data=weather, metadata=metadata)
DU2.pre_process()
```