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

In questa sezione vengono forniti degli esempi di utilizzo delle funzioni e dei moduli presenti nel pacchetto. 
In particolare, si mostra come eseguire tutte le pipeline di analisi descritte in precedenza.
Se si desidera eseguire solo una parte dell'analisi, è possibile utilizzare i singoli moduli e le funzioni presenti all'interno di ciascun modulo.
Fare riferimento a ciascun modulo per ulteriori dettagli.

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

### Individuazione automatica di anomalie energetiche

#### Anomalie sul consumo energetico
```python
import pandas as pd
from src.energy_analytics.anomaly_detection import detect_anomaly_power

load_profile = pd.Series()
aggregate_name = ""
user_type = ""

anomaly = detect_anomaly_power(load_profile, aggregate_name, user_type)

if anomaly:
    print("Anomalia rilevata")
else:
    print("Nessuna anomalia rilevata")
```

####  Anomalie sulla produzione fotovoltaica
```python
import pandas as pd

from src.energy_analytics import Building, Aggregate, run_train, run_evaluation


data_DU1 = pd.DataFrame()
metadata_DU1 = {}

data_DU2 = pd.DataFrame()
metadata_DU2 = {}

data_DU5 = pd.DataFrame()
metadata_DU5 = {}

weather = pd.DataFrame()

DU1 = Building(data=data_DU1, weather_data=weather, metadata=metadata_DU1)
DU2 = Building(data=data_DU2, weather_data=weather, metadata=metadata_DU2)
DU5 = Building(data=data_DU5, weather_data=weather, metadata=metadata_DU5)

aggregate = Aggregate(name="anguillara", buildings=[DU1, DU2, DU5])

for building in aggregate.buildings:
    building.pre_process()

run_train(aggregate)
run_evaluation(aggregate)
```

### Analisi di benchmarking esterno
```python
import pandas as pd

from src.energy_analytics import Building, Aggregate, run_benchmarking

data_DU1 = pd.DataFrame()
metadata_DU1 = {}

data_DU2 = pd.DataFrame()
metadata_DU2 = {}

data_DU5 = pd.DataFrame()
metadata_DU5 = {}

weather = pd.DataFrame()

DU1 = Building(data=data_DU1, weather_data=weather, metadata=metadata_DU1)
DU2 = Building(data=data_DU2, weather_data=weather, metadata=metadata_DU2)
DU5 = Building(data=data_DU5, weather_data=weather, metadata=metadata_DU5)

aggregate = Aggregate(name="anguillara", buildings=[DU1, DU2, DU5])

for building in aggregate.buildings:
    building.pre_process()

run_benchmarking(aggregate)
```
### Calcolo dei KPI

```python
import pandas as pd
import os

from settings import PROJECT_ROOT
from src.energy_analytics import Building, Aggregate, run_kpi


data_DU1 = pd.DataFrame()
metadata_DU1 = {}

data_DU2 = pd.DataFrame()
metadata_DU2 = {}

weather = pd.DataFrame()

DU1 = Building(data=data_DU1, weather_data=weather, metadata=metadata_DU1)
DU2 = Building(data=data_DU2, weather_data=weather, metadata=metadata_DU2)

aggregate = Aggregate(name="anguillara", buildings=[DU1, DU2])

for building in aggregate.buildings:
    building.pre_process()

cluster = pd.read_csv(os.path.join(PROJECT_ROOT, "results", "benchmarking", "cluster_anguillara.csv"))

run_kpi(aggregate, cluster)
```

---
