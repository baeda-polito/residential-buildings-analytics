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

```

####  Anomalie sulla produzione fotovoltaica
```python
import pandas as pd
import json
import os

from settings import PROJECT_ROOT
from src.energy_analytics import Building, Aggregate, run_train, run_evaluation


data_DU1 = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "energy_meter", "7436df46-294b-4c97-bd1b-8aaa3aed97c5.csv"))
metadata_DU1 = json.load(open(os.path.join(PROJECT_ROOT, "data", "metadata", "7436df46-294b-4c97-bd1b-8aaa3aed97c5.json")))

data_DU2 = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "energy_meter", "80c3bedd-8c41-450c-ae52-1864b9ace7aa.csv"))
metadata_DU2 = json.load(open(os.path.join(PROJECT_ROOT, "data", "metadata", "80c3bedd-8c41-450c-ae52-1864b9ace7aa.json")))

data_DU5 = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "energy_meter", "b87be67b-8133-4b7f-a045-c06da08b5416.csv"))
metadata_DU5 = json.load(open(os.path.join(PROJECT_ROOT, "data", "metadata", "b87be67b-8133-4b7f-a045-c06da08b5416.json")))

weather = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "weather", "anguillara.csv"))

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
