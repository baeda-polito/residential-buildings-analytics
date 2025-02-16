# Valutazione di indicatori compatti di prestazione

Il modulo `kpi` contiene le funzionalità per il calcolo degli indicatori di prestazione energetica. Gli indicatori sono calcolati a partire dai dati di potenza consumata e prodotta dall'edificio.

Sono disponibili tre tipologie di indicatori:
- *Prestazione energetica*
- *Flessibilità energetica*
- *Utilizzo della fonte rinnovabile*

La lista degli indicatori per ogni categoria sono i seguenti:

**Prestazione energetica**

![](static/kpi_prestazione_energetica.png)

**Flessibilità energetica**

![](static/kpi_flessibilità.png)

**Utilizzo della fonte rinnovabile**

![](static/kpi_fonte_rinnovabile.png)

Il modulo è organizzato nel seguente modo:
- `kpis.py` contiene le funzioni per il calcolo degli indicatori di prestazione energetica.
- `utils.py` contiene le funzioni di supporto per il calcolo degli indicatori.
- `calculate.py` contiene le funzioni per il calcolo di ognuna delle categorie di indicatori, sia su base giornaliera che aggregata (in base alla tipologia di indicatore).
- `viz.py` contiene le funzioni per visualizzare gli indicatori (boxplot, heatmaps e radar plot).
- `config.json` contiene il dizionario di configurazione per ogni indicatore, utile per formattare correttamente i risultati.
- `main.py` contiene la funzione generale per la pipeline di calcolo degli indicatori per un intero aggregato (`run_kpi`). La pipeline calcola tutti gli indicatori, salva i risultati e le visualizzazioni all'interno delle cartelle `results/kpi` e `figures/kpi`.

## Utilizzo

### Calcolo dei KPI per l'intero aggregato per un determinato periodo di tempo

Se si vuole calcolare gli indicatori per tutti gli edifici di un intero aggregato, per esempio per funzioni di reportistica, è possibile eseguire direttamente tutta la pipeline.

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

cluster = pd.read_csv(os.path.join(PROJECT_ROOT, "results", "benchmarking", "cluster_anguillara_assigned.csv"))

run_kpi(aggregate, cluster)
```

### Calcolo di alcuni KPI per un singolo edificio

Se si vuole calcolare gli indicatori per un singolo edificio, è possibile utilizzare le funzioni di calcolo direttamente.

```python
import pandas as pd

from src.energy_analytics.kpi.kpis import eui

data = pd.DataFrame()  # Dataframe con le colonne Load e timestamp
surface = 100  # Superficie dell'edificio in m^2

EUI = eui(data, surface)
```

È anche possibile calcolare un'intera categoria di KPIs.

```python
import pandas as pd
import os

from settings import PROJECT_ROOT
from src.energy_analytics import Building
from src.energy_analytics.kpi.calculate import calculate_kpi_load

data = pd.DataFrame()
weather = pd.DataFrame()
metadata = {}

building = Building(data=data, weather_data=weather, metadata=metadata)

cluster = pd.read_csv(os.path.join(PROJECT_ROOT, "results", "benchmarking", "cluster_anguillara_assigned.csv"))

kpis_load = calculate_kpi_load(building, cluster)
```
