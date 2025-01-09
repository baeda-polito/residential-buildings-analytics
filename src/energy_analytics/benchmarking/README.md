# Analisi di benchmarking esterno per la caratterizzazione dell'utenza residenziale

Il modulo `benchmarking` contiene le funzionalità per l'analisi di benchmarking esterno per la caratterizzazione dell'utenza.
Il processo include il clustering basato su indicatori di forma calcolati sui profili di carico e la successiva analisi dei cluster ottenuti.
Il processo, inoltre, prevede anche l'assegnazione di nuovi profili all'interno del cluster il cui medioide è più vicino al profilo di carico analizzato.

Il modulo è organizzato nel seguente modo:
- `clustering.py` contiene le funzioni per implementare il clustering (`run_clustering`).
- `utils.py` contiene le funzioni di supporto per il clustering, come il calcolo degli indicatori di forma.
- `shape_factors.py` contiene le funzioni per il calcolo degli indicatori di forma del carico.
- `assign.py` contiene le funzioni per l'assegnazione dei profili di carico ai cluster a cui appartengono (`assign_cluster`).
- `viz.py` contiene le funzioni per visualizzare i risultati del clustering (profili di carico, analisi di cardinalità, etc.).
- `main.py` contiene la funzione generale per la pipeline di analisi di benchmarking esterno (`run_benchmarking`). La pipeline calcola gli indicatori di forma, esegue il clustering, assegna i profili ai cluster e salva i risultati all'interno delle cartelle `results/benchmarking` e `figures/benchmarking`.

## Utilizzo

Il processo di benchmarking può essere eseguito sia in modalità online che offline.

### Modalità offline

In modalità offline, il processo di benchmarking porta alla definizione dei cluster dei profili di carico e dei suoi medioidi. 
Una volta trovati i cluster, vengono estratte le statistiche sulla cardinalità dei cluster e caratterizzati i singoli edifici in base al cluster di appartenenza.
In questo caso, tutto il processo deve essere eseguito.

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

### Modalità online

In modalità online, il processo di benchmarking consiste nell'assegnazione dei profili di carico di un utente all'interno del cluster con il medioide più vicino al profilo di carico analizzato.

```python
import pandas as pd

from src.energy_analytics.benchmarking import assign_to_nearest_or_anomalous


load_profile = pd.Series()  # 96 valori del carico elettrico
medioids = pd.DataFrame()  # Medioidi dei cluster. DataFrame nx96 (n cluster), con indice il numero del cluster.

cluster = assign_to_nearest_or_anomalous(load_profile, medioids, threshold=3)
```