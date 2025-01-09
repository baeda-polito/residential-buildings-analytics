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

```python
# TODO
```