# Pipeline

Il modulo `pipeline` contiene gli script per l'esecuzione delle pipeline di analisi energetica. Le pipeline sono state
sviluppate per l'analisi dei dati relativi al consumo energetico e alla produzione fotovoltaica degli edifici, e sono
state sviluppate per l'analisi delle attività di benchmarking, rilevazione di anomalie e calcolo degli indicatori KPI.

### Struttura del modulo
```plaintext
.
├── anomaly_detection_pv.py
├── benchmarking.py
├── kpi.py
```

Eseguendo tutti gli script presenti nella cartella `pipeline` è possibile eseguire le tre pipeline di analitica descritte
nelle attività, in modalità _offline_. Per l'esecuzione online, fare riferimento alla documentazione di ciascun modulo.
Eseguite le pipeline, i risultati saranno salvati nella cartella `results` e `figures`.
