# Codice sorgente

Il codice sorgente è organizzato in moduli che implementano le funzionalità richieste dalle attività di analisi energetica. I moduli sono organizzati in cartelle all'interno della cartella `src` come mostrato nella struttura del progetto.

### Struttura del progetto
```plaintext
.
├── src                                 # Codice sorgente
│   ├── energy_analytics                # Moduli per l'analisi energetica
│   │   ├── ...
│   │   ├── anomaly_detection
│   │   │   ├── ...
│   │   ├── benchmarking
│   │   │   ├── ...
│   │   ├── kpi
│   │   │   ├── ...
│   │   ├── pre_processing
│   │   │   ├── ...
│   ├── api                             # API per l'interfacciamento con la piattforma DHOMUS e le piattaforme meteo
│   │   ├── smarthome.py                # Interfaccia con la piattaforma DHOMUS
│   │   ├── weather.py                  # Interfaccia con la piattaforma meteo (SOLCAST)
│   │   ├── download_data.py            # Script per il download dei dati  #TODO: da completare
│   ├── pipeline                        # Pipeline di analisi 
│   │   ├── anomaly_detection_pv.py
│   │   ├── benchmarking.py
│   │   ├── kpi.py
├── README.md

```

Per eseguire le pipeline di analisi sviluppate nel pacchetto `energy_analytics` è necessario avere a disposizione i dati relativi al consumo energetico e alla produzione fotovoltaica degli edifici, i metadati degli edifici e i dati relativi alle condizioni meteorologiche. I dati devono essere organizzati in cartelle separate all'interno della cartella `data` come mostrato nella struttura del progetto, e devono strettamente aderire ad una struttura specifica. Per ulteriori dettagli fare riferimento alla documentazione del modulo `data`.

Eseguendo gli script presenti nella cartella `pipeline` è possibile eseguire le tre pipeline di analitica descritte nelle attività, in modalità _offline_. Per l'esecuzione online, fare riferimento alla documentazione di ciascun modulo. Eseguite le pipeline, i risultati saranno salvati nella cartella `results` e `figures`.
