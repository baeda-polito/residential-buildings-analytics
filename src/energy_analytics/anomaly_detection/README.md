# Identificazione e diagnostica di anomalie energetiche in edifici residenziali

Il modulo `anomaly_detection` contiene le funzioni per implementare il processo di identificazione e diagnostica di anomalie energetiche sulla produzione fotovoltaica e sul consumo energetico di un edificio residenziale.

## Identificazione di anomalie sul consumo energetico



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

```python

```


#### Modalità online

