![](figures/logo.png)  

# Processi data-driven per l’individuazione automatica di anomalie energetiche e analisi di benchmarking in edifici residenziali

Questa repository contiene il codice sviluppato per il _Report Ricerca di Sistema Elettrico - Tecnologie per la penetrazione efficiente del vettore elettrico negli usi finali - LA1.10 Processi data-driven di individuazione automatica di anomalie energetiche e benchmarking per edifici residenziali_ sviluppato in collaborazione tra [BAEDA Lab](https://baeda.polito.it/) ed ENEA. Le attività sviluppate riguardano:

- Attività 1: ***Identificazione e diagnostica di anomalie energetiche negli edifici***: Questa attività ha lo scopo di definire processi data-driven per l’identificazione automatica e la diagnostica di anomalie energetiche in edifici residenziali sia rispetto ai profili di domanda di energia che rispetto ai profili di produzione dei sistemi di generazione (e.g., applicabile a sistemi di produzione fotovoltaici)
- Attività 2: ***Analisi di benchmarking esterno per la caratterizzazione dell’utenza residenziale***: Questa attività ha lo scopo di definire benchmark energetici che consentano di condurre analisi comparative esterne di trend di domanda di energia tra diversi utenti residenziali al fine di abilitare politiche di user engagement e di customer classification e valutare il relativo potenziale di flessibilità e resilienza dell’edificio.
- Attività 3: ***Valutazione di indicatori compatti di prestazione***: Questa attività ha lo scopo di definire ed implementare KPIs e analisi a livello di singolo utente residenziale.

Il progetto utilizza i dati forniti dalla piattaforma DHOMUS e i dati meteo estratti da SOLCAST per sviluppare le metodologie analitiche descritte nelle attività.
In particolare, i test sono stati condotti su un aggregato di edifici residenziali situati a Roma.

### Struttura del progetto
Il progetto è strutturato nei seguenti moduli:
```plaintext
.
├── data                    
│   ├── energy_meter              # Dati relativi a consumo e produzione energetica degli edifici
│   │   ├── ...
│   ├── metadata                  # Metadati degli edifici
│   │   ├── ...
│   ├── weather_data              # Dati meteo per l'aggregato
│   │   ├── ...
├── figures                       # Immagini e grafici
│   ├── anomaly_detection
│   │   ├── ...
│   ├── benchmarking
│   │   ├── ...
│   ├── kpi
│   │   ├── ...
├── results                       # Risultati delle analisi
│   ├── anomaly_detection
│   │   ├── ...
│   ├── benchmarking
│   │   ├── ...
│   ├── kpi
│   │   ├── ...
├── src                           # Codice sorgente
│   ├── energy_analytics          # Moduli per l'analisi energetica
│   │   ├── anomaly_detection
│   │   │   ├── ...
│   │   ├── benchmarking
│   │   │   ├── ...
│   │   ├── kpi
│   │   │   ├── ...
│   ├── api                       # API per l'interfacciamento con la piattforma DHOMUS e le piattaforme meteo
│   ├── pipeline                  # Pipeline di analisi 
├── README.md
```

Per eseguire le pipeline di analisi sviluppate nel pacchetto `energy_analytics` è necessario avere a disposizione i dati relativi al consumo energetico e alla produzione fotovoltaica degli edifici, i metadati degli edifici e i dati relativi alle condizioni meteorologiche. I dati devono essere organizzati in cartelle separate all'interno della cartella `data` come mostrato nella struttura del progetto, e devono strettamente aderire ad una struttura specifica. Per ulteriori dettagli fare riferimento alla documentazione di ciascun modulo.

Eseguendo gli script presenti nella cartella `pipeline` è possibile eseguire le tre pipeline di analitica descritte nelle attività, in modalità _offline_. Per l'esecuzione online, fare riferimento alla documentazione di ciascun modulo. Eseguite le pipeline, i risultati saranno salvati nella cartella `results` e `figures`.
 

###  Configurazione del progetto

- Clonare il progetto da Github, aprendo il terminale ed eseguendo il seguente comando:
    ```bash
  git clone https://github.com/Giudice7/ENEA24
    ```

- Aprire il terminare nella repository python del progetto.

- Controllare la versione python installata.:
    ```bash
  python --version
    ```

- Per non avere problemi di compilazione la versione necessaria è la 3.11. Se si dispone di una versione diversa scaricare la 3.11 dal seguente [link](https://www.python.org/downloads/release/python-3110/).

- Creare un virtual environment:
    ```bash
    python -m venv venv
    ```
- Attivare l'ambiente virtuale:
    ```bash
    venv/Scripts/activate
    ```
- Installare le dipendenze:
    ```bash
    pip install -r requirements.txt
    ```