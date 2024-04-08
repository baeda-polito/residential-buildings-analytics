# Processi data-driven per l’individuazione automatica di anomalie energetiche e analisi di benchmarking in edifici residenziali

Questa repository contiene il codice sviluppato per il contratto di ricerca tra BAEDA Lab ed ENEA 2022-24. Gli obiettivi del contratto di ricerca sono:

- Attività 1: ***Identificazione e diagnostica di anomalie energetiche negli edifici***: Questa attività ha lo scopo di definire processi data-driven per l’identificazione automatica e la diagnostica di anomalie energetiche in edifici residenziali sia rispetto ai profili di domanda di energia che rispetto ai profili di produzione dei sistemi di generazione (e.g., applicabile a sistemi di produzione fotovoltaici). Il suddetto processo, per quanto riguarda l’analisi dei dati di consumo energetico che pertiene gli edifici in analisi, sarà finalizzato all’individuazione di anomalie legate a specifici schemi comportamentali dell’utente e condizioni di qualità dell’ambiente interno in esercizio.  Similmente, per quanto concerne l’analisi dei dati di produzione di sistemi di generazione, l’obiettivo è quello di definire un processo che consenta l’individuazione automatica di profili di produzione infrequenti e\o anomali rispetto alle condizioni al contorno che ne hanno determinato sia forma che magnitudo.
- Attività 2: ***Analisi di benchmarking esterno per la caratterizzazione dell’utenza residenziale***: Questa attività ha lo scopo di definire benchmark energetici che consentano di condurre analisi comparative esterne di trend di domanda di energia tra diversi utenti residenziali al fine di abilitare politiche di user engagement e di customer classification e valutare il relativo potenziale di flessibilità e resilienza dell’edificio. Tale processo consentirà di segmentare e classificare gli utenti residenziali a seconda del grado di similarità rispetto a pattern di consumo energetico tipologici estratti per mezzo di analisi non supervisionate e di time series analytics. La suddetta attività consentirà una completa caratterizzazione degli utenti residenziali abilitando la conseguente definizione di indicatori compatti rappresentativi di informazioni su come e quando il consumo di energia nell’edificio cambia nel corso di uno specifico intervallo di tempo (e.g., giorno, settimana) per diversi usi finali quali illuminazione, carichi di spina, riscaldamento e raffrescamento. Il processo che sarà sviluppato coerentemente con gli obiettivi di tale attività consentirà, inoltre, di prevedere per nuovi edifici inclusi nel parco edilizio di interesse il pattern tipologico di consumo energetico atteso, di fatto classificando un utente residenziale sfruttando la conoscenza di un numero limitato di variabili che lo caratterizzano.
- Attività 3: ***Valutazione di indicatori compatti di prestazione***: Questa attività ha lo scopo di definire ed implementare KPIs e analisi e valutazione dello Smart Readiness Indicator (SRI) a livello di singolo utente residenziale. Inoltre sulla base delle risultanze delle attività condotte nelle scorse annualità relativamente all’ analisi critica di KPIs di flessibilità energetica in ambito residenziale, saranno implementati con un approccio data driven metriche in grado di quantificare il livello e la disponibilità all’ implementazione di strategie di gestione della domanda attiva al fine di fornire servizi alla rete e ad altri utenti connessi per mezzo di uno schema di comunità energetica.

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
