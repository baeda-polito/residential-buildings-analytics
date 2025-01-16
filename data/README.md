# Data

Questo modulo contiene i dati necessari all'esecuzione della pipeline.
Per ottenere i dati è necessario eseguire lo script `download_data.py` presente nel modulo `src/api`, il quale scarica i dati degli energy meter e degli edifici e i dati meteo. Tuttavia i metadati devono essere caricati manualmente nella cartella `metadata`.

La struttura della cartella `data` è la seguente:

```plaintext
.
├── energy_meter              # Dati relativi a consumo e produzione energetica degli edifici
│   ├── id1.csv
│   ├── ...
├── metadata                  # Metadati degli edifici
│   ├── id1.json
│   ├── ...
├── weather_data              # Dati meteo per l'aggregato
│   ├── aggregate.csv
```
La formattazione necessaria è la seguente:

- `energy_meter`: i dati devono essere campionati ogni 15 minuti in formato csv con le seguenti colonne:
    - `timestamp`: data e ora della rilevazione
    - `power`: potenza attiva in W
    - `productionPower`: potenza prodotta in W

    Un esempio di file è il seguente:
    ```csv
    timestamp,power,productionPower
    2021-01-01 00:00:00,1000,0
    2021-01-01 00:15:00,1000,0
    2021-01-01 00:30:00,1000,0
    ```
- `metadata`: i metadati degli edifici devono essere in formato json.
I metadati necessari sono:
  - `id`: identificativo univoco dell'edificio
  - `name`: nome dell'edificio
  - `user_type`: tipologia di utente (producer, consumer, prosumer)
  - `persons`: numero di persone
  - `occupancy`: orari di occupazione dell'edificio
  - `rated_power`: potenza nominale dell'edificio
  - `surface`: superficie dell'edificio
  - `coordinates`: coordinate geografiche dell'edificio
  
  Un esempio di file è il seguente:
    ```json
    
    {
      "id": "3d956901-f5ea-4094-9c85-333cc68183d4",
      "name": "DU_9",
      "user_type": "producer",
      "persons": 1,
      "occupancy": {
        "8-13": 1,
        "13-19": 0,
        "19-24": 0,
        "24-8": 1
      },
      "rated_power": 3,
      "tariff": "monorary",
      "surface": 156,
      "coordinates": [
        12.2830132,
        42.0837101
      ],
      "ac": {
        "n_ac": 0,
        "type": null
      },
      "pv": {
        "rated_power": 3,
        "tilt": 30,
        "azimuth": 40,
        "storage": null
      },
      "retrofit": 0
    }
    ```

- `weather_data`: i dati meteo devono essere campionati ogni 15 minuti in formato csv con le seguenti colonne:
    - `timestamp`: data e ora della rilevazione
    - `air_temp`: temperatura dell'aria in °C
    - `dni`: radiazione diretta normale in W/m^2
    - `ghi`: radiazione globale orizzontale in W/m^2

    Il nome del file csv deve essere lo stesso dell'aggregato (e.g. `anguillara.csv`).
    Un esempio di file è il seguente:
    ```csv
    timestamp,air_temp,dni,ghi
    2021-01-01 00:00:00,10,0,0
    2021-01-01 00:15:00,10,0,0
    2021-01-01 00:30:00,10,0,0
    ```