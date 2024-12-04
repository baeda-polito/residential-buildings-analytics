from src.anomaly_detection.train import train
from src.anomaly_detection.evaluate import evaluate_pv_model
from src.anomaly_detection.anomaly_detection_functions import calculate_threshold
from src.building import load_anguillara, load_garda


def run_train(aggregate: str):
    """
    Esegue la pipeline di addestramento dei modelli di previsione della produzione fotovoltaica per tutti gli edifici
    di un intero aggregato.
    :param aggregate: il nome dell'aggregato ("anguillara" o "garda")
    """

    building_list = []
    if aggregate == "anguillara":
        building_list = load_anguillara()
    elif aggregate == "garda":
        building_list = load_garda()

    for building in building_list:
        if building.building_info["user_type"] != "consumer":
            print(f"Training model for {building.building_info['id']} --- {building.building_info['name']}")
            train(building.building_info["id"])


def run_evaluation(aggregate: str):
    """
    Esegue la pipeline di valutazione dei modelli di previsione della produzione fotovoltaica per tutti gli edifici
    di un intero aggregato.
    :param aggregate: il nome dell'aggregato ("anguillara" o "garda")
    """

    building_list = []
    if aggregate == "anguillara":
        building_list = load_anguillara()
    elif aggregate == "garda":
        building_list = load_garda()

    for building in building_list:
        if building.building_info["user_type"] != "consumer":
            print(f"Evaluating model for {building.building_info['id']} --- {building.building_info['name']}")
            evaluate_pv_model(building.building_info["id"], aggregate)
            calculate_threshold(building.building_info["id"], aggregate)


if __name__ == "__main__":
    run_train("anguillara")
    run_evaluation("anguillara")
