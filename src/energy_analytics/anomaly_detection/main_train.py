from loguru import logger

from .train import train
from .evaluate import evaluate_pv_model
from .anomaly_detection_functions import calculate_threshold
from ..aggregate import Aggregate


def run_train(aggregate: Aggregate):
    """
    Esegue la pipeline di addestramento dei modelli di previsione della produzione fotovoltaica per tutti gli edifici
    di un intero aggregato.

    Args:
        aggregate (Aggregate): oggetto Aggregate con le informazioni sull'aggregato.

    Returns:
        None
    """
    logger.info("Inizio della pipeline di addestramento dei modelli di previsione della produzione fotovoltaica.")

    for building in aggregate.buildings:
        if building.building_info["user_type"] != "consumer":
            logger.info(f"Training del modello per l'edificio {building.building_info['id']} --- {building.building_info['name']}")
            train(building)


def run_evaluation(aggregate: Aggregate):
    """
    Esegue la pipeline di valutazione dei modelli di previsione della produzione fotovoltaica per tutti gli edifici
    di un intero aggregato.

    Args:
        aggregate (Aggregate): oggetto Aggregate con le informazioni sull'aggregato.

    Returns:
        None
    """
    logger.info("Inizio della pipeline di valutazione dei modelli di previsione della produzione fotovoltaica.")

    for building in aggregate.buildings:
        if building.building_info["user_type"] != "consumer":
            logger.info(f"Valutazione del modello per l'edificio {building.building_info['id']} --- {building.building_info['name']}")
            calculate_threshold(building)
            evaluate_pv_model(building)
