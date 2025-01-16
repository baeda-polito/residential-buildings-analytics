import os


def create_directories(base_path, directories):
    """
    Crea una directory se non esiste.

    Args:
        base_path (str): path della directory base.
        directories (list): lista di directory da creare.

    Returns:
        None
    """
    for directory in directories:
        path = os.path.join(base_path, directory)
        os.makedirs(path, exist_ok=True)
