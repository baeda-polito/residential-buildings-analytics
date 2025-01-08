

class Aggregate:
    """
    Classe che contiene le informazioni utili per un intero aggregato.
    Un aggregato è composto da una lista di edifici
    """

    def __init__(self, name: str, buildings: list):
        """
        Costruttore della classe Aggregate
        Args:
            name (str): nome dell'aggregato
            buildings (list): lista di edifici
        """

        self.name = name
        self.buildings = buildings
