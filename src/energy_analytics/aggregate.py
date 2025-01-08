

class Aggregate:
    """
    Classe che contiene le informazioni utili per un intero aggregato.
    Un aggregato è composto da una lista di edifici
    """

    def __init__(self, buildings: list):
        """
        Costruttore della classe Aggregate
        Args:
            buildings (list): lista di edifici
        """

        self.buildings = buildings
