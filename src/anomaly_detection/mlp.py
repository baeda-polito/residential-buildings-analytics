import torch
import torch.nn as nn


class MultiLayerPerceptron(nn.Module):
    """
    Classe che definisce un modello di rete del tipo Multi Layer Perceptron.
    Nel metodo forward, ovvero durante la predizione, si controlla se i primi due elementi del tensore in input, ovvero
    Production e ghi, sono entrambi uguali a zero. In tal caso, si restituisce un tensore di zeri per quelle righe,
    poiché non è possibile predire la produzione in assenza di radiazione solare (molto spesso può succedere di predire
    un valore molto piccolo. In tal caso, si preferisce restituire un valore nullo). Se la condizione non è verificata,
    si effettua il forward pass attraverso il modello.
    """
    def __init__(self, input_size, hidden_layers, output_size):
        super(MultiLayerPerceptron, self).__init__()

        layers = []

        # Input to the first hidden layer
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        layers.append(nn.ReLU())

        # Add intermediate hidden layers
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_layers[-1], output_size))

        # Register all layers as a sequential module
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Check if the first and second elements are zeros
        zero_mask = (x[:, 0] == 0) & (x[:, 1] == 0)

        if zero_mask.any():
            # If both the first and second elements are zero, return a tensor of zeros for those rows
            output = torch.zeros(x.size(0), 1)
            output[~zero_mask] = self.model(x[~zero_mask])  # Only forward pass for rows where the condition is False
        else:
            # If the condition is not met, forward pass through the model
            output = self.model(x)

        # Clip the output to ensure no negative values (clipping to zero)
        output = torch.clamp(output, min=0)

        return output

