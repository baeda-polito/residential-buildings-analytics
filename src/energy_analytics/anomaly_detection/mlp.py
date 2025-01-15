import torch.nn as nn


class MultiLayerPerceptron(nn.Module):
    """
    Classe che definisce un modello di rete del tipo Multi Layer Perceptron.
    """

    def __init__(self, input_size, hidden_layers, output_size):
        super(MultiLayerPerceptron, self).__init__()

        # Input to the first hidden layer
        layers = [nn.Linear(input_size, hidden_layers[0]), nn.ReLU()]

        # Add intermediate hidden layers
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_layers[-1], output_size))

        # Register all layers as a sequential module
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        output = self.model(x)
        return output
