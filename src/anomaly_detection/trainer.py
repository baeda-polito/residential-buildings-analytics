import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class Trainer:
    """
    Classe che si occupa dell'addestramento di un modello di regressione.
    :param model: il modello da addestrare (di default MultiLayerPerceptron)
    :param criterion: la funzione di loss da minimizzare (di default MSELoss)
    :param config: dizionario con i parametri di configurazione dell'addestramento (max_epochs, lr,
    early_stopping_delta, min_epochs)
    """

    def __init__(self, model: nn.Module, criterion: torch.nn, config: dict):
        self.model = model
        self.criterion = criterion
        self.config = config
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.epoch = 0
        self.loss_list_train = []
        self.loss_list_valid = []

    def train(self, train_loader, validation_loader):
        """
        Funzione che addestra il modello per un numero massimo di epoche definito in config. Durante l'addestramento,
        calcola la loss sul train set e sul validation set e stampa il risultato a video ogni 10 epoche. Inoltre,
        implementa l'early stopping, ovvero ferma l'addestramento se la loss validation non migliora di almeno
        early_stopping_delta.
        :param train_loader: dataloader con i dati di addestramento
        :param validation_loader:  dataloader con i dati di validazione
        """
        for epoch in range(self.config['max_epochs']):
            self.model.train()
            batch_loss = 0.0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                y_pred = self.model(batch_X)
                loss = self.criterion(y_pred, batch_y)
                loss.backward()
                self.optimizer.step()
                batch_loss += loss.item()
            batch_loss /= len(train_loader)
            self.loss_list_train.append(batch_loss)

            self.model.eval()
            val_loss = 0.0
            for batch_X, batch_y in validation_loader:
                y_pred = self.model(batch_X)
                loss = self.criterion(y_pred, batch_y)
                val_loss += loss.item()
            val_loss /= len(validation_loader)
            self.loss_list_valid.append(val_loss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch} - Loss train: {batch_loss:.6f} - Loss validation: {val_loss:.6f}")

            # Early stopping
            if epoch > self.config['min_epochs'] and abs(self.loss_list_valid[-2] - self.loss_list_valid[-1]) < \
                    self.config['early_stopping_delta']:
                break

    def evaluate(self, X_test_tensor, y_test, y_scaler):
        """
        Funzione che valuta il modello sul test set. Calcola le metriche di errore MAE, RMSE e R2 tra i valori predetti
        e quelli reali, già denormalizzati.
        :param X_test_tensor: tensor con i dati delle features sul test set
        :param y_test: array con i valori di produzione reali sul test set
        :param y_scaler: lo scaler per la variabile target
        :return y_pred_rescaled: array con i valori di produzione predetti sul test set, già denormalizzati.
        """
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_test_tensor).numpy()
            y_pred_rescaled = y_scaler.inverse_transform(y_pred)
            y_test_rescaled = y_scaler.inverse_transform(y_test)

            mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
            rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
            r2 = r2_score(y_test_rescaled, y_pred_rescaled)
            print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")

        return y_pred_rescaled
