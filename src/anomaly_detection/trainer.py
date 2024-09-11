import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class Trainer:
    def __init__(self, model, criterion, config):
        self.model = model
        self.criterion = criterion
        self.config = config
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.epoch = 0
        self.loss_list_train = []
        self.loss_list_valid = []

    def train(self, train_loader, validation_loader):
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
                print(f"Epoch {epoch} - Loss train: {batch_loss} - Loss validation: {val_loss}")

            # Early stopping
            if epoch > 100 and abs(self.loss_list_valid[-2] - self.loss_list_valid[-1]) < self.config[
                'early_stopping_delta']:
                break

    def evaluate(self, X_test_tensor, y_test, y_scaler):
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
