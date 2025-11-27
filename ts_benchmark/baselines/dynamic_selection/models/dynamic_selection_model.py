import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error as MAE

from ts_benchmark.models.model_base import ModelBase


class DynamicSelectionModel(ModelBase):
    """
    DynamicSelection class.
    """

    def __init__(self, **kwargs):
        self.config = kwargs
        self.models = None
        self.x_training_windows = None
        self.y_training = None

    @property
    def model_name(self):
        """
        Returns the name of the model.
        """
        return "DynamicSelection"

    @staticmethod
    def required_hyper_params() -> dict:
        """
        Return the hyperparameters required by DynamicSelection.
        """
        return {"k": 2, "n": 2, "h": 3}

    def _select_k_similar_windows(self, training_windows, test_window, k):
        """"
        Returns the indice of the 'k' most similar windows
        """
        distances = np.linalg.norm(training_windows - test_window, axis=1)
        k_similar_indices = np.argsort(distances)[:k]
        return k_similar_indices

    def _ola_ds_selection(self, x_training_windows, y_training, test_window, k, n, models):
        """
        Returns the 'n' selected models and errors associated to each model
        """
        k_similar_indices = self._select_k_similar_windows(x_training_windows, test_window, k)
        X_roc = x_training_windows[k_similar_indices]
        y_roc = y_training[k_similar_indices]

        errors_models = []
        for model in models:
            pred = model.predict(X_roc)
            mae = float(MAE(pred, y_roc))
            errors_models.append((model, mae))

        selected_models_erros = sorted(errors_models, key=lambda x: x[1])[:n]
        selected_models = [model for model, _ in selected_models_erros]
        selected_maes = [mae for _, mae in selected_models_erros]
        return selected_models, selected_maes

    def _proxy_ds(self, x_training_windows, y_training, last_window, k, n, models, h):
        '''
        1ª Ideia: seleciona os h-step ahead models utilizando a previsão t como um substituto do valor real
        '''
        preds = []
        for _ in range(h):
            selected_models, _ = self._ola_ds_selection(x_training_windows, y_training, last_window, k, n, models)
            prev = []
            for model in selected_models:
                prev.append(model.predict(last_window.reshape(1, -1)))

            prev_avg = np.array(np.ceil(np.sum(prev) / len(prev))).item()
            preds.append(prev_avg)
            last_window = np.roll(last_window, -1)
            last_window[-1] = prev_avg
        return preds

    def forecast_fit(
        self, train_data: pd.DataFrame, *, train_ratio_in_tv: float = 1.0, **kwargs
    ) -> "ModelBase":
        """
        Train the model.
        """
        # This is a placeholder for the actual training logic.
        # The original file does not have a clear training part,
        # so we are initializing the models and data here.
        # Dummy models for demonstration
        self.models = [ModeloTeste(i) for i in range(10)]
        # Dummy training data
        self.x_training_windows = train_data.values[:-1]
        self.y_training = train_data.values[1:, 0]  # Assuming we predict the first column

        return self

    def forecast(self, horizon: int, series: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Make predictions.
        """
        last_window = series.values[-1]
        preds = self._proxy_ds(
            self.x_training_windows,
            self.y_training,
            last_window,
            self.config.get("k"),
            self.config.get("n"),
            self.models,
            horizon,
        )
        return np.array(preds)


class ModeloTeste:
    def __init__(self, bias):
        self.bias = bias

    def predict(self, X):
        """
        Dummy prediction: returns the last values of each array (+ bias)
        """
        preds = []
        for i in range(X.shape[0]):
            pred = X[i][-1] + self.bias
            preds.append(pred)
        return np.array(preds)

    def __str__(self):
        return "Model (" + str(self.bias) + ")"

    def __repr__(self):
        return self.__str__()

