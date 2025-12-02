import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error as MAE
from darts import TimeSeries
from darts.models import AutoARIMA, LinearRegressionModel, NBEATSModel, NHiTSModel

from ts_benchmark.models.model_base import ModelBase

logger = logging.getLogger(__name__)


class DynamicSelectionModel(ModelBase):
    """
    DynamicSelection class.
    """

    def __init__(self, **kwargs):
        self.config = kwargs
        self.models = None
        self.x_training_windows = None
        self.y_training = None
        self.window_size = None

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
        
        Parameters:
            similar_windows: Number of similar windows to select for model evaluation
            n_models: Number of models to select from the pool
            window_size: Size of the sliding window for creating training samples
            h: Forecast horizon (number of steps to predict)
        """
        return {"similar_windows": 3, "n_models": 1, "window_size": 30, "h": 30}

    def _create_sliding_windows(self, data: np.ndarray, window_size: int):
        """
        Create sliding windows from time series data.
        
        Args:
            data: Time series data as numpy array
            window_size: Size of each window
            
        Returns:
            X: Array of windows (n_windows, window_size)
            y: Array of target values (n_windows,)
        """
        n_samples = len(data) - window_size
        X = np.array([data[i:i+window_size].flatten() for i in range(n_samples)])
        y = data[window_size:].flatten()
        return X, y
    
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

    def _proxy_ds(self, x_training_windows, y_training, last_window, similar_windows, n_models, pool_of_models, h):
        '''
        1ª Ideia: seleciona os h-step ahead models utilizando a previsão t como um substituto do valor real
        '''
        preds = []
        for _ in range(h):
            selected_models, _ = self._ola_ds_selection(x_training_windows, y_training, last_window, similar_windows, n_models, pool_of_models)
            logger.info(f"SELECTED MODEL: {selected_models}")
            
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
        Train the model with a pool of darts models.
        """
        # logger.info(train_data)
        
        # Create pool of models
        self.pool_of_models = []
        
        # Add AutoARIMA models
        for seasonal in [True, False]:
            for m in range(1, 5):
                try:
                    model = DartsModelWrapper(AutoARIMA(seasonal=seasonal, m=m), train_data)
                    if model.is_fitted:
                        self.pool_of_models.append(model)
                except Exception as e:
                    logger.warning(f"Failed AutoARIMA(seasonal={seasonal}, m={m}): {e}")
        
        # Add LinearRegressionModel
        for lags in range(5, 10):
            try:
                model = DartsModelWrapper(LinearRegressionModel(lags=lags, output_chunk_length=1), train_data)
                if model.is_fitted:
                    self.pool_of_models.append(model)
            except Exception as e:
                logger.warning(f"Failed LinearRegressionModel(lags={lags}): {e}")
        
        # Add NBEATSModel
        try:
            model = DartsModelWrapper(
                NBEATSModel(
                    input_chunk_length=self.config.get("window_size", 30),
                    output_chunk_length=1,
                    n_epochs=10,
                    pl_trainer_kwargs={"enable_progress_bar": False}
                ),
                train_data
            )
            if model.is_fitted:
                self.pool_of_models.append(model)
        except Exception as e:
            logger.warning(f"Failed NBEATSModel: {e}")
        
        # # Add NHiTSModel
        try:
            model = DartsModelWrapper(
                NHiTSModel(
                    input_chunk_length=self.config.get("window_size", 30),
                    output_chunk_length=1,
                    n_epochs=10,
                    pl_trainer_kwargs={"enable_progress_bar": False}
                ),
                train_data
            )
            if model.is_fitted:
                self.pool_of_models.append(model)
        except Exception as e:
            logger.warning(f"Failed NHiTSModel: {e}")
        
        logger.info(f"Total models in pool: {len(self.pool_of_models)}")
        
        # Get window size from config
        self.window_size = self.config.get("window_size", 30)
        logger.info(f"Using window_size: {self.window_size}")
        
        # Prepare training windows using sliding window approach
        train_values = train_data.values.flatten()
        self.x_training_windows, self.y_training = self._create_sliding_windows(
            train_values, self.window_size
        )
        
        logger.info(f"Created {len(self.x_training_windows)} training windows of size {self.window_size}")

        return self

    def forecast(self, horizon: int, series: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Make predictions.
        """
        # Extract the last window_size values from the series as the initial window
        series_values = series.values.flatten()
        if len(series_values) < self.window_size:
            raise ValueError(f"Series length ({len(series_values)}) must be >= window_size ({self.window_size})")
        
        last_window = series_values[-self.window_size:]
        logger.debug(f"Initial window shape: {last_window.shape}, horizon: {horizon}")
        
        preds = self._proxy_ds(
            self.x_training_windows,
            self.y_training,
            last_window,
            self.config.get("similar_windows"),
            self.config.get("n_models"),
            self.pool_of_models,
            horizon,
        )
        return np.array(preds)


class DartsModelWrapper:
    """Wrapper to adapt darts models for dynamic selection."""
    
    def __init__(self, darts_model, train_data: pd.DataFrame):
        self.darts_model = darts_model
        self.model_name = darts_model.__class__.__name__
        self.is_fitted = False
        
        # Store model parameters for display
        self.model_params = self._extract_model_params()
        
        try:
            train_series = TimeSeries.from_dataframe(train_data)
            self.darts_model.fit(train_series)
            self.is_fitted = True
        except Exception as e:
            logger.warning(f"Failed to fit {self.model_name}: {e}")
    
    def _extract_model_params(self):
        """Extract relevant parameters from the darts model."""
        params = {}
        
        # Common parameters to extract
        param_names = ['seasonal', 'm', 'lags', 'output_chunk_length', 'input_chunk_length',
                      'n_epochs', 'batch_size', 'learning_rate']
        
        for param in param_names:
            if hasattr(self.darts_model, param):
                value = getattr(self.darts_model, param)
                # Only include non-None values
                if value is not None:
                    params[param] = value
        
        return params
    
    def predict(self, X):
        """Make predictions on multiple windows."""
        if not self.is_fitted:
            return np.full(X.shape[0], np.nan)
        
        preds = []
        for i in range(X.shape[0]):
            try:
                window_df = pd.DataFrame(X[i].reshape(-1, 1))
                window_series = TimeSeries.from_dataframe(window_df)
                forecast = self.darts_model.predict(n=1, series=window_series)
                preds.append(forecast.values()[0, 0])
            except Exception:
                preds.append(X[i][-1])  # Fallback to last value
        
        return np.array(preds)
    
    def __str__(self):
        # Format parameters as a compact string
            return f"DartsModelWrapper({self.model_name})"
    
    def __repr__(self):
        return self.__str__()

