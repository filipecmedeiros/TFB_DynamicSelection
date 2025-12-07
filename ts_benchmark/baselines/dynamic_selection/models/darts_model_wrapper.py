import logging

import numpy as np
import pandas as pd
from darts import TimeSeries

logger = logging.getLogger(__name__)


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
