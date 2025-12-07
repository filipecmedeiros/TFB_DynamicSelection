import logging

import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from darts.models import AutoARIMA, LinearRegressionModel, NBEATSModel, NHiTSModel, RegressionModel, Prophet

from .darts_model_wrapper import DartsModelWrapper

logger = logging.getLogger(__name__)


class ModelPoolFactory:
    """Factory class for creating pools of forecasting models."""
    
    def __init__(self, window_size: int):
        """
        Initialize the model pool factory.
        
        Args:
            window_size: Size of the sliding window for lag-based models
        """
        self.window_size = window_size
    
    def _get_autoarima_configs(self):
        """Get AutoARIMA model configurations."""
        configs = []
        for seasonal in [True, False]:
            for m in range(0, 5):
                configs.append((
                    f"AutoARIMA(seasonal={seasonal}, m={m})",
                    lambda s=seasonal, m_val=m: AutoARIMA(seasonal=s, m=m_val)
                ))
        return configs
    
    def _get_linear_configs(self):
        """Get Linear Regression model configurations."""
        return [(
            "LinearRegressionModel",
            lambda: LinearRegressionModel(
                lags=self.window_size,
                output_chunk_length=1
            )
        )]
    
    def _get_neural_network_configs(self):
        """Get neural network model configurations."""
        return [
            (
                "NBEATSModel",
                lambda: NBEATSModel(
                    input_chunk_length=self.window_size,
                    output_chunk_length=1,
                    n_epochs=10,
                    pl_trainer_kwargs={"enable_progress_bar": False}
                )
            ),
            (
                "NHiTSModel",
                lambda: NHiTSModel(
                    input_chunk_length=self.window_size,
                    output_chunk_length=1,
                    n_epochs=10,
                    pl_trainer_kwargs={"enable_progress_bar": False}
                )
            )
        ]
    
    def _get_svr_configs(self):
        """Get SVR model configurations."""
        configs = []
        for kernel in ['rbf', 'linear', 'poly']:
            configs.append((
                f"SVR(kernel={kernel})",
                lambda k=kernel: RegressionModel(
                    lags=self.window_size,
                    output_chunk_length=1,
                    model=SVR(kernel=k, C=1.0, epsilon=0.1)
                )
            ))
        return configs
    
    def _get_prophet_configs(self):
        """Get Prophet model configurations."""
        configs = []
        for mode in ['additive', 'multiplicative']:
            configs.append((
                f"Prophet(seasonality_mode={mode})",
                lambda m=mode: Prophet(
                    seasonality_mode=m,
                    add_seasonalities=None
                )
            ))
        return configs
    
    def _get_gradient_boosting_configs(self):
        """Get Gradient Boosting model configurations."""
        configs = []
        for n_est in [50, 100, 1000]:
            configs.append((
                f"GradientBoosting(n_estimators={n_est})",
                lambda n=n_est: RegressionModel(
                    lags=self.window_size,
                    output_chunk_length=1,
                    model=GradientBoostingRegressor(
                        n_estimators=n,
                        learning_rate=0.1,
                        max_depth=3,
                        random_state=42
                    )
                )
            ))
        return configs
    
    def get_all_model_configs(self):
        """
        Get all model configurations.
        
        Returns:
            List of tuples (model_name, model_factory)
        """
        configs = []
        configs.extend(self._get_autoarima_configs())
        configs.extend(self._get_linear_configs())
        configs.extend(self._get_neural_network_configs())
        configs.extend(self._get_svr_configs())
        configs.extend(self._get_prophet_configs())
        configs.extend(self._get_gradient_boosting_configs())
        return configs
    
    def create_pool(self, train_data: pd.DataFrame):
        """
        Create a pool of diverse forecasting models.
        
        Args:
            train_data: Training data for fitting models
            
        Returns:
            List of successfully fitted DartsModelWrapper instances
        """
        pool = []
        model_configs = self.get_all_model_configs()
        
        # Fit models and add to pool
        for model_name, model_factory in model_configs:
            try:
                model = DartsModelWrapper(model_factory(), train_data)
                if model.is_fitted:
                    pool.append(model)
            except Exception as e:
                logger.warning(f"Failed {model_name}: {e}")
        
        logger.info(f"Total models in pool: {len(pool)}")
        return pool
