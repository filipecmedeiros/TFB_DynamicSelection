from ts_benchmark.baselines.dynamic_selection.models.dynamic_selection_model import DynamicSelectionModel
from ts_benchmark.models.model_base import ModelBase


class DynamicSelectionConfig:
    def __init__(self, **kwargs):
        self.params = {
            **kwargs,
        }

    def __getattr__(self, key: str):
        return self.get(key)

    def __getitem__(self, key: str):
        return self.get(key)

    def get(self, key: str, default=None):
        return self.params.get(key, default)


class DynamicSelection(ModelBase):
    """
    DynamicSelection Adapter Class
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.config = DynamicSelectionConfig(**kwargs)
        self.model = DynamicSelectionModel(**self.config.params)

    @property
    def model_name(self):
        return "DynamicSelection"

    def forecast_fit(self, train_data, *, train_ratio_in_tv=1, **kwargs):
        self.model.forecast_fit(train_data, train_ratio_in_tv=train_ratio_in_tv, **kwargs)
        return self

    def forecast(self, horizon, series, **kwargs):
        return self.model.forecast(horizon, series, **kwargs)

    @staticmethod
    def required_hyper_params():
        return DynamicSelectionModel.required_hyper_params()
