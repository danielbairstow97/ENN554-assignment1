import numpy as np
from enum import Enum
from ass1.modeling.turbine import WindTurbine


class ConfigurationOption(Enum):
    NINE = 9
    SIXTEEN = 16
    TWENTY_FIVE = 25


class WindFarm:
    turbine: WindTurbine
    configuration: ConfigurationOption

    def __init__(self, configuration: ConfigurationOption, turbine: WindTurbine):
        self.configuration = configuration
        self.turbine = turbine

    @property
    def n_turbines(self) -> int:
        return self.configuration.value

    @property
    def nameplate_capacity_mw(self) -> float:
        return self.turbine.rated_power_kw * self.n_turbines / 1000.0

    def power_at_50m(self, ws_50: np.ndarray, wd: np.ndarray) -> np.ndarray:
        per_turbine_kw = self.turbine.power_at_50m(ws_50)
        return per_turbine_kw * self.n_turbines