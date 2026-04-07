import numpy as np
from zmq import Enum

from ass1.modeling.turbine import WindTurbine


class ConfigurationOption(Enum):
    NINE = 9
    SIXTEEN = 16
    TWENTY_FIVE = 25


class WindFarm:
    # Turbine chosen for the wind farm
    turbine: WindTurbine
    configuration: ConfigurationOption

    def __init__(self, configuration: ConfigurationOption, turbine: WindTurbine):
        self.configuration = configuration
        self.turbine = turbine

        # Build the WindFarm

    def get_output(self, ws_50: np.ndarray) -> np.ndarray:
        """Calculate the output of the windfarm based on the input wind speed at 50m."""

        # This just returns a list of zeros
        return np.zeros_like(ws_50)

    def nameplate_capacity_mw(self) -> float:
        """Nameplate capacity of the wind farm in MW"""
        return 0.0
