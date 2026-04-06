import numpy as np

from ass1.modeling.turbine import WindTurbine


class WindFarm:
    # Turbine chosen for the wind farm
    turbine: WindTurbine

    def get_output(self, ws_50: np.ndarray) -> np.ndarray:
        """Calculate the output of the windfarm based on the input wind speed at 50m."""

        # This just returns a list of zeros
        return np.zeros_like(ws_50)
