import pandas as pd
import pypsa

from ass1.modeling.farm import WindFarm


class Microgrid:
    def __init__(self, farm: WindFarm):
        self.farm = farm
        n = pypsa.Network()

        # Add network components
        n.add("Load", "demand")

        self.n = n

    def prepare_network(self, site: pd.DataFrame):
        """Prepare network for solving. This requires setting the _t values of loads and wind turbines."""
        # This prepares the timeseries of the network
        self.n.set_snapshots(time)

        # Sets the demand that must be met
        self.n.loads_t.p_set["demand"] = 0.0

        # This has to be incorporated into the network
        windfarm_output = self.farm.get_output(ws_50)

    def solve_network(self) -> float:
        """Solve network, returning battery size"""
        # If prepare_network correctly implemented this will succeed
        self.n.optimize()

        # Extract battery optimised battery size
        return 0.0
