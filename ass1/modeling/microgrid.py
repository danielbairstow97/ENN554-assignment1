import pandas as pd
import pypsa

from ass1.modeling.farm import WindFarm


class Microgrid:
    demand_df: pd.DataFrame
    farm: WindFarm

    n: pypsa.Network | None = None

    def build_network(self):
        """Build network components that will be solved for."""
        n = pypsa.Network()

        n.add("Load", "demand")

        self.n = n

    def prepare_network(self, demand: pd.Series, ws_50: pd.Series):
        """Prepare network for solving. This requires setting the _t values of loads and wind turbines."""
        if self.n is None:
            raise ValueError("Network should be set")

        if not demand.index.equals(ws_50.index):
            raise ValueError("demand and wind speed data should share the same datetime index")

        # This prepares the timeseries of the network
        self.n.set_snapshots(demand.index)

        # Sets the demand that must be met
        self.n.loads_t.p_set["demand"] = 0.0

        # This has to be incorporated into the network
        windfarm_output = self.farm.get_output(ws_50.values)

    def solve_network(self) -> float:
        """Solve network, returning battery size"""
        if self.n is None:
            raise ValueError("Network should be set")

        # If prepare_network correctly implemented this will succeed
        self.n.optimise()

        # Extract battery optimised battery size
        return 0.0
